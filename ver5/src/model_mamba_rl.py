"""Efficient multi-agent, LoRA-adapted selective-state recommender.

The expensive pretrained Mamba is used once to cache item semantics.  Online
training and ranking then use separate selective recurrences for the short and
preference agents. Each agent owns a disjoint LoRA bank.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.model import LightGCN


class LoRAUpdate(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Linear(input_dim, rank, bias=False)
        self.b = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(self.dropout(inputs))) * self.scale


class FullRankUpdate(nn.Module):
    """Dense residual update used when LoRA is disabled."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Match LoRA's initial no-op behaviour for a fair/stable switch.
        nn.init.zeros_(self.projection.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.dropout(inputs))


def adaptation_update(
    input_dim: int, output_dim: int, rank: int, alpha: float,
    dropout: float, enable_lora: bool,
) -> nn.Module:
    if enable_lora:
        return LoRAUpdate(input_dim, output_dim, rank, alpha, dropout)
    return FullRankUpdate(input_dim, output_dim, dropout)


class SelectiveMambaAgent(nn.Module):
    """Mamba-style diagonal selective SSM with an agent-owned LoRA delta."""

    def __init__(
        self, dim: int, rank: int, alpha: float, dropout: float,
        initial_timescale: float, enable_lora: bool,
        candidate_scoring: bool = True,
    ):
        super().__init__()
        self.candidate_lora = adaptation_update(dim, dim, rank, alpha, dropout, enable_lora)
        self.delta_lora = adaptation_update(dim, dim, rank, alpha, dropout, enable_lora)
        self.gate_lora = adaptation_update(dim, dim, rank, alpha, dropout, enable_lora)
        self.output_lora = adaptation_update(dim, dim, rank, alpha, dropout, enable_lora)
        self.delta_bias = nn.Parameter(torch.full((dim,), math.log(math.expm1(initial_timescale))))
        self.log_temperature = nn.Parameter(torch.tensor(0.0)) if candidate_scoring else None

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, width, dim = sequence.shape
        state = sequence.new_zeros(batch, dim)
        output = state
        for index in range(width):
            inputs = sequence[:, index]
            delta = F.softplus(self.delta_bias + self.delta_lora(inputs)).clamp(max=12.0)
            decay = torch.exp(-delta)
            proposal = torch.tanh(inputs + self.candidate_lora(inputs))
            updated = decay * state + (1.0 - decay) * proposal
            active = (index < lengths).unsqueeze(1)
            state = torch.where(active, updated, state)
            gate = torch.sigmoid(self.gate_lora(inputs))
            current = gate * state + (1.0 - gate) * inputs
            output = torch.where(active, current, output)
        return F.normalize(output + self.output_lora(output), dim=-1)

    def logits(self, state: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        if self.log_temperature is None:
            raise RuntimeError("This selective-state encoder does not score candidates directly")
        temperature = self.log_temperature.exp().clamp(max=20.0)
        if candidate_vectors.ndim == 2:
            return state @ candidate_vectors.T * temperature
        return torch.einsum("bd,bcd->bc", state, candidate_vectors) * temperature


class CoordinatorAgent(nn.Module):
    def __init__(
        self, dim: int, rank: int, alpha: float, dropout: float,
        preference_score_weight: float, enable_lora: bool,
    ):
        super().__init__()
        self.state_lora = adaptation_update(dim * 2, dim, rank, alpha, dropout, enable_lora)
        self.mix_lora = adaptation_update(dim * 2, 2, rank, alpha, dropout, enable_lora)
        self.score_lora = adaptation_update(dim, dim, rank, alpha, dropout, enable_lora)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        bounded_weight = min(max(preference_score_weight, 1e-4), 1.0 - 1e-4)
        self.preference_score_logit = nn.Parameter(
            torch.tensor(math.log(bounded_weight / (1.0 - bounded_weight)))
        )
        self.preference_context_gate = nn.Linear(2, 1)
        nn.init.zeros_(self.preference_context_gate.weight)
        nn.init.zeros_(self.preference_context_gate.bias)

    def forward(
        self, short_state: torch.Tensor, preference_state: torch.Tensor,
        preference_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((short_state, preference_state), dim=-1)
        preference_bias = (
            self.preference_score_logit
            + self.preference_context_gate(preference_context).squeeze(-1)
        )
        mix_logits = self.mix_lora(combined) + torch.stack(
            (torch.zeros_like(preference_bias), preference_bias), dim=-1
        )
        weights = torch.softmax(mix_logits, dim=-1)
        base = weights[:, :1] * short_state + weights[:, 1:] * preference_state
        state = F.normalize(base + self.state_lora(combined), dim=-1)
        return state, weights

    def logits(self, state: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        adapted = F.normalize(state + self.score_lora(state), dim=-1)
        temperature = self.log_temperature.exp().clamp(max=20.0)
        if candidate_vectors.ndim == 2:
            return adapted @ candidate_vectors.T * temperature
        return torch.einsum("bd,bcd->bc", adapted, candidate_vectors) * temperature


class PreferenceTransitionAgent(nn.Module):
    """Discover soft preference prototypes and predict the next preference state."""

    def __init__(
        self, dim: int, preference_count: int,
        temperature: float, rank: int, alpha: float,
        dropout: float, enable_lora: bool,
    ):
        super().__init__()
        self.preference_count = preference_count
        self.assignment_temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(preference_count, dim))
        nn.init.orthogonal_(self.prototypes)
        # A separate Mamba-style selective recurrence owns the preference LoRA bank.
        self.transition_encoder = SelectiveMambaAgent(
            dim, rank, alpha, dropout, initial_timescale=0.2,
            enable_lora=enable_lora, candidate_scoring=False,
        )
        self.next_head = nn.Linear(dim, preference_count)
        self.change_head = nn.Linear(dim + preference_count, 1)
        self.state_lora = adaptation_update(
            dim, dim, rank, alpha, dropout, enable_lora
        )
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def assignments(self, vectors: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = torch.einsum("...d,kd->...k", F.normalize(vectors, dim=-1), prototypes)
        return torch.softmax(similarities / self.assignment_temperature, dim=-1)

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor):
        preference_sequence = self.assignments(sequence)
        prototype_sequence = preference_sequence @ F.normalize(self.prototypes, dim=-1)
        last = self.transition_encoder.encode(prototype_sequence, lengths)
        rows = torch.arange(sequence.size(0), device=sequence.device)
        last_index = lengths.clamp_min(1) - 1
        current = preference_sequence[rows, last_index]
        predicted = torch.softmax(self.next_head(last), dim=-1)
        change_logit = self.change_head(torch.cat((last, current), dim=-1)).squeeze(-1)
        change_probability = torch.sigmoid(change_logit)
        preference_state = predicted @ F.normalize(self.prototypes, dim=-1)
        preference_state = F.normalize(preference_state + self.state_lora(preference_state), dim=-1)
        return preference_state, current, predicted, change_logit, change_probability

    def logits(self, state: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(max=20.0)
        if candidate_vectors.ndim == 2:
            return state @ candidate_vectors.T * temperature
        return torch.einsum("bd,bcd->bc", state, candidate_vectors) * temperature


class MultiAgentMambaRecommender(nn.Module):
    """Short and full-history preference Mamba agents with separate LoRA banks."""

    def __init__(
        self,
        item_features: torch.Tensor,
        dim: int = 128,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        short_window: int = 10,
        graph_edges: torch.Tensor | None = None,
        graph_users: int | None = None,
        use_graph_embeddings: bool = True,
        preference_count: int = 64,
        preference_temperature: float = 0.2,
        preference_score_weight: float = 0.2,
        enable_lora: bool = True,
        use_short: bool = True,
        use_preference: bool = True,
        use_coordinator: bool = True,
    ):
        super().__init__()
        if not (use_short or use_preference or use_graph_embeddings):
            raise ValueError("Enable at least one agent or LightGCN")
        self.active_agent_count = sum((use_short, use_preference))
        self.graph_only = self.active_agent_count == 0
        self.use_coordinator = use_coordinator and self.active_agent_count > 1
        self.use_short, self.use_preference = use_short, use_preference
        self.preference_score_weight = preference_score_weight
        if item_features.ndim != 2:
            raise ValueError("item_features must have shape [items, feature_dim]")
        if use_graph_embeddings and (graph_edges is None or graph_users is None):
            raise ValueError("graph_edges and graph_users are required when graph embeddings are enabled")
        self.register_buffer("item_features", item_features, persistent=False)
        self.item_projection = nn.Linear(item_features.size(1), dim, bias=False)
        nn.init.xavier_uniform_(self.item_projection.weight)
        self.use_graph_embeddings = use_graph_embeddings
        self._show_graph_progress = use_graph_embeddings
        if use_graph_embeddings:
            self.graph = LightGCN(graph_users, item_features.size(0), dim)
            self.register_buffer("graph_edges", graph_edges, persistent=False)
        self.enable_lora = enable_lora
        self.short_agent = SelectiveMambaAgent(
            dim, lora_rank, lora_alpha, lora_dropout,
            initial_timescale=0.5, enable_lora=enable_lora,
        )
        self.preference_agent = PreferenceTransitionAgent(
            dim, preference_count, preference_temperature,
            lora_rank, lora_alpha, lora_dropout, enable_lora,
        )
        self.coordinator = CoordinatorAgent(
            dim, lora_rank, lora_alpha, lora_dropout, preference_score_weight,
            enable_lora,
        ) if self.use_coordinator else None
        self.short_window = short_window

    @property
    def num_items(self) -> int:
        return self.item_features.size(0)

    @property
    def adaptation_mode(self) -> str:
        """Name the two compatible update modes used by the launch scripts."""
        return "graph_only" if self.graph_only else "multi_lora" if self.enable_lora else "full_rank"

    def adapter_routes(self) -> dict[str, tuple[str, ...]]:
        """Expose the disjoint adapter bank selected by each agent forward path."""
        routes = {}
        if self.use_short:
            routes["short"] = ("candidate_lora", "delta_lora", "gate_lora", "output_lora")
        if self.use_preference:
            routes["preference"] = (
                "transition_encoder.candidate_lora", "transition_encoder.delta_lora",
                "transition_encoder.gate_lora", "transition_encoder.output_lora",
                "state_lora",
            )
        if self.use_coordinator:
            routes["coordinator"] = ("state_lora", "mix_lora", "score_lora")
        return routes

    def place_devices(self, main_device: str, graph_device: str | None = None):
        """Place the dense agents and LightGCN on separate devices when requested."""
        main = torch.device(main_device)
        self.item_features = self.item_features.to(main)
        for module in (
            self.item_projection,
            self.short_agent,
            self.preference_agent,
            self.coordinator,
        ):
            if module is not None:
                module.to(main)
        if self.use_graph_embeddings:
            graph = torch.device(graph_device or main_device)
            self.graph.to(graph)
            self.graph_edges = self.graph_edges.to(graph)
            self._graph_device = graph
        else:
            self._graph_device = None
        return self

    def graph_item_vectors(self) -> torch.Tensor | None:
        """Propagate the graph once so all item lookups in a batch can share it."""
        if not self.use_graph_embeddings:
            return None
        if self.graph_only:
            users, items = self.graph(self.graph_edges)
            self._graph_user_vectors = users
            return items
        if self._show_graph_progress:
            with tqdm(
                total=self.graph.layers + 1,
                desc="GCN item embeddings (initial pass)",
                unit="layer",
                leave=True,
                dynamic_ncols=True,
            ) as progress:
                _, graph_items = self.graph(self.graph_edges, progress=progress)
            self._show_graph_progress = False
            return graph_items
        return self.graph(self.graph_edges)[1]

    def project_ids(
        self, item_ids: torch.Tensor, graph_items: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.graph_only:
            if graph_items is None:
                graph_items = self.graph_item_vectors()
            return graph_items[item_ids.to(graph_items.device)].to(self.item_features.device)
        features = self.item_features[item_ids]
        projected = self.item_projection(features.to(self.item_projection.weight.dtype))
        projected = F.normalize(projected, dim=-1)
        if not self.use_graph_embeddings:
            return projected
        if graph_items is None:
            graph_items = self.graph_item_vectors()
        assert graph_items is not None
        graph_ids = item_ids.to(graph_items.device)
        graph_projected = F.normalize(graph_items[graph_ids], dim=-1).to(
            projected.device, non_blocking=True
        )
        return F.normalize(projected + graph_projected, dim=-1)

    def project_all(self, graph_items: torch.Tensor | None = None) -> torch.Tensor:
        return self.project_ids(
            torch.arange(self.num_items, device=self.item_features.device), graph_items
        )

    @staticmethod
    def _short_histories(histories: torch.Tensor, lengths: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        short_lengths = lengths.clamp(max=window)
        positions = torch.arange(window, device=histories.device).unsqueeze(0)
        starts = (lengths - short_lengths).unsqueeze(1)
        indices = starts + positions
        indices = indices.clamp(min=0, max=max(histories.size(1) - 1, 0))
        gathered = histories.gather(1, indices)
        gathered = torch.where(positions < short_lengths.unsqueeze(1), gathered, torch.zeros_like(gathered))
        return gathered, short_lengths

    def encode_states(
        self, histories: torch.Tensor, lengths: torch.Tensor,
        graph_items: torch.Tensor | None = None,
        user_ids: torch.Tensor | None = None,
    ):
        if self.use_graph_embeddings and graph_items is None:
            graph_items = self.graph_item_vectors()
        if self.graph_only:
            if user_ids is None:
                raise ValueError("LightGCN-only scoring requires user_ids")
            state = self._graph_user_vectors[user_ids.to(self._graph_user_vectors.device)].to(histories.device)
            zero = torch.zeros_like(state)
            preferences = zero.new_zeros((state.size(0), self.preference_agent.preference_count))
            change = zero.new_zeros(state.size(0))
            return (zero, state, zero, zero.new_zeros((state.size(0), 2)),
                    preferences, preferences, change, change)
        # Preference always reads the full supplied history. Only a short-only
        # model may discard the old prefix before projecting item vectors.
        if not self.use_preference:
            histories, lengths = self._short_histories(
                histories, lengths, min(self.short_window, histories.size(1))
            )
        history_sequence = self.project_ids(histories, graph_items)
        zero = history_sequence.new_zeros((histories.size(0), history_sequence.size(-1)))
        if self.use_preference:
            (preference_state, current_preference, predicted_preference,
             change_logit, change_probability) = self.preference_agent.encode(history_sequence, lengths)
        else:
            preference_state = zero
            current_preference = zero.new_zeros((histories.size(0), self.preference_agent.preference_count))
            predicted_preference = current_preference
            change_logit = change_probability = zero.new_zeros(histories.size(0))
        if self.use_short:
            short_ids, short_lengths = self._short_histories(histories, lengths, min(self.short_window, histories.size(1)))
            short_sequence = self.project_ids(short_ids, graph_items)
            short_state = self.short_agent.encode(short_sequence, short_lengths)
        else:
            short_state = zero
        if self.use_coordinator:
            entropy = -(
                predicted_preference.clamp_min(1e-8)
                * predicted_preference.clamp_min(1e-8).log()
            ).sum(-1) / math.log(predicted_preference.size(-1))
            preference_context = torch.stack((change_probability, 1.0 - entropy), dim=-1)
            coordinator_state, weights = self.coordinator(
                short_state, preference_state, preference_context
            )
        else:
            if self.use_short and self.use_preference:
                preference_weight = self.preference_score_weight
                coordinator_state = F.normalize(
                    (1.0 - preference_weight) * short_state + preference_weight * preference_state,
                    dim=-1,
                )
                weights = zero.new_tensor([1.0 - preference_weight, preference_weight]).expand(histories.size(0), -1)
            else:
                coordinator_state = short_state if self.use_short else preference_state
                weights = zero.new_tensor([int(self.use_short), int(self.use_preference)]).expand(histories.size(0), -1)
        return (
            short_state, coordinator_state, preference_state, weights,
            current_preference, predicted_preference, change_logit, change_probability,
        )

    def logits_from_states(self, states, candidate_vectors: torch.Tensor):
        (
            short_state, coordinator_state, preference_state, weights,
            current_preference, predicted_preference, change_logit, change_probability,
        ) = states
        if self.use_coordinator:
            coordinator_logits = self.coordinator.logits(coordinator_state, candidate_vectors)
        elif candidate_vectors.ndim == 2:
            coordinator_logits = coordinator_state @ candidate_vectors.T
        else:
            coordinator_logits = torch.einsum("bd,bcd->bc", coordinator_state, candidate_vectors)
        zero_logits = torch.zeros_like(coordinator_logits)
        short_logits = self.short_agent.logits(short_state, candidate_vectors) if self.use_short else zero_logits
        preference_logits = self.preference_agent.logits(preference_state, candidate_vectors) if self.use_preference else zero_logits
        if self.use_preference:
            preference_entropy = -(
                predicted_preference.clamp_min(1e-8)
                * predicted_preference.clamp_min(1e-8).log()
            ).sum(-1)
            preference_entropy = preference_entropy / math.log(predicted_preference.size(-1))
            preference_weight = weights[:, 1:2]
        else:
            preference_weight = coordinator_logits.new_zeros((coordinator_logits.size(0), 1))
            preference_entropy = coordinator_logits.new_zeros(coordinator_logits.size(0))
        final_logits = (
            coordinator_logits + weights[:, :1] * short_logits
            + weights[:, 1:] * preference_logits
        )
        if not self.use_coordinator and not self.graph_only:
            if self.use_short and self.use_preference:
                final_logits = weights[:, :1] * short_logits + weights[:, 1:] * preference_logits
            else:
                final_logits = short_logits if self.use_short else preference_logits
        return {
            "short": short_logits,
            "preference": preference_logits,
            "coordinator": final_logits,
            "states": (short_state, coordinator_state, preference_state),
            "weights": weights,
            "preference_current": current_preference,
            "preference_next": predicted_preference,
            "preference_change_logit": change_logit,
            "preference_change": change_probability,
            "preference_weight": preference_weight,
            "preference_uncertainty": preference_entropy,
        }

    def preference_targets(self, item_vectors: torch.Tensor) -> torch.Tensor:
        return self.preference_agent.assignments(item_vectors)

    def forward(self, histories: torch.Tensor, lengths: torch.Tensor, candidates: torch.Tensor, user_ids=None):
        graph_items = self.graph_item_vectors()
        states = self.encode_states(histories, lengths, graph_items, user_ids=user_ids)
        candidate_vectors = self.project_ids(candidates, graph_items)
        return self.logits_from_states(states, candidate_vectors)

    def full_catalog_scores(
        self, histories: torch.Tensor, lengths: torch.Tensor,
        item_vectors: torch.Tensor | None = None,
        graph_items: torch.Tensor | None = None,
        user_ids: torch.Tensor | None = None,
    ):
        if self.use_graph_embeddings and graph_items is None:
            graph_items = self.graph_item_vectors()
        states = self.encode_states(histories, lengths, graph_items, user_ids=user_ids)
        items = self.project_all(graph_items) if item_vectors is None else item_vectors
        return self.logits_from_states(states, items)

    def set_stage(self, stage: str) -> None:
        if stage not in {"specialists", "joint", "all_dl"}:
            raise ValueError(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        graph_modules = (self.graph,) if self.use_graph_embeddings else ()
        modules = (
            (self.short_agent, self.preference_agent, self.item_projection, *graph_modules)
            if stage == "specialists"
            else (self.short_agent, self.preference_agent, self.coordinator, self.item_projection, *graph_modules)
        )
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(True)

        for name in ("short", "preference"):
            if not getattr(self, f"use_{name}"):
                getattr(self, f"{name}_agent").requires_grad_(False)
        if self.graph_only:
            self.item_projection.requires_grad_(False)
    def agent_parameter_counts(self) -> dict[str, int]:
        return {
            "short": sum(p.numel() for p in self.short_agent.parameters()),
            "preference": sum(p.numel() for p in self.preference_agent.parameters()),
            "coordinator": sum(p.numel() for p in self.coordinator.parameters()) if self.coordinator is not None else 0,
            "shared_projection": sum(p.numel() for p in self.item_projection.parameters()),
            "graph": sum(p.numel() for p in self.graph.parameters()) if self.use_graph_embeddings else 0,
        }
