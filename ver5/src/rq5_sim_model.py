"""SLC control for RQ5 similarity cohorts, without changing the existing trainer.

The trainer's second-specialist ('preference') slot is used as a compatibility
alias for LONG. There are no learned prototypes, preference heads or preference
auxiliary losses. States[2] / weights[:,1] refer to the item-level long agent.
The coordinator architecture is shared with SPC; its preference-specific context
is held constant at (rho=0, certainty=0), because SLC has no preference predictor.
"""
import torch
from torch import nn
from src.model_mamba_rl import MultiAgentMambaRecommender, SelectiveMambaAgent


class LongItemAgent(nn.Module):
    def __init__(self, dim, rank, alpha, dropout, enable_lora, reporting_count):
        super().__init__()
        self.preference_count = reporting_count  # Uniform diagnostic placeholders only.
        self.encoder = SelectiveMambaAgent(dim, rank, alpha, dropout,
            initial_timescale=.08, enable_lora=enable_lora)

    def encode(self, sequence, lengths):
        state = self.encoder.encode(sequence, lengths)
        uniform = state.new_full((len(state), self.preference_count), 1 / self.preference_count)
        zero = state.new_zeros(len(state))
        return state, uniform, uniform, zero, zero

    def logits(self, state, candidates):
        return self.encoder.logits(state, candidates)


class SLCRecommender(MultiAgentMambaRecommender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.item_projection.out_features
        # The constructor defaults are explicitly matched to the parent's defaults.
        self.preference_agent = LongItemAgent(dim, kwargs.get('lora_rank', 8),
            kwargs.get('lora_alpha', 16.), kwargs.get('lora_dropout', .05),
            self.enable_lora, kwargs.get('preference_count', 64))

    def encode_states(self, histories, lengths, graph_items=None, user_ids=None):
        if self.use_graph_embeddings and graph_items is None:
            graph_items = self.graph_item_vectors()
        sequence = self.project_ids(histories, graph_items)
        long_state, uniform, predicted, change_logit, change = self.long_agent.encode(sequence, lengths)
        short_ids, short_lengths = self._short_histories(
            histories, lengths, min(self.short_window, histories.size(1)))
        short_state = self.short_agent.encode(self.project_ids(short_ids, graph_items), short_lengths)
        # No learned preference or transition signal is supplied to SLC.
        context = long_state.new_zeros((len(histories), 2))
        coordinator, weights = self.coordinator(short_state, long_state, context)
        return (short_state, coordinator, long_state, weights,
                uniform, predicted, change_logit, change)

    @property
    def long_agent(self):
        return self.preference_agent

    def adapter_routes(self):
        routes = super().adapter_routes()
        routes.pop('preference', None)
        routes['long'] = ('encoder.candidate_lora', 'encoder.delta_lora',
                          'encoder.gate_lora', 'encoder.output_lora')
        return routes

    def agent_parameter_counts(self):
        counts = super().agent_parameter_counts()
        counts['long'] = counts.pop('preference')
        return counts
