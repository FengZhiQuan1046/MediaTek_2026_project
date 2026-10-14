#!/usr/bin/env python3
"""Generate memory_latency_RIBS.pdf beside this script.

Dependencies: numpy, matplotlib.
Latency values are 1000 / full-catalog throughput (ms/user),
not independently timed forward passes. The x-axis uses a declared
piecewise-linear scale. Font source: Microsoft Core Fonts times32.exe,
https://downloads.sourceforge.net/corefonts/times32.exe.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import font_manager

# Load local Times New Roman fonts, or use the installed font on this machine.
_base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if (_base_dir / "MediaTek_2026_project/ver5/overhead").is_dir():
    _base_dir = _base_dir / "MediaTek_2026_project/ver5/overhead"
for _font_file in (_base_dir / "fonts/TimesNewRoman").glob("*.TTF"):
    font_manager.fontManager.addfont(str(_font_file))
font_manager.findfont("Times New Roman", fallback_to_default=False)
# Match preliminary/aggregated/plot_aggregated.py typography and axes.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "axes.edgecolor": "#666666",
    "grid.color": "#D8DCE1", "grid.linewidth": .5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.unicode_minus": False,
})

# The preliminary palette, extended for eight methods; shared across both panels.
method_colors = {
    "BERT4Rec": "#0072B2", "EAGER": "#E69F00",
    "Mamba4Rec": "#009E73", "ReSID": "#CC79A7",
    "SIGMA": "#56B4E9", "LLM-SRec": "#666666",
    "LLMRec": "#000000", "Ours": "#D55E00",
}


# ============================================================
# Data from your tables
# ============================================================

methods = [
    "BERT4Rec",
    "EAGER",
    "Mamba4Rec",
    "ReSID",
    "SIGMA",
    "LLM-SRec",
    "LLMRec",
    "Ours",
]

# Full-catalog throughput (users/sec)
throughput = {
    "BERT4Rec": 9700,
    "EAGER": 79.3,
    "Mamba4Rec": 7900,
    "ReSID": 4400,
    "SIGMA": 16600,
    "LLM-SRec": 29.8,
    "LLMRec": None,
    "Ours": 2500,
}

# Throughput-derived latency proxy (ms/user), not isolated forward-pass timing.
# Replace with measured forward-pass latencies when available.
forward_pass_latency = {
    method: 1000.0 / value if value is not None else None
    for method, value in throughput.items()
}

# Peak GPU memory (GiB)
memory = {
    "BERT4Rec": 0.8,
    "EAGER": 2.4,
    "Mamba4Rec": 2.9,
    "ReSID": 1.4,
    "SIGMA": 28.9,
    "LLM-SRec": 17.9,
    "LLMRec": None,
    "Ours": 1.5,
}

# ============================================================
# Recommendation quality
# ============================================================

quality = {
    "BERT4Rec": [
        .0885, .0832, .1002, .0869,
        .0203, .0157, .0271, .0178,
        .0311, .0234, .0428, .0272,
        .0363, .0279, .0466, .0312
    ],

    "EAGER": [
        .0137, .0090, .0206, .0111,
        .0022, .0014, .0045, .0022,
        .0167, .0107, .0278, .0143,
        .0028, .0017, .0042, .0021
    ],

    "Mamba4Rec": [
        .1057, .1048, .1057, .1048,
        .0251, .0208, .0310, .0227,
        .0375, .0301, .0489, .0337,
        .0297, .0251, .0352, .0268
    ],

    "ReSID": [
        .0398, .0515, .0289, .0327,
        .0093, .0063, .0144, .0079,
        .0156, .0101, .0255, .0133,
        .0132, .0090, .0194, .0110
    ],

    "SIGMA": [
        .0899, .0824, .1042, .0870,
        .0112, .0074, .0177, .0095,
        .0270, .0187, .0402, .0229,
        .0195, .0136, .0280, .0164
    ],

    "LLM-SRec": [
        .0268, .0175, .0453, .0234,
        .0098, .0065, .0160, .0084,
        .0118, .0072, .0205, .0100,
        .0064, .0081, .0095, .0148
    ],

    "LLMRec": [
        .0851, .0836, .0892, .0849,
        .0178, .0143, .0228, .0159,
        .0170, .0132, .0228, .0151,
        .0299, .0226, .0395, .0257
    ],

    "Ours": [
        .1222, .1118, .1332, .1153,
        .0338, .0271, .0428, .0300,
        .0439, .0344, .0567, .0385,
        .0467, .0373, .0579, .0409
    ],
}

# Mean quality across all 16 reported metrics
mean_quality = {
    method: np.mean(values)
    for method, values in quality.items()
}

# ============================================================
# Print mean quality
# ============================================================

print("=" * 60)
print("Mean recommendation quality")
print("=" * 60)

for m in methods:
    print(f"{m:12s}: {mean_quality[m]:.4f}")


# ============================================================
# Generic Pareto functions
# ============================================================

def pareto_frontier_max_max(methods, objective_x, objective_y):
    """
    Both objectives are maximized.
    """
    valid = [
        m for m in methods
        if objective_x[m] is not None
        and objective_y[m] is not None
    ]

    pareto = []

    for method in valid:
        x = objective_x[method]
        y = objective_y[method]

        dominated = False

        for other in valid:
            if other == method:
                continue

            x_other = objective_x[other]
            y_other = objective_y[other]

            if (
                x_other >= x
                and y_other >= y
                and (x_other > x or y_other > y)
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(method)

    return pareto


def pareto_frontier_min_max(methods, objective_x, objective_y):
    """
    X objective is minimized.
    Y objective is maximized.
    """
    valid = [
        m for m in methods
        if objective_x[m] is not None
        and objective_y[m] is not None
    ]

    pareto = []

    for method in valid:
        x = objective_x[method]
        y = objective_y[method]

        dominated = False

        for other in valid:
            if other == method:
                continue

            x_other = objective_x[other]
            y_other = objective_y[other]

            if (
                x_other <= x
                and y_other >= y
                and (x_other < x or y_other > y)
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(method)

    return pareto


# ============================================================
# Calculate both Pareto frontiers
# ============================================================

# Latency: minimize
# Quality: maximize
pareto_latency = pareto_frontier_min_max(
    methods,
    forward_pass_latency,
    mean_quality
)

# Memory: minimize
# Quality: maximize
pareto_memory = pareto_frontier_min_max(
    methods,
    memory,
    mean_quality
)

print("\n" + "=" * 60)
print("Pareto-optimal methods: Forward Pass Latency vs Quality")
print("=" * 60)

for m in pareto_latency:
    print(
        f"{m:12s} | "
        f"Latency = {forward_pass_latency[m]:8.3f} ms/user | "
        f"Quality = {mean_quality[m]:.4f}"
    )

print("\n" + "=" * 60)
print("Pareto-optimal methods: Memory vs Quality")
print("=" * 60)

for m in pareto_memory:
    print(
        f"{m:12s} | "
        f"Memory = {memory[m]:6.1f} GiB | "
        f"Quality = {mean_quality[m]:.4f}"
    )


# ============================================================
# Plot settings
# ============================================================

plot_methods_latency = [
    m for m in methods
    if forward_pass_latency[m] is not None
]

plot_methods_memory = [
    m for m in methods
    if memory[m] is not None
]

# Marker size based on the OTHER efficiency metric
# This keeps the visual information from your original plot.
def marker_size(method):
    return 55 if method == "Ours" else 20 + memory[method] * 1.2


# Place each label just beyond the marker edge, vertically centered on its point.
# Scatter sizes are in points squared, so sqrt(size) / 2 approximates the radius.
offsets_latency = {
    method: ((-1 if method == "LLM-SRec" else 1)
             * (np.sqrt(marker_size(method)) / 2 + 2), 0)
    for method in plot_methods_latency
}
offsets_memory = {
    method: ((-1 if method == "SIGMA" else 1)
             * (np.sqrt(marker_size(method)) / 2 + 2), 0)
    for method in plot_methods_memory
}

# ============================================================
# Side-by-side Pareto plots
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(6, 3),
    constrained_layout=True
)

# ============================================================
# (a) Forward Pass Latency vs Quality
# ============================================================

for method in plot_methods_latency:

    if method == "Ours":
        ax1.scatter(
            forward_pass_latency[method],
            mean_quality[method],
            s=marker_size(method),
            marker="*",
            facecolors=method_colors[method],
            edgecolors=method_colors[method],
            linewidths=1.0,
            zorder=5,
        )
    else:
        ax1.scatter(
            forward_pass_latency[method],
            mean_quality[method],
            s=marker_size(method),
            facecolors=method_colors[method],
            edgecolors=method_colors[method],
            linewidths=0.9,
            zorder=3
        )

# Pareto frontier
pareto_sorted = sorted(
    pareto_latency,
    key=lambda m: forward_pass_latency[m]
)

ax1.plot(
    [forward_pass_latency[m] for m in pareto_sorted],
    [mean_quality[m] for m in pareto_sorted],
    linestyle="--",
    linewidth=1.2,
    color="#777777",
    zorder=2,
    label="Pareto frontier"
)

# Labels
for method in plot_methods_latency:

    dx, dy = offsets_latency.get(method, (8, 8))

    ax1.annotate(
        ("Ours (RIBS)" if method == "Ours" else method),
        (forward_pass_latency[method], mean_quality[method]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7,
        ha="right" if method == "LLM-SRec" else "left",
        va="center",
        fontweight="bold" if method == "Ours" else "normal",
    )

# Piecewise-linear axis: 0–1 ms/user occupies about 20% of the positive axis width.
# Values and tick labels remain in original ms/user units.
latency_low_scale = 8.5

def latency_axis_forward(x):
    x = np.asarray(x)
    return np.where(x <= 1.0, latency_low_scale * x,
                    latency_low_scale + x - 1.0)

def latency_axis_inverse(x):
    x = np.asarray(x)
    return np.where(x <= latency_low_scale, x / latency_low_scale,
                    1.0 + x - latency_low_scale)

ax1.set_xscale("function", functions=(latency_axis_forward, latency_axis_inverse))
ax1.set_xticks([0, 0.5, 1, 10, 20, 30, 35])
ax1.set_xticklabels(["0", ".5", "1", "10", "20", "30", "35"])
ax1.axvline(1, color="#777777", linestyle=":", linewidth=.65, alpha=0.6)
# ax1.text(0.73, 0.90, "Scale changes at 1 ms/user", transform=ax1.transAxes,
#          ha="center", va="top", fontsize=9, color="dimgray")
ax1.set_xlim(-0.2, 35)  # Left padding keeps SIGMA clear of the frame.
ax1.set_ylim(0.005, 0.066)

ax1.set_xlabel(
    "Forward pass latency (ms/user)",
    fontsize=9
)

ax1.set_ylabel(
    "Mean recommendation quality",
    fontsize=9
)

ax1.set_title(
    "(a) Quality vs. Forward Pass Latency",
    fontsize=10,
    loc="left",
    pad=9,
    fontweight="bold"
)

ax1.grid(axis="y", alpha=.8)
ax1.set_axisbelow(True)
ax1.tick_params(direction="out", length=3, width=.6)

# ax1.legend(
#     loc="center right",
#     frameon=True
# )


# ============================================================
# (b) Memory vs Quality
# ============================================================

for method in plot_methods_memory:

    if method == "Ours":
        ax2.scatter(
            memory[method],
            mean_quality[method],
            s=marker_size(method),
            marker="*",
            facecolors=method_colors[method],
            edgecolors=method_colors[method],
            linewidths=1.0,
            zorder=5,
        )
    else:
        ax2.scatter(
            memory[method],
            mean_quality[method],
            s=marker_size(method),
            facecolors=method_colors[method],
            edgecolors=method_colors[method],
            linewidths=0.9,
            zorder=3
        )

# Pareto frontier
pareto_sorted_memory = sorted(
    pareto_memory,
    key=lambda m: memory[m]
)

ax2.plot(
    [memory[m] for m in pareto_sorted_memory],
    [mean_quality[m] for m in pareto_sorted_memory],
    linestyle="--",
    linewidth=1.2,
    color="#777777",
    zorder=2,
    label="Pareto frontier"
)

# Labels
for method in plot_methods_memory:

    dx, dy = offsets_memory.get(method, (8, 8))

    ax2.annotate(
        "Ours (RIBS)" if method == "Ours" else method,
        (memory[method], mean_quality[method]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7,
        ha="right" if method == "SIGMA" else "left",
        va="center",
        fontweight="bold" if method == "Ours" else "normal",
    )

ax2.set_xlabel(
    "Peak GPU memory (GiB)",
    fontsize=10
)

ax2.set_ylabel(
    "Mean recommendation quality",
    fontsize=10
)

ax2.set_title(
    "(b) Quality vs. GPU Memory",
    fontsize=11,
    loc="left",
    pad=9,
    fontweight="bold"
)

ax2.grid(axis="y", alpha=.8)
ax2.set_axisbelow(True)
ax2.tick_params(direction="out", length=3, width=.6)

# ax2.legend(
#     loc="lower right",
#     frameon=True
# )


# ============================================================
# Final formatting
# ============================================================

# The available timings cover full-catalog evaluation, not an isolated forward pass.
# Latencies are throughput-derived proxies, not isolated forward-pass measurements.
from pathlib import Path
output_dir = _base_dir
output_path = output_dir / "memory_latency_RIBS.pdf"
# Keep point labels inside their subplot, including after layout/font changes.
def keep_labels_inside_axes(fig, axes, padding_points=4):
    from matplotlib.text import Annotation

    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        padding = renderer.points_to_pixels(padding_points)
        for ax in axes:
            bounds = ax.get_window_extent(renderer)
            for label in ax.texts:
                if not isinstance(label, Annotation):
                    continue
                box = label.get_window_extent(renderer)
                shift_x = max(bounds.x0 + padding - box.x0, 0)
                shift_x -= max(box.x1 - (bounds.x1 - padding), 0)
                shift_y = max(bounds.y0 + padding - box.y0, 0)
                shift_y -= max(box.y1 - (bounds.y1 - padding), 0)
                dx, dy = label.get_position()
                pixels_per_point = renderer.points_to_pixels(1)
                label.set_position((dx + shift_x / pixels_per_point,
                                    dy + shift_y / pixels_per_point))

keep_labels_inside_axes(fig, (ax1, ax2))
fig.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Saved PDF: {output_path.resolve()}")
plt.close(fig)

