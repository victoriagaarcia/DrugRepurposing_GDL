import json
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load results
# -----------------------------

json_path = "./results/ablation/ablation_20260428_114647/ablation_results_copy.json"

with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

configs = ["full", "no_anatomy", "no_gene", "no_intermediate"]
config_labels = ["FULL", "NO_ANATOMY", "NO_GENE", "NO_INTERMEDIATE"]

encoders = ["rgcn", "han", "sage"]
encoder_labels = {
    "rgcn": "R-GCN",
    "han": "HAN-insp.",
    "sage": "GraphSAGE",
}

mrr_mean = {encoder_labels[e]: [] for e in encoders}
mrr_std = {encoder_labels[e]: [] for e in encoders}

for config in configs:
    for encoder in encoders:
        runs = results[config][encoder]
        values = [run["MRR"] for run in runs]

        label = encoder_labels[encoder]
        mrr_mean[label].append(np.mean(values))
        mrr_std[label].append(np.std(values, ddof=1))

# -----------------------------
# Plot
# -----------------------------

x = np.arange(len(configs))
width = 0.23

fig, ax = plt.subplots(figsize=(8.0, 4.2))

for i, label in enumerate(encoder_labels.values()):
    offset = (i - 1) * width
    ax.bar(
        x + offset,
        mrr_mean[label],
        width,
        yerr=mrr_std[label],
        capsize=3.5,
        label=label,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

# -----------------------------
# Formatting
# -----------------------------

ax.set_ylabel("Mean Reciprocal Rank (MRR)")
ax.set_xlabel("Graph configuration")

ax.set_xticks(x)
ax.set_xticklabels(config_labels, rotation=12, ha="right")

ax.set_ylim(0.40, 0.72)

# Legend inside the plot, away from title/caption
ax.legend(
    frameon=True,
    fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    borderaxespad=0.4,
)

ax.grid(axis="y", linestyle="--", alpha=0.30)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig("mrr_ablation_results.pdf", bbox_inches="tight")
plt.savefig("mrr_ablation_results.png", dpi=300, bbox_inches="tight")

plt.show()