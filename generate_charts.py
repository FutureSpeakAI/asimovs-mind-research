"""
Generate all analysis charts for the Asimov's Mind paper.
Usage: uv run generate_charts.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

results_dir = Path("governed/results")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(results_dir / "all_results.tsv", sep="\t")
df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
df["peak_vram_mb"] = pd.to_numeric(df["peak_vram_mb"], errors="coerce")

histories = {}
for cond in ["A", "B", "C"]:
    p = results_dir / f"history_{cond}.tsv"
    if p.exists():
        histories[cond] = pd.read_csv(p, sep="\t")

baseline_bpb = df[df["phase"] == "baseline"].iloc[0]["val_bpb"]
print(f"Baseline val_bpb: {baseline_bpb:.6f}")

# ---------------------------------------------------------------------------
# Chart 1: Parameter Sensitivity (isolated experiments)
# ---------------------------------------------------------------------------

isolated = df[(df["phase"] == "isolated") & (df["condition"] == "A")].copy()
isolated["delta"] = baseline_bpb - isolated["val_bpb"]
isolated = isolated.sort_values("delta", ascending=False)

fig, ax = plt.subplots(figsize=(12, 5.5))
colors = ["#27ae60" if d > 0 else "#c0392b" for d in isolated["delta"]]
bars = ax.bar(range(len(isolated)), isolated["delta"], color=colors,
              edgecolor="black", linewidth=0.5, width=0.7)

ax.set_xticks(range(len(isolated)))
ax.set_xticklabels(isolated["description"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Improvement over baseline\n(positive = better)")
ax.set_title(f"Isolated Parameter Sensitivity — Each Change vs Baseline ({baseline_bpb:.4f})")
ax.axhline(y=0, color="black", linewidth=1)
ax.grid(axis="y", alpha=0.25)

for bar, delta in zip(bars, isolated["delta"]):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, y,
            f"{delta:+.4f}", ha="center",
            va="bottom" if delta > 0 else "top",
            fontsize=8.5, fontweight="bold",
            color="#1a7a3a" if delta > 0 else "#8b1a1a")

plt.tight_layout()
plt.savefig("chart_param_sensitivity.png", dpi=180, bbox_inches="tight")
print("Saved chart_param_sensitivity.png")
plt.close()

# ---------------------------------------------------------------------------
# Chart 2: Cumulative Progress (all conditions + optimal)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6.5))

cond_styles = {
    "A":       {"color": "#c0392b", "label": "A: Ungoverned",      "marker": "s", "ls": "-"},
    "B":       {"color": "#2980b9", "label": "B: Governed Single",  "marker": "^", "ls": "-"},
    "C":       {"color": "#27ae60", "label": "C: Governed Swarm",   "marker": "o", "ls": "-"},
    "optimal": {"color": "#8e44ad", "label": "Optimal Order",       "marker": "D", "ls": "--"},
}

cumulative = df[df["phase"].isin(["cumulative", "optimal"])]

for cond, style in cond_styles.items():
    cond_data = cumulative[(cumulative["condition"] == cond) & (cumulative["val_bpb"] > 0)]
    if cond_data.empty:
        continue
    steps = list(range(len(cond_data) + 1))
    bpbs = [baseline_bpb] + list(cond_data["val_bpb"])
    ax.plot(steps, bpbs, marker=style["marker"], color=style["color"],
            linewidth=2.2, markersize=7, label=style["label"],
            linestyle=style["ls"], zorder=3)
    # Annotate final
    ax.annotate(f"{bpbs[-1]:.4f}", (steps[-1], bpbs[-1]),
                textcoords="offset points", xytext=(10, -2), fontsize=9,
                color=style["color"], fontweight="bold")

ax.axhline(y=baseline_bpb, color="gray", linestyle=":", alpha=0.6,
           linewidth=1.5, label=f"Baseline ({baseline_bpb:.4f})")
ax.set_xlabel("Cumulative Step")
ax.set_ylabel("val_bpb (lower is better)")
ax.set_title("Cumulative Experiment Progress — Historical Replay vs Optimal")
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.25)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("chart_cumulative_progress.png", dpi=180, bbox_inches="tight")
print("Saved chart_cumulative_progress.png")
plt.close()

# ---------------------------------------------------------------------------
# Chart 3: Crash Rate Comparison (from reconstructed history)
# ---------------------------------------------------------------------------

if histories:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    cond_labels = {
        "A": "A: Ungoverned",
        "B": "B: Governed Single",
        "C": "C: Governed Swarm",
    }
    cond_title_colors = {"A": "#c0392b", "B": "#2980b9", "C": "#27ae60"}

    for ax, cond in zip(axes, ["A", "B", "C"]):
        hist = histories[cond]
        counts = hist["status"].value_counts()
        categories = ["keep", "discard", "crash"]
        values = [counts.get(c, 0) for c in categories]
        cat_colors = ["#27ae60", "#f39c12", "#c0392b"]

        bars = ax.bar(categories, values, color=cat_colors,
                      edgecolor="black", linewidth=0.5, width=0.6)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                        str(v), ha="center", fontweight="bold", fontsize=13)

        total = len(hist)
        crash_pct = counts.get("crash", 0) / total * 100
        ax.set_title(f"{cond_labels[cond]}\n{total} exps · {crash_pct:.0f}% crash rate",
                     fontsize=11, color=cond_title_colors[cond], fontweight="bold")
        ax.set_ylabel("Count" if cond == "A" else "")
        ax.set_ylim(0, max(max(values) + 2, 7))
        ax.tick_params(axis="x", labelsize=10)

    plt.suptitle("Experiment Outcomes — Reconstructed from AI Agent Git History",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("chart_crash_rates.png", dpi=180, bbox_inches="tight")
    print("Saved chart_crash_rates.png")
    plt.close()

# ---------------------------------------------------------------------------
# Chart 4: Degradation Rate Comparison
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

cond_colors = {"A": "#c0392b", "B": "#2980b9", "C": "#27ae60"}
cond_labels_short = {"A": "Ungoverned", "B": "Governed Single", "C": "Governed Swarm"}

degradation_rates = {}
for cond in ["A", "B", "C"]:
    cond_data = cumulative[(cumulative["condition"] == cond) & (cumulative["val_bpb"] > 0)]
    if cond_data.empty:
        continue
    bpbs = list(cond_data["val_bpb"])
    n_steps = len(bpbs)
    total_degradation = bpbs[-1] - baseline_bpb
    rate = total_degradation / n_steps
    degradation_rates[cond] = rate

bars = ax.bar(
    [cond_labels_short[c] for c in degradation_rates],
    list(degradation_rates.values()),
    color=[cond_colors[c] for c in degradation_rates],
    edgecolor="black", linewidth=0.5, width=0.5,
)

for bar, (cond, rate) in zip(bars, degradation_rates.items()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{rate:+.4f}/step", ha="center", fontweight="bold", fontsize=11)

ax.set_ylabel("val_bpb degradation per cumulative step\n(closer to 0 = better)")
ax.set_title("Degradation Rate: How Fast Does Each Condition Get Worse?")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("chart_degradation_rate.png", dpi=180, bbox_inches="tight")
print("Saved chart_degradation_rate.png")
plt.close()

# ---------------------------------------------------------------------------
# Chart 5: Combined Summary (2x2 panel for paper)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel A: Parameter sensitivity
ax = axes[0, 0]
colors = ["#27ae60" if d > 0 else "#c0392b" for d in isolated["delta"]]
ax.bar(range(len(isolated)), isolated["delta"], color=colors,
       edgecolor="black", linewidth=0.4, width=0.7)
ax.set_xticks(range(len(isolated)))
ax.set_xticklabels(isolated["description"], rotation=45, ha="right", fontsize=7.5)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Improvement")
ax.set_title("(a) Isolated Parameter Sensitivity", fontweight="bold")
ax.grid(axis="y", alpha=0.2)

# Panel B: Cumulative progress
ax = axes[0, 1]
for cond, style in cond_styles.items():
    cond_data = cumulative[(cumulative["condition"] == cond) & (cumulative["val_bpb"] > 0)]
    if cond_data.empty:
        continue
    steps = list(range(len(cond_data) + 1))
    bpbs = [baseline_bpb] + list(cond_data["val_bpb"])
    ax.plot(steps, bpbs, marker=style["marker"], color=style["color"],
            linewidth=1.8, markersize=5, label=style["label"], linestyle=style["ls"])
ax.axhline(y=baseline_bpb, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("val_bpb")
ax.set_title("(b) Cumulative Progress by Condition", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.2)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Panel C: Crash rates
ax = axes[1, 0]
if histories:
    conds = ["A", "B", "C"]
    labels = ["Ungoverned", "Governed\nSingle", "Governed\nSwarm"]
    crash_pcts = []
    for c in conds:
        h = histories[c]
        crash_pcts.append((h["status"] == "crash").sum() / len(h) * 100)
    bar_colors = [cond_colors[c] for c in conds]
    b = ax.bar(labels, crash_pcts, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.5)
    for bar, pct in zip(b, crash_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("Crash Rate (%)")
    ax.set_title("(c) Historical Crash Rates", fontweight="bold")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.2)

# Panel D: Degradation rates
ax = axes[1, 1]
labels = [cond_labels_short[c] for c in degradation_rates]
rates = list(degradation_rates.values())
bar_colors = [cond_colors[c] for c in degradation_rates]
b = ax.bar(labels, rates, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.5)
for bar, rate in zip(b, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f"{rate:+.4f}", ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("Degradation per step")
ax.set_title("(d) Degradation Rate per Cumulative Step", fontweight="bold")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.2)

plt.suptitle("Asimov's Mind — Governed Multi-Agent Optimization Results",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("chart_paper_summary.png", dpi=200, bbox_inches="tight")
print("Saved chart_paper_summary.png")
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)
print()
print("Individual charts:")
print("  chart_param_sensitivity.png   — isolated experiment results")
print("  chart_cumulative_progress.png — cumulative replay comparison")
print("  chart_crash_rates.png         — historical crash rates")
print("  chart_degradation_rate.png    — degradation per step")
print()
print("Paper figure:")
print("  chart_paper_summary.png       — 2x2 combined panel")
