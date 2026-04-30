#!/usr/bin/env python3
"""Regenerate all 6 poster figures with larger fonts (not bold), no overlaps, bold titles."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/ziqiwei/Desktop/2026spring/SERC/climate-negotiation-sim/poster_figures"

# ── Global font settings (large, NOT bold) ───────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 18,
    "font.weight": "normal",
    "axes.titlesize": 26,
    "axes.titleweight": "bold",       # only titles are bold
    "axes.labelsize": 22,
    "axes.labelweight": "normal",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 28,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
})

DARK  = "#193856"
MID   = "#286e76"
LIGHT = "#477a53"
BLUE  = "#3b6e9f"
RED   = "#c06060"
GOLD  = "#9e8a4c"
GRAY  = "#888888"


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Radar: Baseline Evaluation Profile
# ═════════════════════════════════════════════════════════════════════════════
def fig1_radar():
    categories = ["BERTScore", "ROUGE-L", "Stance", "Brackets", "Clauses", "Structure"]
    values     = [0.977, 0.894, 1.000, 1.000, 0.667, 0.991]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Perfect circle
    perfect = [1.0] * N + [1.0]
    ax.plot(angles, perfect, "--", color=GRAY, linewidth=1.5, label="Perfect (1.0)")

    # Simulation
    ax.plot(angles, values_plot, "o-", color=BLUE, linewidth=3, markersize=10, label="Simulation")
    ax.fill(angles, values_plot, alpha=0.2, color=BLUE)

    # Labels with values — push outward with padding
    labels = [f"{c}\n({v:.3f})" for c, v in zip(categories, values)]
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=19)
    # Push labels outward
    ax.set_rlabel_position(30)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)

    fig.suptitle("Baseline Evaluation Profile", fontsize=28, fontweight="bold", y=0.98)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=17)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig1_radar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig1_radar.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Ablation: Text Quality + Plenary Votes
# ═════════════════════════════════════════════════════════════════════════════
def fig2_ablation():
    labels   = ["Baseline\n(all 8)", "-EU", "-G77", "-AOSIS", "-Umbrella", "-LDC", "-AFR", "-LMDC", "-EIG"]
    quality  = [0.874, 0.868, 0.874, 0.714, 0.862, 0.572, 0.769, 0.852, 0.862]
    accept   = [3, 2, 5, 5, 2, 5, 5, 3, 3]
    block    = [5, 5, 2, 2, 5, 2, 2, 4, 4]
    baseline = 0.874

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.suptitle("Ablation Study: Text Quality and Plenary Votes", fontsize=28, fontweight="bold", y=1.02)

    # (a) Text Quality
    colors_a = []
    for i, v in enumerate(quality):
        if i == 0:
            colors_a.append(DARK)
        elif v > baseline + 0.005:
            colors_a.append(LIGHT)
        elif v < baseline - 0.05:
            colors_a.append(RED)
        else:
            colors_a.append(BLUE)

    ax1.bar(labels, quality, color=colors_a, width=0.65, edgecolor="white", linewidth=1.5)
    ax1.axhline(y=baseline, color=RED, linestyle="--", linewidth=2, alpha=0.7)
    ax1.text(len(labels) - 0.5, baseline + 0.01, "baseline", color=RED, fontsize=17, ha="right", style="italic")
    ax1.set_ylabel("Text Quality Score", fontsize=22)
    ax1.set_title("(a) Text Quality by Ablation", fontsize=24, fontweight="bold")
    ax1.set_ylim(0.4, 1.05)
    ax1.tick_params(axis="x", labelsize=17)
    ax1.tick_params(axis="y", labelsize=17)

    # (b) Plenary Votes
    x = np.arange(len(labels))
    w = 0.65
    ax2.bar(x, accept, w, label="Accept", color=BLUE, edgecolor="white", linewidth=1.5)
    ax2.bar(x, block, w, bottom=accept, label="Block", color=RED, alpha=0.45, edgecolor="white", linewidth=1.5)
    ax2.axhline(y=3.5, color=RED, linestyle=":", linewidth=2, alpha=0.6)
    ax2.text(len(labels) - 0.5, 3.65, "majority", color=RED, fontsize=17, ha="right", style="italic")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=17)
    ax2.set_ylabel("Agent Count", fontsize=22)
    ax2.set_title("(b) Plenary Votes by Ablation", fontsize=24, fontweight="bold")
    ax2.legend(fontsize=19, loc="upper right")
    ax2.tick_params(axis="y", labelsize=17)

    plt.tight_layout(w_pad=4)
    fig.savefig(f"{OUT}/fig2_ablation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig2_ablation.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Heatmap: Voting Patterns Under Ablation
# ═════════════════════════════════════════════════════════════════════════════
def fig3_heatmap():
    agents = ["EU", "G77", "AOSIS", "Umbrella", "LDC", "AFR", "LMDC", "EIG"]
    ablations = ["-EU", "-G77", "-AOSIS", "-Umbrella", "-LDC", "-AFR", "-LMDC", "-EIG"]

    # 0 = removed, 1 = Accept, -1 = Block
    data = [
        [ 0, -1, -1,  1, -1, -1, -1,  1],  # -EU
        [ 1,  0,  1,  1, -1,  1, -1,  1],  # -G77
        [ 1, -1,  0,  1,  1,  1, -1,  1],  # -AOSIS
        [ 1, -1, -1,  0, -1, -1, -1,  1],  # -Umbrella
        [ 1, -1,  1,  1,  0,  1, -1,  1],  # -LDC
        [ 1, -1,  1,  1,  1,  0, -1,  1],  # -AFR
        [ 1, -1, -1,  1, -1, -1,  0,  1],  # -LMDC
        [ 1, -1, -1,  1,  1, -1, -1,  0],  # -EIG
    ]
    data = np.array(data, dtype=float)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([RED, "#cccccc", "#6dbe7a"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_title("(c) Voting Patterns Under Agent Ablation", fontsize=28, fontweight="bold", pad=10)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, fontsize=19)
    ax.set_yticks(range(len(ablations)))
    ax.set_yticklabels(ablations, fontsize=19)
    ax.set_xlabel("Remaining Agent", fontsize=22, labelpad=14)
    ax.set_ylabel("Ablation Condition", fontsize=22, labelpad=14)

    # Cell labels
    label_map = {1: "A", -1: "B", 0: "-"}
    for i in range(len(ablations)):
        for j in range(len(agents)):
            v = int(data[i, j])
            ax.text(j, i, label_map[v], ha="center", va="center",
                    fontsize=24, color="white")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6dbe7a", label="Accept (A)"),
        Patch(facecolor=RED, label="Block (B)"),
        Patch(facecolor="#cccccc", label="Removed (-)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, fontsize=18, frameon=False)

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig3_heatmap.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Fairness Framing Weights by Coalition
# ═════════════════════════════════════════════════════════════════════════════
def fig4_fairness():
    coalitions = ["AOSIS", "G77+\nChina", "LMDC", "African\nGroup", "LDC", "EU", "EIG", "Umbrella"]
    hist_resp  = [0.50, 0.60, 0.70, 0.55, 0.50, 0.30, 0.30, 0.15]
    curr_cap   = [0.20, 0.20, 0.15, 0.20, 0.15, 0.30, 0.30, 0.35]
    fut_needs  = [0.30, 0.20, 0.15, 0.25, 0.35, 0.40, 0.40, 0.50]

    x = np.arange(len(coalitions))
    w = 0.25

    fig, ax = plt.subplots(figsize=(20, 9))
    fig.suptitle("Fairness Framing Weights by Coalition", fontsize=28, fontweight="bold", y=1.01)

    b1 = ax.bar(x - w, hist_resp, w, label="Historical Responsibility", color=DARK, edgecolor="white", linewidth=1.5)
    b2 = ax.bar(x,     curr_cap,  w, label="Current Capability", color="#8baec4", edgecolor="white", linewidth=1.5)
    b3 = ax.bar(x + w, fut_needs, w, label="Future Needs", color=GOLD, edgecolor="white", linewidth=1.5)

    # Value labels (not bold)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=14)

    # Divider
    ax.axvline(x=4.5, color=GRAY, linestyle="--", linewidth=2, alpha=0.6)
    ax.text(2.0, 0.82, "Developing", fontsize=21, color=MID, ha="center", style="italic")
    ax.text(6.0, 0.82, "Developed", fontsize=21, color=MID, ha="center", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(coalitions, fontsize=19)
    ax.set_ylabel("Weight (0-1)", fontsize=22)
    ax.set_ylim(0, 0.92)
    ax.legend(fontsize=19, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    ax.tick_params(axis="y", labelsize=17)

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_fairness.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig4_fairness.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — System Performance Across Development Iterations
# ═════════════════════════════════════════════════════════════════════════════
def fig5_evolution():
    stages = ["Initial\n(pre-fix)", "Post-audit\nfix", "Post-prompt\ntune", "Post-feature\nadd"]
    overall     = [0.44, 0.73, 0.78, 0.75]
    rouge1      = [0.04, 0.72, 0.78, 0.77]
    bertscore   = [0.92, 0.96, 0.98, 0.98]
    stance      = [0.90, 1.00, 1.00, 1.00]

    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle("System Performance Across Development Iterations", fontsize=28, fontweight="bold", y=1.01)

    ax.plot(x, overall,   "o-",  color=DARK,  linewidth=3, markersize=14, label="Overall Score")
    ax.plot(x, rouge1,    "s--", color=BLUE,  linewidth=3, markersize=14, label="ROUGE-1 F1")
    ax.plot(x, bertscore, "^:",  color=MID,   linewidth=3, markersize=14, label="BERTScore F1")
    ax.plot(x, stance,    "D:",  color=LIGHT, linewidth=3, markersize=14, label="Stance Consistency")

    # Annotations (positioned to avoid overlap)
    ax.annotate("48 bugs fixed\n+ red-line critic", xy=(1, 0.73), xytext=(0.3, 0.50),
                fontsize=18, color=GRAY, style="italic",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("caucus + preservation\n+ sub-item rules", xy=(2, 0.98), xytext=(2.4, 0.85),
                fontsize=18, color=GRAY, style="italic",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.annotate("fairness weights\n+ behavioral params", xy=(3, 0.75), xytext=(2.15, 0.55),
                fontsize=18, color=GRAY, style="italic",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=20)
    ax.set_ylabel("Score", fontsize=24)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=20, loc="lower right")
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig5_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig5_evolution.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Scatter: Text Quality vs. Political Acceptance
# ═════════════════════════════════════════════════════════════════════════════
def fig6_scatter():
    labels  = ["Baseline", "-EU", "-G77", "-AOSIS", "-Umbrella", "-LDC", "-AFR", "-LMDC", "-EIG"]
    tq      = [0.874, 0.815, 0.890, 0.714, 0.862, 0.572, 0.769, 0.852, 0.862]
    ar      = [0.375, 0.286, 0.714, 0.571, 0.286, 0.714, 0.714, 0.429, 0.375]

    fig, ax = plt.subplots(figsize=(14, 13))  # wider, still taller than wide
    ax.set_title("(d) Text Quality vs. Political Acceptance", fontsize=28, fontweight="bold", pad=10)

    ax.scatter(tq, ar, s=350, color=BLUE, zorder=5, edgecolors="white", linewidth=2.5)

    # Labels — large, tight offsets for compact layout
    offsets = {
        "Baseline":   (-14, -28),
        "-EU":        (-12, -25),
        "-G77":       (-70, -5),
        "-AOSIS":     (14, -22),
        "-Umbrella":  (14, 8),
        "-LDC":       (14, 8),
        "-AFR":       (14, 8),
        "-LMDC":      (14, 8),
        "-EIG":       (-70, 8),
    }
    for i, lab in enumerate(labels):
        ox, oy = offsets[lab]
        fs = 22 if lab == "Baseline" else 20
        clr = DARK if lab == "Baseline" else RED
        fw = "bold" if lab == "Baseline" else "normal"
        ax.annotate(lab, (tq[i], ar[i]), textcoords="offset points", xytext=(ox, oy),
                    fontsize=fs, fontweight=fw, color=clr)

    # Threshold lines
    ax.axhline(y=0.5, color=RED, linestyle="--", linewidth=2, alpha=0.5)
    ax.text(0.94, 0.508, "majority threshold", fontsize=17, color=RED, ha="right", style="italic")
    ax.axvline(x=0.874, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.5)
    ax.text(0.879, 0.20, "baseline\ntext quality", fontsize=16, color=GRAY, ha="left", style="italic")

    ax.set_xlabel("Text Quality Score", fontsize=23)
    ax.set_ylabel("Accept Ratio", fontsize=23)
    ax.set_xlim(0.52, 0.95)
    ax.set_ylim(0.18, 0.80)
    ax.tick_params(labelsize=18)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig6_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig6_scatter.png saved")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_radar()
    fig2_ablation()
    fig3_heatmap()
    fig4_fairness()
    fig5_evolution()
    fig6_scatter()
    print("\nAll 6 figures regenerated.")
