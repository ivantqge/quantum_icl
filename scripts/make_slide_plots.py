"""Generate slide-quality figures from completed results_qicl/ runs.

Outputs to paper/figures/*.png and *.pdf.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_qicl")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# --- color palette -------------------------------------------------------
COLOR_MODEL = {
    "gpt-4o-mini": "#9c9c9c",
    "Gemini-3-Flash": "#4c72b0",
    "Qwen-7B (base)": "#dd8452",
    "Qwen-7B + SFT (3600)": "#55a868",
    "Qwen-7B + SFT (1800)": "#8172b3",
}
COLOR_COND = {
    "zero_shot": "#cccccc",
    "feedback_only": "#4c72b0",
    "structural_retrieval_only": "#dd8452",
    "structural_retrieval_plus_feedback": "#55a868",
}
COND_LABEL = {
    "zero_shot": "zero-shot",
    "feedback_only": "+feedback",
    "structural_retrieval_only": "+retrieval",
    "structural_retrieval_plus_feedback": "+retrieval+fb",
}


def load_summary(tag_glob):
    """Most recent summary matching a tag prefix."""
    dirs = sorted(glob.glob(os.path.join(RESULTS, tag_glob)))
    if not dirs:
        return None
    summary_path = os.path.join(dirs[-1], "summary.json")
    if not os.path.exists(summary_path):
        return None
    return json.load(open(summary_path))


def cell(s, tier, condition):
    """Return (n_solved, n_tasks) for a (tier, condition); None if missing."""
    if s is None:
        return None
    for v in s.values():
        if v["tier"] == tier and v["condition"] == condition:
            return v["n_solved"], v["n_tasks"]
    return None


# =========================================================================
# Figure 1: 2x2 ablation across three models, three tiers (the headline)
# =========================================================================
def fig1_main_2x2():
    models = [
        ("gpt-4o-mini", "confirm100_gpt4omini_*"),
        ("Gemini-3-Flash", "confirm100_gemini3flash_*"),
        ("Qwen-7B (base)", "confirm100_base_qwen_*"),
    ]
    summaries = {name: load_summary(g) for name, g in models}
    tiers = [("B", "Tier B (2q stabilizer)"),
             ("C_lite", "Tier C-lite (1q Clifford)"),
             ("D_lite", "Tier D-lite (1q Cliff+T)")]
    conds = ["zero_shot", "feedback_only",
             "structural_retrieval_only",
             "structural_retrieval_plus_feedback"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    width = 0.25
    x = np.arange(len(conds))

    for ax, (tier, title) in zip(axes, tiers):
        for i, (mname, _) in enumerate(models):
            s = summaries[mname]
            rates = []
            for c in conds:
                cv = cell(s, tier, c)
                if cv is None:
                    rates.append(0.0)
                else:
                    n_solved, n_total = cv
                    rates.append(n_solved / n_total)
            ax.bar(x + (i - 1) * width, rates, width,
                   label=mname, color=COLOR_MODEL[mname], edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Solve rate (n=100)")
    axes[-1].legend(loc="upper left", frameon=True)
    fig.suptitle("Verifier-feedback × structural-retrieval ablation across three frozen models",
                 fontsize=14, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_main_2x2.{ext}"))
    plt.close(fig)
    print("wrote fig1_main_2x2")


# =========================================================================
# Figure 2: SFT data-size curve (most novel finding)
# =========================================================================
def fig2_sft_curve():
    sizes = [0, 300, 900, 1800, 3600]
    runs = {
        0: "confirm100_base_qwen_*",
        300: "confirm100_sft_n300_*",
        900: "confirm100_sft_n900_*",
        1800: "confirm100_sft_n1800_*",
        3600: "confirm100_sft_n3600_*",
    }
    summaries = {sz: load_summary(g) for sz, g in runs.items()}
    tiers = [("B", "Tier B"), ("C_lite", "Tier C-lite"), ("D_lite", "Tier D-lite")]
    conds = ["zero_shot", "feedback_only",
             "structural_retrieval_only",
             "structural_retrieval_plus_feedback"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (tier, title) in zip(axes, tiers):
        for cond in conds:
            ys = []
            for sz in sizes:
                cv = cell(summaries[sz], tier, cond)
                if cv is None:
                    ys.append(np.nan)
                else:
                    ys.append(cv[0] / cv[1])
            ax.plot(sizes, ys, marker="o", markersize=7, linewidth=2,
                    color=COLOR_COND[cond], label=COND_LABEL[cond])
        ax.set_title(title)
        ax.set_xlabel("SFT training examples")
        ax.set_xscale("symlog", linthresh=300)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.set_ylim(0, 0.45)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Solve rate (n=100)")
    axes[-1].legend(loc="upper left", frameon=True)
    fig.suptitle("Qwen-7B LoRA-SFT data-size scan × 4 conditions × 3 tiers",
                 fontsize=14, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_sft_curve.{ext}"))
    plt.close(fig)
    print("wrote fig2_sft_curve")


# =========================================================================
# Figure 3: Oracle vs structural retrieval (target features are enough)
# =========================================================================
def fig3_oracle_vs_struct():
    gemini = load_summary("main_gemini3flash_*")
    oracle = load_summary("oracle_gemini3flash_*")
    tiers = ["B", "C_lite", "D_lite"]
    tier_labels = ["B", "C-lite", "D-lite"]

    struct_rates = []
    oracle_rates = []
    for t in tiers:
        sv = cell(gemini, t, "structural_retrieval_plus_feedback")
        ov = cell(oracle, t, "oracle_retrieval")
        struct_rates.append(sv[0] / sv[1] if sv else np.nan)
        oracle_rates.append(ov[0] / ov[1] if ov else np.nan)

    x = np.arange(len(tiers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, struct_rates, width, color="#4c72b0",
           label="Structural (target features)", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, oracle_rates, width, color="#c44e52",
           label="Oracle (hidden generator features)", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Target-only features ≈ oracle features (Gemini-3-Flash, n=30)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for i, (s, o) in enumerate(zip(struct_rates, oracle_rates)):
        if not np.isnan(s):
            ax.text(i - width / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=10)
        if not np.isnan(o):
            ax.text(i + width / 2, o + 0.02, f"{o:.2f}", ha="center", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_oracle_vs_struct.{ext}"))
    plt.close(fig)
    print("wrote fig3_oracle_vs_struct")


# =========================================================================
# Figure 4: Base vs SFT-3600 (paired) — the SFT-rescues-hard-tier story
# =========================================================================
def fig4_base_vs_sft():
    base = load_summary("confirm100_base_qwen_*")
    sft = load_summary("confirm100_sft_n3600_*")
    tiers = [("B", "Tier B"), ("C_lite", "C-lite"), ("D_lite", "D-lite")]
    conds = ["zero_shot", "feedback_only",
             "structural_retrieval_only",
             "structural_retrieval_plus_feedback"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_cells = len(tiers) * len(conds)
    base_y = []
    sft_y = []
    labels = []
    cell_idx = 0
    group_centers = []
    for tier, tlabel in tiers:
        group_centers.append(cell_idx + (len(conds) - 1) / 2)
        for c in conds:
            bv = cell(base, tier, c)
            sv = cell(sft, tier, c)
            base_y.append((bv[0] / bv[1]) if bv else np.nan)
            sft_y.append((sv[0] / sv[1]) if sv else np.nan)
            labels.append(COND_LABEL[c])
            cell_idx += 1
    x = np.arange(n_cells)
    width = 0.4
    ax.bar(x - width / 2, base_y, width, color="#dd8452", label="Qwen-7B base",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, sft_y, width, color="#55a868", label="Qwen-7B + SFT-3600",
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Solve rate (n=100)")
    ax.set_ylim(0, 0.45)
    # tier separators
    for sep in (3.5, 7.5):
        ax.axvline(sep, color="black", alpha=0.25, lw=0.8)
    # tier annotations along the top
    for cx, (_, tlabel) in zip(group_centers, tiers):
        ax.text(cx, 0.42, tlabel, ha="center", fontsize=12,
                fontweight="bold", color="#333")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("LoRA SFT on 3600 verified synthetic examples: base vs SFT, paired by seed")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig4_base_vs_sft.{ext}"))
    plt.close(fig)
    print("wrote fig4_base_vs_sft")


# =========================================================================
# Figure 5: Attempts-per-solved -- feedback's *efficiency* win
# =========================================================================
def fig5_attempts_per_solved():
    runs = [
        ("gpt-4o-mini", "confirm100_gpt4omini_*"),
        ("Gemini-3-Flash", "confirm100_gemini3flash_*"),
        ("Qwen-7B base", "confirm100_base_qwen_*"),
        ("Qwen-7B SFT-3600", "confirm100_sft_n3600_*"),
    ]
    summaries = {m: load_summary(g) for m, g in runs}
    tier = "D_lite"
    conds = ["zero_shot", "feedback_only",
             "structural_retrieval_only",
             "structural_retrieval_plus_feedback"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(conds))
    width = 0.2
    for i, (m, _) in enumerate(runs):
        s = summaries[m]
        ys = []
        for c in conds:
            for v in s.values():
                if v["tier"] == tier and v["condition"] == c:
                    ys.append(v.get("mean_attempts_per_solved", 0))
                    break
            else:
                ys.append(0.0)
        ax.bar(x + (i - 1.5) * width, ys, width,
               label=m, edgecolor="white", linewidth=0.5,
               color={"gpt-4o-mini": "#9c9c9c",
                      "Gemini-3-Flash": "#4c72b0",
                      "Qwen-7B base": "#dd8452",
                      "Qwen-7B SFT-3600": "#55a868"}[m])
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=25, ha="right")
    ax.set_ylabel("Mean attempts per solved task")
    ax.set_title("Tier D-lite: feedback adds, retrieval makes solves *cheaper* (fewer attempts)")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 2.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_attempts_per_solved.{ext}"))
    plt.close(fig)
    print("wrote fig5_attempts_per_solved")


# =========================================================================
# Figure 6: Capability ladder -- best cell per model
# =========================================================================
def fig6_capability_ladder():
    models = [
        ("gpt-4o-mini", "confirm100_gpt4omini_*"),
        ("Qwen-7B base", "confirm100_base_qwen_*"),
        ("Qwen-7B SFT-1800", "confirm100_sft_n1800_*"),
        ("Qwen-7B SFT-3600", "confirm100_sft_n3600_*"),
        ("Gemini-3-Flash", "confirm100_gemini3flash_*"),
    ]
    summaries = [(m, load_summary(g)) for m, g in models]
    tiers = ["B", "C_lite", "D_lite"]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    width = 0.27
    x = np.arange(len(models))
    colors_by_tier = ["#4c72b0", "#dd8452", "#55a868"]
    for i, t in enumerate(tiers):
        ys = []
        for _, s in summaries:
            best = 0
            for v in s.values():
                if v["tier"] == t:
                    best = max(best, v["success_rate"])
            ys.append(best)
        ax.bar(x + (i - 1) * width, ys, width,
               label={"B": "Tier B", "C_lite": "C-lite", "D_lite": "D-lite"}[t],
               color=colors_by_tier[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in models], rotation=20, ha="right")
    ax.set_ylabel("Best solve rate across conditions (n=100)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Capability ladder: best (across 4 conditions) per model × tier")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig6_capability_ladder.{ext}"))
    plt.close(fig)
    print("wrote fig6_capability_ladder")


# =========================================================================
def main():
    print(f"Output dir: {OUT}")
    fig1_main_2x2()
    fig2_sft_curve()
    fig3_oracle_vs_struct()
    fig4_base_vs_sft()
    fig5_attempts_per_solved()
    fig6_capability_ladder()
    print("\nAll figures in", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".png", ".pdf")):
            print(" ", f)


if __name__ == "__main__":
    main()
