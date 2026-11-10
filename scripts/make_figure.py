"""Grouped bar chart of the three runs (R1, R2, R3) on all metrics, read from the saved metrics.csv files.

Usage (project root; needs pandas and matplotlib only):
    python -m scripts.make_figure                      # writes docs/figures/runs_comparison.{pdf,png}
The dashed line in each panel is the model-free first-N-words baseline of run R2 (273 test articles).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = [("R1: 1.5B, 1000 train (200 test)", "experiments/04_final_1000train_200test/metrics.csv"),
        ("R2: 1.5B, 4907 train (273 test)", "outputs/metrics.csv"),
        ("R3: 3B, 4907 train (273 test)", "experiments/05_3b_full_data/metrics.csv")]
CONDITIONS = ["C1_ZeroShot", "C2_FewShot", "C3_CoT", "C4_LoRA"]
LABELS = ["C1", "C2", "C3", "C4"]
PANELS = [("rougeL", "ROUGE-L", "rougeL"), ("chrf_pp", "chrF++", "chrf_pp"),
          ("bertscore_f1", "BERTScore F1", "bertscore_f1"),
          ("unsupported_entity_rate", "Unsupported entities (%) $\\downarrow$", "unsupported_entity_rate")]
COLORS = ["#9ecae1", "#3182bd", "#e6550d"]  # light/dark blue for the 1.5B runs, orange for 3B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="outputs/baselines.csv")
    ap.add_argument("--output-dir", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = pd.read_csv(args.baselines).set_index("baseline").loc["First-28-words"]
    data = [pd.read_csv(path).set_index("condition").loc[CONDITIONS] for _, path in RUNS]

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15))
    width = 0.26
    x = np.arange(len(CONDITIONS))
    for ax, (col, title, base_col) in zip(axes, PANELS):
        for i, ((name, _), df) in enumerate(zip(RUNS, data)):
            bars = ax.bar(x + (i - 1) * width, df[col], width, label=name, color=COLORS[i])
            ax.bar_label(bars, fmt="%.1f", rotation=90, fontsize=4.6, padding=1.5,
                         bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.8))
        ax.axhline(base[base_col], color="black", ls="--", lw=0.8, label="first-28-words baseline (R2 test)")
        ax.set_ylim(0, max(df[col].max() for df in data) * 1.3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir, f"runs_comparison.{ext}"), dpi=300)
    print(f"[INFO] wrote {args.output_dir}/runs_comparison.pdf and .png")


if __name__ == "__main__":
    main()
