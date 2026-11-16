from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2E86AB"
ORANGE = "#EE7733"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot long-tail VFL motivation figures.")
    parser.add_argument("--dataset", default="bank_marketing")
    parser.add_argument(
        "--result-json",
        default="results/strict_vfl_all_ratios_all_methods_ablation_params_active_weak_k10.json",
    )
    parser.add_argument(
        "--result-table",
        default="results/strict_vfl_all_ratios_all_methods_ablation_params_active_weak_k10_table.tsv",
    )
    parser.add_argument("--paper-figures-dir", default="/Users/lin/Desktop/福州大学第三篇论文/figures")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str, result_dir: Path, paper_dir: Path) -> None:
    for directory in (result_dir, paper_dir):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.png", dpi=450)
        fig.savefig(directory / f"{stem}.svg")
    plt.close(fig)


def load_distribution(result_json: Path, dataset: str) -> pd.DataFrame:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    meta = data["datasets"][dataset]["long_tail_meta"]
    rows = []
    for ratio_key in ["0.03", "0.02", "0.01"]:
        mean = meta[ratio_key]["mean"]
        train_n = float(mean["train_n_mean"])
        minority_ratio = float(mean["train_minority_ratio_mean"])
        minority = train_n * minority_ratio
        majority = train_n - minority
        rows.append(
            {
                "ratio": ratio_key,
                "majority": majority,
                "minority": minority,
                "minority_ratio": minority_ratio,
                "train_n": train_n,
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    labels = [f"{float(value) * 100:.0f}%" for value in df["ratio"]]
    x = np.arange(len(labels))
    majority = df["majority"].to_numpy()
    minority = df["minority"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    ax.bar(x, majority, color=LIGHT, edgecolor="white", width=0.62, label="Majority")
    ax.bar(x, minority, bottom=majority, color=ORANGE, edgecolor="white", width=0.62, label="Minority")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target minority ratio (K=10 active-weak VFL)")
    ax.set_ylabel("Samples per aligned party")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    for idx, value in enumerate(minority):
        ax.text(idx, majority[idx] + value + 35, f"{int(round(value))}", ha="center", va="bottom", fontsize=7, color=GRAY)
    ax.set_ylim(0, max(majority + minority) * 1.16)
    fig.tight_layout()
    save(fig, "intro_long_tail_vfl_distribution", result_dir, paper_dir)


def load_result_gap(result_table: Path, dataset: str) -> pd.DataFrame:
    table = pd.read_csv(result_table, sep="\t")
    row = table[(table["dataset"] == dataset) & (table["method"] == "Strict VFL-XGB Hist")].iloc[0]
    rows = []
    for ratio in ["0.03", "0.02", "0.01"]:
        rows.append(
            {
                "ratio": ratio,
                "accuracy": float(row[f"acc_{ratio}"]),
                "minority_f1": float(row[f"minf1_{ratio}"]),
            }
        )
    return pd.DataFrame(rows)


def plot_result_gap(df: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    labels = [f"{float(value) * 100:.0f}%" for value in df["ratio"]]
    x = np.arange(len(labels))
    width = 0.34
    acc = df["accuracy"].to_numpy() * 100.0
    minf1 = df["minority_f1"].to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    ax.bar(x - width / 2.0, acc, width=width, color=BLUE, edgecolor="white", linewidth=0.8, label="Accuracy")
    ax.bar(x + width / 2.0, minf1, width=width, color=ORANGE, edgecolor="white", linewidth=0.8, label="Minority F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target minority ratio (K=10 active-weak VFL)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    for idx, value in enumerate(acc):
        ax.text(idx - width / 2.0, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5, color=GRAY)
    for idx, value in enumerate(minf1):
        ax.text(idx + width / 2.0, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5, color=GRAY)
    fig.tight_layout()
    save(fig, "intro_long_tail_vfl_result_gap", result_dir, paper_dir)


def main() -> None:
    args = parse_args()
    setup_style()
    result_json = Path(args.result_json)
    result_table = Path(args.result_table)
    result_dir = result_json.parent / "intro_long_tail_vfl_figures"
    paper_dir = Path(args.paper_figures_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dist = load_distribution(result_json, args.dataset)
    result_gap = load_result_gap(result_table, args.dataset)
    dist.to_csv(result_dir / "long_tail_vfl_distribution.csv", index=False)
    result_gap.to_csv(result_dir / "long_tail_vfl_result_gap.csv", index=False)
    plot_distribution(dist, result_dir, paper_dir)
    plot_result_gap(result_gap, result_dir, paper_dir)
    print(dist.to_string(index=False))
    print("---")
    print(result_gap.to_string(index=False))


if __name__ == "__main__":
    main()
