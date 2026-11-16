from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BLUE = "#2E86AB"
ORANGE = "#EE7733"
GRAY = "#6B7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot two simple introduction diagnostic figures.")
    parser.add_argument("--result-dir", default="results/intro_vfl_ebrb_diagnostic")
    parser.add_argument("--paper-figures-dir", default="/Users/lin/Desktop/福州大学第三篇论文/figures")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.55,
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


def plot_evidence_line(result_dir: Path, paper_dir: Path) -> None:
    df = pd.read_csv(result_dir / "centralized_rule_evidence_cumulative.csv")
    x = df["top_k"].to_numpy()
    y = df["mean_cumulative_share"].to_numpy()
    tau = 0.80
    reach = int(x[(y >= tau).argmax()])

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(x, y, color=BLUE, marker="o", markersize=3.2, linewidth=1.8)
    ax.axhline(tau, color=GRAY, linestyle="--", linewidth=0.9)
    ax.axvline(reach, color=GRAY, linestyle=":", linewidth=0.9)
    ax.text(reach + 0.12, tau - 0.08, f"k={reach}", color=GRAY, fontsize=7)
    ax.set_xlabel("Top-k clients by rule-match evidence")
    ax.set_ylabel("Cumulative evidence share")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.75, max(x) + 0.25)
    fig.tight_layout()
    save(fig, "intro_diag_line_cross_party_evidence", result_dir, paper_dir)


def plot_disagreement_bar(result_dir: Path, paper_dir: Path) -> None:
    df = pd.read_csv(result_dir / "local_ebrb_conflict_groups.csv")
    order = ["0", "<=0.2", "0.2-0.4", ">0.4"]
    df["disagreement_group"] = pd.Categorical(df["disagreement_group"], categories=order, ordered=True)
    df = df.sort_values("disagreement_group")
    x_labels = df["disagreement_group"].astype(str).tolist()
    y = (df["mean_fusion_error_rate"] * 100.0).to_numpy()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    bars = ax.bar(x_labels, y, color=ORANGE, edgecolor="white", linewidth=0.8, width=0.68)
    ax.set_xlabel("Local EBRB vote disagreement")
    ax.set_ylabel("Simple-mean error rate (%)")
    ax.set_ylim(0, max(y) * 1.24)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(y) * 0.035,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=GRAY,
        )
    fig.tight_layout()
    save(fig, "intro_diag_bar_local_ebrb_disagreement", result_dir, paper_dir)


def main() -> None:
    args = parse_args()
    setup_style()
    result_dir = Path(args.result_dir)
    paper_dir = Path(args.paper_figures_dir)
    plot_evidence_line(result_dir, paper_dir)
    plot_disagreement_bar(result_dir, paper_dir)


if __name__ == "__main__":
    main()
