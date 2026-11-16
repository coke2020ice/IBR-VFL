from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_long_tail_ebrb_experiment as lt  # noqa: E402
from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_vertibench_heterogeneity_experiment import _StrictVFLBoostedEBRB  # noqa: E402


BLUE = "#2E86AB"
ORANGE = "#EE7733"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EBRB belief bias on true minority samples under strict VFL long-tail settings."
    )
    parser.add_argument("--dataset", default="bank_marketing", choices=["bank_marketing", "adult", "default_credit"])
    parser.add_argument("--ratios", default="0.03,0.02,0.01")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--rules-per-party", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="results/intro_long_tail_vfl_figures")
    parser.add_argument("--paper-figures-dir", default="/Users/lin/Desktop/福州大学第三篇论文/figures")
    return parser.parse_args()


def dataset_spec(name: str) -> exp.DatasetSpec:
    specs = {
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    return specs[name]


def parse_ratios(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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


def fit_strict_vfl_ebrb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    *,
    rules_per_party: int,
    seed: int,
) -> _StrictVFLBoostedEBRB:
    rules_per_party = int(max(1, rules_per_party))
    beam_width = int(max(2 * rules_per_party, rules_per_party))
    model = _StrictVFLBoostedEBRB(
        n_estimators=9,
        learning_rate=0.25,
        sample_fraction=0.85,
        n_rules_per_party=rules_per_party,
        bins=6,
        max_antecedents=2,
        beam_width=beam_width,
        beta_smoothing=0.5,
        min_support_fraction=0.02,
        class_balance_power=0.0,
        parameter_iterations=0,
        parameter_update_blend=0.5,
        sampling_class_balance_power=0.5,
        client_weight_method="label_accuracy",
        fusion_method="log",
        weight_update_method="residual",
        decision_threshold_method="train_prior",
        decision_threshold_prior_multiplier=0.5,
        server_logit_la_gamma=0.0,
        server_logit_la_temperature=1.0,
        server_logit_la_blend=0.0,
        server_class_conditioned_fusion=False,
        server_class_weight_power=1.0,
        client_fit_weight_power=0.0,
        random_state=seed,
    )
    model.fit(X_train, y_train, party_features=party_features, active_party_id=active_party_id)
    return model


def run_diagnostic(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_raw, y_raw = exp.load_dataset(dataset_spec(args.dataset))
    y_raw = np.asarray(y_raw, dtype=object).astype(str)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=args.seed)
    ratios = parse_ratios(args.ratios)
    splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    rows = []
    sample_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, exp.stratify_labels(y_raw)), start=1):
        y_train_base = y_raw[train_idx]
        labels, counts = np.unique(y_train_base, return_counts=True)
        minority_label = str(labels[int(np.argmin(counts))])
        for ratio in ratios:
            X_train_tail_raw, y_train_tail = lt.long_tail_resample(
                X_raw[train_idx],
                y_train_base,
                minority_label=minority_label,
                target_minority_ratio=ratio,
                seed=1000 * fold + int(round(ratio * 1000)),
            )
            X_test_tail_raw, y_test_tail = lt.long_tail_resample(
                X_raw[test_idx],
                y_raw[test_idx],
                minority_label=minority_label,
                target_minority_ratio=ratio,
                seed=2000 * fold + int(round(ratio * 1000)),
            )
            encoder = exp.TabularEncoder()
            X_train = encoder.fit_transform(X_train_tail_raw)
            X_test = encoder.transform(X_test_tail_raw)
            split = lt.active_weak_split(
                X_train,
                y_train_tail,
                n_parties=args.n_parties,
                rank_strength=args.rank_strength,
                seed=args.seed + 100 * fold,
            )
            model = fit_strict_vfl_ebrb(
                X_train,
                np.asarray(y_train_tail, dtype=object),
                split.party_features,
                0,
                rules_per_party=args.rules_per_party,
                seed=args.seed + 1000 * fold + int(round(ratio * 1000)),
            )
            proba = model.predict_proba(X_test)
            pred = model.predict(X_test).astype(str)
            classes = np.asarray(model.classes_, dtype=str)
            class_to_idx = {str(label): idx for idx, label in enumerate(classes)}
            minority_idx = class_to_idx[minority_label]
            majority_labels = [label for label in classes if label != minority_label]
            if len(majority_labels) != 1:
                raise ValueError("This diagnostic expects binary classification")
            majority_label = str(majority_labels[0])
            majority_idx = class_to_idx[majority_label]
            y_test_str = np.asarray(y_test_tail, dtype=str)
            y_idx = np.asarray([class_to_idx[value] for value in y_test_str], dtype=int)
            pred_idx = np.asarray([class_to_idx[value] for value in pred], dtype=int)
            minority_mask = y_test_str == minority_label
            majority_belief = proba[minority_mask, majority_idx]
            minority_belief = proba[minority_mask, minority_idx]
            belief_margin = majority_belief - minority_belief
            pred_majority = pred[minority_mask] == majority_label
            argmax_majority = np.argmax(proba[minority_mask], axis=1) == majority_idx

            rows.append(
                {
                    "ratio": ratio,
                    "fold": fold,
                    "minority_n": int(np.sum(minority_mask)),
                    "mean_majority_belief_on_minority": float(np.mean(majority_belief)),
                    "mean_minority_belief_on_minority": float(np.mean(minority_belief)),
                    "belief_margin_majority_minus_minority": float(np.mean(belief_margin)),
                    "majority_belief_dominance_rate": float(np.mean(belief_margin > 0.0)),
                    "predicted_majority_rate": float(np.mean(pred_majority)),
                    "argmax_majority_rate": float(np.mean(argmax_majority)),
                    "accuracy": float(accuracy_score(y_idx, pred_idx)),
                    "minority_f1": float(
                        f1_score(y_idx, pred_idx, labels=[minority_idx], average="macro", zero_division=0)
                    ),
                }
            )
            for sample_offset, (maj, min_, margin, pred_maj, argmax_maj) in enumerate(
                zip(majority_belief, minority_belief, belief_margin, pred_majority, argmax_majority)
            ):
                sample_rows.append(
                    {
                        "ratio": ratio,
                        "fold": fold,
                        "minority_sample_index": sample_offset,
                        "majority_belief": float(maj),
                        "minority_belief": float(min_),
                        "belief_margin": float(margin),
                        "predicted_majority": int(pred_maj),
                        "argmax_majority": int(argmax_maj),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def save(fig: plt.Figure, stem: str, result_dir: Path, paper_dir: Path) -> None:
    for directory in (result_dir, paper_dir):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.png", dpi=450)
        fig.savefig(directory / f"{stem}.svg")
    plt.close(fig)


def plot_belief_margin(summary: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    grouped = summary.groupby("ratio", as_index=False).agg(
        belief_margin_majority_minus_minority=("belief_margin_majority_minus_minority", "mean"),
        majority_belief_dominance_rate=("majority_belief_dominance_rate", "mean"),
        predicted_majority_rate=("predicted_majority_rate", "mean"),
    )
    grouped = grouped.sort_values("ratio", ascending=False)
    labels = [f"{value * 100:.0f}%" for value in grouped["ratio"]]
    margin = grouped["belief_margin_majority_minus_minority"].to_numpy() * 100.0
    x = np.arange(len(labels))

    setup_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    bars = ax.bar(x, margin, width=0.58, color=ORANGE, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True minority samples under target ratio")
    ax.set_ylabel("EBRB belief margin (pp)")
    y_min = min(0.0, float(np.min(margin)) - 5.0)
    y_max = max(10.0, float(np.max(margin)) + 10.0)
    ax.set_ylim(y_min, y_max)
    for idx, bar in enumerate(bars):
        value = bar.get_height()
        va = "bottom" if value >= 0 else "top"
        offset = 1.5 if value >= 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.1f}",
            ha="center",
            va=va,
            fontsize=7,
            color=GRAY,
        )
    fig.tight_layout()
    save(fig, "intro_long_tail_vfl_ebrb_minority_belief_margin", result_dir, paper_dir)


def plot_belief_pair(summary: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    grouped = summary.groupby("ratio", as_index=False).agg(
        mean_majority_belief_on_minority=("mean_majority_belief_on_minority", "mean"),
        mean_minority_belief_on_minority=("mean_minority_belief_on_minority", "mean"),
    )
    grouped = grouped.sort_values("ratio", ascending=False)
    labels = [f"{value * 100:.0f}%" for value in grouped["ratio"]]
    x = np.arange(len(labels))
    width = 0.34
    majority = grouped["mean_majority_belief_on_minority"].to_numpy() * 100.0
    minority = grouped["mean_minority_belief_on_minority"].to_numpy() * 100.0

    setup_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    bars_a = ax.bar(
        x - width / 2.0,
        majority,
        width=width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Majority belief",
    )
    bars_b = ax.bar(
        x + width / 2.0,
        minority,
        width=width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Minority belief",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True minority samples under target ratio")
    ax.set_ylabel("Mean EBRB belief (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=GRAY,
            )
    fig.tight_layout()
    save(fig, "intro_long_tail_vfl_ebrb_minority_belief_pair", result_dir, paper_dir)


def plot_belief_split(summary: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    grouped = summary.groupby("ratio", as_index=False).agg(
        mean_majority_belief_on_minority=("mean_majority_belief_on_minority", "mean"),
        mean_minority_belief_on_minority=("mean_minority_belief_on_minority", "mean"),
    )
    grouped = grouped.sort_values("ratio", ascending=False)
    labels = [f"{value * 100:.0f}%" for value in grouped["ratio"]]
    majority = grouped["mean_majority_belief_on_minority"].to_numpy() * 100.0
    minority = grouped["mean_minority_belief_on_minority"].to_numpy() * 100.0
    y_pos = np.arange(len(labels))

    setup_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    ax.barh(
        y_pos,
        majority,
        height=0.56,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Majority belief",
    )
    ax.barh(
        y_pos,
        minority,
        left=majority,
        height=0.56,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Minority belief",
    )
    ax.axvline(50, color=GRAY, linestyle=(0, (3, 2)), linewidth=0.8, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean EBRB belief on true minority samples (%)")
    ax.set_ylabel("Target minority ratio")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    for idx, (maj, min_) in enumerate(zip(majority, minority)):
        ax.text(maj / 2.0, idx, f"{maj:.0f}%", ha="center", va="center", fontsize=7, color="white")
        ax.text(maj + min_ / 2.0, idx, f"{min_:.0f}%", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    save(fig, "intro_long_tail_vfl_ebrb_minority_belief_split", result_dir, paper_dir)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_figures_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary, sample_rows = run_diagnostic(args)
    summary.to_csv(result_dir / "ebrb_minority_belief_gap_summary.csv", index=False)
    sample_rows.to_csv(result_dir / "ebrb_minority_belief_gap_samples.csv", index=False)
    plot_belief_margin(summary, result_dir, paper_dir)
    plot_belief_pair(summary, result_dir, paper_dir)
    plot_belief_split(summary, result_dir, paper_dir)
    print(summary.groupby("ratio", as_index=False).mean(numeric_only=True).to_string(index=False))


if __name__ == "__main__":
    main()
