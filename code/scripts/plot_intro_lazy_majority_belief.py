from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_long_tail_ebrb_experiment as lt
from scripts import run_vertibench_heterogeneity_experiment as exp
from scripts.run_vertibench_heterogeneity_experiment import _StrictVFLXGBTree


BLUE = "#2E86AB"
ORANGE = "#EE7733"
GRAY = "#6B7280"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose majority-biased belief on true minority samples.")
    parser.add_argument("--dataset", default="bank_marketing", choices=["bank_marketing", "adult", "default_credit"])
    parser.add_argument("--ratios", default="0.03,0.02,0.01")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
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


def parse_ratios(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def strict_vfl_xgb_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    *,
    n_estimators: int = 9,
    max_depth: int = 3,
    learning_rate: float = 0.20,
    max_bins: int = 32,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_child_weight: float = 1.0,
    min_samples_leaf: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    classes, y_idx = np.unique(y_train, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("This diagnostic expects binary classification")
    party_ids = sorted(int(pid) for pid in party_features)
    train_blocks: Dict[int, np.ndarray] = {}
    test_blocks: Dict[int, np.ndarray] = {}
    thresholds: Dict[int, List[np.ndarray]] = {}
    bin_ids: Dict[int, np.ndarray] = {}
    for party_id in party_ids:
        features = list(party_features[party_id])
        train_block = np.nan_to_num(np.asarray(X_train[:, features], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        test_block = np.nan_to_num(np.asarray(X_test[:, features], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        train_blocks[party_id] = train_block
        test_blocks[party_id] = test_block
        party_thresholds: List[np.ndarray] = []
        party_bins = np.zeros(train_block.shape, dtype=np.int16)
        for feature_index in range(train_block.shape[1]):
            values = train_block[:, feature_index]
            unique = np.unique(values)
            if unique.size <= 1:
                th = np.asarray([], dtype=float)
            elif unique.size <= max_bins:
                th = (unique[:-1] + unique[1:]) / 2.0
            else:
                quantiles = np.linspace(0.0, 1.0, int(max_bins) + 1)[1:-1]
                th = np.unique(np.quantile(values, quantiles))
            party_thresholds.append(th)
            if th.size:
                party_bins[:, feature_index] = np.searchsorted(th, values, side="right").astype(np.int16)
        thresholds[party_id] = party_thresholds
        bin_ids[party_id] = party_bins

    pos_rate = float(np.clip(np.mean(y_idx), 1e-6, 1.0 - 1e-6))
    base_logit = np.log(pos_rate / (1.0 - pos_rate))
    logits = np.full(len(y_idx), base_logit, dtype=float)
    trees: List[_StrictVFLXGBTree] = []
    for _ in range(max(1, int(n_estimators))):
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradients = proba - y_idx.astype(float)
        hessians = np.maximum(proba * (1.0 - proba), 1e-6)
        tree = _StrictVFLXGBTree(
            thresholds=thresholds,
            bin_ids=bin_ids,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
        )
        tree.fit(gradients, hessians)
        logits += float(learning_rate) * tree.predict_blocks(train_blocks)
        trees.append(tree)

    test_logits = np.full(X_test.shape[0], base_logit, dtype=float)
    for tree in trees:
        test_logits += float(learning_rate) * tree.predict_blocks(test_blocks)
    test_pos_proba = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30.0, 30.0)))
    proba = np.column_stack([1.0 - test_pos_proba, test_pos_proba])
    return classes, proba


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
            classes, proba = strict_vfl_xgb_proba(X_train, y_train_tail, X_test, split.party_features)
            class_to_idx = {str(label): idx for idx, label in enumerate(classes.astype(str))}
            minority_idx = class_to_idx[minority_label]
            majority_idx = 1 - minority_idx
            pred_idx = np.argmax(proba, axis=1)
            y_test_str = np.asarray(y_test_tail, dtype=str)
            y_idx = np.asarray([class_to_idx[value] for value in y_test_str], dtype=int)
            minority_mask = y_test_str == minority_label
            majority_conf = proba[minority_mask, majority_idx]
            minority_conf = proba[minority_mask, minority_idx]
            predicted_majority = pred_idx[minority_mask] == majority_idx
            rows.append(
                {
                    "ratio": ratio,
                    "fold": fold,
                    "minority_n": int(np.sum(minority_mask)),
                    "predicted_majority_rate": float(np.mean(predicted_majority)),
                    "mean_majority_confidence_on_minority": float(np.mean(majority_conf)),
                    "median_majority_confidence_on_minority": float(np.median(majority_conf)),
                    "mean_minority_confidence_on_minority": float(np.mean(minority_conf)),
                    "accuracy": float(accuracy_score(y_idx, pred_idx)),
                    "minority_f1": float(
                        f1_score(y_idx, pred_idx, labels=[minority_idx], average="macro", zero_division=0)
                    ),
                }
            )
            for sample_offset, (maj_conf, min_conf, pred_maj) in enumerate(
                zip(majority_conf, minority_conf, predicted_majority)
            ):
                sample_rows.append(
                    {
                        "ratio": ratio,
                        "fold": fold,
                        "minority_sample_index": sample_offset,
                        "majority_confidence": float(maj_conf),
                        "minority_confidence": float(min_conf),
                        "predicted_majority": int(pred_maj),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def plot_lazy_majority(summary: pd.DataFrame, result_dir: Path, paper_dir: Path) -> None:
    grouped = summary.groupby("ratio", as_index=False).agg(
        predicted_majority_rate=("predicted_majority_rate", "mean"),
        mean_majority_confidence_on_minority=("mean_majority_confidence_on_minority", "mean"),
        minority_f1=("minority_f1", "mean"),
    )
    grouped = grouped.sort_values("ratio", ascending=False)
    labels = [f"{value * 100:.0f}%" for value in grouped["ratio"]]
    x = np.arange(len(labels))
    width = 0.34
    pred_majority = grouped["predicted_majority_rate"].to_numpy() * 100.0
    maj_conf = grouped["mean_majority_confidence_on_minority"].to_numpy() * 100.0

    setup_style()
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    bars_a = ax.bar(
        x - width / 2.0,
        pred_majority,
        width=width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Predicted as majority",
    )
    bars_b = ax.bar(
        x + width / 2.0,
        maj_conf,
        width=width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.8,
        label="Majority confidence",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True minority samples under target ratio")
    ax.set_ylabel("Rate / confidence (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=GRAY,
            )
    fig.tight_layout()
    for directory in (result_dir, paper_dir):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "intro_long_tail_vfl_lazy_majority_belief.png", dpi=450)
        fig.savefig(directory / "intro_long_tail_vfl_lazy_majority_belief.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_figures_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary, sample_rows = run_diagnostic(args)
    summary.to_csv(result_dir / "lazy_majority_belief_summary.csv", index=False)
    sample_rows.to_csv(result_dir / "lazy_majority_belief_samples.csv", index=False)
    plot_lazy_majority(summary, result_dir, paper_dir)
    print(summary.groupby("ratio", as_index=False).mean(numeric_only=True).to_string(index=False))


if __name__ == "__main__":
    main()
