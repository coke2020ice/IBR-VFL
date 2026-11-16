from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import (  # noqa: E402
    long_tail_metrics,
    long_tail_resample,
    parse_list,
    summarize,
)
from vfl_tuned_dr_mbrb.heterogeneity import evaluate_heterogeneity_split  # noqa: E402


Runner = Callable[
    [np.ndarray, np.ndarray, np.ndarray, Mapping[int, Sequence[int]], int, int],
    Tuple[np.ndarray, Dict[str, float]],
]


def active_weak_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_parties: int,
    rank_strength: float,
    seed: int,
) -> exp.HeterogeneitySplit:
    raw = np.exp(np.linspace(float(rank_strength), 0.0, int(n_parties)))
    skew_weights = (raw / np.sum(raw)).tolist()
    split = exp.make_vertibench_style_split(
        X_train,
        y_train,
        n_parties=n_parties,
        active_party_id=0,
        mode="importance_skew",
        random_state=seed,
        skew_weights=skew_weights,
    )
    ordered = sorted(split.party_importance, key=lambda key: (split.party_importance[key], key))
    party_features = {
        new_id: split.party_features[old_id]
        for new_id, old_id in enumerate(ordered)
    }
    return evaluate_heterogeneity_split("active_weak_importance_skew", party_features, split.feature_importance)


def make_ebrb_runner(
    *,
    n_estimators: int,
    rules_per_party: int,
    learning_rate: float = 0.20,
    tau: float = 1.0,
    sample_fraction: float = 1.0,
    fusion_method: str = "log",
    class_balance_power: float = 0.0,
    sampling_class_balance_power: float = 0.5,
    parameter_iterations: int = 0,
    parameter_update_blend: float = 0.01,
) -> Runner:
    return exp.strict_vfl_ebrb_boost_runner(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        sample_fraction=sample_fraction,
        n_rules_per_party=rules_per_party,
        bins=6,
        max_antecedents=2,
        beam_width=2 * rules_per_party,
        beta_smoothing=0.5,
        min_support_fraction=0.02,
        class_balance_power=class_balance_power,
        sampling_class_balance_power=sampling_class_balance_power,
        client_weight_method="label_accuracy",
        fusion_method=fusion_method,
        weight_update_method="residual",
        decision_threshold_method="train_prior",
        decision_threshold_prior_multiplier=tau,
        parameter_iterations=parameter_iterations,
        parameter_update_blend=parameter_update_blend,
    )


def wrap_scope(runner: Runner, scope: str) -> Runner:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if scope == "local":
            return runner(X_train, y_train, X_test, {0: list(party_features[active_party_id])}, 0, seed)
        if scope == "central":
            return runner(X_train, y_train, X_test, {0: list(range(X_train.shape[1]))}, 0, seed)
        return runner(X_train, y_train, X_test, party_features, active_party_id, seed)

    return run


def majority_runner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del X_train, party_features, active_party_id, seed
    labels, counts = np.unique(y_train, return_counts=True)
    majority_label = labels[int(np.argmax(counts))]
    return np.full(X_test.shape[0], majority_label, dtype=object), {}


def build_methods(rules_per_party: int, centralized_rules: int) -> Dict[str, Runner]:
    iter_kwargs = {
        "class_balance_power": 0.5,
        "sampling_class_balance_power": 0.75,
        "parameter_iterations": 2,
        "parameter_update_blend": 0.01,
    }
    single = make_ebrb_runner(
        n_estimators=1,
        rules_per_party=rules_per_party,
        tau=1.0,
        **iter_kwargs,
    )
    boost = make_ebrb_runner(
        n_estimators=9,
        rules_per_party=rules_per_party,
        learning_rate=0.25,
        tau=1.0,
        sample_fraction=0.85,
        fusion_method="log",
        **iter_kwargs,
    )
    centralized_single = make_ebrb_runner(
        n_estimators=1,
        rules_per_party=centralized_rules,
        tau=1.0,
        **iter_kwargs,
    )
    centralized_boost = make_ebrb_runner(
        n_estimators=9,
        rules_per_party=centralized_rules,
        learning_rate=0.25,
        tau=1.0,
        sample_fraction=0.85,
        fusion_method="log",
        **iter_kwargs,
    )
    return {
        "Majority-only": majority_runner,
        "Local-P0 EBRB-LT-Iter": wrap_scope(single, "local"),
        "Local-P0 EBRB boost-LT-Iter": wrap_scope(boost, "local"),
        "Strict VFL-EBRB-LT-Iter": single,
        "Strict VFL-EBRB boost-LT-Iter": boost,
        "Centralized EBRB-LT-Iter": wrap_scope(centralized_single, "central"),
        "Centralized EBRB boost-LT-Iter": wrap_scope(centralized_boost, "central"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P=10 active-weak EBRB ablation for long-tail VFL.")
    parser.add_argument("--datasets", default="default_credit,adult,bank_marketing")
    parser.add_argument("--rho", type=float, default=0.03)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--total-rule-budget", type=int, default=72)
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "k10_active_weak_ablation_3datasets.json"))
    args = parser.parse_args()

    specs = {
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    rules_per_party = max(4, int(np.ceil(args.total_rule_budget / max(args.n_parties, 1))))
    methods = build_methods(rules_per_party=rules_per_party, centralized_rules=args.total_rule_budget)

    result = {
        "experiment": "p10_active_weak_ablation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rho": float(args.rho),
        "n_parties": int(args.n_parties),
        "rank_strength": float(args.rank_strength),
        "rules_per_party": int(rules_per_party),
        "centralized_rules": int(args.total_rule_budget),
        "splits": int(args.splits),
        "datasets": {},
    }
    start_all = time.perf_counter()
    for dataset_name in parse_list(args.datasets):
        spec = specs[dataset_name]
        X_raw, y_raw = exp.load_dataset(spec)
        y_raw = np.asarray(y_raw, dtype=object).astype(str)
        X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=13)
        splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=13)
        rows = {name: [] for name in methods}
        fold_meta = []
        print(f"\nDataset {dataset_name}", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, exp.stratify_labels(y_raw)), start=1):
            y_train_base = y_raw[train_idx]
            labels, counts = np.unique(y_train_base, return_counts=True)
            minority_label = labels[int(np.argmin(counts))]
            X_train_raw, y_train = long_tail_resample(
                X_raw[train_idx],
                y_train_base,
                minority_label=minority_label,
                target_minority_ratio=args.rho,
                seed=1000 * fold + int(round(args.rho * 1000)),
            )
            X_test_raw, y_test = long_tail_resample(
                X_raw[test_idx],
                y_raw[test_idx],
                minority_label=minority_label,
                target_minority_ratio=args.rho,
                seed=2000 * fold + int(round(args.rho * 1000)),
            )
            encoder = exp.TabularEncoder()
            X_train = encoder.fit_transform(X_train_raw)
            X_test = encoder.transform(X_test_raw)
            split = active_weak_split(
                X_train,
                y_train,
                n_parties=args.n_parties,
                rank_strength=args.rank_strength,
                seed=13 + 100 * fold,
            )
            fold_meta.append(
                {
                    "fold": fold,
                    "minority_label": str(minority_label),
                    "train_n": int(len(y_train)),
                    "test_n": int(len(y_test)),
                    "train_minority_ratio": float(np.mean(y_train == minority_label)),
                    "test_minority_ratio": float(np.mean(y_test == minority_label)),
                    "party_importance": {str(k): float(v) for k, v in split.party_importance.items()},
                    "party_feature_count": {str(k): int(v) for k, v in split.feature_counts.items()},
                    "importance_imbalance_ratio": float(split.importance_imbalance_ratio),
                }
            )
            print(f"  Fold {fold}: n_train={len(y_train)} n_test={len(y_test)}", flush=True)
            for method_name, runner in methods.items():
                t0 = time.perf_counter()
                pred, extra = runner(X_train, y_train, X_test, split.party_features, 0, seed=13 + fold)
                metric = long_tail_metrics(y_test, pred, minority_label)
                metric["wall_time_sec"] = float(time.perf_counter() - t0)
                for key, value in extra.items():
                    try:
                        metric[key] = float(value)
                    except Exception:
                        pass
                rows[method_name].append(metric)
                print(
                    f"    {method_name}: acc={metric['accuracy']:.4f} "
                    f"bal={metric['balanced_accuracy']:.4f} "
                    f"minf1={metric['minority_f1']:.4f}",
                    flush=True,
                )
        result["datasets"][dataset_name] = {
            "fold_meta": fold_meta,
            "results": {
                name: {
                    "folds": method_rows,
                    "mean": summarize(method_rows),
                }
                for name, method_rows in rows.items()
            },
        }
    result["wall_time_sec"] = float(time.perf_counter() - start_all)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\nSaved", args.output)
    print("| Dataset | Method | Acc | Macro-F1 | Bal-Acc | Min-P | Min-R | Min-F1 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset_name, dataset_payload in result["datasets"].items():
        for method_name, payload in dataset_payload["results"].items():
            mean = payload["mean"]
            print(
                f"| {dataset_name} | {method_name} | "
                f"{mean['accuracy_mean']:.4f} | {mean['macro_f1_mean']:.4f} | "
                f"{mean['balanced_accuracy_mean']:.4f} | {mean['minority_precision_mean']:.4f} | "
                f"{mean['minority_recall_mean']:.4f} | {mean['minority_f1_mean']:.4f} |"
            )


if __name__ == "__main__":
    main()
