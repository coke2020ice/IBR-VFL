from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from vfl_tuned_dr_mbrb.heterogeneity import evaluate_heterogeneity_split  # noqa: E402


Runner = Callable[
    [np.ndarray, np.ndarray, np.ndarray, Mapping[int, Sequence[int]], int, int],
    Tuple[np.ndarray, Dict[str, float]],
]


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def long_tail_resample(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: object,
    target_minority_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=object)
    min_idx = np.flatnonzero(y_arr == minority_label)
    maj_idx = np.flatnonzero(y_arr != minority_label)
    if min_idx.size == 0 or maj_idx.size == 0:
        raise ValueError("Long-tail resampling needs both minority and majority samples")

    target = float(np.clip(target_minority_ratio, 1e-4, 0.499))
    desired_min = int(np.floor(target / (1.0 - target) * maj_idx.size))
    if desired_min <= min_idx.size:
        keep_min = rng.choice(min_idx, size=max(1, desired_min), replace=False)
        keep_maj = maj_idx
    else:
        desired_maj = int(np.floor((1.0 - target) / target * min_idx.size))
        keep_min = min_idx
        keep_maj = rng.choice(maj_idx, size=max(1, min(desired_maj, maj_idx.size)), replace=False)
    keep = np.concatenate([keep_min, keep_maj]).astype(int)
    rng.shuffle(keep)
    return X[keep], y_arr[keep]


def long_tail_metrics(y_true: Sequence[object], y_pred: Sequence[object], minority_label: object) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "minority_precision": float(precision_score(y_true_arr, y_pred_arr, labels=[minority_label], average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_true_arr, y_pred_arr, labels=[minority_label], average="macro", zero_division=0)),
        "minority_f1": float(f1_score(y_true_arr, y_pred_arr, labels=[minority_label], average="macro", zero_division=0)),
        "true_minority_ratio": float(np.mean(y_true_arr == minority_label)),
        "pred_minority_ratio": float(np.mean(y_pred_arr == minority_label)),
    }


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


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(values))
        out[f"{key}_std"] = float(np.nanstd(values))
    return out


def evaluate_runner(
    runner: Runner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
    minority_label: object,
) -> Dict[str, float]:
    start = time.perf_counter()
    pred, extra = runner(X_train, y_train, X_test, party_features, active_party_id, seed)
    row = long_tail_metrics(y_test, pred, minority_label)
    row["wall_time_sec"] = float(time.perf_counter() - start)
    for key, value in extra.items():
        row[key] = float(value)
    return row


def make_methods(
    base_payload: Dict[str, Any],
    *,
    rules_per_party: int = 24,
    centralized_rules: int = 72,
    n_parties: int = 3,
    use_ablation_boost_params: bool = False,
) -> Dict[str, Runner]:
    rules_per_party = int(max(1, rules_per_party))
    centralized_rules = int(max(rules_per_party, centralized_rules))
    beam_width = int(max(2 * rules_per_party, rules_per_party))
    centralized_beam_width = int(max(2 * centralized_rules, centralized_rules))
    boost_learning_rate = 0.25 if use_ablation_boost_params else 0.20
    boost_sample_fraction = 0.85 if use_ablation_boost_params else 1.0
    mbrb_rules_per_class = int(max(1, np.ceil(rules_per_party / 2.0)))
    active_only_ebrb_base = exp.strict_vfl_ebrb_boost_runner(
        n_estimators=9,
        learning_rate=boost_learning_rate,
        sample_fraction=boost_sample_fraction,
        n_rules_per_party=rules_per_party,
        bins=6,
        max_antecedents=2,
        beam_width=beam_width,
        beta_smoothing=0.5,
        min_support_fraction=0.02,
        class_balance_power=0.0,
        sampling_class_balance_power=0.5,
        client_weight_method="label_accuracy",
        fusion_method="log",
        weight_update_method="residual",
        decision_threshold_method="train_prior",
        decision_threshold_prior_multiplier=0.5,
    )

    centralized_ebrb_base = exp.strict_vfl_ebrb_boost_runner(
        n_estimators=9,
        learning_rate=boost_learning_rate,
        sample_fraction=boost_sample_fraction,
        n_rules_per_party=centralized_rules,
        bins=6,
        max_antecedents=2,
        beam_width=centralized_beam_width,
        beta_smoothing=0.5,
        min_support_fraction=0.02,
        class_balance_power=0.0,
        sampling_class_balance_power=0.5,
        client_weight_method="label_accuracy",
        fusion_method="log",
        weight_update_method="residual",
        decision_threshold_method="train_prior",
        decision_threshold_prior_multiplier=0.5,
    )

    def active_only_ebrb(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        active_features = list(party_features[active_party_id])
        return active_only_ebrb_base(X_train, y_train, X_test, {0: active_features}, 0, seed)

    def centralized_ebrb(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        del party_features, active_party_id
        full_features = list(range(X_train.shape[1]))
        return centralized_ebrb_base(X_train, y_train, X_test, {0: full_features}, 0, seed)

    def strict_vfl_ebrb_lt_iter_scer(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        from scripts.run_scer_robustness_experiment import (  # noqa: WPS433
            SCERRobustnessConfig,
            StrictVFLEBRBLTIterSCER,
        )

        config = SCERRobustnessConfig(
            name="top4-G64-g0.75",
            group="overall",
            n_parties=int(n_parties),
            total_rule_budget=int(centralized_rules),
            n_estimators=9,
            learning_rate=boost_learning_rate,
            sample_fraction=boost_sample_fraction,
            top_rules_per_party=4,
            combo_budget=64,
            combo_gamma=0.75,
            combo_min_support=0.001,
            parameter_iterations=2,
            parameter_update_blend=0.01,
            threshold_prior_multiplier=1.0,
        )
        model = StrictVFLEBRBLTIterSCER(config, random_state=seed)
        model.fit(X_train, y_train, party_features=party_features, active_party_id=active_party_id)
        return model.predict(X_test), model.diagnostics()

    methods: Dict[str, Runner] = {
        "Strict VFL-LR": exp.strict_vfl_lr,
        "Strict VFL-GaussianNB": exp.strict_vfl_gaussian_nb,
        "Strict VFL-XGBoost Hist 9": exp.strict_vfl_xgboost_hist_runner(
            n_estimators=9,
            max_depth=3,
            learning_rate=0.20,
            max_bins=32,
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=1.0,
            min_samples_leaf=10,
        ),
        "Strict VFL-XGBoost Hist 9 LT": exp.strict_vfl_xgboost_hist_runner(
            n_estimators=9,
            max_depth=3,
            learning_rate=0.20,
            max_bins=32,
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=1.0,
            min_samples_leaf=10,
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=1.0,
        ),
        "FedTree Vertical GBDT 9": exp.fedtree_vertical_runner(
            fedtree_root="../external/FedTree",
            n_trees=9,
            depth=3,
            learning_rate=0.20,
            timeout_sec=300,
        ),
        "Strict VFL-Tuned DR-MBRB boost": exp.boosted_dr_mbrb_runner(
            base_payload,
            n_estimators=9,
            sample_fraction=boost_sample_fraction,
            learning_rate=boost_learning_rate,
            fusion_method="log",
            weight_update_method="residual",
            sampling_class_balance_power=0.5,
            class_conditioned_fusion=True,
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=0.5,
            final_la_gamma=1.2,
            final_la_temperature=0.8,
            final_la_blend=0.75,
            prior_gated_final_la=False,
            min_class_prior_for_la=0.15,
            strict_vfl=True,
            n_modules_per_party=1,
            n_rules_per_class=mbrb_rules_per_class,
            n_rules_per_module=rules_per_party,
        ),
        "Strict VFL-Tuned DR-MBRB boost LT": exp.boosted_dr_mbrb_runner(
            base_payload,
            n_estimators=9,
            sample_fraction=boost_sample_fraction,
            learning_rate=boost_learning_rate,
            fusion_method="log",
            weight_update_method="residual",
            sampling_class_balance_power=1.0,
            class_conditioned_fusion=True,
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=1.0,
            final_la_gamma=1.2,
            final_la_temperature=0.8,
            final_la_blend=0.75,
            prior_gated_final_la=False,
            min_class_prior_for_la=0.15,
            strict_vfl=True,
            n_modules_per_party=1,
            n_rules_per_class=mbrb_rules_per_class,
            n_rules_per_module=rules_per_party,
        ),
        "Active-only EBRB": active_only_ebrb,
        "Centralized-EBRB": centralized_ebrb,
        "Strict VFL-EBRB boost": exp.strict_vfl_ebrb_boost_runner(
            n_estimators=9,
            learning_rate=boost_learning_rate,
            sample_fraction=boost_sample_fraction,
            n_rules_per_party=rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=beam_width,
            beta_smoothing=0.5,
            min_support_fraction=0.02,
            class_balance_power=0.0,
            sampling_class_balance_power=0.5,
            client_weight_method="label_accuracy",
            fusion_method="log",
            weight_update_method="residual",
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=0.5,
        ),
        "Strict VFL-EBRB boost LT-Iter": exp.strict_vfl_ebrb_boost_runner(
            n_estimators=9,
            learning_rate=boost_learning_rate,
            sample_fraction=boost_sample_fraction,
            n_rules_per_party=rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=beam_width,
            beta_smoothing=0.5,
            min_support_fraction=0.02,
            class_balance_power=0.5,
            parameter_iterations=2,
            parameter_update_blend=0.01,
            sampling_class_balance_power=0.75,
            client_weight_method="label_accuracy",
            fusion_method="log",
            weight_update_method="residual",
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=1.0,
        ),
        "Strict VFL-EBRB boost LT-Iter+SCER": strict_vfl_ebrb_lt_iter_scer,
        "Strict VFL-EBRB boost LT-Iter-p0.75": exp.strict_vfl_ebrb_boost_runner(
            n_estimators=9,
            learning_rate=boost_learning_rate,
            sample_fraction=boost_sample_fraction,
            n_rules_per_party=rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=beam_width,
            beta_smoothing=0.5,
            min_support_fraction=0.02,
            class_balance_power=0.5,
            parameter_iterations=2,
            parameter_update_blend=0.01,
            sampling_class_balance_power=0.75,
            client_weight_method="label_accuracy",
            fusion_method="log",
            weight_update_method="residual",
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=0.75,
        ),
        "Strict VFL-EBRB boost LT-Iter-p0.5": exp.strict_vfl_ebrb_boost_runner(
            n_estimators=9,
            learning_rate=boost_learning_rate,
            sample_fraction=boost_sample_fraction,
            n_rules_per_party=rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=beam_width,
            beta_smoothing=0.5,
            min_support_fraction=0.02,
            class_balance_power=0.5,
            parameter_iterations=2,
            parameter_update_blend=0.01,
            sampling_class_balance_power=0.75,
            client_weight_method="label_accuracy",
            fusion_method="log",
            weight_update_method="residual",
            decision_threshold_method="train_prior",
            decision_threshold_prior_multiplier=0.5,
        ),
    }
    return methods


PAPER_METHOD_ORDER = [
    "Strict VFL-LR",
    "Strict VFL-GaussianNB",
    "Strict VFL-XGBoost Hist 9",
    "Strict VFL-XGBoost Hist 9 LT",
    "FedTree Vertical GBDT 9",
    "Strict VFL-Tuned DR-MBRB boost",
    "Strict VFL-Tuned DR-MBRB boost LT",
    "Strict VFL-EBRB boost",
    "Strict VFL-EBRB boost LT-Iter",
    "Strict VFL-EBRB boost LT-Iter+SCER",
    "Strict VFL-EBRB boost LT-Iter-p0.75",
    "Strict VFL-EBRB boost LT-Iter-p0.5",
]

PAPER_METHOD_DISPLAY = {
    "Strict VFL-GaussianNB": "Strict VFL-GNB",
    "Strict VFL-XGBoost Hist 9": "Strict VFL-XGB Hist",
    "Strict VFL-XGBoost Hist 9 LT": "Strict VFL-XGB Hist-LT",
    "FedTree Vertical GBDT 9": "Vertical FedTree-GBDT",
    "Strict VFL-Tuned DR-MBRB boost": "Strict VFL-MBRB boost",
    "Strict VFL-Tuned DR-MBRB boost LT": "Strict VFL-MBRB boost-LT",
    "Strict VFL-EBRB boost LT-Iter": "Strict VFL-EBRB boost-LT-Iter",
    "Strict VFL-EBRB boost LT-Iter+SCER": "Strict VFL-EBRB-LT-Iter+SCER",
    "Strict VFL-EBRB boost LT-Iter-p0.75": "Strict VFL-EBRB boost-LT-Iter-p0.75",
    "Strict VFL-EBRB boost LT-Iter-p0.5": "Strict VFL-EBRB boost-LT-Iter-p0.5",
}


def collect_method_summary(
    dataset_result: Mapping[str, Any],
    ratio: str,
    method_name: str,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for mode_payload in dataset_result["results"][ratio].values():
        payload = mode_payload.get(method_name)
        if payload is not None:
            rows.extend(payload["folds"])
    if not rows:
        return {}
    return summarize(rows)


def write_paper_table_tsv(path: str, result: Mapping[str, Any]) -> None:
    ratios = [f"{float(value):.2f}" for value in result["ratios"]]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        header = ["dataset", "method"]
        for ratio in ratios:
            header.extend([f"acc_{ratio}", f"minf1_{ratio}"])
        handle.write("\t".join(header) + "\n")
        for dataset_name, dataset_result in result["datasets"].items():
            for method_name in PAPER_METHOD_ORDER:
                display_name = PAPER_METHOD_DISPLAY.get(method_name, method_name)
                values = [dataset_name, display_name]
                for ratio in ratios:
                    mean = collect_method_summary(dataset_result, ratio, method_name)
                    if not mean:
                        values.extend(["nan", "nan"])
                    else:
                        values.extend(
                            [
                                f"{mean['accuracy_mean']:.4f}",
                                f"{mean['minority_f1_mean']:.4f}",
                            ]
                        )
                handle.write("\t".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict VFL long-tail EBRB analysis.")
    parser.add_argument("--config", default="configs/vfl_tuned_dr_mbrb_demo.json")
    parser.add_argument("--datasets", default="default_credit,adult,bank_marketing")
    parser.add_argument("--ratios", default="0.20,0.10,0.05")
    parser.add_argument("--heterogeneity", default="balanced,importance_skew")
    parser.add_argument("--n-parties", type=int, default=3)
    parser.add_argument("--split-strategy", choices=["standard", "active_weak"], default="standard")
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--total-rule-budget", type=int, default=0)
    parser.add_argument("--use-ablation-boost-params", action="store_true")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--output", default="outputs/strict_vfl_long_tail_ebrb_3datasets.json")
    parser.add_argument("--table-output", default="")
    args = parser.parse_args()

    specs = {
        "breast_cancer": exp.DatasetSpec("breast_cancer", sklearn_loader="breast_cancer"),
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    base_payload = exp.load_config(args.config)
    ratios = [float(item) for item in parse_list(args.ratios)]
    heterogeneity_modes = parse_list(args.heterogeneity)
    if args.split_strategy == "active_weak" and heterogeneity_modes == ["balanced", "importance_skew"]:
        heterogeneity_modes = ["active_weak"]
    total_rule_budget = int(args.total_rule_budget)
    if total_rule_budget > 0:
        rules_per_party = int(max(4, np.ceil(total_rule_budget / max(int(args.n_parties), 1))))
        centralized_rules = total_rule_budget
    else:
        rules_per_party = 24
        centralized_rules = max(72, 24 * max(int(args.n_parties), 1))
    methods = make_methods(
        base_payload,
        rules_per_party=rules_per_party,
        centralized_rules=centralized_rules,
        n_parties=int(args.n_parties),
        use_ablation_boost_params=bool(args.use_ablation_boost_params),
    )

    result: Dict[str, Any] = {
        "experiment": "strict_vfl_long_tail_ebrb_analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": int(args.splits),
        "max_samples": int(args.max_samples),
        "ratios": ratios,
        "heterogeneity_modes": heterogeneity_modes,
        "n_parties": int(args.n_parties),
        "split_strategy": args.split_strategy,
        "rank_strength": float(args.rank_strength),
        "total_rule_budget": int(total_rule_budget) if total_rule_budget > 0 else None,
        "rules_per_party": int(rules_per_party),
        "centralized_rules": int(centralized_rules),
        "use_ablation_boost_params": bool(args.use_ablation_boost_params),
        "notes": {
            "long_tail_protocol": "For each fold, both train and test are resampled to the target minority ratio using the train-fold minority label.",
            "strict_vfl": "Passive parties do not receive plaintext labels; EBRB/DR-MBRB return belief evidence to the server.",
            "lt_variant": "LT variants use stronger class-balanced sampling and train-prior decision threshold multiplier=1.0.",
        },
        "datasets": {},
    }

    for dataset_name in parse_list(args.datasets):
        spec = specs[dataset_name]
        X_raw, y_raw = exp.load_dataset(spec)
        y_raw = np.asarray(y_raw, dtype=object).astype(str)
        X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=13)
        splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=13)
        dataset_rows: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]] = {
            f"{ratio:.2f}": {mode: {name: [] for name in methods} for mode in heterogeneity_modes}
            for ratio in ratios
        }
        dataset_meta: Dict[str, List[Dict[str, float]]] = {f"{ratio:.2f}": [] for ratio in ratios}

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, exp.stratify_labels(y_raw)), start=1):
            y_train_base = y_raw[train_idx]
            labels, counts = np.unique(y_train_base, return_counts=True)
            minority_label = labels[int(np.argmin(counts))]
            for ratio in ratios:
                X_train_tail_raw, y_train_tail = long_tail_resample(
                    X_raw[train_idx],
                    y_train_base,
                    minority_label=minority_label,
                    target_minority_ratio=ratio,
                    seed=1000 * fold + int(round(ratio * 1000)),
                )
                X_test_tail_raw, y_test_tail = long_tail_resample(
                    X_raw[test_idx],
                    y_raw[test_idx],
                    minority_label=minority_label,
                    target_minority_ratio=ratio,
                    seed=2000 * fold + int(round(ratio * 1000)),
                )
                encoder = exp.TabularEncoder()
                X_train_base = encoder.fit_transform(X_train_tail_raw)
                X_test_base = encoder.transform(X_test_tail_raw)
                dataset_meta[f"{ratio:.2f}"].append(
                    {
                        "fold": float(fold),
                        "train_n": float(len(y_train_tail)),
                        "test_n": float(len(y_test_tail)),
                        "train_minority_ratio": float(np.mean(y_train_tail == minority_label)),
                        "test_minority_ratio": float(np.mean(y_test_tail == minority_label)),
                    }
                )

                for mode in heterogeneity_modes:
                    if args.split_strategy == "active_weak" or mode == "active_weak":
                        split = active_weak_split(
                            X_train_base,
                            y_train_tail,
                            n_parties=int(args.n_parties),
                            rank_strength=float(args.rank_strength),
                            seed=13 + 100 * fold,
                        )
                    else:
                        split_mode = "balanced" if mode == "noisy_party" else mode
                        split = exp.make_vertibench_style_split(
                            X_train_base,
                            y_train_tail,
                            n_parties=int(args.n_parties),
                            active_party_id=0,
                            mode=split_mode,
                            random_state=13 + 100 * fold,
                        )
                    X_train, X_test = X_train_base, X_test_base
                    if mode == "noisy_party":
                        X_train, X_test = exp.apply_noisy_party(
                            X_train_base,
                            X_test_base,
                            split.party_features,
                            noisy_party_id=1,
                            noise_std=0.75,
                            dropout_rate=0.30,
                            random_state=13 + 1000 * fold,
                        )
                    for method_name, runner in methods.items():
                        dataset_rows[f"{ratio:.2f}"][mode][method_name].append(
                            evaluate_runner(
                                runner,
                                X_train,
                                y_train_tail,
                                X_test,
                                y_test_tail,
                                split.party_features,
                                active_party_id=0,
                                seed=13 + fold,
                                minority_label=minority_label,
                            )
                        )

        result["datasets"][dataset_name] = {
            "long_tail_meta": {
                ratio: {
                    "folds": rows,
                    "mean": summarize(rows),
                }
                for ratio, rows in dataset_meta.items()
            },
            "results": {
                ratio: {
                    mode: {
                        method_name: {
                            "folds": rows,
                            "mean": summarize(rows),
                        }
                        for method_name, rows in methods_rows.items()
                    }
                    for mode, methods_rows in modes.items()
                }
                for ratio, modes in dataset_rows.items()
            },
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    if args.table_output:
        write_paper_table_tsv(args.table_output, result)

    print("| Dataset | Ratio | Heterogeneity | Method | Acc | Bal-Acc | Min-Recall | Min-F1 | Pred-Min-Ratio |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|")
    for dataset_name, dataset_result in result["datasets"].items():
        for ratio, ratio_payload in dataset_result["results"].items():
            for mode, mode_payload in ratio_payload.items():
                for method_name, payload in mode_payload.items():
                    mean = payload["mean"]
                    print(
                        f"| {dataset_name} | {ratio} | {mode} | {method_name} | "
                        f"{mean['accuracy_mean']:.4f} | {mean['balanced_accuracy_mean']:.4f} | "
                        f"{mean['minority_recall_mean']:.4f} | {mean['minority_f1_mean']:.4f} | "
                        f"{mean['pred_minority_ratio_mean']:.4f} |"
                    )
    print(f"\nSaved JSON: {args.output}")
    if args.table_output:
        print(f"Saved paper table TSV: {args.table_output}")


if __name__ == "__main__":
    main()
