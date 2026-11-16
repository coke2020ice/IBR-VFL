from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_k10_active_weak_ablation import active_weak_split  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import long_tail_metrics, long_tail_resample  # noqa: E402
from vfl_tuned_dr_mbrb.belief import EPS  # noqa: E402


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        numeric_keys = sorted(
            {
                item_key
                for item in group
                for item_key, value in item.items()
                if item_key not in group_keys and isinstance(value, (int, float))
            }
        )
        for item_key in numeric_keys:
            vals = np.asarray([float(item[item_key]) for item in group if item_key in item], dtype=float)
            if vals.size:
                summary[f"{item_key}_mean"] = float(np.mean(vals))
                summary[f"{item_key}_std"] = float(np.std(vals))
        summary["n"] = len(group)
        out.append(summary)
    return out


def make_model(args: argparse.Namespace, seed: int) -> Any:
    rules_per_party = max(4, int(math.ceil(args.total_rule_budget / max(args.n_parties, 1))))
    return exp._StrictVFLBoostedEBRB(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        sample_fraction=args.sample_fraction,
        n_rules_per_party=rules_per_party,
        bins=6,
        max_antecedents=2,
        beam_width=2 * rules_per_party,
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
        server_logit_la_gamma=0.0,
        server_logit_la_temperature=1.0,
        server_logit_la_blend=0.0,
        server_class_conditioned_fusion=False,
        server_class_weight_power=1.0,
        client_fit_weight_power=0.0,
        random_state=seed,
    )


def choose_target_party(party_features: Mapping[int, Sequence[int]], active_party_id: int = 0) -> int:
    candidates = [party_id for party_id in sorted(party_features) if party_id != active_party_id]
    if not candidates:
        raise ValueError("Attribute inference needs at least one passive party")
    return max(candidates, key=lambda party_id: (len(party_features[party_id]), party_id))


def normalized_activation(party: Any, local_X: np.ndarray) -> np.ndarray:
    match = party._rule_match_matrix(local_X)
    theta = np.asarray([max(rule.theta, EPS) for rule in party.rules_], dtype=float)
    reliability = np.asarray([max(rule.reliability, EPS) for rule in party.rules_], dtype=float)
    strength = match * theta[None, :] * reliability[None, :]
    return strength / np.maximum(np.sum(strength, axis=1, keepdims=True), EPS)


def intermediate_views(model: Any, X: np.ndarray, target_party: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    beliefs = []
    activations = []
    metadata = []
    for estimator in model.estimators_:
        if target_party not in estimator.parties_:
            continue
        party = estimator.parties_[target_party]
        local_X = X[:, estimator.party_features_[target_party]]
        belief = party.predict_proba(local_X)
        activation = normalized_activation(party, local_X)
        betas = np.stack([rule.beta for rule in party.rules_], axis=0)
        scores = np.asarray([rule.score for rule in party.rules_], dtype=float)
        reliability = np.asarray([rule.reliability for rule in party.rules_], dtype=float)
        supports = np.asarray([rule.support for rule in party.rules_], dtype=float)
        activation_meta = np.concatenate(
            [
                activation,
                activation * scores[None, :],
                activation * reliability[None, :],
                activation * supports[None, :],
                activation @ betas,
                np.max(activation[:, :, None] * betas[None, :, :], axis=1),
            ],
            axis=1,
        )
        beliefs.append(belief)
        activations.append(activation)
        metadata.append(activation_meta)
    if not beliefs:
        raise ValueError(f"Target party {target_party} is not present in the fitted model")
    belief_view = np.concatenate(beliefs, axis=1)
    mean_belief_view = np.mean(np.stack(beliefs, axis=1), axis=1)
    activation_view = np.concatenate(activations, axis=1)
    metadata_view = np.concatenate(metadata, axis=1)
    return {
        "Random baseline": rng.normal(0.0, 1.0, size=(X.shape[0], max(activation_view.shape[1], 1))),
        "Mean belief only": mean_belief_view,
        "Quantized mean belief": np.round(mean_belief_view, 2),
        "Belief only": belief_view,
        "Activation summary": activation_view,
        "Activation + metadata": metadata_view,
    }


def make_feature_bins(train_values: np.ndarray, test_values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray] | None:
    train = np.asarray(train_values, dtype=float)
    test = np.asarray(test_values, dtype=float)
    if np.unique(train).size < 3:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    thresholds = np.unique(np.quantile(train, quantiles))
    if thresholds.size == 0:
        return None
    y_train = np.digitize(train, thresholds, right=False)
    y_test = np.digitize(test, thresholds, right=False)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None
    return y_train.astype(int), y_test.astype(int)


def attack_models(seed: int) -> Dict[str, Any]:
    return {
        "Dummy-stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_attack(
    X_attack_train: np.ndarray,
    y_attack_train: np.ndarray,
    X_attack_test: np.ndarray,
    y_attack_test: np.ndarray,
    *,
    seed: int,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for attack_name, attack in attack_models(seed).items():
        try:
            attack.fit(X_attack_train, y_attack_train)
            pred = attack.predict(X_attack_test)
            rows.append(
                {
                    "attack_model": attack_name,
                    "attack_accuracy": float(accuracy_score(y_attack_test, pred)),
                    "attack_macro_f1": float(f1_score(y_attack_test, pred, average="macro", zero_division=0)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "attack_model": attack_name,
                    "attack_accuracy": float("nan"),
                    "attack_macro_f1": float("nan"),
                    "error": str(exc),
                }
            )
    return rows


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    spec = exp.DatasetSpec("bank_marketing", ("bank-marketing",))
    X_raw, y_raw = exp.load_dataset(spec)
    y_raw = np.asarray(y_raw, dtype=object).astype(str)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=13)
    splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=13)
    rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    start_all = time.perf_counter()

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
        target_party = choose_target_party(split.party_features, active_party_id=0)
        target_features = list(split.party_features[target_party])
        model = make_model(args, seed=8100 + fold)
        print(
            f"fold={fold} train={len(y_train)} test={len(y_test)} minority={minority_label} "
            f"target_party={target_party} target_features={target_features}",
            flush=True,
        )
        model.fit(X_train, y_train, party_features=split.party_features, active_party_id=0)
        pred = model.predict(X_test)
        perf = long_tail_metrics(y_test, pred, minority_label)
        fold_rows.append(
            {
                "fold": fold,
                "target_party": target_party,
                "target_feature_count": len(target_features),
                "model_accuracy": perf["accuracy"],
                "model_minority_f1": perf["minority_f1"],
                "model_minority_recall": perf["minority_recall"],
                "model_pred_minority_ratio": perf["pred_minority_ratio"],
            }
        )
        rng_train = np.random.default_rng(10000 + fold)
        rng_test = np.random.default_rng(20000 + fold)
        train_views = intermediate_views(model, X_train, target_party, rng_train)
        test_views = intermediate_views(model, X_test, target_party, rng_test)

        for local_pos, global_feature_idx in enumerate(target_features):
            binned = make_feature_bins(
                X_train[:, global_feature_idx],
                X_test[:, global_feature_idx],
                n_bins=args.feature_bins,
            )
            if binned is None:
                continue
            y_attack_train, y_attack_test = binned
            n_classes = int(np.unique(y_attack_train).size)
            for view_name, X_attack_train in train_views.items():
                X_attack_test = test_views[view_name]
                for attack_result in evaluate_attack(
                    X_attack_train,
                    y_attack_train,
                    X_attack_test,
                    y_attack_test,
                    seed=30000 + fold + local_pos,
                ):
                    row = {
                        "fold": fold,
                        "target_party": target_party,
                        "target_global_feature": int(global_feature_idx),
                        "target_local_feature_position": int(local_pos),
                        "feature_bins": int(args.feature_bins),
                        "target_bucket_classes": n_classes,
                        "view": view_name,
                        "view_dim": int(X_attack_train.shape[1]),
                        "model_accuracy": perf["accuracy"],
                        "model_minority_f1": perf["minority_f1"],
                    }
                    row.update(attack_result)
                    rows.append(row)
        print(f"  model MinF1={perf['minority_f1']:.4f}; attack rows so far={len(rows)}", flush=True)

    summary_by_view_model = summarize(rows, ["view", "attack_model"])
    rf_rows = [row for row in rows if row["attack_model"] == "RandomForest"]
    summary_by_view = summarize(rf_rows, ["view"])
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_sec": float(time.perf_counter() - start_all),
        "settings": vars(args),
        "folds": fold_rows,
        "attacks": rows,
        "summary_by_view": summary_by_view,
        "summary_by_view_model": summary_by_view_model,
    }
    return result


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: str, result: Mapping[str, Any]) -> None:
    summary = sorted(result["summary_by_view"], key=lambda row: str(row["view"]))
    fold_rows = result["folds"]
    model_minf1 = float(np.mean([row["model_minority_f1"] for row in fold_rows]))
    report_rows = []
    for row in summary:
        report_rows.append(
            {
                "Visible evidence": row["view"],
                "Attack Acc": row.get("attack_accuracy_mean", float("nan")),
                "Attack Macro-F1": row.get("attack_macro_f1_mean", float("nan")),
                "Std Macro-F1": row.get("attack_macro_f1_std", float("nan")),
                "View Dim": row.get("view_dim_mean", float("nan")),
                "Model MinF1": model_minf1,
            }
        )
    lines = [
        "# Privacy attribute inference experiment",
        "",
        "Task: infer the target passive client's local feature buckets from intermediate evidence on Bank Marketing under rho=0.03 and K=10 active-weak VFL.",
        "",
        "Main attacker: RandomForest. Attack training uses the training fold; attack evaluation uses the held-out test fold.",
        "",
        markdown_table(
            report_rows,
            ["Visible evidence", "Attack Acc", "Attack Macro-F1", "Std Macro-F1", "View Dim", "Model MinF1"],
        ),
        "",
        "Interpretation: this is an empirical leakage-risk test, not a formal cryptographic proof. Full activation-level evidence and per-estimator belief evidence both expose substantial attribute information. Averaging belief over estimators reduces leakage but does not eliminate it. Coarse quantization of the averaged belief sharply reduces the attack signal in this attack-only view; if quantized belief is used inside the actual inference protocol, its predictive-performance impact should be evaluated separately. In deployment, activation-level evidence should be minimized or protected with secure aggregation, quantization, noise, or encryption.",
    ]
    with open(os.path.join(output_dir, "privacy_attribute_inference_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute inference attack on VFL-EBRB intermediate evidence.")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "privacy_attribute_inference"))
    parser.add_argument("--rho", type=float, default=0.03)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--total-rule-budget", type=int, default=72)
    parser.add_argument("--n-estimators", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--sample-fraction", type=float, default=0.85)
    parser.add_argument("--feature-bins", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result = run_experiment(args)
    write_csv(os.path.join(args.output_dir, "attribute_inference_attack_rows.csv"), result["attacks"])
    write_csv(os.path.join(args.output_dir, "attribute_inference_summary_by_view.csv"), result["summary_by_view"])
    write_csv(os.path.join(args.output_dir, "attribute_inference_summary_by_view_model.csv"), result["summary_by_view_model"])
    write_csv(os.path.join(args.output_dir, "attribute_inference_folds.csv"), result["folds"])
    with open(os.path.join(args.output_dir, "attribute_inference_experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    write_report(args.output_dir, result)
    print(f"Done. Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
