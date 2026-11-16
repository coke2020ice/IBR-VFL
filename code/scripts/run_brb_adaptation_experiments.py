from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.model_selection import StratifiedKFold

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_k10_active_weak_ablation import active_weak_split  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import long_tail_metrics, long_tail_resample  # noqa: E402
from scripts.run_scer_robustness_experiment import (  # noqa: E402
    SCERRobustnessConfig,
    StrictVFLEBRBLTIterSCER,
)
from vfl_tuned_dr_mbrb.belief import EPS  # noqa: E402


METHOD_LABELS = {
    "Strict VFL-GaussianNB": "Strict VFL-GNB",
    "Strict VFL-XGBoost Hist 9": "Strict VFL-XGB Hist",
    "Strict VFL-XGBoost Hist 9 LT": "Strict VFL-XGB Hist-LT",
    "FedTree Vertical GBDT 9": "Vertical FedTree-GBDT",
    "Strict VFL-Tuned DR-MBRB boost": "Strict VFL-MBRB boost",
    "Strict VFL-Tuned DR-MBRB boost LT": "Strict VFL-MBRB boost-LT",
    "Strict VFL-EBRB boost LT-Iter": "Strict VFL-EBRB boost-LT-Iter",
    "Strict VFL-EBRB boost LT-Iter-p0.75": "Strict VFL-EBRB boost-LT-Iter-p0.75",
    "Strict VFL-EBRB boost LT-Iter-p0.5": "Strict VFL-EBRB boost-LT-Iter-p0.5",
}


def canonical_method(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def entropy(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    return -np.sum(arr * np.log(arr + EPS), axis=-1)


def beta_entropy(betas: np.ndarray) -> np.ndarray:
    denom = max(math.log(max(betas.shape[1], 2)), EPS)
    return entropy(betas) / denom


def pairwise_rule_conflict(activation: np.ndarray, betas: np.ndarray) -> np.ndarray:
    n_rules = betas.shape[0]
    if n_rules <= 1:
        return np.zeros(activation.shape[0], dtype=float)
    distances = np.zeros((n_rules, n_rules), dtype=float)
    for left, right in itertools.combinations(range(n_rules), 2):
        distances[left, right] = 0.5 * float(np.sum(np.abs(betas[left] - betas[right])))
        distances[right, left] = distances[left, right]
    mass = np.einsum("nr,ns->nrs", activation, activation)
    numerator = 0.5 * np.sum(mass * distances[None, :, :], axis=(1, 2))
    denom = 0.5 * (1.0 - np.sum(activation * activation, axis=1))
    return np.where(denom > EPS, numerator / np.maximum(denom, EPS), 0.0)


def pairwise_client_disagreement(stack: np.ndarray) -> np.ndarray:
    n_clients = stack.shape[1]
    if n_clients <= 1:
        return np.zeros(stack.shape[0], dtype=float)
    values = []
    for left, right in itertools.combinations(range(n_clients), 2):
        values.append(0.5 * np.sum(np.abs(stack[:, left, :] - stack[:, right, :]), axis=1))
    return np.mean(np.stack(values, axis=1), axis=1)


def summarize_numeric(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


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


def format_mean_std(row: Mapping[str, float], key: str, digits: int = 4) -> str:
    return f"{row.get(key + '_mean', float('nan')):.{digits}f} +/- {row.get(key + '_std', 0.0):.{digits}f}"


def make_boost_model(
    *,
    name: str,
    n_parties: int,
    rules_per_party: int,
    seed: int,
) -> Any:
    if name == "Strict VFL-EBRB":
        return exp._StrictVFLBoostedEBRB(
            n_estimators=1,
            learning_rate=0.25,
            sample_fraction=1.0,
            n_rules_per_party=rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=2 * rules_per_party,
            beta_smoothing=0.5,
            min_support_fraction=0.02,
            class_balance_power=0.0,
            parameter_iterations=0,
            parameter_update_blend=0.0,
            sampling_class_balance_power=0.0,
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
    if name == "Strict VFL-EBRB-LT-Iter":
        return exp._StrictVFLBoostedEBRB(
            n_estimators=1,
            learning_rate=0.25,
            sample_fraction=1.0,
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
    if name == "Strict VFL-EBRB boost-LT-Iter":
        return exp._StrictVFLBoostedEBRB(
            n_estimators=9,
            learning_rate=0.25,
            sample_fraction=0.85,
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
    if name == "Strict VFL-EBRB-LT-Iter+SCER":
        config = SCERRobustnessConfig(
            name="top4-G64-g0.75",
            group="quality",
            n_parties=n_parties,
            total_rule_budget=rules_per_party * n_parties,
            n_estimators=9,
            learning_rate=0.25,
            sample_fraction=0.85,
            top_rules_per_party=4,
            combo_budget=64,
            combo_gamma=0.75,
            parameter_iterations=2,
            parameter_update_blend=0.01,
            threshold_prior_multiplier=1.0,
        )
        return StrictVFLEBRBLTIterSCER(config, random_state=seed)
    raise ValueError(f"Unknown quality experiment method: {name}")


def base_boost_model(model: Any) -> Any:
    return model.model if isinstance(model, StrictVFLEBRBLTIterSCER) else model


def rule_to_text(rule: Any, global_features: Sequence[int], max_terms: int = 2) -> str:
    terms = []
    for literal in rule.antecedents[:max_terms]:
        feature = global_features[int(literal.feature_index)]
        op = "<=" if literal.direction == "le" else ">"
        terms.append(f"x{feature} {op} {literal.threshold:.3f}")
    return " AND ".join(terms) if terms else "<base prior>"


def rule_quality_metrics(model: Any, X: np.ndarray, minority_label: object) -> Dict[str, float]:
    boost = base_boost_model(model)
    if boost.classes_ is None:
        raise ValueError("Model has not been fitted")
    class_to_idx = {label: idx for idx, label in enumerate(boost.classes_)}
    minority_idx = int(class_to_idx[minority_label])
    rule_rows: List[Dict[str, float]] = []
    effective_counts: List[float] = []
    conflicts: List[float] = []
    disagreements: List[float] = []
    party_confidence: List[float] = []
    update_shifts: List[float] = []

    for estimator in boost.estimators_:
        party_beliefs = []
        for party_id in sorted(estimator.parties_):
            party = estimator.parties_[party_id]
            features = estimator.party_features_[party_id]
            local_X = X[:, features]
            match = party._rule_match_matrix(local_X)
            betas = np.stack([rule.beta for rule in party.rules_], axis=0)
            theta = np.asarray([max(rule.theta, EPS) for rule in party.rules_], dtype=float)
            reliability = np.asarray([max(rule.reliability, EPS) for rule in party.rules_], dtype=float)
            strength = match * theta[None, :] * reliability[None, :]
            activation = strength / np.maximum(np.sum(strength, axis=1, keepdims=True), EPS)
            effective = np.exp(entropy(activation))
            effective_counts.extend(effective.tolist())
            conflicts.extend(pairwise_rule_conflict(activation, betas).tolist())
            proba = party.predict_proba(local_X)
            party_beliefs.append(proba)
            party_confidence.extend(np.max(proba, axis=1).tolist())
            update_shifts.append(float(getattr(party, "parameter_update_shift_", 0.0)))
            for rule in party.rules_:
                row = {
                    "antecedent_len": float(len(rule.antecedents)),
                    "support": float(rule.support),
                    "score": float(rule.score),
                    "theta": float(rule.theta),
                    "reliability": float(rule.reliability),
                    "beta_entropy": float(beta_entropy(np.asarray(rule.beta, dtype=float)[None, :])[0]),
                    "beta_concentration": float(np.max(rule.beta)),
                    "minority_dominant": float(int(np.argmax(rule.beta) == minority_idx)),
                }
                rule_rows.append(row)
        if len(party_beliefs) > 1:
            stack = np.stack(party_beliefs, axis=1)
            disagreements.extend(pairwise_client_disagreement(stack).tolist())

    out = summarize_numeric(rule_rows)
    out["local_rule_count"] = float(len(rule_rows))
    out["estimator_count"] = float(len(boost.estimators_))
    out["effective_active_rules_mean"] = float(np.mean(effective_counts)) if effective_counts else 0.0
    out["rule_conflict_mean"] = float(np.mean(conflicts)) if conflicts else 0.0
    out["client_belief_disagreement_mean"] = float(np.mean(disagreements)) if disagreements else 0.0
    out["party_confidence_mean"] = float(np.mean(party_confidence)) if party_confidence else 0.0
    out["parameter_update_shift_mean"] = float(np.mean(update_shifts)) if update_shifts else 0.0

    if isinstance(model, StrictVFLEBRBLTIterSCER):
        combos = model.combo_rules_
        out["scer_combo_rule_count"] = float(len(combos))
        out["total_evidence_rule_count"] = float(len(rule_rows) + len(combos))
        if combos:
            combo_betas = np.stack([rule.beta for rule in combos], axis=0)
            out["scer_combo_support_mean"] = float(np.mean([rule.support for rule in combos]))
            out["scer_combo_score_mean"] = float(np.mean([rule.score for rule in combos]))
            out["scer_combo_reliability_mean"] = float(np.mean([rule.reliability for rule in combos]))
            out["scer_combo_beta_entropy_mean"] = float(np.mean(beta_entropy(combo_betas)))
            out["scer_combo_beta_concentration_mean"] = float(np.mean(np.max(combo_betas, axis=1)))
            out["scer_combo_minority_dominant_share"] = float(np.mean(np.argmax(combo_betas, axis=1) == minority_idx))
    else:
        out["scer_combo_rule_count"] = 0.0
        out["total_evidence_rule_count"] = float(len(rule_rows))
    return out


def top_rule_rows(model: Any, minority_label: object, fold: int, method: str, limit: int = 12) -> List[Dict[str, Any]]:
    boost = base_boost_model(model)
    class_to_idx = {label: idx for idx, label in enumerate(boost.classes_)}
    minority_idx = int(class_to_idx[minority_label])
    rows: List[Dict[str, Any]] = []
    for estimator_idx, estimator in enumerate(boost.estimators_[:1]):
        for party_id in sorted(estimator.parties_):
            party = estimator.parties_[party_id]
            features = estimator.party_features_[party_id]
            scored = sorted(
                enumerate(party.rules_),
                key=lambda item: (item[1].score, item[1].support),
                reverse=True,
            )
            for rule_idx, rule in scored[: max(1, limit // max(len(estimator.parties_), 1))]:
                rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "estimator": estimator_idx,
                        "party": party_id,
                        "rule": rule_idx,
                        "antecedent": rule_to_text(rule, features),
                        "beta": ";".join(f"{float(value):.4f}" for value in rule.beta),
                        "minority_beta": float(rule.beta[minority_idx]),
                        "support": float(rule.support),
                        "score": float(rule.score),
                        "theta": float(rule.theta),
                        "reliability": float(rule.reliability),
                    }
                )
    rows.sort(key=lambda row: (float(row["minority_beta"]), float(row["score"])), reverse=True)
    return rows[:limit]


def run_rule_quality_experiment(args: argparse.Namespace, output_dir: str) -> Dict[str, Any]:
    specs = {
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    spec = specs[args.quality_dataset]
    X_raw, y_raw = exp.load_dataset(spec)
    y_raw = np.asarray(y_raw, dtype=object).astype(str)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=13)
    splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=13)
    rows: List[Dict[str, Any]] = []
    rule_examples: List[Dict[str, Any]] = []
    methods = [
        "Strict VFL-EBRB",
        "Strict VFL-EBRB-LT-Iter",
        "Strict VFL-EBRB boost-LT-Iter",
    ]
    rules_per_party = max(4, int(math.ceil(args.total_rule_budget / max(args.n_parties, 1))))

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
        print(f"quality fold={fold} train={len(y_train)} test={len(y_test)} minority={minority_label}", flush=True)
        for method in methods:
            seed = 9000 + 97 * fold + len(method)
            start = time.perf_counter()
            model = make_boost_model(
                name=method,
                n_parties=args.n_parties,
                rules_per_party=rules_per_party,
                seed=seed,
            )
            model.fit(X_train, y_train, party_features=split.party_features, active_party_id=0)
            pred = model.predict(X_test)
            perf = long_tail_metrics(y_test, pred, minority_label)
            quality = rule_quality_metrics(model, X_test, minority_label)
            row = {
                "dataset": args.quality_dataset,
                "fold": fold,
                "method": method,
                "minority_label": str(minority_label),
                "train_n": int(len(y_train)),
                "test_n": int(len(y_test)),
                "rules_per_party": int(rules_per_party),
                "wall_time_sec": float(time.perf_counter() - start),
            }
            row.update(perf)
            row.update(quality)
            rows.append(row)
            if fold == 1 and method == "Strict VFL-EBRB boost-LT-Iter":
                rule_examples.extend(top_rule_rows(model, minority_label, fold=fold, method=method))
            print(f"  {method}: MinF1={perf['minority_f1']:.4f} rules={quality['total_evidence_rule_count']:.0f}", flush=True)

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary_rows = []
    for method in methods:
        summary = {"method": method}
        summary.update(summarize_numeric(grouped[method]))
        summary_rows.append(summary)

    write_csv(os.path.join(output_dir, "rule_quality_folds.csv"), rows)
    write_csv(os.path.join(output_dir, "rule_quality_summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "rule_quality_top_rules.csv"), rule_examples)
    return {"folds": rows, "summary": summary_rows, "top_rules": rule_examples}


def flatten_statistical_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: List[Dict[str, Any]] = []
    for dataset, dataset_payload in payload["datasets"].items():
        for ratio, ratio_payload in dataset_payload["results"].items():
            for mode, mode_payload in ratio_payload.items():
                for raw_method, method_payload in mode_payload.items():
                    method = canonical_method(raw_method)
                    for fold_idx, fold in enumerate(method_payload["folds"], start=1):
                        row = {
                            "dataset": dataset,
                            "ratio": float(ratio),
                            "mode": mode,
                            "fold": fold_idx,
                            "scenario": f"{dataset}|{ratio}|{mode}|{fold_idx}",
                            "method": method,
                        }
                        row.update({key: float(value) for key, value in fold.items() if isinstance(value, (int, float))})
                        rows.append(row)
    return rows


def metric_matrix(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[list[str], list[str], np.ndarray]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    lookup = {(str(row["scenario"]), str(row["method"])): float(row[metric]) for row in rows}
    complete_scenarios = [
        scenario for scenario in scenarios if all((scenario, method) in lookup for method in methods)
    ]
    matrix = np.asarray(
        [[lookup[(scenario, method)] for method in methods] for scenario in complete_scenarios],
        dtype=float,
    )
    return complete_scenarios, methods, matrix


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * float(p_value))
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def run_statistical_experiment(args: argparse.Namespace, output_dir: str) -> Dict[str, Any]:
    rows = flatten_statistical_rows(args.stats_source)
    write_csv(os.path.join(output_dir, "statistical_test_folds.csv"), rows)
    metrics = ["minority_f1", "accuracy", "macro_f1", "balanced_accuracy"]
    all_rank_rows: List[Dict[str, Any]] = []
    all_holm_rows: List[Dict[str, Any]] = []
    friedman_rows: List[Dict[str, Any]] = []

    for metric in metrics:
        scenarios, methods, matrix = metric_matrix(rows, metric)
        ranks = np.vstack([rankdata(-row, method="average") for row in matrix])
        stat, p_value = friedmanchisquare(*[matrix[:, idx] for idx in range(matrix.shape[1])])
        friedman_rows.append(
            {
                "metric": metric,
                "n_scenarios": len(scenarios),
                "n_methods": len(methods),
                "friedman_statistic": float(stat),
                "friedman_p": float(p_value),
            }
        )
        for method_idx, method in enumerate(methods):
            all_rank_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "mean_value": float(np.mean(matrix[:, method_idx])),
                    "std_value": float(np.std(matrix[:, method_idx])),
                    "average_rank": float(np.mean(ranks[:, method_idx])),
                    "best_count": int(np.sum(ranks[:, method_idx] == 1.0)),
                    "zero_count": int(np.sum(np.isclose(matrix[:, method_idx], 0.0))),
                }
            )

        if args.control_method in methods:
            control_idx = methods.index(args.control_method)
            p_values = []
            pairs = []
            for method_idx, method in enumerate(methods):
                if method_idx == control_idx:
                    continue
                diff = matrix[:, control_idx] - matrix[:, method_idx]
                try:
                    test = wilcoxon(matrix[:, control_idx], matrix[:, method_idx], zero_method="wilcox")
                    raw_p = float(test.pvalue)
                except ValueError:
                    raw_p = 1.0
                p_values.append(raw_p)
                pairs.append(
                    {
                        "metric": metric,
                        "control": args.control_method,
                        "method": method,
                        "control_mean": float(np.mean(matrix[:, control_idx])),
                        "method_mean": float(np.mean(matrix[:, method_idx])),
                        "mean_delta": float(np.mean(diff)),
                        "median_delta": float(np.median(diff)),
                        "wilcoxon_p": raw_p,
                    }
                )
            adjusted = holm_adjust(p_values)
            for pair, adjusted_p in zip(pairs, adjusted):
                pair["holm_adjusted_p"] = float(adjusted_p)
                pair["reject_alpha_0.05"] = bool(adjusted_p < 0.05)
                all_holm_rows.append(pair)

    all_rank_rows.sort(key=lambda row: (row["metric"], row["average_rank"]))
    write_csv(os.path.join(output_dir, "average_ranks.csv"), all_rank_rows)
    write_csv(os.path.join(output_dir, "friedman_tests.csv"), friedman_rows)
    write_csv(os.path.join(output_dir, "holm_vs_control.csv"), all_holm_rows)
    return {"ranks": all_rank_rows, "friedman": friedman_rows, "holm": all_holm_rows}


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], limit: int | None = None) -> str:
    selected = list(rows[:limit] if limit is not None else rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output_dir: str, quality: Mapping[str, Any], stats: Mapping[str, Any]) -> None:
    quality_rows = list(quality["summary"])
    rank_minf1 = [row for row in stats["ranks"] if row["metric"] == "minority_f1"]
    holm_minf1 = [row for row in stats["holm"] if row["metric"] == "minority_f1"]
    friedman_minf1 = [row for row in stats["friedman"] if row["metric"] == "minority_f1"][0]

    report_lines = [
        "# BRB adaptation experiments",
        "",
        "## Experiment 1: rule quality and interpretability",
        "",
        "Difficult setting: Bank Marketing, rho=0.03, K=10 active-weak VFL split, three stratified folds.",
        "",
        markdown_table(
            [
                {
                    "method": row["method"],
                    "Acc": row.get("accuracy_mean", float("nan")),
                    "MinF1": row.get("minority_f1_mean", float("nan")),
                    "RuleCnt": row.get("total_evidence_rule_count_mean", float("nan")),
                    "BetaEnt": row.get("beta_entropy_mean_mean", float("nan")),
                    "Conflict": row.get("rule_conflict_mean_mean", float("nan")),
                    "EffActive": row.get("effective_active_rules_mean_mean", float("nan")),
                    "ClientDis": row.get("client_belief_disagreement_mean_mean", float("nan")),
                }
                for row in quality_rows
            ],
            ["method", "Acc", "MinF1", "RuleCnt", "BetaEnt", "Conflict", "EffActive", "ClientDis"],
        ),
        "",
        "Interpretation: BetaEnt measures consequent-belief entropy; in long-tail settings, a higher value can indicate that LT-Iter has reduced majority-class over-concentration rather than simply making rules less useful. Conflict measures activation-weighted disagreement among co-active local rules, so a lower value indicates more consistent evidence. EffActive is the entropy-effective number of active local rules per sample.",
        "",
        "## Experiment 2: average ranking and statistical test",
        "",
        f"Source: 54 fold-level scenarios. Friedman statistic for MinF1 = {friedman_minf1['friedman_statistic']:.4f}, p = {friedman_minf1['friedman_p']:.4g}.",
        "",
        markdown_table(
            rank_minf1,
            ["method", "mean_value", "std_value", "average_rank", "best_count", "zero_count"],
        ),
        "",
        "Holm-adjusted Wilcoxon comparisons against Strict VFL-EBRB boost-LT-Iter for MinF1:",
        "",
        markdown_table(
            holm_minf1,
            ["method", "mean_delta", "wilcoxon_p", "holm_adjusted_p", "reject_alpha_0.05"],
        ),
        "",
        "Generated files: rule_quality_folds.csv, rule_quality_summary.csv, rule_quality_top_rules.csv, average_ranks.csv, friedman_tests.csv, holm_vs_control.csv.",
    ]
    with open(os.path.join(output_dir, "manuscript_ready_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two BRB-adaptation experiments for the VFL-EBRB manuscript.")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "brb_adaptation_experiments"))
    parser.add_argument("--stats-source", default=os.path.join(PROJECT_ROOT, "results", "strict_vfl_all_ratios_all_methods_ebrb_lt_iter_3datasets.json"))
    parser.add_argument("--control-method", default="Strict VFL-EBRB boost-LT-Iter")
    parser.add_argument("--quality-dataset", default="bank_marketing", choices=["bank_marketing"])
    parser.add_argument("--rho", type=float, default=0.03)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--total-rule-budget", type=int, default=72)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()
    quality = run_rule_quality_experiment(args, args.output_dir)
    stats = run_statistical_experiment(args, args.output_dir)
    write_report(args.output_dir, quality, stats)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_sec": float(time.perf_counter() - start),
        "quality_summary": quality["summary"],
        "statistical_friedman": stats["friedman"],
    }
    with open(os.path.join(args.output_dir, "brb_adaptation_experiments.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Done. Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
