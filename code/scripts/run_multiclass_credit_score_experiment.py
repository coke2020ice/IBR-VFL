from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts import run_additional_ml_vfl_baselines as old_vfl  # noqa: E402
from scripts.run_k10_active_weak_ablation import active_weak_split  # noqa: E402
from scripts.run_scer_robustness_experiment import (  # noqa: E402
    SCERRobustnessConfig,
    StrictVFLEBRBLTIterSCER,
)
from vfl_tuned_dr_mbrb import TabularEncoder  # noqa: E402
from vfl_tuned_dr_mbrb.belief import EPS, softmax  # noqa: E402


IDENTIFIER_COLUMNS = ["ID", "Customer_ID", "Name", "SSN", "Month"]
CATEGORICAL_COLUMNS = {
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
}


def load_credit_score(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"Customer_ID", "Credit_Score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Each customer occurs in eight monthly rows. Keeping one row per customer
    # prevents the same person from appearing in both training and test folds.
    frame = frame.dropna(subset=["Customer_ID", "Credit_Score"]).copy()
    label_conflicts = int((frame.groupby("Customer_ID")["Credit_Score"].nunique() > 1).sum())
    frame = frame.sort_values(["Customer_ID", "Month"]).drop_duplicates("Customer_ID", keep="first")
    y = frame.pop("Credit_Score").astype(str).str.strip().to_numpy(dtype=object)
    frame = frame.drop(columns=[column for column in IDENTIFIER_COLUMNS if column in frame], errors="ignore")

    for column in frame.columns:
        if column in CATEGORICAL_COLUMNS:
            frame[column] = frame[column].astype("string").replace({"_": pd.NA, "!@9#%8": pd.NA})
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        else:
            cleaned = frame[column].astype("string").str.replace("_", "", regex=False)
            frame[column] = pd.to_numeric(cleaned, errors="coerce").astype(float)

    metadata = {
        "raw_rows": 100000,
        "unique_customers": int(len(frame)),
        "label_conflict_customers": label_conflicts,
        "features_after_identifier_removal": int(frame.shape[1]),
        "class_counts_before_cap": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    return frame.to_numpy(dtype=object), y, metadata


def multiclass_long_tail_resample(
    X: np.ndarray,
    y: np.ndarray,
    class_order: Sequence[object],
    imbalance_factor: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample C classes to n_j/n_1 = IF^(-j/(C-1))."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=object)
    classes = list(class_order)
    if len(classes) < 3:
        raise ValueError("The multiclass experiment requires at least three classes")
    factor = max(float(imbalance_factor), 1.0)
    weights = np.power(factor, -np.arange(len(classes), dtype=float) / (len(classes) - 1.0))
    available = np.asarray([np.sum(y_arr == label) for label in classes], dtype=int)
    base = int(np.floor(np.min(available / weights)))
    desired = np.maximum(1, np.floor(base * weights).astype(int))
    selected = []
    for label, count in zip(classes, desired):
        indices = np.flatnonzero(y_arr == label)
        selected.append(rng.choice(indices, size=min(int(count), len(indices)), replace=False))
    keep = np.concatenate(selected).astype(int)
    rng.shuffle(keep)
    return X[keep], y_arr[keep]


def multiclass_target_tail_resample(
    X: np.ndarray,
    y: np.ndarray,
    tail_label: object,
    target_tail_proportion: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample only the tail class to a specified proportion of all samples."""
    rho = float(target_tail_proportion)
    if not 0.0 < rho < 1.0:
        raise ValueError("target_tail_proportion must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=object)
    tail_indices = np.flatnonzero(y_arr == tail_label)
    non_tail_indices = np.flatnonzero(y_arr != tail_label)
    desired_tail = max(1, int(np.floor(rho * len(non_tail_indices) / (1.0 - rho))))
    desired_tail = min(desired_tail, len(tail_indices))
    selected_tail = rng.choice(tail_indices, size=desired_tail, replace=False)
    keep = np.concatenate([non_tail_indices, selected_tail]).astype(int)
    rng.shuffle(keep)
    return X[keep], y_arr[keep]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels, counts = np.unique(y_true, return_counts=True)
    tail_label = labels[int(np.argmin(counts))]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tail_f1": float(f1_score(y_true, y_pred, labels=[tail_label], average="macro", zero_division=0)),
    }


def summarize(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in sorted({key for row in rows for key in row}):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def centralized_runner(factory):
    def run(X_train, y_train, X_test, party_features, active_party_id, seed):
        del party_features, active_party_id
        classes, y_idx = np.unique(y_train, return_inverse=True)
        model = factory(seed)
        model.fit(X_train, y_idx)
        return classes[np.asarray(model.predict(X_test), dtype=int)], {}

    return run


def strict_vfl_softmax_lr(X_train, y_train, X_test, party_features, active_party_id, seed):
    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    n_classes = len(classes)
    rng = np.random.default_rng(seed)
    blocks_train, blocks_test = old_vfl._party_standardize(X_train, X_test, party_features)
    weights = {
        pid: rng.normal(0.0, 1e-3, size=(block.shape[1], n_classes))
        for pid, block in blocks_train.items()
    }
    bias = np.zeros(n_classes, dtype=float)
    counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    class_weights = np.max(counts) / np.maximum(counts, 1.0)
    sample_weights = class_weights[y_idx]
    sample_weights /= np.mean(sample_weights)
    one_hot = np.eye(n_classes)[y_idx]
    lr, l2, steps = 0.08, 1e-3, 450
    for _ in range(steps):
        logits = np.broadcast_to(bias[None, :], (len(y_idx), n_classes)).copy()
        for pid, block in blocks_train.items():
            logits += block @ weights[pid]
        proba = softmax(logits)
        residual = (proba - one_hot) * sample_weights[:, None]
        bias -= lr * np.mean(residual, axis=0)
        for pid, block in blocks_train.items():
            weights[pid] -= lr * (block.T @ residual / len(y_idx) + l2 * weights[pid])
    logits = np.broadcast_to(bias[None, :], (X_test.shape[0], n_classes)).copy()
    for pid, block in blocks_test.items():
        logits += block @ weights[pid]
    return classes[np.argmax(logits, axis=1)], {"strict_vfl_softmax_steps": float(steps)}


def strict_vfl_splitnn_multiclass(X_train, y_train, X_test, party_features, active_party_id, seed):
    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    n_classes = len(classes)
    rng = np.random.default_rng(seed)
    train_blocks, test_blocks = old_vfl._party_standardize(X_train, X_test, party_features)
    hidden = 4
    encoders = {pid: rng.normal(0.0, 0.04, (x.shape[1], hidden)) for pid, x in train_blocks.items()}
    heads = {pid: rng.normal(0.0, 0.04, (hidden, n_classes)) for pid in train_blocks}
    bias = np.zeros(n_classes, dtype=float)
    counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    class_weights = np.max(counts) / np.maximum(counts, 1.0)
    sample_weights = class_weights[y_idx] / np.mean(class_weights[y_idx])
    targets = np.eye(n_classes)[y_idx]
    lr, l2, steps = 0.025, 1e-4, 260
    for _ in range(steps):
        reps = {pid: np.tanh(x @ encoders[pid]) for pid, x in train_blocks.items()}
        logits = np.broadcast_to(bias[None, :], (len(y_idx), n_classes)).copy()
        for pid in reps:
            logits += reps[pid] @ heads[pid]
        residual = (softmax(logits) - targets) * sample_weights[:, None] / len(y_idx)
        bias -= lr * np.sum(residual, axis=0)
        old_heads = {pid: value.copy() for pid, value in heads.items()}
        for pid, block in train_blocks.items():
            heads[pid] -= lr * (reps[pid].T @ residual + l2 * heads[pid])
            local = (residual @ old_heads[pid].T) * (1.0 - reps[pid] ** 2)
            encoders[pid] -= lr * (block.T @ local + l2 * encoders[pid])
    logits = np.broadcast_to(bias[None, :], (X_test.shape[0], n_classes)).copy()
    for pid, block in test_blocks.items():
        logits += np.tanh(block @ encoders[pid]) @ heads[pid]
    return classes[np.argmax(logits, axis=1)], {"strict_vfl_splitnn_steps": float(steps)}


def build_methods(n_parties: int, total_rule_budget: int):
    rules_per_party = max(4, int(np.ceil(total_rule_budget / n_parties)))
    common = dict(
        n_estimators=9,
        learning_rate=0.25,
        sample_fraction=0.85,
        n_rules_per_party=rules_per_party,
        bins=6,
        max_antecedents=2,
        beam_width=2 * rules_per_party,
        beta_smoothing=0.5,
        min_support_fraction=0.02,
        sampling_class_balance_power=0.75,
        client_weight_method="label_accuracy",
        fusion_method="log",
        weight_update_method="residual",
        decision_threshold_method="argmax",
        decision_threshold_prior_multiplier=1.0,
    )
    base = exp.strict_vfl_ebrb_boost_runner(
        **common,
        class_balance_power=0.0,
        parameter_iterations=0,
        parameter_update_blend=0.0,
    )
    lt_iter = exp.strict_vfl_ebrb_boost_runner(
        **common,
        class_balance_power=0.5,
        parameter_iterations=2,
        parameter_update_blend=0.01,
    )
    lt_iter_calibrated = exp.strict_vfl_ebrb_boost_runner(
        **common,
        class_balance_power=0.5,
        parameter_iterations=2,
        parameter_update_blend=0.01,
        server_logit_la_gamma=0.5,
        server_logit_la_temperature=1.0,
        server_logit_la_blend=1.0,
    )
    with open("configs/vfl_tuned_dr_mbrb_demo.json", "r", encoding="utf-8") as handle:
        mbrb_payload = json.load(handle)
    vfl_mbrb = exp.boosted_dr_mbrb_runner(
        mbrb_payload,
        n_estimators=9,
        sample_fraction=0.85,
        learning_rate=0.25,
        fusion_method="log",
        weight_update_method="residual",
        sampling_class_balance_power=0.75,
        class_conditioned_fusion=True,
        decision_threshold_method="argmax",
        decision_threshold_prior_multiplier=1.0,
        final_la_gamma=0.5,
        final_la_temperature=1.0,
        final_la_blend=1.0,
        prior_gated_final_la=False,
        min_class_prior_for_la=0.0,
        strict_vfl=True,
        n_modules_per_party=1,
        n_rules_per_class=max(2, int(np.ceil(rules_per_party / 3))),
        n_rules_per_module=rules_per_party,
    )

    def make_scer_variant(parameter_iterations: int, calibrated: bool):
        def run(X_train, y_train, X_test, party_features, active_party_id, seed):
            def make_config(
                top_rules: int,
                combo_budget: int,
                combo_gamma: float,
                rule_budget: int = total_rule_budget,
                class_balance_power: float = 1.0,
                sampling_balance_power: float = 0.75,
                combo_quota_prior_power: float = 0.0,
            ):
                return SCERRobustnessConfig(
                    name=f"multiclass-scer-iter{parameter_iterations}-cal{int(calibrated)}",
                    group="multiclass_ablation",
                    n_parties=n_parties,
                    total_rule_budget=rule_budget,
                    n_estimators=9,
                    learning_rate=0.25,
                    sample_fraction=0.85,
                    top_rules_per_party=top_rules,
                    combo_budget=combo_budget,
                    combo_gamma=combo_gamma,
                    parameter_iterations=parameter_iterations,
                    parameter_update_blend=0.01 if parameter_iterations else 0.0,
                    class_balanced_selection=True,
                    class_balance_power=class_balance_power,
                    sampling_class_balance_power=sampling_balance_power,
                    combo_quota_prior_power=combo_quota_prior_power,
                )

            config = make_config(4, 64, 0.25)
            if not calibrated:
                model = StrictVFLEBRBLTIterSCER(config, random_state=seed)
                model.fit(X_train, y_train, party_features=party_features, active_party_id=active_party_id)
                diagnostics = model.diagnostics()
                return model.predict(X_test), diagnostics
            indices = np.arange(len(y_train))
            inner_repeats = max(1, int(os.environ.get("IBR_INNER_REPEATS", "1")))
            inner_cv_folds = max(0, int(os.environ.get("IBR_INNER_CV_FOLDS", "0")))
            inner_splits = []
            if inner_cv_folds >= 2:
                inner_splitter = StratifiedKFold(
                    n_splits=inner_cv_folds, shuffle=True, random_state=seed + 7919
                )
                split_iterator = inner_splitter.split(indices, np.asarray(y_train, dtype=object))
            else:
                split_iterator = (
                    train_test_split(
                        indices,
                        test_size=0.2,
                        random_state=seed + 7919 * repeat,
                        stratify=np.asarray(y_train, dtype=object),
                    )
                    for repeat in range(inner_repeats)
                )
            for repeat, (fit_idx, val_idx) in enumerate(split_iterator):
                # Feature ownership is a supervised construction in this benchmark.
                # Recompute it from the inner-training subset so that validation
                # labels do not influence the candidate models.
                inner_split = active_weak_split(
                    X_train[fit_idx],
                    np.asarray(y_train, dtype=object)[fit_idx],
                    n_parties=n_parties,
                    rank_strength=3.0,
                    seed=seed + 7919 * repeat,
                )
                inner_splits.append((fit_idx, val_idx, inner_split.party_features))
            if os.environ.get("IBR_EXPANDED_SEARCH", "0") == "1":
                # A single, predeclared union of the structure families used in
                # the earlier balance, quota, capacity, and default searches.
                # Candidate selection remains confined to the inner validation
                # splits; the held-out outer test fold is never used for tuning.
                structure_candidates = (
                    (4, 64, 0.10, 72, 0.50, 0.50, 0.00),
                    (4, 64, 0.10, 72, 0.75, 0.50, 0.00),
                    (4, 64, 0.10, 72, 0.75, 0.75, 0.00),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.00),
                    (4, 64, 0.25, 72, 0.50, 0.50, 0.00),
                    (4, 64, 0.25, 72, 0.75, 0.75, 0.00),
                    (4, 64, 0.10, 120, 0.50, 0.50, 0.00),
                    (4, 64, 0.10, 120, 0.75, 0.75, 0.00),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.15),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.30),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.50),
                    (4, 64, 0.25, 72, 1.00, 0.75, 0.15),
                    (4, 64, 0.25, 72, 1.00, 0.75, 0.30),
                    (4, 64, 0.10, 120, 1.00, 0.75, 0.00),
                    (6, 96, 0.25, 120, 1.00, 0.75, 0.00),
                    (6, 96, 0.25, 180, 1.00, 0.75, 0.00),
                    (4, 96, 0.25, 72, 1.00, 0.75, 0.00),
                    (6, 64, 0.25, 72, 1.00, 0.75, 0.00),
                    (6, 96, 0.25, 72, 1.00, 0.75, 0.00),
                    (6, 96, 0.40, 72, 1.00, 0.75, 0.00),
                )
            elif os.environ.get("IBR_BALANCE_SEARCH", "0") == "1":
                structure_candidates = (
                    (4, 64, 0.10, 72, 0.50, 0.50),
                    (4, 64, 0.10, 72, 0.75, 0.50),
                    (4, 64, 0.10, 72, 0.75, 0.75),
                    (4, 64, 0.10, 72, 1.00, 0.75),
                    (4, 64, 0.25, 72, 0.50, 0.50),
                    (4, 64, 0.25, 72, 0.75, 0.75),
                    (4, 64, 0.10, 120, 0.50, 0.50),
                    (4, 64, 0.10, 120, 0.75, 0.75),
                )
            elif os.environ.get("IBR_QUOTA_SEARCH", "0") == "1":
                structure_candidates = (
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.00),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.15),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.30),
                    (4, 64, 0.10, 72, 1.00, 0.75, 0.50),
                    (4, 64, 0.25, 72, 1.00, 0.75, 0.15),
                    (4, 64, 0.25, 72, 1.00, 0.75, 0.30),
                )
            elif os.environ.get("IBR_CAPACITY_SEARCH", "0") == "1":
                structure_candidates = (
                    (4, 64, 0.10, 72, 1.00, 0.75),
                    (4, 64, 0.10, 120, 1.00, 0.75),
                    (6, 96, 0.25, 120, 1.00, 0.75),
                    (6, 96, 0.25, 180, 1.00, 0.75),
                )
            else:
                structure_candidates = (
                    (4, 64, 0.10, total_rule_budget, 1.00, 0.75),
                    (4, 64, 0.25, total_rule_budget, 1.00, 0.75),
                    (4, 96, 0.25, total_rule_budget, 1.00, 0.75),
                    (6, 64, 0.25, total_rule_budget, 1.00, 0.75),
                    (6, 96, 0.25, total_rule_budget, 1.00, 0.75),
                    (6, 96, 0.40, total_rule_budget, 1.00, 0.75),
                )
            validation_objective = os.environ.get("IBR_VALIDATION_OBJECTIVE", "macro_f1").strip().lower()
            gamma_candidates = tuple(
                float(value.strip())
                for value in os.environ.get("IBR_GAMMA_CANDIDATES", "0.0,0.1,0.2,0.3").split(",")
                if value.strip()
            )
            tail_delta_candidates = tuple(
                float(value.strip())
                for value in os.environ.get("IBR_TAIL_LOGIT_DELTAS", "0.0").split(",")
                if value.strip()
            )
            best_gamma = 0.0
            best_tail_delta = 0.0
            best_structure = structure_candidates[0]
            best_score = (-np.inf, -np.inf, -np.inf)
            # Older search grids omit quota power and retain equal class quotas.
            structure_candidates = tuple(
                candidate if len(candidate) == 7 else (*candidate, 0.0)
                for candidate in structure_candidates
            )
            for (top_rules, candidate_budget, combo_gamma, candidate_rule_budget,
                 class_balance_power, sampling_balance_power, combo_quota_prior_power) in structure_candidates:
                candidate_config = make_config(
                    top_rules,
                    candidate_budget,
                    combo_gamma,
                    candidate_rule_budget,
                    class_balance_power,
                    sampling_balance_power,
                    combo_quota_prior_power,
                )
                scores_by_calibration = {
                    (gamma, tail_delta): []
                    for gamma in gamma_candidates
                    for tail_delta in tail_delta_candidates
                }
                for repeat, (fit_idx, val_idx, inner_party_features) in enumerate(inner_splits):
                    calibration_model = StrictVFLEBRBLTIterSCER(
                        candidate_config, random_state=seed + 7919 * repeat
                    )
                    calibration_model.fit(
                        X_train[fit_idx],
                        np.asarray(y_train, dtype=object)[fit_idx],
                        party_features=inner_party_features,
                        active_party_id=active_party_id,
                    )
                    fit_proba = calibration_model.predict_proba(X_train[fit_idx])
                    val_proba = calibration_model.predict_proba(X_train[val_idx])
                    calibration_reference = np.mean(fit_proba, axis=0)
                    y_val = np.asarray(y_train, dtype=object)[val_idx]
                    labels, counts = np.unique(
                        np.asarray(y_train, dtype=object)[fit_idx], return_counts=True
                    )
                    tail_label = labels[int(np.argmin(counts))]
                    tail_class_idx = int(np.flatnonzero(calibration_model.classes_ == tail_label)[0])
                    for gamma in gamma_candidates:
                        for tail_delta in tail_delta_candidates:
                            val_logits = (
                                np.log(val_proba + EPS)
                                - gamma * np.log(calibration_reference[None, :] + EPS)
                            )
                            val_logits[:, tail_class_idx] += tail_delta
                            val_corrected = softmax(val_logits)
                            val_pred = calibration_model.classes_[np.argmax(val_corrected, axis=1)]
                            scores_by_calibration[(gamma, tail_delta)].append(
                                (
                                    f1_score(y_val, val_pred, average="macro", zero_division=0),
                                    accuracy_score(y_val, val_pred),
                                    f1_score(
                                        y_val,
                                        val_pred,
                                        labels=[tail_label],
                                        average="macro",
                                        zero_division=0,
                                    ),
                                )
                            )
                for gamma in gamma_candidates:
                    for tail_delta in tail_delta_candidates:
                        macro_f1, accuracy, tail_f1 = np.mean(
                            scores_by_calibration[(gamma, tail_delta)], axis=0
                        )
                        if validation_objective == "tail_f1":
                            score = (tail_f1, macro_f1, accuracy)
                        elif validation_objective == "macro_f1":
                            score = (macro_f1, accuracy, tail_f1)
                        elif validation_objective == "joint":
                            # Geometric mean penalizes candidates that gain one target
                            # by collapsing the other, while remaining scale-free.
                            score = (
                                float(np.sqrt(max(accuracy, 0.0) * max(tail_f1, 0.0))),
                                macro_f1,
                                accuracy,
                            )
                        else:
                            raise ValueError(
                                f"Unknown IBR_VALIDATION_OBJECTIVE: {validation_objective}"
                            )
                        if score > best_score:
                            best_score = score
                            best_gamma = float(gamma)
                            best_tail_delta = float(tail_delta)
                            best_structure = (
                                top_rules,
                                candidate_budget,
                                combo_gamma,
                                candidate_rule_budget,
                                class_balance_power,
                                sampling_balance_power,
                                combo_quota_prior_power,
                            )

            config = make_config(*best_structure)
            model = StrictVFLEBRBLTIterSCER(config, random_state=seed)
            model.fit(X_train, y_train, party_features=party_features, active_party_id=active_party_id)
            diagnostics = model.diagnostics()
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)
            reference = np.mean(train_proba, axis=0)
            corrected_logits = np.log(test_proba + EPS) - best_gamma * np.log(reference[None, :] + EPS)
            final_labels, final_counts = np.unique(np.asarray(y_train, dtype=object), return_counts=True)
            final_tail_label = final_labels[int(np.argmin(final_counts))]
            final_tail_idx = int(np.flatnonzero(model.classes_ == final_tail_label)[0])
            corrected_logits[:, final_tail_idx] += best_tail_delta
            corrected = softmax(corrected_logits)
            if os.environ.get("IBR_OOF_STACK_CALIBRATION", "0") == "1":
                stack_folds = int(os.environ.get("IBR_OOF_STACK_FOLDS", "3"))
                stack_power_candidates = tuple(
                    float(value.strip())
                    for value in os.environ.get(
                        "IBR_OOF_STACK_CLASS_POWERS", "0.5,0.75,1.0,1.25"
                    ).split(",")
                    if value.strip()
                )
                stack_splitter = StratifiedKFold(
                    n_splits=stack_folds, shuffle=True, random_state=seed + 104729
                )
                oof_proba = np.zeros((len(y_train), len(model.classes_)), dtype=float)
                for stack_train_idx, stack_val_idx in stack_splitter.split(X_train, y_train):
                    stack_model = StrictVFLEBRBLTIterSCER(config, random_state=seed)
                    stack_model.fit(
                        X_train[stack_train_idx],
                        np.asarray(y_train, dtype=object)[stack_train_idx],
                        party_features=party_features,
                        active_party_id=active_party_id,
                    )
                    stack_fit_proba = stack_model.predict_proba(X_train[stack_train_idx])
                    stack_reference = np.mean(stack_fit_proba, axis=0)
                    oof_proba[stack_val_idx] = softmax(
                        np.log(stack_model.predict_proba(X_train[stack_val_idx]) + EPS)
                        - best_gamma * np.log(stack_reference[None, :] + EPS)
                    )
                class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
                y_stack = np.asarray([class_to_idx[label] for label in y_train], dtype=int)
                stack_counts = np.bincount(y_stack, minlength=len(model.classes_)).astype(float)
                meta_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 130363)
                best_stack_power = stack_power_candidates[0]
                best_stack_score = (-np.inf, -np.inf)
                tail_idx = int(np.argmin(stack_counts))
                for candidate_power in stack_power_candidates:
                    meta_scores = []
                    candidate_weights = np.power(
                        np.max(stack_counts) / np.maximum(stack_counts, 1.0), candidate_power
                    )
                    for meta_train_idx, meta_val_idx in meta_splitter.split(oof_proba, y_stack):
                        meta_calibrator = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                        meta_calibrator.fit(
                            np.log(oof_proba[meta_train_idx] + EPS),
                            y_stack[meta_train_idx],
                            sample_weight=candidate_weights[y_stack[meta_train_idx]],
                        )
                        meta_pred = meta_calibrator.predict(np.log(oof_proba[meta_val_idx] + EPS))
                        meta_accuracy = accuracy_score(y_stack[meta_val_idx], meta_pred)
                        meta_tail_f1 = f1_score(
                            y_stack[meta_val_idx], meta_pred, labels=[tail_idx],
                            average="macro", zero_division=0,
                        )
                        meta_scores.append((meta_accuracy, meta_tail_f1))
                    mean_meta_accuracy, mean_meta_tail_f1 = np.mean(meta_scores, axis=0)
                    candidate_score = (
                        float(np.sqrt(max(mean_meta_accuracy, 0.0) * max(mean_meta_tail_f1, 0.0))),
                        mean_meta_accuracy,
                    )
                    if candidate_score > best_stack_score:
                        best_stack_score = candidate_score
                        best_stack_power = candidate_power
                stack_weights = np.power(
                    np.max(stack_counts) / np.maximum(stack_counts, 1.0), best_stack_power
                )
                stack_calibrator = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                stack_calibrator.fit(
                    np.log(oof_proba + EPS),
                    y_stack,
                    sample_weight=stack_weights[y_stack],
                )
                corrected = stack_calibrator.predict_proba(np.log(corrected + EPS))
                diagnostics["multiclass_oof_stack_calibration"] = 1.0
                diagnostics["multiclass_oof_stack_folds"] = float(stack_folds)
                diagnostics["multiclass_oof_stack_class_power"] = float(best_stack_power)
                diagnostics["multiclass_oof_stack_validation_score"] = float(best_stack_score[0])
            diagnostics["multiclass_calibration_gamma"] = best_gamma
            diagnostics["multiclass_tail_logit_delta"] = best_tail_delta
            diagnostics["multiclass_validation_tail_objective"] = float(validation_objective == "tail_f1")
            diagnostics["multiclass_validation_joint_objective"] = float(validation_objective == "joint")
            diagnostics["multiclass_inner_repeats"] = float(inner_repeats)
            diagnostics["multiclass_inner_cv_folds"] = float(inner_cv_folds)
            diagnostics["multiclass_calibration_validation_score"] = float(best_score[0])
            diagnostics["multiclass_selected_top_rules"] = float(best_structure[0])
            diagnostics["multiclass_selected_combo_budget"] = float(best_structure[1])
            diagnostics["multiclass_selected_combo_gamma"] = float(best_structure[2])
            diagnostics["multiclass_selected_rule_budget"] = float(best_structure[3])
            diagnostics["multiclass_selected_class_balance_power"] = float(best_structure[4])
            diagnostics["multiclass_selected_sampling_balance_power"] = float(best_structure[5])
            diagnostics["multiclass_selected_combo_quota_prior_power"] = float(best_structure[6])
            return model.classes_[np.argmax(corrected, axis=1)], diagnostics

        return run

    def local_p0(X_train, y_train, X_test, party_features, active_party_id, seed):
        active = {0: list(party_features[active_party_id])}
        return base(X_train, y_train, X_test, active, 0, seed)

    def local_p0_mbrb(X_train, y_train, X_test, party_features, active_party_id, seed):
        active = {0: list(party_features[active_party_id])}
        return vfl_mbrb(X_train, y_train, X_test, active, 0, seed)

    fedtree = exp.fedtree_vertical_runner(
        fedtree_root="../external/FedTree",
        n_trees=100,
        depth=3,
        learning_rate=0.08,
        timeout_sec=300,
    )

    return {
        "Centralized LR": centralized_runner(
            lambda seed: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "Centralized Random Forest": centralized_runner(
            lambda seed: RandomForestClassifier(
                n_estimators=140, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=4, random_state=seed
            )
        ),
        "Centralized Extra Trees": centralized_runner(
            lambda seed: ExtraTreesClassifier(
                n_estimators=140, min_samples_leaf=2, class_weight="balanced", n_jobs=4, random_state=seed
            )
        ),
        "Centralized HistGradientBoosting": centralized_runner(
            lambda seed: HistGradientBoostingClassifier(
                max_iter=160, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=0.01,
                class_weight="balanced", random_state=seed
            )
        ),
        "Centralized MLP": centralized_runner(
            lambda seed: MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=100, early_stopping=True, random_state=seed
            )
        ),
        "Active-only LR": exp.active_only_lr,
        "Strict VFL-Softmax LR": strict_vfl_softmax_lr,
        "Strict VFL-Linear SVM": old_vfl.strict_vfl_linear_svm,
        "Strict VFL-KNN": old_vfl.strict_vfl_knn,
        "Strict VFL-SplitNN": strict_vfl_splitnn_multiclass,
        "Strict VFL-XGBoost": exp.vfl_xgboost_proxy,
        "Strict VFL-Vertical RF": old_vfl.strict_vfl_vertical_rf,
        "Strict VFL-FedTree GBDT": fedtree,
        "Strict VFL-GaussianNB": exp.strict_vfl_gaussian_nb,
        "VFL-MBRB": vfl_mbrb,
        "Local-P0 EBRB": local_p0,
        "Local-P0 MBRB": local_p0_mbrb,
        "IBR-VFL-base": base,
        "IBR-VFL-base+LT-Iter": lt_iter,
        "IBR-VFL-base+LT-Iter+MC-Cal": lt_iter_calibrated,
        "IBR-VFL-base+SCER": make_scer_variant(parameter_iterations=0, calibrated=False),
        "IBR-VFL-base+LT-Iter+SCER": make_scer_variant(parameter_iterations=2, calibrated=False),
        "IBR-VFL": make_scer_variant(parameter_iterations=2, calibrated=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiclass long-tailed Credit Score VFL experiment")
    parser.add_argument("--data", default="data/credit_score_classification/train.csv")
    parser.add_argument("--imbalance-factors", default="10")
    parser.add_argument(
        "--target-tail-proportions",
        default="",
        help="Comma-separated tail-class proportions; when set, overrides --imbalance-factors.",
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--total-rule-budget", type=int, default=72)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--methods", default="", help="Optional comma-separated method names to run.")
    parser.add_argument("--fold-indices", default="", help="Optional comma-separated 1-based folds to run.")
    parser.add_argument("--output", default="results/multiclass_credit_score_experiment.json")
    args = parser.parse_args()

    X_raw, y_raw, metadata = load_credit_score(args.data)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, args.max_samples, args.seed)
    labels, counts = np.unique(y_raw, return_counts=True)
    class_order = labels[np.argsort(-counts)]
    factors = [float(value.strip()) for value in args.imbalance_factors.split(",") if value.strip()]
    target_tail_proportions = [
        float(value.strip()) for value in args.target_tail_proportions.split(",") if value.strip()
    ]
    splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    methods = build_methods(args.n_parties, args.total_rule_budget)
    requested_methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    requested_folds = {int(value.strip()) for value in args.fold_indices.split(",") if value.strip()}
    if requested_methods:
        missing_methods = [name for name in requested_methods if name not in methods]
        if missing_methods:
            raise ValueError(f"Unknown methods: {missing_methods}")
        methods = {name: methods[name] for name in requested_methods}

    result: Dict[str, Any] = {
        "experiment": "multiclass_long_tailed_credit_score_vfl",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": metadata,
        "sample_cap": args.max_samples,
        "class_order_head_to_tail": [str(label) for label in class_order],
        "imbalance_factors": factors,
        "target_tail_proportions": target_tail_proportions,
        "decision_rule": "argmax over the three output belief degrees",
        "results": {},
    }

    settings = (
        [(f"rho={rho:g}", rho, True) for rho in target_tail_proportions]
        if target_tail_proportions
        else [(f"IF={factor:g}", factor, False) for factor in factors]
    )
    for setting_name, setting_value, uses_target_rho in settings:
        method_rows: Dict[str, list[Dict[str, float]]] = {name: [] for name in methods}
        fold_distributions = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, y_raw), start=1):
            if requested_folds and fold not in requested_folds:
                continue
            # Define the head-to-tail order from this fold's training data only.
            # The held-out labels must not determine which class is treated as tail.
            fold_labels, fold_counts = np.unique(y_raw[train_idx], return_counts=True)
            fold_class_order = fold_labels[np.argsort(-fold_counts)]
            fold_tail_label = fold_class_order[-1]
            if uses_target_rho:
                X_train_raw, y_train = multiclass_target_tail_resample(
                    X_raw[train_idx], y_raw[train_idx], fold_tail_label, setting_value, args.seed + 100 * fold
                )
                X_test_raw, y_test = multiclass_target_tail_resample(
                    X_raw[test_idx], y_raw[test_idx], fold_tail_label, setting_value, args.seed + 1000 * fold
                )
            else:
                X_train_raw, y_train = multiclass_long_tail_resample(
                    X_raw[train_idx], y_raw[train_idx], fold_class_order, setting_value, args.seed + 100 * fold
                )
                X_test_raw, y_test = multiclass_long_tail_resample(
                    X_raw[test_idx], y_raw[test_idx], fold_class_order, setting_value, args.seed + 1000 * fold
                )
            encoder = TabularEncoder()
            X_train = encoder.fit_transform(X_train_raw)
            X_test = encoder.transform(X_test_raw)
            split = active_weak_split(
                X_train,
                y_train,
                n_parties=args.n_parties,
                rank_strength=3.0,
                seed=args.seed + fold,
            )
            fold_distributions.append(
                {
                    "fold": fold,
                    "tail_label_selected_from_train": str(fold_tail_label),
                    "train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                    "test": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
                }
            )
            for name, runner in methods.items():
                started = time.perf_counter()
                pred, diagnostics = runner(
                    X_train, y_train, X_test, split.party_features, 0, args.seed + fold
                )
                if "FedTree" in name:
                    time.sleep(2.0)
                row = metrics(y_test, np.asarray(pred, dtype=object))
                row["wall_time_sec"] = float(time.perf_counter() - started)
                row["rule_count"] = float(
                    diagnostics.get("ebrb_boost_rule_count", diagnostics.get("scer_combo_rule_count", 0.0))
                )
                for key, value in diagnostics.items():
                    if key.startswith("scer_") or key.startswith("multiclass_"):
                        row[key] = float(value)
                method_rows[name].append(row)
                print(
                    f"{setting_name} fold={fold} {name}: "
                    f"Acc={row['accuracy']:.4f} MacroF1={row['macro_f1']:.4f} "
                    f"BalAcc={row['balanced_accuracy']:.4f} TailF1={row['tail_f1']:.4f}"
                )
        result["results"][setting_name] = {
            "fold_distributions": fold_distributions,
            "methods": {
                name: {"folds": rows, "mean": summarize(rows)} for name, rows in method_rows.items()
            },
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
