from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vfl_tuned_dr_mbrb import (  # noqa: E402
    BaggedMBRBConfig,
    BoostedMBRBConfig,
    TabularEncoder,
    VFLBaggedDRMBRB,
    VFLBoostedDRMBRB,
    VFLTunedDRMBRB,
    apply_noisy_party,
    config_from_dict,
    make_vertibench_style_split,
)
from vfl_tuned_dr_mbrb.belief import EPS, er_fuse, normalize_rows, softmax  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_candidates: Tuple[str, ...] = ()
    sklearn_loader: str | None = None


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_openml_candidates(candidates: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    last_error: Exception | None = None
    for name in candidates:
        try:
            data = fetch_openml(name=name, version=1, as_frame=False, parser="liac-arff")
            return np.asarray(data.data, dtype=object), np.asarray(data.target, dtype=object)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load OpenML candidates {candidates}: {last_error}")


def load_dataset(spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray]:
    if spec.sklearn_loader == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(object), data.target.astype(object)
    return fetch_openml_candidates(spec.openml_candidates)


def stratified_limit(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    sample_idx, _ = next(splitter.split(X, stratify_labels(y)))
    return X[sample_idx], y[sample_idx]


def stratify_labels(y: Sequence[object]) -> np.ndarray:
    return np.unique(np.asarray(y, dtype=object).astype(str), return_inverse=True)[1]


def metric_report(y_true: Sequence[object], y_pred: Sequence[object]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    try:
        y_pred_arr = y_pred_arr.astype(y_true_arr.dtype)
    except (TypeError, ValueError):
        y_true_arr = y_true_arr.astype(str)
        y_pred_arr = y_pred_arr.astype(str)
    labels, counts = np.unique(y_true_arr, return_counts=True)
    minority_label = labels[int(np.argmin(counts))]
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "minority_f1": float(f1_score(y_true_arr, y_pred_arr, labels=[minority_label], average="macro", zero_division=0)),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    summary: Dict[str, float] = {}
    for key in keys:
        missing_value = 0.0 if key.startswith("selected_") else np.nan
        values = np.asarray([row.get(key, missing_value) for row in rows], dtype=float)
        if np.all(np.isnan(values)):
            continue
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def active_only_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    features = list(party_features[active_party_id])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X_train[:, features], y_train)
    return model.predict(X_test[:, features]), {}


def strict_vfl_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Strict vertical LR simulation.

    The active party owns labels and computes the scalar residual. Passive
    parties only receive residuals and update local weights on their own feature
    blocks; no party trains on the full feature matrix.
    """

    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Strict VFL-LR currently supports binary classification")
    rng = np.random.default_rng(seed)
    party_ids = sorted(int(pid) for pid in party_features)
    train_blocks: Dict[int, np.ndarray] = {}
    test_blocks: Dict[int, np.ndarray] = {}
    weights: Dict[int, np.ndarray] = {}
    for party_id in party_ids:
        features = list(party_features[party_id])
        Xp = np.asarray(X_train[:, features], dtype=float)
        mean = np.mean(Xp, axis=0)
        scale = np.std(Xp, axis=0)
        scale[scale < 1e-8] = 1.0
        train_blocks[party_id] = (Xp - mean) / scale
        test_blocks[party_id] = (np.asarray(X_test[:, features], dtype=float) - mean) / scale
        weights[party_id] = rng.normal(0.0, 1e-3, size=len(features))

    counts = np.bincount(y_idx, minlength=2).astype(float)
    sample_weight = np.ones_like(y_idx, dtype=float)
    class_weight = np.max(counts) / np.maximum(counts, 1.0)
    sample_weight *= class_weight[y_idx]
    sample_weight /= max(float(np.mean(sample_weight)), 1e-12)

    bias = 0.0
    n_samples = float(len(y_idx))
    learning_rate = 0.08
    l2 = 1e-3
    n_steps = 450
    for _ in range(n_steps):
        logits = np.full(len(y_idx), bias, dtype=float)
        for party_id in party_ids:
            logits += train_blocks[party_id] @ weights[party_id]
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        residual = (proba - y_idx.astype(float)) * sample_weight
        bias -= learning_rate * float(np.mean(residual))
        for party_id in party_ids:
            grad = train_blocks[party_id].T @ residual / n_samples + l2 * weights[party_id]
            weights[party_id] -= learning_rate * grad

    test_logits = np.full(X_test.shape[0], bias, dtype=float)
    for party_id in party_ids:
        test_logits += test_blocks[party_id] @ weights[party_id]
    pred_idx = (test_logits >= 0.0).astype(int)
    return classes[pred_idx], {
        "strict_vfl_lr_steps": float(n_steps),
        "strict_vfl_lr_l2": float(l2),
    }


def vfl_split_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del party_features, active_party_id
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)
    return model.predict(X_test), {}


class StrictVFLGaussianNB:
    """GaussianNB through active-label secure-statistic simulation.

    In a deployed protocol the active party's class indicators would be used
    through secure aggregation or HE to obtain per-class feature sums on each
    passive party. This simulation records the same statistics without exposing
    raw features outside party-local blocks.
    """

    def __init__(self, party_features: Mapping[int, Sequence[int]]) -> None:
        self.party_features = {int(key): list(values) for key, values in party_features.items()}
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: np.ndarray | None = None
        self.party_stats_: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StrictVFLGaussianNB":
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        self.class_prior_log_ = np.log((counts + 1.0) / (np.sum(counts) + len(self.classes_)))
        for party_id, features in self.party_features.items():
            Xp = np.asarray(X[:, features], dtype=float)
            means = np.zeros((len(self.classes_), Xp.shape[1]), dtype=float)
            variances = np.zeros_like(means)
            for class_idx in range(len(self.classes_)):
                mask = (y_idx == class_idx).astype(float)
                denom = max(float(np.sum(mask)), 1.0)
                sums = mask @ Xp
                sq_sums = mask @ (Xp * Xp)
                mean = sums / denom
                var = sq_sums / denom - mean * mean
                means[class_idx] = mean
                variances[class_idx] = np.maximum(var, 1e-6)
            self.party_stats_[party_id] = (means, variances)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_prior_log_ is None:
            raise ValueError("StrictVFLGaussianNB has not been fitted")
        fused = np.tile(self.class_prior_log_[None, :], (X.shape[0], 1))
        for party_id, features in self.party_features.items():
            means, variances = self.party_stats_[party_id]
            Xp = np.asarray(X[:, features], dtype=float)
            for class_idx in range(len(self.classes_)):
                fused[:, class_idx] += -0.5 * np.sum(
                    np.log(2.0 * np.pi * variances[class_idx])
                    + ((Xp - means[class_idx]) ** 2) / variances[class_idx],
                    axis=1,
                )
        return self.classes_[np.argmax(fused, axis=1)]


class VFLGaussianNB:
    def __init__(self, party_features: Mapping[int, Sequence[int]]) -> None:
        self.party_features = {int(key): list(values) for key, values in party_features.items()}
        self.models: Dict[int, GaussianNB] = {}
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VFLGaussianNB":
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float) + 1.0
        self.class_prior_log_ = np.log(counts / np.sum(counts))
        for party_id, features in self.party_features.items():
            model = GaussianNB()
            model.fit(X[:, features], y)
            self.models[party_id] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_prior_log_ is None:
            raise ValueError("VFLGaussianNB has not been fitted")
        fused = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for party_id, model in self.models.items():
            fused += model._joint_log_likelihood(X[:, self.party_features[party_id]])
        fused -= (len(self.models) - 1) * self.class_prior_log_[None, :]
        return self.classes_[np.argmax(fused, axis=1)]


def vfl_gaussian_nb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id, seed
    model = VFLGaussianNB(party_features)
    model.fit(X_train, y_train)
    return model.predict(X_test), {}


def strict_vfl_gaussian_nb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id, seed
    model = StrictVFLGaussianNB(party_features)
    model.fit(X_train, y_train)
    return model.predict(X_test), {"secure_stat_simulation": 1.0}


@dataclass
class _StrictVFLXGBNode:
    value: float
    party_id: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "_StrictVFLXGBNode | None" = None
    right: "_StrictVFLXGBNode | None" = None


class _StrictVFLXGBTree:
    def __init__(
        self,
        thresholds: Mapping[int, Sequence[np.ndarray]],
        bin_ids: Mapping[int, np.ndarray],
        max_depth: int,
        reg_lambda: float,
        gamma: float,
        min_child_weight: float,
        min_samples_leaf: int,
    ) -> None:
        self.thresholds = {int(pid): list(values) for pid, values in thresholds.items()}
        self.bin_ids = {int(pid): np.asarray(values, dtype=np.int16) for pid, values in bin_ids.items()}
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.root_: _StrictVFLXGBNode | None = None
        self.n_splits_ = 0

    def fit(self, gradients: np.ndarray, hessians: np.ndarray) -> "_StrictVFLXGBTree":
        self.gradients_ = np.asarray(gradients, dtype=float)
        self.hessians_ = np.asarray(hessians, dtype=float)
        self.root_ = self._build(np.arange(len(self.gradients_), dtype=int), depth=0)
        return self

    def predict_blocks(self, blocks: Mapping[int, np.ndarray]) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Tree has not been fitted")
        out = np.zeros(next(iter(blocks.values())).shape[0], dtype=float)
        self._predict_node(self.root_, np.arange(out.shape[0], dtype=int), out, blocks)
        return out

    def _build(self, indices: np.ndarray, depth: int) -> _StrictVFLXGBNode:
        grad = self.gradients_[indices]
        hess = self.hessians_[indices]
        g_sum = float(np.sum(grad))
        h_sum = float(np.sum(hess))
        value = -g_sum / max(h_sum + self.reg_lambda, 1e-12)
        node = _StrictVFLXGBNode(value=value)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(indices, g_sum, h_sum)
        if split is None:
            return node
        party_id, feature_index, bin_threshold, threshold = split
        bins = self.bin_ids[party_id][indices, feature_index]
        left_mask = bins <= bin_threshold
        if int(np.sum(left_mask)) < self.min_samples_leaf or int(np.sum(~left_mask)) < self.min_samples_leaf:
            return node
        self.n_splits_ += 1
        node.party_id = party_id
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build(indices[left_mask], depth + 1)
        node.right = self._build(indices[~left_mask], depth + 1)
        return node

    def _best_split(
        self,
        indices: np.ndarray,
        parent_g: float,
        parent_h: float,
    ) -> Tuple[int, int, int, float] | None:
        parent_score = parent_g * parent_g / max(parent_h + self.reg_lambda, 1e-12)
        best_gain = -np.inf
        best: Tuple[int, int, int, float] | None = None
        grad = self.gradients_[indices]
        hess = self.hessians_[indices]
        for party_id in sorted(self.bin_ids):
            party_bins = self.bin_ids[party_id]
            for feature_index, thresholds in enumerate(self.thresholds[party_id]):
                n_bins = len(thresholds) + 1
                if n_bins <= 1:
                    continue
                bins = party_bins[indices, feature_index]
                g_hist = np.bincount(bins, weights=grad, minlength=n_bins)
                h_hist = np.bincount(bins, weights=hess, minlength=n_bins)
                g_left = np.cumsum(g_hist)[:-1]
                h_left = np.cumsum(h_hist)[:-1]
                g_right = parent_g - g_left
                h_right = parent_h - h_left
                valid = (h_left >= self.min_child_weight) & (h_right >= self.min_child_weight)
                if not np.any(valid):
                    continue
                gains = 0.5 * (
                    g_left * g_left / np.maximum(h_left + self.reg_lambda, 1e-12)
                    + g_right * g_right / np.maximum(h_right + self.reg_lambda, 1e-12)
                    - parent_score
                ) - self.gamma
                gains = np.where(valid, gains, -np.inf)
                split_idx = int(np.argmax(gains))
                gain = float(gains[split_idx])
                if gain > best_gain and gain > 0.0:
                    best_gain = gain
                    best = (party_id, feature_index, split_idx, float(thresholds[split_idx]))
        return best

    def _predict_node(
        self,
        node: _StrictVFLXGBNode,
        indices: np.ndarray,
        out: np.ndarray,
        blocks: Mapping[int, np.ndarray],
    ) -> None:
        if node.party_id is None or node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
            out[indices] = node.value
            return
        values = blocks[node.party_id][indices, node.feature_index]
        left_mask = values <= node.threshold
        if np.any(left_mask):
            self._predict_node(node.left, indices[left_mask], out, blocks)
        if np.any(~left_mask):
            self._predict_node(node.right, indices[~left_mask], out, blocks)


def strict_vfl_xgboost_hist_runner(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    max_bins: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    decision_threshold_method: str = "argmax",
    decision_threshold_prior_multiplier: float = 1.0,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        del active_party_id, seed
        classes, y_idx = np.unique(y_train, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("Strict VFL-XGBoost Hist currently supports binary classification")
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
        logits = np.full(len(y_idx), np.log(pos_rate / (1.0 - pos_rate)), dtype=float)
        trees: List[_StrictVFLXGBTree] = []
        split_counts: List[float] = []
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
            update = tree.predict_blocks(train_blocks)
            logits += float(learning_rate) * update
            trees.append(tree)
            split_counts.append(float(tree.n_splits_))

        train_proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        test_logits = np.full(X_test.shape[0], np.log(pos_rate / (1.0 - pos_rate)), dtype=float)
        for tree in trees:
            test_logits += float(learning_rate) * tree.predict_blocks(test_blocks)
        test_proba = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30.0, 30.0)))
        threshold = 0.5
        threshold_class = 1
        if decision_threshold_method == "argmax":
            pred_idx = (test_proba >= threshold).astype(int)
        elif decision_threshold_method == "train_prior":
            counts = np.bincount(y_idx, minlength=2).astype(float)
            minority_idx = int(np.argmin(counts))
            target_rate = float(
                np.clip(
                    counts[minority_idx] / max(float(np.sum(counts)), EPS) * float(decision_threshold_prior_multiplier),
                    EPS,
                    1.0 - EPS,
                )
            )
            threshold_class = minority_idx
            if minority_idx == 1:
                threshold = float(np.quantile(train_proba, 1.0 - target_rate))
                pred_idx = np.zeros(test_proba.shape[0], dtype=int)
                pred_idx[test_proba >= threshold] = 1
            else:
                train_minority_proba = 1.0 - train_proba
                test_minority_proba = 1.0 - test_proba
                threshold = float(np.quantile(train_minority_proba, 1.0 - target_rate))
                pred_idx = np.ones(test_proba.shape[0], dtype=int)
                pred_idx[test_minority_proba >= threshold] = 0
        else:
            raise ValueError(f"Unknown strict VFL-XGBoost decision threshold method: {decision_threshold_method}")
        return classes[pred_idx], {
            "strict_vfl_xgboost_trees": float(len(trees)),
            "strict_vfl_xgboost_depth": float(max_depth),
            "strict_vfl_xgboost_max_bins": float(max_bins),
            "strict_vfl_xgboost_avg_splits": float(np.mean(split_counts)),
            "strict_vfl_xgboost_decision_threshold": float(threshold),
            "strict_vfl_xgboost_threshold_class": float(threshold_class),
        }

    return run


@dataclass(frozen=True)
class _StrictVFLEBRBLiteral:
    feature_index: int
    direction: str
    threshold: float
    scale: float


@dataclass
class _StrictVFLEBRBRule:
    antecedents: Tuple[_StrictVFLEBRBLiteral, ...]
    theta: float
    beta: np.ndarray
    reliability: float
    support: float
    score: float


@dataclass
class _StrictVFLEBRBCandidate:
    antecedents: Tuple[_StrictVFLEBRBLiteral, ...]
    activation: np.ndarray
    beta: np.ndarray
    theta: float
    reliability: float
    support: float
    score: float


class _StrictVFLEBRBParty:
    """Client-side EBRB antecedents with active/server-side consequent calibration.

    The party owns only a local feature block. It can compute rule activation
    summaries from local features; the active/server side calibrates consequent
    belief degrees against labels. This simulates a strict VFL evidence protocol
    without handing plaintext labels to passive parties.
    """

    def __init__(
        self,
        party_id: int,
        n_rules: int,
        bins: int,
        max_antecedents: int,
        beam_width: int,
        beta_smoothing: float,
        min_support_fraction: float,
        class_balance_power: float,
        parameter_iterations: int,
        parameter_update_blend: float,
    ) -> None:
        self.party_id = int(party_id)
        self.n_rules = int(n_rules)
        self.bins = int(bins)
        self.max_antecedents = int(max_antecedents)
        self.beam_width = int(beam_width)
        self.beta_smoothing = float(beta_smoothing)
        self.min_support_fraction = float(min_support_fraction)
        self.class_balance_power = float(class_balance_power)
        self.parameter_iterations = int(parameter_iterations)
        self.parameter_update_blend = float(parameter_update_blend)
        self.rules_: List[_StrictVFLEBRBRule] = []
        self.n_features_in_: int | None = None
        self.parameter_update_shift_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        class_prior: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "_StrictVFLEBRBParty":
        arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ValueError("EBRB party needs a two-dimensional local feature block")
        self.n_features_in_ = int(arr.shape[1])
        weights = np.asarray(sample_weight, dtype=float)
        weights = weights / max(float(np.sum(weights)), EPS)
        total_weight = float(np.sum(weights))
        class_counts = np.bincount(y_idx, weights=weights, minlength=n_classes).astype(float)
        if self.class_balance_power > 0.0:
            class_weights = np.power(
                np.max(class_counts) / np.maximum(class_counts, EPS),
                float(np.clip(self.class_balance_power, 0.0, 1.0)),
            )
        else:
            class_weights = np.ones(n_classes, dtype=float)
        base_beta = class_counts + self.beta_smoothing * np.asarray(class_prior, dtype=float)
        base_beta = base_beta / max(float(np.sum(base_beta)), EPS)
        base = _StrictVFLEBRBRule(
            antecedents=tuple(),
            theta=0.05,
            beta=base_beta,
            reliability=1.0,
            support=total_weight,
            score=0.0,
        )

        single_candidates: List[_StrictVFLEBRBCandidate] = []
        quantiles = np.linspace(0.0, 1.0, max(3, self.bins) + 1)[1:-1]
        for feature_index in range(arr.shape[1]):
            values = arr[:, feature_index]
            unique = np.unique(values)
            if unique.size <= 1:
                continue
            thresholds = np.unique(np.quantile(values, quantiles))
            feature_scale = float(np.std(values))
            if feature_scale <= EPS:
                feature_scale = 1.0
            literal_scale = max(feature_scale / max(float(self.bins), 1.0), 1e-3)
            for threshold in thresholds:
                for direction in ("le", "gt"):
                    literal = _StrictVFLEBRBLiteral(
                        feature_index=int(feature_index),
                        direction=direction,
                        threshold=float(threshold),
                        scale=float(literal_scale),
                    )
                    activation = self._literal_activation(values, literal)
                    candidate = self._candidate_from_activation(
                        antecedents=(literal,),
                        activation=activation,
                        y_idx=y_idx,
                        n_classes=n_classes,
                        class_prior=class_prior,
                        sample_weight=weights,
                        class_weights=class_weights,
                    )
                    if candidate is not None:
                        single_candidates.append(candidate)

        candidates = list(single_candidates)
        if self.max_antecedents >= 2 and single_candidates:
            literal_pool = sorted(single_candidates, key=lambda item: item.score, reverse=True)[: max(2, self.beam_width)]
            for left_idx, left in enumerate(literal_pool):
                for right in literal_pool[left_idx + 1 :]:
                    left_lit, right_lit = left.antecedents[0], right.antecedents[0]
                    if left_lit.feature_index == right_lit.feature_index:
                        continue
                    activation = left.activation * right.activation
                    candidate = self._candidate_from_activation(
                        antecedents=(left_lit, right_lit),
                        activation=activation,
                        y_idx=y_idx,
                        n_classes=n_classes,
                        class_prior=class_prior,
                        sample_weight=weights,
                        class_weights=class_weights,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        ranked_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        if n_classes > 2 and self.class_balance_power > 0.0:
            # Reserve local-rule capacity for every consequent class before
            # SCER construction. Otherwise, under extreme multiclass imbalance,
            # globally ranked head-class rules can exhaust the small local
            # budget and leave SCER with no tail-class rule to combine.
            capacity = max(1, self.n_rules - 1)
            grouped_candidates = {
                class_idx: [item for item in ranked_candidates if int(np.argmax(item.beta)) == class_idx]
                for class_idx in range(n_classes)
            }
            base_quota, remainder = divmod(capacity, n_classes)
            balanced_candidates: List[_StrictVFLEBRBCandidate] = []
            leftovers: List[_StrictVFLEBRBCandidate] = []
            for class_idx in range(n_classes):
                quota = base_quota + (1 if class_idx < remainder else 0)
                balanced_candidates.extend(grouped_candidates[class_idx][:quota])
                leftovers.extend(grouped_candidates[class_idx][quota:])
            if len(balanced_candidates) < capacity:
                leftovers.sort(key=lambda item: item.score, reverse=True)
                balanced_candidates.extend(leftovers[: capacity - len(balanced_candidates)])
            ranked_candidates = balanced_candidates

        self.rules_ = [base]
        seen = {tuple()}
        for candidate in ranked_candidates:
            key = tuple(
                (literal.feature_index, literal.direction, round(literal.threshold, 10))
                for literal in candidate.antecedents
            )
            if key in seen:
                continue
            self.rules_.append(
                _StrictVFLEBRBRule(
                    antecedents=candidate.antecedents,
                    theta=candidate.theta,
                    beta=candidate.beta,
                    reliability=candidate.reliability,
                    support=candidate.support,
                    score=candidate.score,
                )
            )
            seen.add(key)
            if len(self.rules_) >= max(2, self.n_rules):
                break
        self._iterate_rule_parameters(
            arr=arr,
            y_idx=y_idx,
            n_classes=n_classes,
            class_prior=class_prior,
            sample_weight=weights,
            class_weights=class_weights,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.rules_ or self.n_features_in_ is None:
            raise ValueError("EBRB party has not been fitted")
        arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2 or arr.shape[1] != self.n_features_in_:
            raise ValueError("Input feature dimension does not match fitted EBRB party")
        activation = np.zeros((arr.shape[0], len(self.rules_)), dtype=float)
        betas = np.stack([rule.beta for rule in self.rules_], axis=0)
        for rule_idx, rule in enumerate(self.rules_):
            match = np.ones(arr.shape[0], dtype=float)
            for literal in rule.antecedents:
                match *= self._literal_activation(arr[:, literal.feature_index], literal)
            activation[:, rule_idx] = match * max(rule.theta, EPS) * max(rule.reliability, EPS)
        activation = activation / np.maximum(np.sum(activation, axis=1, keepdims=True), EPS)
        evidence = np.broadcast_to(betas[None, :, :], (arr.shape[0], betas.shape[0], betas.shape[1]))
        return er_fuse(evidence, activation)

    def _candidate_from_activation(
        self,
        antecedents: Tuple[_StrictVFLEBRBLiteral, ...],
        activation: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        class_prior: np.ndarray,
        sample_weight: np.ndarray,
        class_weights: np.ndarray,
    ) -> _StrictVFLEBRBCandidate | None:
        weighted_activation = np.asarray(activation, dtype=float) * sample_weight
        support = float(np.sum(weighted_activation))
        min_support = max(float(self.min_support_fraction), 0.0) * float(np.sum(sample_weight))
        if support < max(min_support, EPS):
            return None
        mass = np.bincount(y_idx, weights=weighted_activation, minlength=n_classes).astype(float)
        observed = mass / max(support, EPS)
        weighted_mass = mass * class_weights
        beta = weighted_mass + self.beta_smoothing * np.asarray(class_prior, dtype=float)
        beta = beta / max(float(np.sum(beta)), EPS)
        support_fraction = support / max(float(np.sum(sample_weight)), EPS)
        gini_prior = 1.0 - float(np.sum(np.asarray(class_prior, dtype=float) ** 2))
        gini_rule = 1.0 - float(np.sum(observed**2))
        dominant = int(np.argmax(observed))
        lift = max(0.0, float(observed[dominant] - class_prior[dominant]))
        gain = max(0.0, gini_prior - gini_rule) * support_fraction
        purity = float(np.max(observed))
        reliability = float(np.clip(0.15 + 0.85 * purity * np.sqrt(max(support_fraction, EPS)), 0.05, 1.0))
        theta = float(np.clip(np.power(max(support_fraction, EPS), 0.25), 0.05, 1.0))
        score = float(gain + 0.35 * lift * np.sqrt(max(support_fraction, EPS)) + 0.05 * purity)
        return _StrictVFLEBRBCandidate(
            antecedents=antecedents,
            activation=activation,
            beta=beta,
            theta=theta,
            reliability=reliability,
            support=support,
            score=score,
        )

    def _iterate_rule_parameters(
        self,
        arr: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        class_prior: np.ndarray,
        sample_weight: np.ndarray,
        class_weights: np.ndarray,
    ) -> None:
        iterations = max(0, int(self.parameter_iterations))
        if iterations <= 0 or len(self.rules_) <= 1:
            self.parameter_update_shift_ = 0.0
            return
        blend = float(np.clip(self.parameter_update_blend, 0.0, 1.0))
        if blend <= 0.0:
            self.parameter_update_shift_ = 0.0
            return

        match = self._rule_match_matrix(arr)
        weights = np.asarray(sample_weight, dtype=float)
        weights = weights / max(float(np.sum(weights)), EPS)
        update_shifts: List[float] = []
        class_prior_arr = np.asarray(class_prior, dtype=float)
        class_prior_arr = class_prior_arr / max(float(np.sum(class_prior_arr)), EPS)
        class_weights_arr = np.asarray(class_weights, dtype=float)

        for _ in range(iterations):
            betas = np.stack([rule.beta for rule in self.rules_], axis=0)
            theta = np.asarray([max(rule.theta, EPS) for rule in self.rules_], dtype=float)
            reliability = np.asarray([max(rule.reliability, EPS) for rule in self.rules_], dtype=float)
            strength = match * theta[None, :] * reliability[None, :]
            activation = strength / np.maximum(np.sum(strength, axis=1, keepdims=True), EPS)
            evidence = np.broadcast_to(betas[None, :, :], (arr.shape[0], betas.shape[0], betas.shape[1]))
            belief = er_fuse(evidence, activation)
            true_belief = np.clip(belief[np.arange(arr.shape[0]), y_idx], EPS, 1.0)
            difficulty = np.maximum(1.0 - true_belief, EPS)
            difficulty = difficulty / max(float(np.average(difficulty, weights=weights)), EPS)
            difficulty = np.clip(difficulty, 0.25, 4.0)
            label_likelihood = betas[:, y_idx].T
            responsibility = strength * np.maximum(label_likelihood, EPS)
            responsibility = responsibility / np.maximum(np.sum(responsibility, axis=1, keepdims=True), EPS)
            weighted_resp = responsibility * weights[:, None] * difficulty[:, None]
            support = np.sum(weighted_resp, axis=0)

            shift = 0.0
            for rule_idx, rule in enumerate(self.rules_):
                if not rule.antecedents:
                    continue
                class_mass = np.bincount(
                    y_idx,
                    weights=weighted_resp[:, rule_idx],
                    minlength=n_classes,
                ).astype(float)
                observed = class_mass / max(float(np.sum(class_mass)), EPS)
                weighted_mass = class_mass * class_weights_arr
                new_beta = weighted_mass + self.beta_smoothing * class_prior_arr
                new_beta = new_beta / max(float(np.sum(new_beta)), EPS)
                support_fraction = float(support[rule_idx] / max(float(np.sum(weights)), EPS))
                purity = float(np.max(observed))
                new_reliability = float(
                    np.clip(0.15 + 0.85 * purity * np.sqrt(max(support_fraction, EPS)), 0.05, 1.0)
                )
                new_theta = float(np.clip(np.power(max(support_fraction, EPS), 0.25), 0.05, 1.0))
                gini_prior = 1.0 - float(np.sum(class_prior_arr**2))
                gini_rule = 1.0 - float(np.sum(observed**2))
                dominant = int(np.argmax(observed))
                lift = max(0.0, float(observed[dominant] - class_prior_arr[dominant]))
                gain = max(0.0, gini_prior - gini_rule) * support_fraction
                new_score = float(gain + 0.35 * lift * np.sqrt(max(support_fraction, EPS)) + 0.05 * purity)

                old_beta = rule.beta.copy()
                old_theta = float(rule.theta)
                old_reliability = float(rule.reliability)
                rule.beta = normalize_rows(((1.0 - blend) * old_beta + blend * new_beta)[None, :])[0]
                rule.theta = float((1.0 - blend) * old_theta + blend * new_theta)
                rule.reliability = float((1.0 - blend) * old_reliability + blend * new_reliability)
                rule.support = float((1.0 - blend) * rule.support + blend * support_fraction)
                rule.score = float((1.0 - blend) * rule.score + blend * new_score)
                shift += float(np.mean(np.abs(rule.beta - old_beta)))
                shift += abs(float(rule.theta) - old_theta)
                shift += abs(float(rule.reliability) - old_reliability)
            update_shifts.append(shift / max(float(len(self.rules_) - 1), 1.0))
        self.parameter_update_shift_ = float(np.mean(update_shifts)) if update_shifts else 0.0

    def _rule_match_matrix(self, arr: np.ndarray) -> np.ndarray:
        match_matrix = np.ones((arr.shape[0], len(self.rules_)), dtype=float)
        for rule_idx, rule in enumerate(self.rules_):
            match = np.ones(arr.shape[0], dtype=float)
            for literal in rule.antecedents:
                match *= self._literal_activation(arr[:, literal.feature_index], literal)
            match_matrix[:, rule_idx] = match
        return match_matrix

    @staticmethod
    def _literal_activation(values: np.ndarray, literal: _StrictVFLEBRBLiteral) -> np.ndarray:
        scaled = (np.asarray(values, dtype=float) - literal.threshold) / max(float(literal.scale), EPS)
        scaled = np.clip(scaled, -30.0, 30.0)
        if literal.direction == "le":
            return 1.0 / (1.0 + np.exp(scaled))
        if literal.direction == "gt":
            return 1.0 / (1.0 + np.exp(-scaled))
        raise ValueError(f"Unknown literal direction: {literal.direction}")


class _StrictVFLEBRB:
    def __init__(
        self,
        n_rules_per_party: int,
        bins: int,
        max_antecedents: int,
        beam_width: int,
        beta_smoothing: float,
        min_support_fraction: float,
        class_balance_power: float,
        parameter_iterations: int,
        parameter_update_blend: float,
        client_weight_method: str,
        decision_threshold_method: str,
        decision_threshold_prior_multiplier: float,
        random_state: int,
    ) -> None:
        self.n_rules_per_party = int(n_rules_per_party)
        self.bins = int(bins)
        self.max_antecedents = int(max_antecedents)
        self.beam_width = int(beam_width)
        self.beta_smoothing = float(beta_smoothing)
        self.min_support_fraction = float(min_support_fraction)
        self.class_balance_power = float(class_balance_power)
        self.parameter_iterations = int(parameter_iterations)
        self.parameter_update_blend = float(parameter_update_blend)
        self.client_weight_method = client_weight_method
        self.decision_threshold_method = decision_threshold_method
        self.decision_threshold_prior_multiplier = float(decision_threshold_prior_multiplier)
        self.random_state = int(random_state)
        self.classes_: np.ndarray | None = None
        self.party_features_: Dict[int, List[int]] = {}
        self.parties_: Dict[int, _StrictVFLEBRBParty] = {}
        self.client_weights_: np.ndarray | None = None
        self.decision_threshold_class_: int | None = None
        self.decision_threshold_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        sample_weight: np.ndarray | None = None,
    ) -> "_StrictVFLEBRB":
        del active_party_id
        arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        n_classes = len(self.classes_)
        weights = np.ones(arr.shape[0], dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = np.maximum(weights, EPS)
        weights = weights / max(float(np.sum(weights)), EPS)
        class_prior = np.bincount(y_idx, weights=weights, minlength=n_classes).astype(float)
        class_prior = (class_prior + 1.0) / max(float(np.sum(class_prior) + n_classes), EPS)
        self.party_features_ = {int(key): [int(value) for value in values] for key, values in party_features.items()}
        self.parties_.clear()
        for offset, party_id in enumerate(sorted(self.party_features_)):
            party = _StrictVFLEBRBParty(
                party_id=party_id,
                n_rules=self.n_rules_per_party,
                bins=self.bins,
                max_antecedents=self.max_antecedents,
                beam_width=self.beam_width,
                beta_smoothing=self.beta_smoothing,
                min_support_fraction=self.min_support_fraction,
                class_balance_power=self.class_balance_power,
                parameter_iterations=self.parameter_iterations,
                parameter_update_blend=self.parameter_update_blend,
            )
            party.fit(
                arr[:, self.party_features_[party_id]],
                y_idx=y_idx,
                n_classes=n_classes,
                class_prior=class_prior,
                sample_weight=weights,
            )
            self.parties_[party_id] = party

        train_stack = self._party_stack(arr)
        pred_idx = np.argmax(train_stack, axis=2)
        confidence = np.mean(np.max(train_stack, axis=2), axis=0)
        accuracy = np.mean(pred_idx == y_idx[:, None], axis=0)
        if self.client_weight_method == "label_accuracy":
            raw_weights = np.maximum(accuracy, EPS)
        elif self.client_weight_method == "confidence":
            raw_weights = np.maximum(confidence, EPS)
        else:
            raw_weights = np.maximum(0.5 * accuracy + 0.5 * confidence, EPS)
        self.client_weights_ = raw_weights / max(float(np.sum(raw_weights)), EPS)
        self._fit_decision_threshold(arr, y_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.client_weights_ is None:
            raise ValueError("Strict VFL-EBRB has not been fitted")
        arr = np.asarray(X, dtype=float)
        stack = self._party_stack(arr)
        confidence = np.maximum(np.max(stack, axis=2), EPS)
        weights = confidence * self.client_weights_[None, :]
        return er_fuse(stack, weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Strict VFL-EBRB has not been fitted")
        proba = self.predict_proba(X)
        if self.decision_threshold_class_ is not None and self.decision_threshold_ is not None and len(self.classes_) == 2:
            target = int(self.decision_threshold_class_)
            other = 1 - target
            pred_idx = np.full(proba.shape[0], other, dtype=int)
            pred_idx[proba[:, target] >= float(self.decision_threshold_)] = target
            return self.classes_[pred_idx]
        return self.classes_[np.argmax(proba, axis=1)]

    def diagnostics(self) -> Dict[str, float]:
        if not self.parties_ or self.client_weights_ is None:
            raise ValueError("Strict VFL-EBRB has not been fitted")
        rule_counts = np.asarray([len(party.rules_) for party in self.parties_.values()], dtype=float)
        update_shifts = np.asarray([party.parameter_update_shift_ for party in self.parties_.values()], dtype=float)
        return {
            "ebrb_party_count": float(len(self.parties_)),
            "ebrb_rule_count": float(np.sum(rule_counts)),
            "ebrb_rules_per_party_mean": float(np.mean(rule_counts)),
            "ebrb_parameter_iterations": float(self.parameter_iterations),
            "ebrb_parameter_update_blend": float(self.parameter_update_blend),
            "ebrb_parameter_update_shift_mean": float(np.mean(update_shifts)),
            "ebrb_client_weight_min": float(np.min(self.client_weights_)),
            "ebrb_client_weight_max": float(np.max(self.client_weights_)),
            "ebrb_decision_threshold_enabled": float(self.decision_threshold_ is not None),
        }

    def _party_stack(self, X: np.ndarray) -> np.ndarray:
        aligned = []
        for party_id in sorted(self.parties_):
            party = self.parties_[party_id]
            aligned.append(party.predict_proba(X[:, self.party_features_[party_id]]))
        return np.stack(aligned, axis=1)

    def _fit_decision_threshold(self, X: np.ndarray, y_idx: np.ndarray) -> None:
        if self.decision_threshold_method == "argmax":
            return
        if self.decision_threshold_method != "train_prior":
            raise ValueError(f"Unknown EBRB decision threshold method: {self.decision_threshold_method}")
        if self.classes_ is None or len(self.classes_) != 2:
            return
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        minority_idx = int(np.argmin(counts))
        minority_prior = float(counts[minority_idx] / max(float(np.sum(counts)), EPS))
        target_rate = float(
            np.clip(
                minority_prior * self.decision_threshold_prior_multiplier,
                EPS,
                1.0 - EPS,
            )
        )
        proba = self.predict_proba(X)
        self.decision_threshold_class_ = minority_idx
        self.decision_threshold_ = float(np.quantile(proba[:, minority_idx], 1.0 - target_rate))


class _StrictVFLBoostedEBRB:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        sample_fraction: float,
        n_rules_per_party: int,
        bins: int,
        max_antecedents: int,
        beam_width: int,
        beta_smoothing: float,
        min_support_fraction: float,
        class_balance_power: float,
        parameter_iterations: int,
        parameter_update_blend: float,
        sampling_class_balance_power: float,
        client_weight_method: str,
        fusion_method: str,
        weight_update_method: str,
        decision_threshold_method: str,
        decision_threshold_prior_multiplier: float,
        server_logit_la_gamma: float,
        server_logit_la_temperature: float,
        server_logit_la_blend: float,
        server_class_conditioned_fusion: bool,
        server_class_weight_power: float,
        client_fit_weight_power: float,
        random_state: int,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.sample_fraction = float(sample_fraction)
        self.n_rules_per_party = int(n_rules_per_party)
        self.bins = int(bins)
        self.max_antecedents = int(max_antecedents)
        self.beam_width = int(beam_width)
        self.beta_smoothing = float(beta_smoothing)
        self.min_support_fraction = float(min_support_fraction)
        self.class_balance_power = float(class_balance_power)
        self.parameter_iterations = int(parameter_iterations)
        self.parameter_update_blend = float(parameter_update_blend)
        self.sampling_class_balance_power = float(sampling_class_balance_power)
        self.client_weight_method = client_weight_method
        self.fusion_method = fusion_method
        self.weight_update_method = weight_update_method
        self.decision_threshold_method = decision_threshold_method
        self.decision_threshold_prior_multiplier = float(decision_threshold_prior_multiplier)
        self.server_logit_la_gamma = float(server_logit_la_gamma)
        self.server_logit_la_temperature = float(server_logit_la_temperature)
        self.server_logit_la_blend = float(server_logit_la_blend)
        self.server_class_conditioned_fusion = bool(server_class_conditioned_fusion)
        self.server_class_weight_power = float(server_class_weight_power)
        self.client_fit_weight_power = float(client_fit_weight_power)
        self.random_state = int(random_state)
        self.estimators_: List[_StrictVFLEBRB] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self.estimator_mean_beliefs_: List[np.ndarray] = []
        self.estimator_class_scores_: List[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.decision_threshold_class_: int | None = None
        self.decision_threshold_: float | None = None
        self.base_logit_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
    ) -> "_StrictVFLBoostedEBRB":
        arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        if self.fusion_method == "additive_logit":
            return self._fit_additive_logit(arr, y_arr, y_idx, party_features, active_party_id)
        rng = np.random.default_rng(self.random_state)
        sample_weights = self._initial_sample_weights(y_idx, len(self.classes_))
        self.estimators_.clear()
        self.estimator_weights_.clear()
        self.estimator_errors_.clear()
        self.estimator_mean_beliefs_.clear()
        self.estimator_class_scores_.clear()
        for estimator_idx in range(max(1, self.n_estimators)):
            sample_idx = self._sample_rows(y_idx, sample_weights, rng)
            model = _StrictVFLEBRB(
                n_rules_per_party=self.n_rules_per_party,
                bins=self.bins,
                max_antecedents=self.max_antecedents,
                beam_width=self.beam_width,
                beta_smoothing=self.beta_smoothing,
                min_support_fraction=self.min_support_fraction,
                class_balance_power=self.class_balance_power,
                parameter_iterations=self.parameter_iterations,
                parameter_update_blend=self.parameter_update_blend,
                client_weight_method=self.client_weight_method,
                decision_threshold_method="argmax",
                decision_threshold_prior_multiplier=1.0,
                random_state=self.random_state + 1009 * estimator_idx,
            )
            model.fit(
                arr[sample_idx],
                y_arr[sample_idx],
                party_features=party_features,
                active_party_id=active_party_id,
                sample_weight=self._client_fit_weights(sample_weights, sample_idx),
            )
            proba = self._align_model_proba(model, arr)
            mean_belief = np.mean(proba, axis=0)
            mean_belief = mean_belief / max(float(np.sum(mean_belief)), EPS)
            class_scores = self._estimator_class_scores(proba, y_idx, sample_weights, len(self.classes_))
            pred_idx = np.argmax(proba, axis=1)
            incorrect = np.asarray(pred_idx != y_idx, dtype=float)
            weighted_error = float(np.sum(sample_weights * incorrect))
            estimator_weight = self._estimator_weight(weighted_error, len(self.classes_))
            self.estimators_.append(model)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(weighted_error)
            self.estimator_mean_beliefs_.append(mean_belief)
            self.estimator_class_scores_.append(class_scores)
            update_proba = self.predict_proba(arr) if self.weight_update_method == "ensemble_residual" else proba
            sample_weights = self._updated_sample_weights(
                sample_weights,
                incorrect,
                update_proba,
                y_idx,
                estimator_weight,
            )
            sample_weights = sample_weights / max(float(np.sum(sample_weights)), EPS)
        self._fit_decision_threshold(arr, y_idx)
        return self

    def _fit_additive_logit(
        self,
        arr: np.ndarray,
        y_arr: np.ndarray,
        y_idx: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
    ) -> "_StrictVFLBoostedEBRB":
        if self.classes_ is None or len(self.classes_) != 2:
            raise ValueError("additive_logit EBRB boosting currently supports binary classification only")
        rng = np.random.default_rng(self.random_state)
        y_binary = (y_idx == 1).astype(float)
        base_weights = self._initial_sample_weights(y_idx, len(self.classes_))
        prior = float(np.average(y_binary, weights=base_weights))
        prior = float(np.clip(prior, EPS, 1.0 - EPS))
        self.base_logit_ = float(np.log(prior / (1.0 - prior)))
        current_logit = np.full(arr.shape[0], self.base_logit_, dtype=float)
        self.estimators_.clear()
        self.estimator_weights_.clear()
        self.estimator_errors_.clear()
        self.estimator_mean_beliefs_.clear()
        self.estimator_class_scores_.clear()

        for estimator_idx in range(max(1, self.n_estimators)):
            p_current = 1.0 / (1.0 + np.exp(-np.clip(current_logit, -30.0, 30.0)))
            gradient = p_current - y_binary
            hessian = np.maximum(p_current * (1.0 - p_current), EPS)
            newton_residual = np.abs(gradient) / np.maximum(hessian, EPS)
            fit_weights = base_weights * np.maximum(np.abs(gradient), EPS) * np.sqrt(hessian + EPS) * (
                1.0 + 0.25 * np.clip(newton_residual, 0.0, 8.0)
            )
            fit_weights = fit_weights / max(float(np.sum(fit_weights)), EPS)
            sample_idx = self._sample_rows(y_idx, fit_weights, rng)
            model = _StrictVFLEBRB(
                n_rules_per_party=self.n_rules_per_party,
                bins=self.bins,
                max_antecedents=self.max_antecedents,
                beam_width=self.beam_width,
                beta_smoothing=self.beta_smoothing,
                min_support_fraction=self.min_support_fraction,
                class_balance_power=self.class_balance_power,
                parameter_iterations=self.parameter_iterations,
                parameter_update_blend=self.parameter_update_blend,
                client_weight_method=self.client_weight_method,
                decision_threshold_method="argmax",
                decision_threshold_prior_multiplier=1.0,
                random_state=self.random_state + 1009 * estimator_idx,
            )
            model.fit(
                arr[sample_idx],
                y_arr[sample_idx],
                party_features=party_features,
                active_party_id=active_party_id,
                sample_weight=fit_weights[sample_idx],
            )
            proba = self._align_model_proba(model, arr)
            delta = self._binary_delta_logit(proba)
            step = float(self.learning_rate) * self._line_search_step(current_logit, delta, y_binary, base_weights)
            current_logit = current_logit + step * delta
            mean_belief = np.mean(proba, axis=0)
            mean_belief = mean_belief / max(float(np.sum(mean_belief)), EPS)
            class_scores = self._estimator_class_scores(proba, y_idx, fit_weights, len(self.classes_))
            pred_idx = (p_current >= 0.5).astype(int)
            incorrect = np.asarray(pred_idx != y_idx, dtype=float)
            weighted_error = float(np.sum(fit_weights * incorrect))
            self.estimators_.append(model)
            self.estimator_weights_.append(step)
            self.estimator_errors_.append(weighted_error)
            self.estimator_mean_beliefs_.append(mean_belief)
            self.estimator_class_scores_.append(class_scores)
        self._fit_decision_threshold(arr, y_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_ or self.classes_ is None:
            raise ValueError("Strict VFL-EBRB boost has not been fitted")
        arr = np.asarray(X, dtype=float)
        if self.fusion_method == "additive_logit":
            if self.classes_ is None or len(self.classes_) != 2:
                raise ValueError("additive_logit EBRB boosting currently supports binary classification only")
            logits = np.full(arr.shape[0], float(self.base_logit_), dtype=float)
            for model, weight in zip(self.estimators_, self.estimator_weights_):
                proba = self._align_model_proba(model, arr)
                logits += float(weight) * self._binary_delta_logit(proba)
            p_one = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            return normalize_rows(np.column_stack([1.0 - p_one, p_one]))
        aligned = []
        for estimator_idx, model in enumerate(self.estimators_):
            proba = self._align_model_proba(model, arr)
            if estimator_idx < len(self.estimator_mean_beliefs_):
                proba = self._apply_server_logit_la(proba, self.estimator_mean_beliefs_[estimator_idx])
            aligned.append(proba)
        stack = np.stack(aligned, axis=1)
        weights = self._fusion_weights()
        if self.fusion_method == "mean":
            return normalize_rows(np.sum(stack * weights[None, :, :], axis=1))
        if self.fusion_method == "log":
            logits = np.sum(weights[None, :, :] * np.log(stack + EPS), axis=1)
            return softmax(logits)
        if self.fusion_method == "additive_logit":
            raise ValueError("additive_logit should have been handled before stack fusion")
        raise ValueError(f"Unknown EBRB boost fusion method: {self.fusion_method}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Strict VFL-EBRB boost has not been fitted")
        proba = self.predict_proba(X)
        if self.decision_threshold_class_ is not None and self.decision_threshold_ is not None and len(self.classes_) == 2:
            target = int(self.decision_threshold_class_)
            other = 1 - target
            pred_idx = np.full(proba.shape[0], other, dtype=int)
            pred_idx[proba[:, target] >= float(self.decision_threshold_)] = target
            return self.classes_[pred_idx]
        return self.classes_[np.argmax(proba, axis=1)]

    def diagnostics(self) -> Dict[str, float]:
        if not self.estimators_:
            raise ValueError("Strict VFL-EBRB boost has not been fitted")
        weights = np.asarray(self.estimator_weights_, dtype=float)
        errors = np.asarray(self.estimator_errors_, dtype=float)
        normalized = self._normalized_estimator_weights()
        estimator_reports = [estimator.diagnostics() for estimator in self.estimators_]
        rule_counts = [report["ebrb_rule_count"] for report in estimator_reports]
        update_shifts = [
            report.get("ebrb_parameter_update_shift_mean", 0.0)
            for report in estimator_reports
        ]
        report = {
            "ebrb_boost_estimators": float(len(self.estimators_)),
            "ebrb_boost_rule_count": float(np.sum(rule_counts)),
            "ebrb_boost_weight_mean": float(np.mean(weights)),
            "ebrb_boost_weight_max": float(np.max(weights)),
            "ebrb_boost_weight_entropy": float(
                -np.sum(normalized * np.log(normalized + EPS)) / max(np.log(len(normalized)), EPS)
            ),
            "ebrb_boost_error_mean": float(np.mean(errors)),
            "ebrb_boost_error_min": float(np.min(errors)),
            "ebrb_boost_error_max": float(np.max(errors)),
            "ebrb_boost_decision_threshold_enabled": float(self.decision_threshold_ is not None),
            "ebrb_boost_parameter_iterations": float(self.parameter_iterations),
            "ebrb_boost_parameter_update_blend": float(self.parameter_update_blend),
            "ebrb_boost_parameter_update_shift_mean": float(np.mean(update_shifts)),
            "ebrb_boost_server_logit_la_gamma": float(self.server_logit_la_gamma),
            "ebrb_boost_server_logit_la_temperature": float(self.server_logit_la_temperature),
            "ebrb_boost_server_logit_la_blend": float(self.server_logit_la_blend),
            "ebrb_boost_server_class_conditioned_fusion": float(self.server_class_conditioned_fusion),
            "ebrb_boost_server_class_weight_power": float(self.server_class_weight_power),
            "ebrb_boost_ensemble_residual_update": float(self.weight_update_method == "ensemble_residual"),
            "ebrb_boost_client_fit_weight_power": float(self.client_fit_weight_power),
        }
        if self.estimator_mean_beliefs_:
            mean_beliefs = np.stack(self.estimator_mean_beliefs_, axis=0)
            report["ebrb_boost_estimator_mean_belief_min"] = float(np.min(mean_beliefs))
            report["ebrb_boost_estimator_mean_belief_max"] = float(np.max(mean_beliefs))
        if self.decision_threshold_ is not None and self.decision_threshold_class_ is not None:
            report["ebrb_boost_decision_threshold"] = float(self.decision_threshold_)
            report["ebrb_boost_decision_threshold_class"] = float(self.decision_threshold_class_)
        return report

    @staticmethod
    def _estimator_class_scores(
        proba: np.ndarray,
        y_idx: np.ndarray,
        sample_weights: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        global_score = float(np.average(np.max(proba, axis=1), weights=sample_weights))
        scores = np.full(n_classes, global_score, dtype=float)
        for class_idx in range(n_classes):
            mask = y_idx == class_idx
            if not np.any(mask):
                continue
            local_weights = sample_weights[mask]
            local_weights = local_weights / max(float(np.sum(local_weights)), EPS)
            class_confidence = float(np.sum(local_weights * proba[mask, class_idx]))
            class_recall = float(np.sum(local_weights * (np.argmax(proba[mask], axis=1) == class_idx)))
            scores[class_idx] = 0.65 * class_confidence + 0.35 * class_recall
        return np.clip(scores, EPS, 1.0)

    def _apply_server_logit_la(self, proba: np.ndarray, mean_belief: np.ndarray) -> np.ndarray:
        gamma = float(self.server_logit_la_gamma)
        blend = float(np.clip(self.server_logit_la_blend, 0.0, 1.0))
        if gamma <= 0.0 or blend <= 0.0:
            return proba
        logits = (
            np.log(proba + EPS)
            - gamma * np.log(np.asarray(mean_belief, dtype=float)[None, :] + EPS)
        ) / max(float(self.server_logit_la_temperature), EPS)
        adjusted = softmax(logits)
        return normalize_rows((1.0 - blend) * proba + blend * adjusted)

    def _align_model_proba(self, model: _StrictVFLEBRB, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or model.classes_ is None:
            raise ValueError("Strict VFL-EBRB boost has not been fitted")
        proba = model.predict_proba(X)
        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        for source_idx, label in enumerate(model.classes_):
            if label in class_to_idx:
                aligned[:, class_to_idx[label]] = proba[:, source_idx]
        return normalize_rows(aligned)

    def _binary_delta_logit(self, proba: np.ndarray) -> np.ndarray:
        if self.classes_ is None or len(self.classes_) != 2:
            raise ValueError("Binary logit deltas require exactly two classes")
        p_one = np.clip(np.asarray(proba, dtype=float)[:, 1], EPS, 1.0 - EPS)
        delta = np.log(p_one / (1.0 - p_one)) - float(self.base_logit_)
        return np.clip(delta, -4.0, 4.0)

    @staticmethod
    def _line_search_step(
        current_logit: np.ndarray,
        delta: np.ndarray,
        y_binary: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        if float(np.max(np.abs(delta))) <= EPS:
            return 0.0
        gamma = 1.0
        for _ in range(6):
            logits = np.clip(current_logit + gamma * delta, -30.0, 30.0)
            proba = 1.0 / (1.0 + np.exp(-logits))
            grad = float(np.sum(weights * (proba - y_binary) * delta))
            hess = float(np.sum(weights * proba * (1.0 - proba) * (delta**2))) + EPS
            gamma -= grad / hess
            gamma = float(np.clip(gamma, 0.0, 2.0))
        return gamma

    def _initial_sample_weights(self, y_idx: np.ndarray, n_classes: int) -> np.ndarray:
        weights = np.full(y_idx.shape[0], 1.0 / max(y_idx.shape[0], 1), dtype=float)
        balance_power = float(np.clip(self.sampling_class_balance_power, 0.0, 1.0))
        if balance_power <= 0:
            return weights
        counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        class_weights = np.power(np.max(counts) / np.maximum(counts, EPS), balance_power)
        weights = weights * class_weights[y_idx]
        return weights / max(float(np.sum(weights)), EPS)

    def _client_fit_weights(self, sample_weights: np.ndarray, sample_idx: np.ndarray) -> np.ndarray | None:
        power = float(np.clip(self.client_fit_weight_power, 0.0, 2.0))
        if power <= 0.0:
            return None
        selected = np.power(np.maximum(sample_weights[np.asarray(sample_idx, dtype=int)], EPS), power)
        return selected / max(float(np.sum(selected)), EPS)

    def _sample_rows(self, y_idx: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fraction = float(np.clip(self.sample_fraction, EPS, 1.0))
        n_draw = max(len(np.unique(y_idx)), int(round(len(y_idx) * fraction)))
        weights = weights / max(float(np.sum(weights)), EPS)
        sampled = rng.choice(np.arange(len(y_idx)), size=n_draw, replace=True, p=weights)
        sampled_labels = set(y_idx[sampled].tolist())
        missing = [label for label in np.unique(y_idx) if label not in sampled_labels]
        for offset, label in enumerate(missing):
            label_idx = np.flatnonzero(y_idx == label)
            label_weights = weights[label_idx] / max(float(np.sum(weights[label_idx])), EPS)
            sampled[offset % len(sampled)] = rng.choice(label_idx, size=1, replace=True, p=label_weights)[0]
        rng.shuffle(sampled)
        return sampled

    def _updated_sample_weights(
        self,
        current: np.ndarray,
        incorrect: np.ndarray,
        proba: np.ndarray,
        y_idx: np.ndarray,
        estimator_weight: float,
    ) -> np.ndarray:
        if estimator_weight <= EPS:
            return current
        if self.weight_update_method == "none":
            return current
        if self.weight_update_method == "hard":
            return current * np.exp(estimator_weight * incorrect)
        if self.weight_update_method == "residual":
            true_proba = np.clip(proba[np.arange(proba.shape[0]), y_idx], EPS, 1.0)
            residual = 1.0 - true_proba
            scaled = residual / max(float(np.average(residual, weights=current)), EPS)
            return current * np.exp(float(self.learning_rate) * scaled)
        if self.weight_update_method == "ensemble_residual":
            true_proba = np.clip(proba[np.arange(proba.shape[0]), y_idx], EPS, 1.0)
            residual = 1.0 - true_proba
            scaled = residual / max(float(np.average(residual, weights=current)), EPS)
            return current * np.exp(float(self.learning_rate) * scaled)
        raise ValueError(f"Unknown EBRB boost weight update method: {self.weight_update_method}")

    def _estimator_weight(self, weighted_error: float, n_classes: int) -> float:
        error = float(np.clip(weighted_error, EPS, 1.0 - EPS))
        random_guess_error = 1.0 - 1.0 / max(float(n_classes), 2.0)
        if error >= random_guess_error:
            return 0.0
        class_term = 0.0 if n_classes <= 2 else np.log(float(n_classes - 1))
        return float(self.learning_rate * max(0.0, np.log((1.0 - error) / error) + class_term))

    def _normalized_estimator_weights(self) -> np.ndarray:
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if weights.size == 0:
            raise ValueError("Strict VFL-EBRB boost has not been fitted")
        if float(np.sum(weights)) <= EPS:
            return np.full(weights.shape[0], 1.0 / weights.shape[0], dtype=float)
        return weights / np.sum(weights)

    def _fusion_weights(self) -> np.ndarray:
        base_weights = self._normalized_estimator_weights()
        if (
            not self.server_class_conditioned_fusion
            or not self.estimator_class_scores_
            or float(self.server_class_weight_power) <= 0.0
        ):
            return np.repeat(base_weights[:, None], len(self.classes_) if self.classes_ is not None else 1, axis=1)
        class_scores = np.stack(self.estimator_class_scores_, axis=0)
        adjusted = np.power(np.maximum(class_scores, EPS), float(self.server_class_weight_power))
        raw = base_weights[:, None] * adjusted
        return raw / np.maximum(np.sum(raw, axis=0, keepdims=True), EPS)

    def _fit_decision_threshold(self, X: np.ndarray, y_idx: np.ndarray) -> None:
        if self.decision_threshold_method == "argmax":
            return
        if self.decision_threshold_method != "train_prior":
            raise ValueError(f"Unknown EBRB boost decision threshold method: {self.decision_threshold_method}")
        if self.classes_ is None or len(self.classes_) != 2:
            return
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        minority_idx = int(np.argmin(counts))
        minority_prior = float(counts[minority_idx] / max(float(np.sum(counts)), EPS))
        target_rate = float(
            np.clip(
                minority_prior * self.decision_threshold_prior_multiplier,
                EPS,
                1.0 - EPS,
            )
        )
        proba = self.predict_proba(X)
        self.decision_threshold_class_ = minority_idx
        self.decision_threshold_ = float(np.quantile(proba[:, minority_idx], 1.0 - target_rate))


def strict_vfl_ebrb_boost_runner(
    n_estimators: int,
    learning_rate: float,
    sample_fraction: float,
    n_rules_per_party: int,
    bins: int,
    max_antecedents: int,
    beam_width: int,
    beta_smoothing: float,
    min_support_fraction: float,
    class_balance_power: float,
    sampling_class_balance_power: float,
    client_weight_method: str,
    fusion_method: str,
    weight_update_method: str,
    decision_threshold_method: str,
    decision_threshold_prior_multiplier: float,
    server_logit_la_gamma: float = 0.0,
    server_logit_la_temperature: float = 1.0,
    server_logit_la_blend: float = 0.0,
    server_class_conditioned_fusion: bool = False,
    server_class_weight_power: float = 1.0,
    client_fit_weight_power: float = 0.0,
    parameter_iterations: int = 0,
    parameter_update_blend: float = 0.5,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        model = _StrictVFLBoostedEBRB(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            sample_fraction=sample_fraction,
            n_rules_per_party=n_rules_per_party,
            bins=bins,
            max_antecedents=max_antecedents,
            beam_width=beam_width,
            beta_smoothing=beta_smoothing,
            min_support_fraction=min_support_fraction,
            class_balance_power=class_balance_power,
            parameter_iterations=parameter_iterations,
            parameter_update_blend=parameter_update_blend,
            sampling_class_balance_power=sampling_class_balance_power,
            client_weight_method=client_weight_method,
            fusion_method=fusion_method,
            weight_update_method=weight_update_method,
            decision_threshold_method=decision_threshold_method,
            decision_threshold_prior_multiplier=decision_threshold_prior_multiplier,
            server_logit_la_gamma=server_logit_la_gamma,
            server_logit_la_temperature=server_logit_la_temperature,
            server_logit_la_blend=server_logit_la_blend,
            server_class_conditioned_fusion=server_class_conditioned_fusion,
            server_class_weight_power=server_class_weight_power,
            client_fit_weight_power=client_fit_weight_power,
            random_state=seed,
        )
        model.fit(X_train, y_train, party_features=party_features, active_party_id=active_party_id)
        return model.predict(X_test), model.diagnostics()

    return run


def vfl_xgboost_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del party_features, active_party_id
    from xgboost import XGBClassifier  # type: ignore

    classes, y_idx = np.unique(y_train, return_inverse=True)
    params: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": seed,
        "verbosity": 0,
        "objective": "binary:logistic" if len(classes) == 2 else "multi:softprob",
    }
    if len(classes) > 2:
        params["num_class"] = len(classes)
    model = XGBClassifier(**params)
    model.fit(X_train, y_idx)
    return classes[model.predict(X_test).astype(int)], {}


def _fedtree_abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _fedtree_dyld_path(fedtree_root: str) -> str:
    candidates = [
        os.path.join(fedtree_root, "build", "lib"),
        os.path.join(ROOT, "external", "local_ntl", "lib"),
        os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Python",
            "3.9",
            "lib",
            "python",
            "site-packages",
            "sklearn",
            ".dylibs",
        ),
    ]
    return ":".join(path for path in candidates if os.path.isdir(path))


def _write_fedtree_csv(path: str, X: np.ndarray, y: np.ndarray | None) -> None:
    X_num = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    headers = ["id"]
    if y is not None:
        headers.append("y")
    headers.extend(f"x{idx}" for idx in range(X_num.shape[1]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row_idx, row in enumerate(X_num):
            values = [str(row_idx)]
            if y is not None:
                values.append(str(int(y[row_idx])))
            values.extend(f"{float(value):.10g}" for value in row)
            handle.write(",".join(values) + "\n")


def fedtree_vertical_runner(
    fedtree_root: str,
    n_trees: int,
    depth: int,
    learning_rate: float,
    timeout_sec: int,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    fedtree_root_abs = _fedtree_abs_path(fedtree_root)
    train_bin = os.path.join(fedtree_root_abs, "build", "bin", "FedTree-train")

    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if active_party_id != 0:
            raise ValueError("FedTree vertical simulation expects party 0 to be the label/active party.")
        if not os.path.exists(train_bin):
            raise FileNotFoundError(f"FedTree-train not found: {train_bin}")
        classes, y_train_idx = np.unique(y_train, return_inverse=True)
        n_classes = len(classes)

        party_ids = sorted(int(pid) for pid in party_features)
        ordered_features: List[int] = []
        for party_id in party_ids:
            ordered_features.extend(int(feature) for feature in party_features[party_id])

        # Keep the directory name short: FedTree's configuration parser uses a
        # fixed line buffer, and a 10-party comma-separated data path can be
        # truncated when verbose temporary paths are used.
        run_dir = os.path.join(
            "/tmp",
            f"f{os.getpid() % 4096:03x}{seed % 256:02x}{time.time_ns() % 256:02x}",
        )
        os.makedirs(run_dir, exist_ok=True)
        train_paths: List[str] = []
        for party_id in party_ids:
            features = list(party_features[party_id])
            party_path = os.path.join(run_dir, f"p{party_id}.csv")
            labels = y_train_idx if party_id == active_party_id else None
            _write_fedtree_csv(party_path, X_train[:, features], labels)
            train_paths.append(party_path)

        test_path = os.path.join(run_dir, "t.csv")
        _write_fedtree_csv(test_path, X_test[:, ordered_features], np.zeros(X_test.shape[0], dtype=int))
        model_path = os.path.join(run_dir, "m")
        conf_path = os.path.join(run_dir, "c.conf")
        with open(conf_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        f"data={','.join(train_paths)}",
                        f"test_data={test_path}",
                        f"model_path={model_path}",
                        f"n_parties={len(party_ids)}",
                        "mode=vertical",
                        "partition=0",
                        "data_format=csv",
                        "privacy_tech=none",
                        f"objective={'binary:logistic' if n_classes == 2 else 'multi:softmax'}",
                        f"metric={'error' if n_classes == 2 else 'macc'}",
                        f"num_class={n_classes}",
                        "reorder_label=0",
                        f"n_trees={int(n_trees)}",
                        f"depth={int(depth)}",
                        f"learning_rate={float(learning_rate)}",
                        "verbose=0",
                    ]
                )
                + "\n"
            )

        env = os.environ.copy()
        dyld = _fedtree_dyld_path(fedtree_root_abs)
        if dyld:
            env["DYLD_LIBRARY_PATH"] = f"{dyld}:{env['DYLD_LIBRARY_PATH']}" if env.get("DYLD_LIBRARY_PATH") else dyld
        completed = subprocess.run(
            [train_bin, conf_path],
            cwd=fedtree_root_abs,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )

        pred_path = model_path + ".pred"
        if not os.path.exists(pred_path):
            tail = "\n".join(completed.stdout.splitlines()[-20:])
            raise RuntimeError(f"FedTree did not produce predictions; rc={completed.returncode}\n{tail}")
        raw_pred = np.loadtxt(pred_path, dtype=float)
        raw_pred = np.atleast_1d(raw_pred)
        if raw_pred.shape[0] != X_test.shape[0]:
            raise RuntimeError(f"FedTree prediction length mismatch: {raw_pred.shape[0]} != {X_test.shape[0]}")
        pred_idx = np.clip(np.rint(raw_pred).astype(int), 0, len(classes) - 1)
        diagnostics = {
            "fedtree_return_code": float(completed.returncode),
            "fedtree_n_trees": float(n_trees),
            "fedtree_depth": float(depth),
            "fedtree_learning_rate": float(learning_rate),
        }
        return classes[pred_idx], diagnostics

    return run


def mbrb_payload(base_payload: Dict[str, Any], variant: str, seed: int, strict_vfl: bool = False) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    payload["passive_label_mode"] = "unsupervised" if strict_vfl else "simulation_supervised"
    payload["random_state"] = seed
    payload.setdefault("brb", {})
    payload["brb"]["n_modules_per_party"] = 3
    payload["brb"]["n_rules_per_class"] = 8
    payload["brb"]["n_rules_per_module"] = 16
    payload["brb"]["class_balance_power"] = 1.0
    payload.setdefault("server", {})
    payload["server"]["lambda_d"] = 0.5
    payload["server"]["kappa"] = 0.0
    payload["server"]["eta_c"] = 0.5
    payload["server"]["eta_d"] = 0.25
    payload["server"]["decision_prior_correction"] = 0.5
    payload.setdefault("tuning", {})
    payload["tuning"]["max_trials"] = 2
    payload["tuning"]["grid"] = {
        "gamma_la": [0.15, 0.35],
        "temperature_la": [0.8],
    }
    if variant == "no_la":
        payload["brb"]["gamma_la"] = 0.0
        payload["brb"]["temperature_la"] = 1.0
        payload["tuning"]["max_trials"] = 1
        payload["tuning"]["grid"] = {"gamma_la": [0.0], "temperature_la": [1.0]}
    elif variant == "no_dr":
        payload["server"]["lambda_d"] = 0.0
        payload["server"]["eta_c"] = 0.0
        payload["server"]["eta_d"] = 0.0
        payload["server"]["use_train_difficulty_stats"] = False
    elif variant != "full":
        raise ValueError(f"Unknown MBRB variant: {variant}")
    return payload


def mbrb_candidate_grid() -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    for n_modules in [1, 2, 3]:
        for rules_per_class in [8, 12, 16]:
            for class_balance_power in [0.0, 0.5, 1.0]:
                for decision_prior_correction in [0.0, 0.5, 1.0]:
                    for gamma_la in [0.0, 0.15, 0.35]:
                        candidates.append(
                            {
                                "n_modules_per_party": float(n_modules),
                                "n_rules_per_class": float(rules_per_class),
                                "class_balance_power": float(class_balance_power),
                                "decision_prior_correction": float(decision_prior_correction),
                                "gamma_la": float(gamma_la),
                            }
                        )
    return candidates


def sample_mbrb_candidates(max_trials: int, seed: int) -> List[Dict[str, float]]:
    candidates = mbrb_candidate_grid()
    if max_trials >= len(candidates):
        return candidates
    anchors = [
        {
            "n_modules_per_party": 1.0,
            "n_rules_per_class": 8.0,
            "class_balance_power": 0.0,
            "decision_prior_correction": 0.0,
            "gamma_la": 0.15,
        },
        {
            "n_modules_per_party": 1.0,
            "n_rules_per_class": 12.0,
            "class_balance_power": 0.5,
            "decision_prior_correction": 0.5,
            "gamma_la": 0.15,
        },
        {
            "n_modules_per_party": 2.0,
            "n_rules_per_class": 12.0,
            "class_balance_power": 0.5,
            "decision_prior_correction": 0.5,
            "gamma_la": 0.35,
        },
        {
            "n_modules_per_party": 3.0,
            "n_rules_per_class": 8.0,
            "class_balance_power": 0.0,
            "decision_prior_correction": 1.0,
            "gamma_la": 0.15,
        },
        {
            "n_modules_per_party": 3.0,
            "n_rules_per_class": 16.0,
            "class_balance_power": 1.0,
            "decision_prior_correction": 0.0,
            "gamma_la": 0.35,
        },
    ]
    selected = anchors[: max_trials]
    remaining = [candidate for candidate in candidates if candidate not in selected]
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(len(remaining)).tolist():
        if len(selected) >= max_trials:
            break
        selected.append(remaining[idx])
    return selected


def mbrb_payload_from_params(
    base_payload: Dict[str, Any],
    params: Mapping[str, float],
    seed: int,
    strict_vfl: bool = False,
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    payload["passive_label_mode"] = "unsupervised" if strict_vfl else "simulation_supervised"
    payload["random_state"] = seed
    payload.setdefault("brb", {})
    payload["brb"]["n_modules_per_party"] = int(params["n_modules_per_party"])
    payload["brb"]["n_rules_per_class"] = int(params["n_rules_per_class"])
    payload["brb"]["n_rules_per_module"] = int(params["n_rules_per_class"]) * 2
    payload["brb"]["class_balance_power"] = float(params["class_balance_power"])
    payload["brb"]["gamma_la"] = float(params["gamma_la"])
    payload["brb"]["temperature_la"] = 0.8
    payload.setdefault("server", {})
    payload["server"]["lambda_d"] = 0.5
    payload["server"]["kappa"] = 0.0
    payload["server"]["eta_c"] = 0.5
    payload["server"]["eta_d"] = 0.25
    payload["server"]["decision_prior_correction"] = float(params["decision_prior_correction"])
    payload.setdefault("tuning", {})
    payload["tuning"]["validation_ratio"] = 0.0
    payload["tuning"]["max_trials"] = 1
    payload["tuning"]["grid"] = {}
    return payload


def tuned_dr_mbrb_runner(
    base_payload: Dict[str, Any],
    variant: str,
    strict_vfl: bool = False,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        payload = mbrb_payload(base_payload, variant, seed, strict_vfl=strict_vfl)
        payload["active_party_id"] = active_party_id
        config = config_from_dict(payload)
        model = VFLTunedDRMBRB(config)
        model.fit(X_train, y_train, party_features=party_features)
        pred = model.predict(X_test)
        diag = model.diagnostics(X_test)
        if model.state_ is not None:
            weights = np.asarray(model.state_.server.client_weights_, dtype=float)
            entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / max(float(np.log(len(weights))), 1e-12)
            diag["client_weight_entropy"] = entropy
            diag["client_weight_gap"] = float(np.max(weights) - np.min(weights))
            for idx, weight in enumerate(weights):
                diag[f"client_weight_{idx}"] = float(weight)
        return pred, diag

    return run


def tuned_dr_mbrb_search_runner(
    base_payload: Dict[str, Any],
    max_trials: int,
    strict_vfl: bool = False,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner_train_idx, val_idx = next(inner.split(X_train, stratify_labels(y_train)))
        candidates = sample_mbrb_candidates(max_trials=max_trials, seed=seed)
        best_score = -1.0
        best_params: Dict[str, float] = candidates[0]
        for params in candidates:
            payload = mbrb_payload_from_params(base_payload, params, seed, strict_vfl=strict_vfl)
            payload["active_party_id"] = active_party_id
            model = VFLTunedDRMBRB(config_from_dict(payload))
            model.fit(X_train[inner_train_idx], y_train[inner_train_idx], party_features=party_features)
            pred = model.predict(X_train[val_idx])
            score = metric_report(y_train[val_idx], pred)["macro_f1"]
            if score > best_score:
                best_score = score
                best_params = dict(params)

        payload = mbrb_payload_from_params(base_payload, best_params, seed, strict_vfl=strict_vfl)
        payload["active_party_id"] = active_party_id
        final = VFLTunedDRMBRB(config_from_dict(payload))
        final.fit(X_train, y_train, party_features=party_features)
        pred = final.predict(X_test)
        diag = final.diagnostics(X_test)
        diag["validation_macro_f1"] = float(best_score)
        diag["search_trials"] = float(len(candidates))
        for key, value in best_params.items():
            diag[f"selected_{key}"] = float(value)
        if final.state_ is not None:
            weights = np.asarray(final.state_.server.client_weights_, dtype=float)
            entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / max(float(np.log(len(weights))), 1e-12)
            diag["client_weight_entropy"] = entropy
            diag["client_weight_gap"] = float(np.max(weights) - np.min(weights))
            for idx, weight in enumerate(weights):
                diag[f"client_weight_{idx}"] = float(weight)
        return pred, diag

    return run


def accuracy_tuned_dr_mbrb_runner(
    base_payload: Dict[str, Any],
    n_modules: int,
    rules_per_class: int,
    class_balance_power: float,
    prototype_method: str,
    rule_scale_method: str,
    rule_scale_neighbors: int,
    rule_scale_multiplier: float,
    reasoning_steps_grid: Sequence[int],
    client_weight_method: str,
    use_stacking: bool,
    stacking_c: float,
    stacking_class_weight: str,
    validation_ratio: float,
    fixed_final_la_gamma: float,
    fixed_final_la_temperature: float,
    fixed_final_la_blend: float,
    prior_gated_final_la: bool,
    min_class_prior_for_la: float,
    strict_vfl: bool = False,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        payload = json.loads(json.dumps(base_payload))
        payload["passive_label_mode"] = "unsupervised" if strict_vfl else "simulation_supervised"
        payload["active_party_id"] = active_party_id
        payload["random_state"] = seed
        payload.setdefault("brb", {})
        payload["brb"]["n_modules_per_party"] = int(n_modules)
        payload["brb"]["n_rules_per_class"] = int(rules_per_class)
        payload["brb"]["n_rules_per_module"] = int(rules_per_class) * 2
        payload["brb"]["class_balance_power"] = float(class_balance_power)
        prototype_grid = ["quantile", "centroid"] if prototype_method == "auto" else [prototype_method]
        payload["brb"]["prototype_method"] = prototype_grid[0]
        payload["brb"]["rule_scale_method"] = rule_scale_method
        payload["brb"]["rule_scale_neighbors"] = int(rule_scale_neighbors)
        payload["brb"]["rule_scale_multiplier"] = float(rule_scale_multiplier)
        payload["brb"]["gamma_la"] = 0.0
        payload["brb"]["temperature_la"] = 0.8
        payload.setdefault("server", {})
        payload["server"]["client_fusion_method"] = "er"
        payload["server"]["client_weight_method"] = client_weight_method
        payload["server"]["lambda_d"] = 0.5
        payload["server"]["kappa"] = 0.0
        payload["server"]["eta_c"] = 0.5
        payload["server"]["eta_d"] = 0.25
        payload["server"]["n_reasoning_steps"] = int(reasoning_steps_grid[0])
        payload["server"]["decision_prior_correction"] = 0.0
        use_validation = float(validation_ratio) > 0.0
        payload["final_la"] = {
            "enabled": True,
            "gamma": 0.0 if use_validation else float(fixed_final_la_gamma),
            "temperature": 0.8 if use_validation else float(fixed_final_la_temperature),
            "blend": 1.0 if use_validation else float(fixed_final_la_blend),
            "apply_on_difficult_only": True,
            "prior_gate_enabled": bool(prior_gated_final_la),
            "min_class_prior_for_la": float(min_class_prior_for_la),
            "tune_on_validation": use_validation,
            "grid": {
                "gamma": [0.8, 1.0, 1.2],
                "temperature": [0.8, 1.0],
                "blend": [0.25, 0.5, 0.75, 1.0],
            } if use_validation else {},
        }
        payload["stacking"] = {
            "enabled": bool(use_stacking and use_validation),
            "C": float(stacking_c),
            "class_weight": None if stacking_class_weight == "none" else stacking_class_weight,
            "max_iter": 1000,
        }
        payload.setdefault("tuning", {})
        payload["tuning"]["validation_ratio"] = float(validation_ratio)
        payload["tuning"]["primary_metric"] = "accuracy"
        if use_validation:
            payload["tuning"]["max_trials"] = 72 * max(1, len(reasoning_steps_grid)) * max(1, len(prototype_grid))
            payload["tuning"]["grid"] = {
                "kappa": [0.0, 0.5, 1.0],
                "n_reasoning_steps": [int(value) for value in reasoning_steps_grid],
            }
            if len(prototype_grid) > 1:
                payload["tuning"]["grid"]["prototype_method"] = prototype_grid
        else:
            payload["tuning"]["max_trials"] = 1
            payload["tuning"]["grid"] = {}
        model = VFLTunedDRMBRB(config_from_dict(payload))
        model.fit(X_train, y_train, party_features=party_features)
        pred = model.predict(X_test)
        diag = model.diagnostics(X_test)
        for key, value in model.best_params_.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                diag[f"selected_{key}"] = float(value)
            else:
                diag[f"selected_{key}_{value}"] = 1.0
        if model.state_ is not None:
            weights = np.asarray(model.state_.server.client_weights_, dtype=float)
            entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / max(float(np.log(len(weights))), 1e-12)
            diag["client_weight_entropy"] = entropy
            diag["client_weight_gap"] = float(np.max(weights) - np.min(weights))
            for idx, weight in enumerate(weights):
                diag[f"client_weight_{idx}"] = float(weight)
        return pred, diag

    return run


def fixed_fair_mbrb_payload(
    base_payload: Dict[str, Any],
    active_party_id: int,
    seed: int,
    final_la_gamma: float,
    final_la_temperature: float,
    final_la_blend: float,
    prior_gated_final_la: bool,
    min_class_prior_for_la: float,
    strict_vfl: bool = False,
    n_modules_per_party: int = 1,
    n_rules_per_class: int = 12,
    n_rules_per_module: int = 24,
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    payload["passive_label_mode"] = "unsupervised" if strict_vfl else "simulation_supervised"
    payload["active_party_id"] = active_party_id
    payload["random_state"] = seed
    payload.setdefault("brb", {})
    payload["brb"]["n_modules_per_party"] = int(n_modules_per_party)
    payload["brb"]["n_rules_per_class"] = int(n_rules_per_class)
    payload["brb"]["n_rules_per_module"] = int(n_rules_per_module)
    payload["brb"]["class_balance_power"] = 0.0
    payload["brb"]["prototype_method"] = "quantile"
    payload["brb"]["rule_scale_method"] = "global"
    payload["brb"]["gamma_la"] = 0.0
    payload["brb"]["temperature_la"] = 0.8
    payload.setdefault("server", {})
    payload["server"]["client_fusion_method"] = "er"
    payload["server"]["client_weight_method"] = "confidence"
    payload["server"]["lambda_d"] = 0.5
    payload["server"]["kappa"] = 0.0
    payload["server"]["eta_c"] = 0.5
    payload["server"]["eta_d"] = 0.25
    payload["server"]["n_reasoning_steps"] = 1
    payload["server"]["decision_prior_correction"] = 0.0
    payload["final_la"] = {
        "enabled": True,
        "gamma": float(final_la_gamma),
        "temperature": float(final_la_temperature),
        "blend": float(final_la_blend),
        "apply_on_difficult_only": True,
        "prior_gate_enabled": bool(prior_gated_final_la),
        "min_class_prior_for_la": float(min_class_prior_for_la),
        "tune_on_validation": False,
        "grid": {},
    }
    payload["stacking"] = {
        "enabled": False,
        "C": 1.0,
        "class_weight": None,
        "max_iter": 1000,
    }
    payload["tuning"] = {
        "validation_ratio": 0.0,
        "max_trials": 1,
        "primary_metric": "accuracy",
        "grid": {},
    }
    return payload


def forest_dr_mbrb_runner(
    base_payload: Dict[str, Any],
    n_estimators: int,
    sample_fraction: float,
    feature_fraction: float,
    fusion_method: str,
    final_la_gamma: float,
    final_la_temperature: float,
    final_la_blend: float,
    prior_gated_final_la: bool,
    min_class_prior_for_la: float,
    strict_vfl: bool = False,
    n_modules_per_party: int = 1,
    n_rules_per_class: int = 12,
    n_rules_per_module: int = 24,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        payload = fixed_fair_mbrb_payload(
            base_payload,
            active_party_id=active_party_id,
            seed=seed,
            final_la_gamma=final_la_gamma,
            final_la_temperature=final_la_temperature,
            final_la_blend=final_la_blend,
            prior_gated_final_la=prior_gated_final_la,
            min_class_prior_for_la=min_class_prior_for_la,
            strict_vfl=strict_vfl,
            n_modules_per_party=n_modules_per_party,
            n_rules_per_class=n_rules_per_class,
            n_rules_per_module=n_rules_per_module,
        )
        base_config = config_from_dict(payload)
        ensemble = VFLBaggedDRMBRB(
            base_config=base_config,
            ensemble_config=BaggedMBRBConfig(
                n_estimators=n_estimators,
                sample_fraction=sample_fraction,
                feature_fraction=feature_fraction,
                fusion_method=fusion_method,
                random_state=seed,
            ),
        )
        ensemble.fit(X_train, y_train, party_features=party_features)
        pred = ensemble.predict(X_test)
        return pred, ensemble.diagnostics()

    return run


def boosted_dr_mbrb_runner(
    base_payload: Dict[str, Any],
    n_estimators: int,
    sample_fraction: float,
    learning_rate: float,
    fusion_method: str,
    weight_update_method: str,
    sampling_class_balance_power: float,
    class_conditioned_fusion: bool,
    decision_threshold_method: str,
    decision_threshold_prior_multiplier: float,
    final_la_gamma: float,
    final_la_temperature: float,
    final_la_blend: float,
    prior_gated_final_la: bool,
    min_class_prior_for_la: float,
    strict_vfl: bool = False,
    n_modules_per_party: int = 1,
    n_rules_per_class: int = 12,
    n_rules_per_module: int = 24,
) -> Callable[..., Tuple[np.ndarray, Dict[str, float]]]:
    def run(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
        seed: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        payload = fixed_fair_mbrb_payload(
            base_payload,
            active_party_id=active_party_id,
            seed=seed,
            final_la_gamma=final_la_gamma,
            final_la_temperature=final_la_temperature,
            final_la_blend=final_la_blend,
            prior_gated_final_la=prior_gated_final_la,
            min_class_prior_for_la=min_class_prior_for_la,
            strict_vfl=strict_vfl,
            n_modules_per_party=n_modules_per_party,
            n_rules_per_class=n_rules_per_class,
            n_rules_per_module=n_rules_per_module,
        )
        base_config = config_from_dict(payload)
        ensemble = VFLBoostedDRMBRB(
            base_config=base_config,
            ensemble_config=BoostedMBRBConfig(
                n_estimators=n_estimators,
                sample_fraction=sample_fraction,
                learning_rate=learning_rate,
                fusion_method=fusion_method,
                weight_update_method=weight_update_method,
                sampling_class_balance_power=sampling_class_balance_power,
                class_conditioned_fusion=class_conditioned_fusion,
                decision_threshold_method=decision_threshold_method,
                decision_threshold_prior_multiplier=decision_threshold_prior_multiplier,
                random_state=seed,
            ),
        )
        ensemble.fit(X_train, y_train, party_features=party_features)
        pred = ensemble.predict(X_test)
        return pred, ensemble.diagnostics()

    return run


def evaluate_method(
    runner: Callable[..., Tuple[np.ndarray, Dict[str, float]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Dict[str, float]:
    start = time.perf_counter()
    pred, extra = runner(X_train, y_train, X_test, party_features, active_party_id, seed)
    wall = time.perf_counter() - start
    row = metric_report(y_test, pred)
    row["wall_time_sec"] = float(wall)
    for key, value in extra.items():
        row[key] = float(value)
    return row


def evaluate_dataset(
    dataset: DatasetSpec,
    base_payload: Dict[str, Any],
    splits: int,
    max_samples: int,
    heterogeneity_modes: Sequence[str],
    strict_vfl: bool,
    include_xgboost: bool,
    include_strict_xgboost: bool,
    include_fedtree: bool,
    include_ablations: bool,
    mbrb_search_trials: int,
    include_accuracy_tuned: bool,
    include_mbrb_forest: bool,
    include_mbrb_boost: bool,
    include_ebrb_boost: bool,
    accuracy_tuned_n_modules: int,
    accuracy_tuned_rules_per_class: int,
    accuracy_tuned_class_balance_power: float,
    accuracy_tuned_prototype_method: str,
    accuracy_tuned_rule_scale_method: str,
    accuracy_tuned_rule_scale_neighbors: int,
    accuracy_tuned_rule_scale_multiplier: float,
    accuracy_tuned_reasoning_steps_grid: Sequence[int],
    accuracy_tuned_client_weight_method: str,
    accuracy_tuned_use_stacking: bool,
    accuracy_tuned_stacking_c: float,
    accuracy_tuned_stacking_class_weight: str,
    accuracy_tuned_validation_ratio: float,
    accuracy_tuned_fixed_final_la_gamma: float,
    accuracy_tuned_fixed_final_la_temperature: float,
    accuracy_tuned_fixed_final_la_blend: float,
    accuracy_tuned_prior_gated_final_la: bool,
    accuracy_tuned_min_class_prior_for_la: float,
    mbrb_forest_estimators: int,
    mbrb_forest_sample_fraction: float,
    mbrb_forest_feature_fraction: float,
    mbrb_forest_fusion_method: str,
    mbrb_boost_estimators: int,
    mbrb_boost_sample_fraction: float,
    mbrb_boost_learning_rate: float,
    mbrb_boost_fusion_method: str,
    mbrb_boost_weight_update_method: str,
    mbrb_boost_sampling_class_balance_power: float,
    mbrb_boost_class_conditioned_fusion: bool,
    mbrb_boost_decision_threshold_method: str,
    mbrb_boost_decision_threshold_prior_multiplier: float,
    ebrb_boost_estimators: int,
    ebrb_boost_sample_fraction: float,
    ebrb_boost_learning_rate: float,
    ebrb_boost_rules_per_party: int,
    ebrb_boost_bins: int,
    ebrb_boost_max_antecedents: int,
    ebrb_boost_beam_width: int,
    ebrb_boost_beta_smoothing: float,
    ebrb_boost_min_support_fraction: float,
    ebrb_boost_class_balance_power: float,
    ebrb_boost_parameter_iterations: int,
    ebrb_boost_parameter_update_blend: float,
    ebrb_boost_sampling_class_balance_power: float,
    ebrb_boost_client_weight_method: str,
    ebrb_boost_fusion_method: str,
    ebrb_boost_weight_update_method: str,
    ebrb_boost_client_fit_weight_power: float,
    ebrb_boost_decision_threshold_method: str,
    ebrb_boost_decision_threshold_prior_multiplier: float,
    strict_xgboost_n_estimators: int,
    strict_xgboost_depth: int,
    strict_xgboost_learning_rate: float,
    strict_xgboost_max_bins: int,
    strict_xgboost_reg_lambda: float,
    strict_xgboost_gamma: float,
    strict_xgboost_min_child_weight: float,
    strict_xgboost_min_samples_leaf: int,
    fedtree_root: str,
    fedtree_n_trees: int,
    fedtree_depth: int,
    fedtree_learning_rate: float,
    fedtree_timeout_sec: int,
) -> Dict[str, Any]:
    config = config_from_dict(base_payload)
    X_raw, y_raw = load_dataset(dataset)
    y_raw = np.asarray(y_raw, dtype=object).astype(str)
    X_raw, y_raw = stratified_limit(X_raw, y_raw, max_samples=max_samples, seed=config.random_state)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.random_state)

    if strict_vfl:
        methods: Dict[str, Callable[..., Tuple[np.ndarray, Dict[str, float]]]] = {
            "Strict VFL-LR": strict_vfl_lr,
            "Strict VFL-GaussianNB": strict_vfl_gaussian_nb,
        }
    else:
        methods = {
            "Active-only LR": active_only_lr,
            "VFL-SplitLR": vfl_split_lr,
            "VFL-GaussianNB": vfl_gaussian_nb,
        }
    if include_fedtree:
        methods["FedTree Vertical GBDT"] = fedtree_vertical_runner(
            fedtree_root=fedtree_root,
            n_trees=fedtree_n_trees,
            depth=fedtree_depth,
            learning_rate=fedtree_learning_rate,
            timeout_sec=fedtree_timeout_sec,
        )
    if include_strict_xgboost:
        methods["Strict VFL-XGBoost Hist"] = strict_vfl_xgboost_hist_runner(
            n_estimators=strict_xgboost_n_estimators,
            max_depth=strict_xgboost_depth,
            learning_rate=strict_xgboost_learning_rate,
            max_bins=strict_xgboost_max_bins,
            reg_lambda=strict_xgboost_reg_lambda,
            gamma=strict_xgboost_gamma,
            min_child_weight=strict_xgboost_min_child_weight,
            min_samples_leaf=strict_xgboost_min_samples_leaf,
        )
    if mbrb_search_trials > 0:
        methods["Strict VFL-Tuned DR-MBRB best" if strict_vfl else "VFL-Tuned DR-MBRB best"] = tuned_dr_mbrb_search_runner(
            base_payload,
            mbrb_search_trials,
            strict_vfl=strict_vfl,
        )
    else:
        methods["Strict VFL-Tuned DR-MBRB full" if strict_vfl else "VFL-Tuned DR-MBRB full"] = tuned_dr_mbrb_runner(
            base_payload,
            "full",
            strict_vfl=strict_vfl,
        )
    if include_accuracy_tuned:
        if float(accuracy_tuned_validation_ratio) <= 0.0:
            accuracy_method_name = "Strict VFL-Tuned DR-MBRB fixed" if strict_vfl else "VFL-Tuned DR-MBRB fixed"
        else:
            accuracy_method_name = (
                "Strict VFL-Tuned DR-MBRB accuracy-tuned"
                if strict_vfl
                else "VFL-Tuned DR-MBRB accuracy-tuned"
            )
        methods[accuracy_method_name] = accuracy_tuned_dr_mbrb_runner(
            base_payload,
            n_modules=accuracy_tuned_n_modules,
            rules_per_class=accuracy_tuned_rules_per_class,
            class_balance_power=accuracy_tuned_class_balance_power,
            prototype_method=accuracy_tuned_prototype_method,
            rule_scale_method=accuracy_tuned_rule_scale_method,
            rule_scale_neighbors=accuracy_tuned_rule_scale_neighbors,
            rule_scale_multiplier=accuracy_tuned_rule_scale_multiplier,
            reasoning_steps_grid=accuracy_tuned_reasoning_steps_grid,
            client_weight_method=accuracy_tuned_client_weight_method,
            use_stacking=accuracy_tuned_use_stacking,
            stacking_c=accuracy_tuned_stacking_c,
            stacking_class_weight=accuracy_tuned_stacking_class_weight,
            validation_ratio=accuracy_tuned_validation_ratio,
            fixed_final_la_gamma=accuracy_tuned_fixed_final_la_gamma,
            fixed_final_la_temperature=accuracy_tuned_fixed_final_la_temperature,
            fixed_final_la_blend=accuracy_tuned_fixed_final_la_blend,
            prior_gated_final_la=accuracy_tuned_prior_gated_final_la,
            min_class_prior_for_la=accuracy_tuned_min_class_prior_for_la,
            strict_vfl=strict_vfl,
        )
    if include_mbrb_forest:
        methods["Strict VFL-Tuned DR-MBRB forest" if strict_vfl else "VFL-Tuned DR-MBRB forest"] = forest_dr_mbrb_runner(
            base_payload,
            n_estimators=mbrb_forest_estimators,
            sample_fraction=mbrb_forest_sample_fraction,
            feature_fraction=mbrb_forest_feature_fraction,
            fusion_method=mbrb_forest_fusion_method,
            final_la_gamma=accuracy_tuned_fixed_final_la_gamma,
            final_la_temperature=accuracy_tuned_fixed_final_la_temperature,
            final_la_blend=accuracy_tuned_fixed_final_la_blend,
            prior_gated_final_la=accuracy_tuned_prior_gated_final_la,
            min_class_prior_for_la=accuracy_tuned_min_class_prior_for_la,
            strict_vfl=strict_vfl,
        )
    if include_mbrb_boost:
        methods["Strict VFL-Tuned DR-MBRB boost" if strict_vfl else "VFL-Tuned DR-MBRB boost"] = boosted_dr_mbrb_runner(
            base_payload,
            n_estimators=mbrb_boost_estimators,
            sample_fraction=mbrb_boost_sample_fraction,
            learning_rate=mbrb_boost_learning_rate,
            fusion_method=mbrb_boost_fusion_method,
            weight_update_method=mbrb_boost_weight_update_method,
            sampling_class_balance_power=mbrb_boost_sampling_class_balance_power,
            class_conditioned_fusion=mbrb_boost_class_conditioned_fusion,
            decision_threshold_method=mbrb_boost_decision_threshold_method,
            decision_threshold_prior_multiplier=mbrb_boost_decision_threshold_prior_multiplier,
            final_la_gamma=accuracy_tuned_fixed_final_la_gamma,
            final_la_temperature=accuracy_tuned_fixed_final_la_temperature,
            final_la_blend=accuracy_tuned_fixed_final_la_blend,
            prior_gated_final_la=accuracy_tuned_prior_gated_final_la,
            min_class_prior_for_la=accuracy_tuned_min_class_prior_for_la,
            strict_vfl=strict_vfl,
        )
    if include_ebrb_boost:
        methods["Strict VFL-EBRB boost" if strict_vfl else "VFL-EBRB boost"] = strict_vfl_ebrb_boost_runner(
            n_estimators=ebrb_boost_estimators,
            learning_rate=ebrb_boost_learning_rate,
            sample_fraction=ebrb_boost_sample_fraction,
            n_rules_per_party=ebrb_boost_rules_per_party,
            bins=ebrb_boost_bins,
            max_antecedents=ebrb_boost_max_antecedents,
            beam_width=ebrb_boost_beam_width,
            beta_smoothing=ebrb_boost_beta_smoothing,
            min_support_fraction=ebrb_boost_min_support_fraction,
            class_balance_power=ebrb_boost_class_balance_power,
            sampling_class_balance_power=ebrb_boost_sampling_class_balance_power,
            client_weight_method=ebrb_boost_client_weight_method,
            fusion_method=ebrb_boost_fusion_method,
            weight_update_method=ebrb_boost_weight_update_method,
            client_fit_weight_power=ebrb_boost_client_fit_weight_power,
            decision_threshold_method=ebrb_boost_decision_threshold_method,
            decision_threshold_prior_multiplier=ebrb_boost_decision_threshold_prior_multiplier,
            parameter_iterations=ebrb_boost_parameter_iterations,
            parameter_update_blend=ebrb_boost_parameter_update_blend,
        )
    if include_xgboost and not strict_vfl:
        methods["VFL-XGBoost proxy"] = vfl_xgboost_proxy
    if include_ablations:
        prefix = "Strict VFL-Tuned DR-MBRB" if strict_vfl else "VFL-Tuned DR-MBRB"
        methods[f"{prefix} w/o logit LA"] = tuned_dr_mbrb_runner(base_payload, "no_la", strict_vfl=strict_vfl)
        methods[f"{prefix} w/o dynamic reasoning"] = tuned_dr_mbrb_runner(base_payload, "no_dr", strict_vfl=strict_vfl)

    aggregate: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        mode: {name: [] for name in methods}
        for mode in heterogeneity_modes
    }
    split_stats: Dict[str, List[Dict[str, float]]] = {mode: [] for mode in heterogeneity_modes}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, stratify_labels(y_raw)), start=1):
        encoder = TabularEncoder()
        X_train_base = encoder.fit_transform(X_raw[train_idx])
        X_test_base = encoder.transform(X_raw[test_idx])
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        for mode in heterogeneity_modes:
            split_mode = "balanced" if mode == "noisy_party" else mode
            split = make_vertibench_style_split(
                X_train_base,
                y_train,
                n_parties=config.n_parties,
                active_party_id=config.active_party_id,
                mode=split_mode,
                random_state=config.random_state + 100 * fold,
            )
            X_train, X_test = X_train_base, X_test_base
            if mode == "noisy_party":
                X_train, X_test = apply_noisy_party(
                    X_train_base,
                    X_test_base,
                    split.party_features,
                    noisy_party_id=1 if config.n_parties > 1 else 0,
                    noise_std=0.75,
                    dropout_rate=0.30,
                    random_state=config.random_state + 1000 * fold,
                )
            split_stats[mode].append(
                {
                    "fold": float(fold),
                    "importance_imbalance_ratio": split.importance_imbalance_ratio,
                    "importance_imbalance_std": split.importance_imbalance_std,
                    "correlation_beta": split.correlation_beta,
                    "within_correlation": split.within_correlation,
                    "cross_correlation": split.cross_correlation,
                    **{f"party_{idx}_features": float(split.feature_counts[idx]) for idx in sorted(split.feature_counts)},
                    **{f"party_{idx}_importance": float(split.party_importance[idx]) for idx in sorted(split.party_importance)},
                }
            )
            for name, runner in methods.items():
                aggregate[mode][name].append(
                    evaluate_method(
                        runner,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        split.party_features,
                        config.active_party_id,
                        config.random_state + fold,
                    )
                )

    return {
        "n_samples": int(len(y_raw)),
        "n_raw_features": int(np.asarray(X_raw).shape[1]),
        "heterogeneity": {
            mode: {
                "folds": rows,
                "mean": summarize(rows),
            }
            for mode, rows in split_stats.items()
        },
        "results": {
            mode: {
                name: {
                    "folds": rows,
                    "mean": summarize(rows),
                }
                for name, rows in method_rows.items()
            }
            for mode, method_rows in aggregate.items()
        },
    }


def compact_table(result: Dict[str, Any]) -> str:
    lines = [
        "| Dataset | Heterogeneity | Method | Acc | Macro-F1 | Bal-Acc | Minority-F1 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for dataset_name, dataset_result in result["datasets"].items():
        for mode, mode_results in dataset_result["results"].items():
            for method, payload in mode_results.items():
                mean = payload["mean"]
                lines.append(
                    f"| {dataset_name} | {mode} | {method} | "
                    f"{mean['accuracy_mean']:.4f} | {mean['macro_f1_mean']:.4f} | "
                    f"{mean['balanced_accuracy_mean']:.4f} | {mean['minority_f1_mean']:.4f} |"
                )
    return "\n".join(lines)


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VertiBench-style VFL heterogeneity comparison experiment.")
    parser.add_argument("--config", default="configs/vfl_tuned_dr_mbrb_demo.json")
    parser.add_argument("--datasets", default="default_credit,adult,bank_marketing")
    parser.add_argument("--heterogeneity", default="balanced,importance_skew,correlation_skew,noisy_party")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--strict-vfl", action="store_true")
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--include-strict-xgboost", action="store_true")
    parser.add_argument("--strict-xgboost-n-estimators", type=int, default=100)
    parser.add_argument("--strict-xgboost-depth", type=int, default=3)
    parser.add_argument("--strict-xgboost-learning-rate", type=float, default=0.08)
    parser.add_argument("--strict-xgboost-max-bins", type=int, default=32)
    parser.add_argument("--strict-xgboost-reg-lambda", type=float, default=1.0)
    parser.add_argument("--strict-xgboost-gamma", type=float, default=0.0)
    parser.add_argument("--strict-xgboost-min-child-weight", type=float, default=1.0)
    parser.add_argument("--strict-xgboost-min-samples-leaf", type=int, default=10)
    parser.add_argument("--include-fedtree", action="store_true")
    parser.add_argument("--fedtree-root", default="external/FedTree")
    parser.add_argument("--fedtree-n-trees", type=int, default=100)
    parser.add_argument("--fedtree-depth", type=int, default=3)
    parser.add_argument("--fedtree-learning-rate", type=float, default=0.08)
    parser.add_argument("--fedtree-timeout-sec", type=int, default=300)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--include-accuracy-tuned", action="store_true")
    parser.add_argument("--include-mbrb-forest", action="store_true")
    parser.add_argument("--include-mbrb-boost", action="store_true")
    parser.add_argument("--include-ebrb-boost", action="store_true")
    parser.add_argument("--accuracy-tuned-n-modules", type=int, default=1)
    parser.add_argument("--accuracy-tuned-rules-per-class", type=int, default=8)
    parser.add_argument("--accuracy-tuned-class-balance-power", type=float, default=0.0)
    parser.add_argument("--accuracy-tuned-prototype-method", default="quantile", choices=["quantile", "centroid", "kmeans", "farthest", "hybrid", "auto"])
    parser.add_argument("--accuracy-tuned-rule-scale-method", default="global", choices=["global", "local_knn"])
    parser.add_argument("--accuracy-tuned-rule-scale-neighbors", type=int, default=48)
    parser.add_argument("--accuracy-tuned-rule-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--accuracy-tuned-reasoning-steps-grid", default="1")
    parser.add_argument(
        "--accuracy-tuned-client-weight-method",
        default="confidence",
        choices=["confidence", "label_accuracy", "label_log_loss"],
    )
    parser.add_argument("--accuracy-tuned-use-stacking", action="store_true")
    parser.add_argument("--accuracy-tuned-stacking-c", type=float, default=1.0)
    parser.add_argument("--accuracy-tuned-stacking-class-weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--accuracy-tuned-validation-ratio", type=float, default=0.2)
    parser.add_argument("--accuracy-tuned-fixed-final-la-gamma", type=float, default=1.2)
    parser.add_argument("--accuracy-tuned-fixed-final-la-temperature", type=float, default=0.8)
    parser.add_argument("--accuracy-tuned-fixed-final-la-blend", type=float, default=0.75)
    parser.add_argument("--accuracy-tuned-prior-gated-final-la", action="store_true")
    parser.add_argument("--accuracy-tuned-min-class-prior-for-la", type=float, default=0.15)
    parser.add_argument("--mbrb-forest-estimators", type=int, default=5)
    parser.add_argument("--mbrb-forest-sample-fraction", type=float, default=0.9)
    parser.add_argument("--mbrb-forest-feature-fraction", type=float, default=0.8)
    parser.add_argument("--mbrb-forest-fusion-method", default="mean", choices=["mean", "er"])
    parser.add_argument("--mbrb-boost-estimators", type=int, default=5)
    parser.add_argument("--mbrb-boost-sample-fraction", type=float, default=1.0)
    parser.add_argument("--mbrb-boost-learning-rate", type=float, default=0.35)
    parser.add_argument("--mbrb-boost-fusion-method", default="log", choices=["mean", "log"])
    parser.add_argument("--mbrb-boost-weight-update-method", default="hard", choices=["hard", "residual"])
    parser.add_argument("--mbrb-boost-sampling-class-balance-power", type=float, default=0.0)
    parser.add_argument("--mbrb-boost-class-conditioned-fusion", action="store_true")
    parser.add_argument("--mbrb-boost-decision-threshold-method", default="argmax", choices=["argmax", "train_prior"])
    parser.add_argument("--mbrb-boost-decision-threshold-prior-multiplier", type=float, default=1.0)
    parser.add_argument("--ebrb-boost-estimators", type=int, default=9)
    parser.add_argument("--ebrb-boost-sample-fraction", type=float, default=1.0)
    parser.add_argument("--ebrb-boost-learning-rate", type=float, default=0.20)
    parser.add_argument("--ebrb-boost-rules-per-party", type=int, default=24)
    parser.add_argument("--ebrb-boost-bins", type=int, default=6)
    parser.add_argument("--ebrb-boost-max-antecedents", type=int, default=2)
    parser.add_argument("--ebrb-boost-beam-width", type=int, default=48)
    parser.add_argument("--ebrb-boost-beta-smoothing", type=float, default=0.5)
    parser.add_argument("--ebrb-boost-min-support-fraction", type=float, default=0.02)
    parser.add_argument("--ebrb-boost-class-balance-power", type=float, default=0.0)
    parser.add_argument("--ebrb-boost-parameter-iterations", type=int, default=0)
    parser.add_argument("--ebrb-boost-parameter-update-blend", type=float, default=0.5)
    parser.add_argument("--ebrb-boost-sampling-class-balance-power", type=float, default=0.5)
    parser.add_argument("--ebrb-boost-client-fit-weight-power", type=float, default=0.0)
    parser.add_argument(
        "--ebrb-boost-client-weight-method",
        default="label_accuracy",
        choices=["confidence", "label_accuracy", "hybrid"],
    )
    parser.add_argument("--ebrb-boost-fusion-method", default="log", choices=["mean", "log"])
    parser.add_argument(
        "--ebrb-boost-weight-update-method",
        default="residual",
        choices=["none", "hard", "residual", "ensemble_residual"],
    )
    parser.add_argument("--ebrb-boost-decision-threshold-method", default="train_prior", choices=["argmax", "train_prior"])
    parser.add_argument("--ebrb-boost-decision-threshold-prior-multiplier", type=float, default=0.5)
    parser.add_argument("--mbrb-search-trials", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    specs = {
        "breast_cancer": DatasetSpec("breast_cancer", sklearn_loader="breast_cancer"),
        "default_credit": DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": DatasetSpec("adult", ("adult",)),
        "bank_marketing": DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    base_payload = load_config(args.config)
    selected_datasets = [specs[name] for name in parse_list(args.datasets)]
    heterogeneity_modes = parse_list(args.heterogeneity)
    accuracy_tuned_reasoning_steps_grid = parse_int_list(args.accuracy_tuned_reasoning_steps_grid)
    include_strict_xgboost = bool(args.include_strict_xgboost or (args.strict_vfl and args.include_xgboost))

    result: Dict[str, Any] = {
        "experiment": "vertibench_style_vfl_heterogeneity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": args.splits,
        "max_samples": args.max_samples,
        "strict_vfl": args.strict_vfl,
        "mbrb_search_trials": args.mbrb_search_trials,
        "include_xgboost_proxy": bool(args.include_xgboost and not args.strict_vfl),
        "include_strict_xgboost": include_strict_xgboost,
        "strict_xgboost_n_estimators": args.strict_xgboost_n_estimators,
        "strict_xgboost_depth": args.strict_xgboost_depth,
        "strict_xgboost_learning_rate": args.strict_xgboost_learning_rate,
        "strict_xgboost_max_bins": args.strict_xgboost_max_bins,
        "strict_xgboost_reg_lambda": args.strict_xgboost_reg_lambda,
        "strict_xgboost_gamma": args.strict_xgboost_gamma,
        "strict_xgboost_min_child_weight": args.strict_xgboost_min_child_weight,
        "strict_xgboost_min_samples_leaf": args.strict_xgboost_min_samples_leaf,
        "include_fedtree": args.include_fedtree,
        "fedtree_root": args.fedtree_root,
        "fedtree_n_trees": args.fedtree_n_trees,
        "fedtree_depth": args.fedtree_depth,
        "fedtree_learning_rate": args.fedtree_learning_rate,
        "include_accuracy_tuned": args.include_accuracy_tuned,
        "include_mbrb_forest": args.include_mbrb_forest,
        "include_mbrb_boost": args.include_mbrb_boost,
        "include_ebrb_boost": args.include_ebrb_boost,
        "accuracy_tuned_n_modules": args.accuracy_tuned_n_modules,
        "accuracy_tuned_rules_per_class": args.accuracy_tuned_rules_per_class,
        "accuracy_tuned_class_balance_power": args.accuracy_tuned_class_balance_power,
        "accuracy_tuned_prototype_method": args.accuracy_tuned_prototype_method,
        "accuracy_tuned_rule_scale_method": args.accuracy_tuned_rule_scale_method,
        "accuracy_tuned_rule_scale_neighbors": args.accuracy_tuned_rule_scale_neighbors,
        "accuracy_tuned_rule_scale_multiplier": args.accuracy_tuned_rule_scale_multiplier,
        "accuracy_tuned_reasoning_steps_grid": accuracy_tuned_reasoning_steps_grid,
        "accuracy_tuned_client_weight_method": args.accuracy_tuned_client_weight_method,
        "accuracy_tuned_use_stacking": args.accuracy_tuned_use_stacking,
        "accuracy_tuned_stacking_c": args.accuracy_tuned_stacking_c,
        "accuracy_tuned_stacking_class_weight": args.accuracy_tuned_stacking_class_weight,
        "accuracy_tuned_validation_ratio": args.accuracy_tuned_validation_ratio,
        "accuracy_tuned_fixed_final_la_gamma": args.accuracy_tuned_fixed_final_la_gamma,
        "accuracy_tuned_fixed_final_la_temperature": args.accuracy_tuned_fixed_final_la_temperature,
        "accuracy_tuned_fixed_final_la_blend": args.accuracy_tuned_fixed_final_la_blend,
        "accuracy_tuned_prior_gated_final_la": args.accuracy_tuned_prior_gated_final_la,
        "accuracy_tuned_min_class_prior_for_la": args.accuracy_tuned_min_class_prior_for_la,
        "mbrb_forest_estimators": args.mbrb_forest_estimators,
        "mbrb_forest_sample_fraction": args.mbrb_forest_sample_fraction,
        "mbrb_forest_feature_fraction": args.mbrb_forest_feature_fraction,
        "mbrb_forest_fusion_method": args.mbrb_forest_fusion_method,
        "mbrb_boost_estimators": args.mbrb_boost_estimators,
        "mbrb_boost_sample_fraction": args.mbrb_boost_sample_fraction,
        "mbrb_boost_learning_rate": args.mbrb_boost_learning_rate,
        "mbrb_boost_fusion_method": args.mbrb_boost_fusion_method,
        "mbrb_boost_weight_update_method": args.mbrb_boost_weight_update_method,
        "mbrb_boost_sampling_class_balance_power": args.mbrb_boost_sampling_class_balance_power,
        "mbrb_boost_class_conditioned_fusion": args.mbrb_boost_class_conditioned_fusion,
        "mbrb_boost_decision_threshold_method": args.mbrb_boost_decision_threshold_method,
        "mbrb_boost_decision_threshold_prior_multiplier": args.mbrb_boost_decision_threshold_prior_multiplier,
        "ebrb_boost_estimators": args.ebrb_boost_estimators,
        "ebrb_boost_sample_fraction": args.ebrb_boost_sample_fraction,
        "ebrb_boost_learning_rate": args.ebrb_boost_learning_rate,
        "ebrb_boost_rules_per_party": args.ebrb_boost_rules_per_party,
        "ebrb_boost_bins": args.ebrb_boost_bins,
        "ebrb_boost_max_antecedents": args.ebrb_boost_max_antecedents,
        "ebrb_boost_beam_width": args.ebrb_boost_beam_width,
        "ebrb_boost_beta_smoothing": args.ebrb_boost_beta_smoothing,
        "ebrb_boost_min_support_fraction": args.ebrb_boost_min_support_fraction,
        "ebrb_boost_class_balance_power": args.ebrb_boost_class_balance_power,
        "ebrb_boost_parameter_iterations": args.ebrb_boost_parameter_iterations,
        "ebrb_boost_parameter_update_blend": args.ebrb_boost_parameter_update_blend,
        "ebrb_boost_sampling_class_balance_power": args.ebrb_boost_sampling_class_balance_power,
        "ebrb_boost_client_weight_method": args.ebrb_boost_client_weight_method,
        "ebrb_boost_fusion_method": args.ebrb_boost_fusion_method,
        "ebrb_boost_weight_update_method": args.ebrb_boost_weight_update_method,
        "ebrb_boost_client_fit_weight_power": args.ebrb_boost_client_fit_weight_power,
        "ebrb_boost_decision_threshold_method": args.ebrb_boost_decision_threshold_method,
        "ebrb_boost_decision_threshold_prior_multiplier": args.ebrb_boost_decision_threshold_prior_multiplier,
        "heterogeneity_modes": heterogeneity_modes,
        "notes": {
            "split_design": "Local implementation of VertiBench-style importance/correlation feature distribution diversity.",
            "noisy_party": "Balanced importance split plus Gaussian noise std=0.75 and feature dropout=0.30 on party 1.",
            "mbrb_protocol": (
                "strict unsupervised passive mode; only active/server uses labels."
                if args.strict_vfl
                else "simulation_supervised passive mode is used for fair non-deep VFL comparison."
            ),
            "xgboost_proxy": "Optional centralized tree proxy, not a secure VFL protocol implementation.",
            "strict_xgboost": "Strict VFL-XGBoost-style histogram boosting: active computes gradients/hessians; each party builds local histograms and exposes split candidates, not raw features or labels.",
            "strict_ebrb": "Strict VFL-EBRB boost uses party-local antecedent activations and active/server-side consequent calibration; passive parties do not receive plaintext labels.",
            "fedtree": "Optional strict vertical FedTree GBDT simulation from Xtra-Computing/FedTree; each party receives only its local feature block.",
            "strict_vfl": "When enabled, passive parties do not receive plaintext labels; centralized proxies are excluded.",
        },
        "datasets": {},
    }
    for spec in selected_datasets:
        result["datasets"][spec.name] = evaluate_dataset(
            spec,
            base_payload=base_payload,
            splits=args.splits,
            max_samples=args.max_samples,
            heterogeneity_modes=heterogeneity_modes,
            strict_vfl=args.strict_vfl,
            include_xgboost=args.include_xgboost,
            include_strict_xgboost=include_strict_xgboost,
            include_fedtree=args.include_fedtree,
            include_ablations=args.include_ablations,
            mbrb_search_trials=args.mbrb_search_trials,
            include_accuracy_tuned=args.include_accuracy_tuned,
            include_mbrb_forest=args.include_mbrb_forest,
            include_mbrb_boost=args.include_mbrb_boost,
            include_ebrb_boost=args.include_ebrb_boost,
            accuracy_tuned_n_modules=args.accuracy_tuned_n_modules,
            accuracy_tuned_rules_per_class=args.accuracy_tuned_rules_per_class,
            accuracy_tuned_class_balance_power=args.accuracy_tuned_class_balance_power,
            accuracy_tuned_prototype_method=args.accuracy_tuned_prototype_method,
            accuracy_tuned_rule_scale_method=args.accuracy_tuned_rule_scale_method,
            accuracy_tuned_rule_scale_neighbors=args.accuracy_tuned_rule_scale_neighbors,
            accuracy_tuned_rule_scale_multiplier=args.accuracy_tuned_rule_scale_multiplier,
            accuracy_tuned_reasoning_steps_grid=accuracy_tuned_reasoning_steps_grid,
            accuracy_tuned_client_weight_method=args.accuracy_tuned_client_weight_method,
            accuracy_tuned_use_stacking=args.accuracy_tuned_use_stacking,
            accuracy_tuned_stacking_c=args.accuracy_tuned_stacking_c,
            accuracy_tuned_stacking_class_weight=args.accuracy_tuned_stacking_class_weight,
            accuracy_tuned_validation_ratio=args.accuracy_tuned_validation_ratio,
            accuracy_tuned_fixed_final_la_gamma=args.accuracy_tuned_fixed_final_la_gamma,
            accuracy_tuned_fixed_final_la_temperature=args.accuracy_tuned_fixed_final_la_temperature,
            accuracy_tuned_fixed_final_la_blend=args.accuracy_tuned_fixed_final_la_blend,
            accuracy_tuned_prior_gated_final_la=args.accuracy_tuned_prior_gated_final_la,
            accuracy_tuned_min_class_prior_for_la=args.accuracy_tuned_min_class_prior_for_la,
            mbrb_forest_estimators=args.mbrb_forest_estimators,
            mbrb_forest_sample_fraction=args.mbrb_forest_sample_fraction,
            mbrb_forest_feature_fraction=args.mbrb_forest_feature_fraction,
            mbrb_forest_fusion_method=args.mbrb_forest_fusion_method,
            mbrb_boost_estimators=args.mbrb_boost_estimators,
            mbrb_boost_sample_fraction=args.mbrb_boost_sample_fraction,
            mbrb_boost_learning_rate=args.mbrb_boost_learning_rate,
            mbrb_boost_fusion_method=args.mbrb_boost_fusion_method,
            mbrb_boost_weight_update_method=args.mbrb_boost_weight_update_method,
            mbrb_boost_sampling_class_balance_power=args.mbrb_boost_sampling_class_balance_power,
            mbrb_boost_class_conditioned_fusion=args.mbrb_boost_class_conditioned_fusion,
            mbrb_boost_decision_threshold_method=args.mbrb_boost_decision_threshold_method,
            mbrb_boost_decision_threshold_prior_multiplier=args.mbrb_boost_decision_threshold_prior_multiplier,
            ebrb_boost_estimators=args.ebrb_boost_estimators,
            ebrb_boost_sample_fraction=args.ebrb_boost_sample_fraction,
            ebrb_boost_learning_rate=args.ebrb_boost_learning_rate,
            ebrb_boost_rules_per_party=args.ebrb_boost_rules_per_party,
            ebrb_boost_bins=args.ebrb_boost_bins,
            ebrb_boost_max_antecedents=args.ebrb_boost_max_antecedents,
            ebrb_boost_beam_width=args.ebrb_boost_beam_width,
            ebrb_boost_beta_smoothing=args.ebrb_boost_beta_smoothing,
            ebrb_boost_min_support_fraction=args.ebrb_boost_min_support_fraction,
            ebrb_boost_class_balance_power=args.ebrb_boost_class_balance_power,
            ebrb_boost_parameter_iterations=args.ebrb_boost_parameter_iterations,
            ebrb_boost_parameter_update_blend=args.ebrb_boost_parameter_update_blend,
            ebrb_boost_sampling_class_balance_power=args.ebrb_boost_sampling_class_balance_power,
            ebrb_boost_client_weight_method=args.ebrb_boost_client_weight_method,
            ebrb_boost_fusion_method=args.ebrb_boost_fusion_method,
            ebrb_boost_weight_update_method=args.ebrb_boost_weight_update_method,
            ebrb_boost_client_fit_weight_power=args.ebrb_boost_client_fit_weight_power,
            ebrb_boost_decision_threshold_method=args.ebrb_boost_decision_threshold_method,
            ebrb_boost_decision_threshold_prior_multiplier=args.ebrb_boost_decision_threshold_prior_multiplier,
            strict_xgboost_n_estimators=args.strict_xgboost_n_estimators,
            strict_xgboost_depth=args.strict_xgboost_depth,
            strict_xgboost_learning_rate=args.strict_xgboost_learning_rate,
            strict_xgboost_max_bins=args.strict_xgboost_max_bins,
            strict_xgboost_reg_lambda=args.strict_xgboost_reg_lambda,
            strict_xgboost_gamma=args.strict_xgboost_gamma,
            strict_xgboost_min_child_weight=args.strict_xgboost_min_child_weight,
            strict_xgboost_min_samples_leaf=args.strict_xgboost_min_samples_leaf,
            fedtree_root=args.fedtree_root,
            fedtree_n_trees=args.fedtree_n_trees,
            fedtree_depth=args.fedtree_depth,
            fedtree_learning_rate=args.fedtree_learning_rate,
            fedtree_timeout_sec=args.fedtree_timeout_sec,
        )

    output = args.output
    if not output:
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(ROOT, "outputs", f"vertibench_heterogeneity_{stamp}.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(compact_table(result))
    print(f"\nSaved JSON: {output}")


if __name__ == "__main__":
    main()
