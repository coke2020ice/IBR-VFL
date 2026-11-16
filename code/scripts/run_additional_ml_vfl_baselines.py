from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import (  # noqa: E402
    active_weak_split,
    evaluate_runner,
    long_tail_resample,
    summarize,
)


EPS = 1e-12


def _party_standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    train_blocks: Dict[int, np.ndarray] = {}
    test_blocks: Dict[int, np.ndarray] = {}
    for party_id in sorted(int(pid) for pid in party_features):
        features = list(party_features[party_id])
        train = np.asarray(X_train[:, features], dtype=float)
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
        scale[scale < 1e-8] = 1.0
        train_blocks[party_id] = (train - mean) / scale
        test_blocks[party_id] = (np.asarray(X_test[:, features], dtype=float) - mean) / scale
    return train_blocks, test_blocks


def _class_balanced_weights(y_idx: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=int(np.max(y_idx)) + 1).astype(float)
    weights = np.max(counts) / np.maximum(counts, 1.0)
    sample_weight = weights[y_idx]
    return sample_weight / max(float(np.mean(sample_weight)), EPS)


def strict_vfl_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    if len(classes) > 2:
        sample_weight = _class_balanced_weights(y_idx)
        train_blocks, test_blocks = _party_standardize(X_train, X_test, party_features)
        n_classes = len(classes)
        rng = np.random.default_rng(seed)
        weights = {
            party_id: rng.normal(0.0, 1e-3, size=(block.shape[1], n_classes))
            for party_id, block in train_blocks.items()
        }
        bias = np.zeros(n_classes, dtype=float)
        learning_rate, l2, n_epochs = 0.025, 1e-3, 260
        batch_size = min(256, max(32, X_train.shape[0]))
        indices = np.arange(X_train.shape[0])
        for _ in range(n_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                scores = np.broadcast_to(bias[None, :], (len(batch), n_classes)).copy()
                for party_id, block in train_blocks.items():
                    scores += block[batch] @ weights[party_id]
                true = y_idx[batch]
                wrong_scores = scores.copy()
                wrong_scores[np.arange(len(batch)), true] = -np.inf
                rival = np.argmax(wrong_scores, axis=1)
                active = 1.0 + scores[np.arange(len(batch)), rival] - scores[np.arange(len(batch)), true] > 0.0
                residual = np.zeros_like(scores)
                coeff = sample_weight[batch] * active.astype(float)
                residual[np.arange(len(batch)), rival] += coeff
                residual[np.arange(len(batch)), true] -= coeff
                bias -= learning_rate * np.mean(residual, axis=0)
                for party_id, block in train_blocks.items():
                    grad = block[batch].T @ residual / max(float(len(batch)), 1.0) + l2 * weights[party_id]
                    weights[party_id] -= learning_rate * grad
        scores = np.broadcast_to(bias[None, :], (X_test.shape[0], n_classes)).copy()
        for party_id, block in test_blocks.items():
            scores += block @ weights[party_id]
        return classes[np.argmax(scores, axis=1)], {
            "strict_vfl_svm_epochs": float(n_epochs),
            "strict_vfl_svm_l2": float(l2),
        }
    y_signed = np.where(y_idx == 1, 1.0, -1.0)
    sample_weight = _class_balanced_weights(y_idx)
    train_blocks, test_blocks = _party_standardize(X_train, X_test, party_features)

    rng = np.random.default_rng(seed)
    weights = {
        party_id: rng.normal(0.0, 1e-3, size=block.shape[1])
        for party_id, block in train_blocks.items()
    }
    bias = 0.0
    learning_rate = 0.035
    l2 = 1e-3
    n_epochs = 260
    batch_size = min(256, max(32, X_train.shape[0]))
    indices = np.arange(X_train.shape[0])
    for _ in range(n_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            margin = np.full(batch.shape[0], bias, dtype=float)
            for party_id, block in train_blocks.items():
                margin += block[batch] @ weights[party_id]
            active = y_signed[batch] * margin < 1.0
            if not np.any(active):
                for party_id in weights:
                    weights[party_id] *= 1.0 - learning_rate * l2
                continue
            coeff = -y_signed[batch] * sample_weight[batch] * active.astype(float)
            bias -= learning_rate * float(np.mean(coeff))
            denom = max(float(batch.shape[0]), 1.0)
            for party_id, block in train_blocks.items():
                grad = block[batch].T @ coeff / denom + l2 * weights[party_id]
                weights[party_id] -= learning_rate * grad

    score = np.full(X_test.shape[0], bias, dtype=float)
    for party_id, block in test_blocks.items():
        score += block @ weights[party_id]
    pred_idx = (score >= 0.0).astype(int)
    return classes[pred_idx], {
        "strict_vfl_svm_epochs": float(n_epochs),
        "strict_vfl_svm_l2": float(l2),
    }


def strict_vfl_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id, seed
    classes, y_idx = np.unique(y_train, return_inverse=True)
    train_blocks, test_blocks = _party_standardize(X_train, X_test, party_features)
    n_classes = len(classes)
    counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    class_vote_weight = np.max(counts) / np.maximum(counts, 1.0)
    k = int(min(15, max(1, X_train.shape[0])))
    chunk_size = 128
    pred_idx = np.zeros(X_test.shape[0], dtype=int)
    for start in range(0, X_test.shape[0], chunk_size):
        stop = min(start + chunk_size, X_test.shape[0])
        dist = np.zeros((stop - start, X_train.shape[0]), dtype=float)
        for party_id, train_block in train_blocks.items():
            test_block = test_blocks[party_id][start:stop]
            dist += np.sum((test_block[:, None, :] - train_block[None, :, :]) ** 2, axis=2)
        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        nn_dist = np.take_along_axis(dist, nn, axis=1)
        nn_label = y_idx[nn]
        weights = 1.0 / (np.sqrt(np.maximum(nn_dist, 0.0)) + 1e-6)
        weights *= class_vote_weight[nn_label]
        vote = np.zeros((stop - start, n_classes), dtype=float)
        for class_idx in range(n_classes):
            vote[:, class_idx] = np.sum(weights * (nn_label == class_idx), axis=1)
        pred_idx[start:stop] = np.argmax(vote, axis=1)
    return classes[pred_idx], {
        "strict_vfl_knn_k": float(k),
        "strict_vfl_knn_class_balanced_vote": 1.0,
    }


def strict_vfl_splitnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Strict VFL-SplitNN currently supports binary classification")
    sample_weight = _class_balanced_weights(y_idx)
    train_blocks, test_blocks = _party_standardize(X_train, X_test, party_features)

    rng = np.random.default_rng(seed)
    hidden_dim = 4
    encoders = {
        party_id: rng.normal(0.0, 0.04, size=(block.shape[1], hidden_dim))
        for party_id, block in train_blocks.items()
    }
    heads = {
        party_id: rng.normal(0.0, 0.04, size=hidden_dim)
        for party_id in train_blocks
    }
    bias = 0.0
    learning_rate = 0.018
    l2 = 1e-4
    n_epochs = 220
    batch_size = min(256, max(32, X_train.shape[0]))
    indices = np.arange(X_train.shape[0])
    y_float = y_idx.astype(float)

    for _ in range(n_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            reps: Dict[int, np.ndarray] = {}
            logits = np.full(batch.shape[0], bias, dtype=float)
            for party_id, block in train_blocks.items():
                rep = np.tanh(block[batch] @ encoders[party_id])
                reps[party_id] = rep
                logits += rep @ heads[party_id]
            proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            residual = (proba - y_float[batch]) * sample_weight[batch]
            denom = max(float(batch.shape[0]), 1.0)
            bias -= learning_rate * float(np.mean(residual))
            old_heads = {party_id: heads[party_id].copy() for party_id in heads}
            for party_id, block in train_blocks.items():
                grad_head = reps[party_id].T @ residual / denom + l2 * heads[party_id]
                heads[party_id] -= learning_rate * grad_head
                local_grad = (residual[:, None] * old_heads[party_id][None, :]) * (1.0 - reps[party_id] ** 2)
                grad_encoder = block[batch].T @ local_grad / denom + l2 * encoders[party_id]
                encoders[party_id] -= learning_rate * grad_encoder

    logits = np.full(X_test.shape[0], bias, dtype=float)
    for party_id, block in test_blocks.items():
        logits += np.tanh(block @ encoders[party_id]) @ heads[party_id]
    pred_idx = (logits >= 0.0).astype(int)
    return classes[pred_idx], {
        "strict_vfl_splitnn_epochs": float(n_epochs),
        "strict_vfl_splitnn_hidden_dim": float(hidden_dim),
    }


@dataclass
class _RFNode:
    proba: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: "_RFNode | None" = None
    right: "_RFNode | None" = None


def _weighted_gini(y_idx: np.ndarray, sample_weight: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    counts = np.bincount(
        y_idx[indices], weights=sample_weight[indices], minlength=int(np.max(y_idx)) + 1
    ).astype(float)
    total = float(np.sum(counts))
    if total <= EPS:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _leaf_proba(y_idx: np.ndarray, sample_weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
    counts = np.bincount(
        y_idx[indices], weights=sample_weight[indices], minlength=int(np.max(y_idx)) + 1
    ).astype(float) + 1.0
    return counts / np.sum(counts)


def _build_tree(
    X: np.ndarray,
    y_idx: np.ndarray,
    sample_weight: np.ndarray,
    indices: np.ndarray,
    candidate_features: np.ndarray,
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    min_leaf: int,
    max_bins: int,
) -> _RFNode:
    proba = _leaf_proba(y_idx, sample_weight, indices)
    if depth >= max_depth or indices.size < 2 * min_leaf or np.max(proba) > 0.995:
        return _RFNode(proba=proba)

    n_features = int(candidate_features.size)
    max_features = max(1, int(np.sqrt(n_features)))
    feature_subset = rng.choice(candidate_features, size=max_features, replace=False)
    parent_weight = float(np.sum(sample_weight[indices]))
    parent_impurity = _weighted_gini(y_idx, sample_weight, indices)
    best_gain = 0.0
    best_feature: int | None = None
    best_threshold: float | None = None
    best_left: np.ndarray | None = None
    best_right: np.ndarray | None = None

    for feature in feature_subset:
        values = X[indices, feature]
        unique = np.unique(values)
        if unique.size <= 1:
            continue
        if unique.size <= max_bins:
            thresholds = (unique[:-1] + unique[1:]) / 2.0
        else:
            qs = np.linspace(0.0, 1.0, max_bins + 2)[1:-1]
            thresholds = np.unique(np.quantile(values, qs))
        for threshold in thresholds:
            left_mask = values <= threshold
            left = indices[left_mask]
            right = indices[~left_mask]
            if left.size < min_leaf or right.size < min_leaf:
                continue
            left_weight = float(np.sum(sample_weight[left]))
            right_weight = float(np.sum(sample_weight[right]))
            gain = parent_impurity
            gain -= (left_weight / max(parent_weight, EPS)) * _weighted_gini(y_idx, sample_weight, left)
            gain -= (right_weight / max(parent_weight, EPS)) * _weighted_gini(y_idx, sample_weight, right)
            if gain > best_gain:
                best_gain = float(gain)
                best_feature = int(feature)
                best_threshold = float(threshold)
                best_left = left
                best_right = right

    if best_feature is None or best_left is None or best_right is None:
        return _RFNode(proba=proba)
    return _RFNode(
        proba=proba,
        feature=best_feature,
        threshold=best_threshold,
        left=_build_tree(X, y_idx, sample_weight, best_left, candidate_features, rng, depth + 1, max_depth, min_leaf, max_bins),
        right=_build_tree(X, y_idx, sample_weight, best_right, candidate_features, rng, depth + 1, max_depth, min_leaf, max_bins),
    )


def _predict_tree(node: _RFNode, x: np.ndarray) -> np.ndarray:
    current = node
    while current.feature is not None and current.threshold is not None and current.left is not None and current.right is not None:
        current = current.left if x[current.feature] <= current.threshold else current.right
    return current.proba


def strict_vfl_vertical_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    active_party_id: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    del active_party_id
    classes, y_idx = np.unique(y_train, return_inverse=True)
    X_train = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(np.asarray(X_test, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    candidate_features = np.asarray(
        sorted({int(feature) for features in party_features.values() for feature in features}),
        dtype=int,
    )
    if candidate_features.size == 0:
        raise ValueError("Strict VFL-RF needs at least one feature from party_features")
    sample_weight = _class_balanced_weights(y_idx)
    rng = np.random.default_rng(seed)
    n_estimators = 15
    max_depth = 5
    min_leaf = max(8, int(0.006 * X_train.shape[0]))
    max_bins = 16
    trees: List[_RFNode] = []
    all_indices = np.arange(X_train.shape[0])
    for _ in range(n_estimators):
        bootstrap = rng.choice(all_indices, size=all_indices.size, replace=True)
        trees.append(_build_tree(X_train, y_idx, sample_weight, bootstrap, candidate_features, rng, 0, max_depth, min_leaf, max_bins))
    proba = np.zeros((X_test.shape[0], len(classes)), dtype=float)
    for tree in trees:
        for row_idx in range(X_test.shape[0]):
            proba[row_idx] += _predict_tree(tree, X_test[row_idx])
    proba /= float(len(trees))
    pred_idx = np.argmax(proba, axis=1)
    return classes[pred_idx], {
        "strict_vfl_rf_trees": float(n_estimators),
        "strict_vfl_rf_depth": float(max_depth),
        "strict_vfl_rf_parties": float(len(party_features)),
        "strict_vfl_rf_candidate_features": float(candidate_features.size),
    }


def main() -> None:
    specs = {
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    methods = {
        "Strict VFL-Linear SVM": strict_vfl_linear_svm,
        "Strict VFL-KNN": strict_vfl_knn,
        "Strict VFL-SplitNN": strict_vfl_splitnn,
        "Strict VFL-Vertical RF": strict_vfl_vertical_rf,
    }
    ratios = [0.03, 0.02, 0.01]
    n_parties = 10
    result: Dict[str, Any] = {
        "experiment": "additional_strict_ml_vfl_baselines_active_weak_k10",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": 3,
        "max_samples": 6000,
        "ratios": ratios,
        "heterogeneity_modes": ["active_weak"],
        "n_parties": n_parties,
        "split_strategy": "active_weak",
        "rank_strength": 3.0,
        "methods": list(methods),
        "notes": {
            "protocol": "Same active-weak K=10, three-fold, three-ratio protocol as the broad baseline tables.",
            "strict_vfl_linear_svm": "Active party aggregates scalar margins and class-balanced hinge residuals; passive parties update only local linear weights.",
            "strict_vfl_knn": "Each party computes local squared-distance contributions; the server aggregates distances and applies class-balanced nearest-neighbor voting.",
            "strict_vfl_splitnn": "Each party trains a local tanh encoder; the active side fuses local representations through a binary classifier.",
            "strict_vfl_vertical_rf": "Tree split search is simulated as party-local split proposals scored by active-label impurity statistics.",
        },
        "datasets": {},
    }
    start_all = time.perf_counter()
    for dataset_name, spec in specs.items():
        print(f"[dataset] {dataset_name}", flush=True)
        X_raw, y_raw = exp.load_dataset(spec)
        y_raw = np.asarray(y_raw, dtype=object).astype(str)
        X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=6000, seed=13)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
        dataset_rows: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]] = {
            f"{ratio:.2f}": {"active_weak": {name: [] for name in methods}} for ratio in ratios
        }
        dataset_meta: Dict[str, List[Dict[str, float]]] = {f"{ratio:.2f}": [] for ratio in ratios}
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, exp.stratify_labels(y_raw)), start=1):
            y_train_base = y_raw[train_idx]
            labels, counts = np.unique(y_train_base, return_counts=True)
            minority_label = labels[int(np.argmin(counts))]
            for ratio in ratios:
                key = f"{ratio:.2f}"
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
                X_train = encoder.fit_transform(X_train_tail_raw)
                X_test = encoder.transform(X_test_tail_raw)
                dataset_meta[key].append(
                    {
                        "fold": float(fold),
                        "train_n": float(len(y_train_tail)),
                        "test_n": float(len(y_test_tail)),
                        "train_minority_ratio": float(np.mean(y_train_tail == minority_label)),
                        "test_minority_ratio": float(np.mean(y_test_tail == minority_label)),
                    }
                )
                split = active_weak_split(
                    X_train,
                    y_train_tail,
                    n_parties=n_parties,
                    rank_strength=3.0,
                    seed=13 + 100 * fold,
                )
                for method_name, runner in methods.items():
                    print(f"  fold={fold} ratio={key} method={method_name}", flush=True)
                    row = evaluate_runner(
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
                    dataset_rows[key]["active_weak"][method_name].append(row)
                    print(
                        f"    acc={row['accuracy']:.4f} minf1={row['minority_f1']:.4f} time={row['wall_time_sec']:.1f}s",
                        flush=True,
                    )
        result["datasets"][dataset_name] = {
            "long_tail_meta": {
                ratio: {"folds": rows, "mean": summarize(rows)}
                for ratio, rows in dataset_meta.items()
            },
            "results": {
                ratio: {
                    mode: {
                        method_name: {"folds": rows, "mean": summarize(rows)}
                        for method_name, rows in methods_rows.items()
                    }
                    for mode, methods_rows in modes.items()
                }
                for ratio, modes in dataset_rows.items()
            },
        }
    out_path = Path(__file__).resolve().parents[2] / "results" / "additional_strict_ml_vfl_active_weak_k10.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}", flush=True)
    print(f"[done] elapsed {time.perf_counter() - start_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
