from __future__ import annotations

from typing import Sequence

import numpy as np


EPS = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = np.sum(arr, axis=1, keepdims=True)
    return arr / np.maximum(total, EPS)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return normalize_rows(exp_values)


def er_fuse(evidence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fuse belief evidence with sample-wise ER-style normalized log pooling.

    Parameters
    ----------
    evidence:
        Belief tensor with shape ``(n_samples, n_items, n_classes)``.
    weights:
        Nonnegative fusion weights with shape ``(n_samples, n_items)``.

    Notes
    -----
    The paper writes local rule and module aggregation as ``ER(...)``. This
    implementation uses a normalized weighted log-opinion pool, which preserves
    the key BRB/MBRB behavior needed here: evidence remains in belief space,
    high-confidence evidence can dominate when weights support it, and the
    output is a valid belief distribution.
    """

    evidence_arr = np.asarray(evidence, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    if evidence_arr.ndim != 3:
        raise ValueError("evidence must have shape (n_samples, n_items, n_classes)")
    if weights_arr.shape != evidence_arr.shape[:2]:
        raise ValueError("weights must have shape (n_samples, n_items)")
    weights_arr = weights_arr / np.maximum(np.sum(weights_arr, axis=1, keepdims=True), EPS)
    logits = np.sum(weights_arr[:, :, None] * np.log(evidence_arr + EPS), axis=1)
    return softmax(logits)


def smooth_prior(y: Sequence[object], classes: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    y_arr = np.asarray(y, dtype=object)
    counts = np.array([np.sum(y_arr == label) for label in classes], dtype=float)
    counts = counts + float(alpha)
    return counts / np.maximum(np.sum(counts), EPS)


def macro_f1_score(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    scores = []
    for label in labels:
        tp = float(np.sum((y_true_arr == label) & (y_pred_arr == label)))
        fp = float(np.sum((y_true_arr != label) & (y_pred_arr == label)))
        fn = float(np.sum((y_true_arr == label) & (y_pred_arr != label)))
        precision = tp / max(tp + fp, EPS)
        recall = tp / max(tp + fn, EPS)
        scores.append(2.0 * precision * recall / max(precision + recall, EPS))
    return float(np.mean(scores)) if scores else 0.0


def accuracy_score(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    return float(np.mean(y_true_arr == y_pred_arr))


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))
