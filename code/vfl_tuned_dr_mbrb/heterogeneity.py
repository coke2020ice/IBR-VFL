from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif


EPS = 1e-12


@dataclass(frozen=True)
class HeterogeneitySplit:
    """Feature split plus VertiBench-style heterogeneity diagnostics."""

    mode: str
    party_features: Dict[int, List[int]]
    feature_importance: np.ndarray
    party_importance: Dict[int, float]
    feature_counts: Dict[int, int]
    importance_imbalance_ratio: float
    importance_imbalance_std: float
    correlation_beta: float
    within_correlation: float
    cross_correlation: float


def make_vertibench_style_split(
    X: Sequence[Sequence[float]],
    y: Sequence[object],
    n_parties: int,
    active_party_id: int = 0,
    mode: str = "balanced",
    random_state: int | None = None,
    skew_weights: Sequence[float] | None = None,
) -> HeterogeneitySplit:
    """Create a vertical feature split inspired by VertiBench.

    The real VertiBench package provides ImportanceSplitter and
    CorrelationSplitter. This local implementation keeps the same experimental
    knobs without pulling in a heavy dependency that may downgrade the current
    research environment.
    """

    arr = _validate_X(X)
    if n_parties <= 0:
        raise ValueError("n_parties must be positive")
    if active_party_id < 0 or active_party_id >= n_parties:
        raise ValueError("active_party_id must be a valid party index")
    if arr.shape[1] < n_parties:
        raise ValueError("The encoded dataset must have at least one feature per party")

    importance = estimate_feature_importance(arr, y, random_state=random_state)
    corr = absolute_correlation(arr)

    if mode == "balanced":
        party_features = _balanced_importance_split(importance, n_parties)
    elif mode == "importance_skew":
        weights = _default_skew_weights(n_parties) if skew_weights is None else np.asarray(skew_weights, dtype=float)
        party_features = _importance_skew_split(importance, n_parties, weights)
    elif mode == "correlation_skew":
        party_features = _correlation_skew_split(importance, corr, n_parties)
    elif mode == "random":
        rng = np.random.default_rng(random_state)
        chunks = np.array_split(rng.permutation(arr.shape[1]), n_parties)
        party_features = {idx: [int(value) for value in chunk.tolist()] for idx, chunk in enumerate(chunks)}
    else:
        raise ValueError(f"Unknown heterogeneity split mode: {mode}")

    party_features = _repair_empty_parties(party_features, importance, n_parties)
    return evaluate_heterogeneity_split(mode, party_features, importance, corr)


def estimate_feature_importance(
    X: Sequence[Sequence[float]],
    y: Sequence[object],
    random_state: int | None = None,
) -> np.ndarray:
    arr = _validate_X(X)
    y_idx = np.unique(np.asarray(y, dtype=object), return_inverse=True)[1]
    try:
        scores = mutual_info_classif(arr, y_idx, random_state=random_state, discrete_features=False)
    except Exception:
        scores = _label_correlation_importance(arr, y_idx)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    if float(np.sum(scores)) <= EPS:
        scores = np.var(arr, axis=0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.sum(scores)) <= EPS:
        scores = np.ones(arr.shape[1], dtype=float)
    return scores


def absolute_correlation(X: Sequence[Sequence[float]]) -> np.ndarray:
    arr = _validate_X(X)
    if arr.shape[1] == 1:
        return np.zeros((1, 1), dtype=float)
    std = np.std(arr, axis=0)
    safe_std = np.where(std <= EPS, 1.0, std)
    z = (arr - np.mean(arr, axis=0, keepdims=True)) / safe_std[None, :]
    # Matrix multiplication avoids np.corrcoef warnings for constant columns;
    # those columns are already represented by all-zero standardized vectors.
    denominator = max(arr.shape[0], 1)
    corr = np.abs(z.T @ z / denominator)
    np.fill_diagonal(corr, 0.0)
    return corr


def evaluate_heterogeneity_split(
    mode: str,
    party_features: Mapping[int, Sequence[int]],
    feature_importance: Sequence[float],
    corr: np.ndarray | None = None,
) -> HeterogeneitySplit:
    importance = np.asarray(feature_importance, dtype=float)
    partitions = {int(key): [int(value) for value in values] for key, values in party_features.items()}
    if corr is None:
        corr = np.zeros((len(importance), len(importance)), dtype=float)

    party_importance = {
        party_id: float(np.sum(importance[features])) if features else 0.0
        for party_id, features in partitions.items()
    }
    feature_counts = {party_id: int(len(features)) for party_id, features in partitions.items()}
    totals = np.array([party_importance[party_id] for party_id in sorted(partitions)], dtype=float)
    positive = totals[totals > EPS]
    if positive.size == 0:
        ratio = 1.0
    elif positive.size < totals.size:
        ratio = float(np.max(positive) / EPS)
    else:
        ratio = float(np.max(positive) / max(float(np.min(positive)), EPS))
    imbalance_std = float(np.std(totals / max(float(np.sum(totals)), EPS)))
    within, cross = _within_cross_correlation(partitions, np.asarray(corr, dtype=float))
    beta = float(within / max(cross, EPS)) if cross > EPS else float(within / EPS if within > EPS else 1.0)
    return HeterogeneitySplit(
        mode=mode,
        party_features=partitions,
        feature_importance=importance,
        party_importance=party_importance,
        feature_counts=feature_counts,
        importance_imbalance_ratio=ratio,
        importance_imbalance_std=imbalance_std,
        correlation_beta=beta,
        within_correlation=float(within),
        cross_correlation=float(cross),
    )


def apply_noisy_party(
    X_train: Sequence[Sequence[float]],
    X_test: Sequence[Sequence[float]],
    party_features: Mapping[int, Sequence[int]],
    noisy_party_id: int = 1,
    noise_std: float = 0.75,
    dropout_rate: float = 0.25,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=float).copy()
    test = np.asarray(X_test, dtype=float).copy()
    features = list(party_features.get(noisy_party_id, []))
    if not features:
        return train, test
    rng = np.random.default_rng(random_state)
    for arr in (train, test):
        local_shape = (arr.shape[0], len(features))
        if noise_std > 0:
            arr[:, features] += rng.normal(0.0, noise_std, size=local_shape)
        if dropout_rate > 0:
            mask = rng.random(local_shape) < dropout_rate
            arr[:, features] = np.where(mask, 0.0, arr[:, features])
    return train, test


def _balanced_importance_split(importance: np.ndarray, n_parties: int) -> Dict[int, List[int]]:
    partitions = {party_id: [] for party_id in range(n_parties)}
    totals = np.zeros(n_parties, dtype=float)
    counts = np.zeros(n_parties, dtype=int)
    for feature in np.argsort(-importance):
        party_id = int(np.lexsort((counts, totals))[0])
        partitions[party_id].append(int(feature))
        totals[party_id] += float(importance[feature])
        counts[party_id] += 1
    return partitions


def _importance_skew_split(
    importance: np.ndarray,
    n_parties: int,
    skew_weights: np.ndarray,
) -> Dict[int, List[int]]:
    if skew_weights.size != n_parties:
        raise ValueError("skew_weights must contain one value per party")
    weights = np.maximum(skew_weights.astype(float), EPS)
    weights = weights / max(float(np.sum(weights)), EPS)
    targets = weights * max(float(np.sum(importance)), EPS)
    partitions = {party_id: [] for party_id in range(n_parties)}
    totals = np.zeros(n_parties, dtype=float)
    for feature in np.argsort(-importance):
        ratios = totals / np.maximum(targets, EPS)
        party_id = int(np.argmin(ratios))
        partitions[party_id].append(int(feature))
        totals[party_id] += float(importance[feature])
    return partitions


def _correlation_skew_split(
    importance: np.ndarray,
    corr: np.ndarray,
    n_parties: int,
) -> Dict[int, List[int]]:
    n_features = len(importance)
    target_counts = [len(chunk) for chunk in np.array_split(np.arange(n_features), n_parties)]
    partitions = {party_id: [] for party_id in range(n_parties)}
    unassigned = set(range(n_features))
    seed_order = [int(value) for value in np.argsort(-importance)]
    for party_id, target in enumerate(target_counts):
        if not unassigned:
            break
        seed = next((feature for feature in seed_order if feature in unassigned), min(unassigned))
        partitions[party_id].append(seed)
        unassigned.remove(seed)
        while len(partitions[party_id]) < target and unassigned:
            candidates = np.array(sorted(unassigned), dtype=int)
            current = np.array(partitions[party_id], dtype=int)
            corr_score = np.mean(corr[np.ix_(candidates, current)], axis=1)
            tie_score = importance[candidates] / max(float(np.max(importance)), EPS)
            best_idx = int(np.argmax(corr_score + 1e-6 * tie_score))
            chosen = int(candidates[best_idx])
            partitions[party_id].append(chosen)
            unassigned.remove(chosen)

    for feature in sorted(unassigned):
        party_id = min(partitions, key=lambda idx: len(partitions[idx]))
        partitions[party_id].append(int(feature))
    return partitions


def _repair_empty_parties(
    party_features: Mapping[int, Sequence[int]],
    importance: np.ndarray,
    n_parties: int,
) -> Dict[int, List[int]]:
    partitions = {party_id: [int(value) for value in party_features.get(party_id, [])] for party_id in range(n_parties)}
    for empty_party in [party_id for party_id, features in partitions.items() if not features]:
        donors = [party_id for party_id, features in partitions.items() if len(features) > 1]
        if not donors:
            raise ValueError("Unable to repair empty parties with the available features")
        donor = max(donors, key=lambda party_id: len(partitions[party_id]))
        donor_features = partitions[donor]
        move_idx = int(np.argmin(importance[donor_features]))
        moved = donor_features.pop(move_idx)
        partitions[empty_party].append(int(moved))
    return partitions


def _within_cross_correlation(
    party_features: Mapping[int, Sequence[int]],
    corr: np.ndarray,
) -> tuple[float, float]:
    within_values: List[float] = []
    cross_values: List[float] = []
    parties = sorted(party_features)
    for party_id in parties:
        features = list(party_features[party_id])
        for i, feature_i in enumerate(features):
            for feature_j in features[i + 1 :]:
                within_values.append(float(corr[feature_i, feature_j]))
    for left_index, party_i in enumerate(parties):
        for party_j in parties[left_index + 1 :]:
            for feature_i in party_features[party_i]:
                for feature_j in party_features[party_j]:
                    cross_values.append(float(corr[feature_i, feature_j]))
    within = float(np.mean(within_values)) if within_values else 0.0
    cross = float(np.mean(cross_values)) if cross_values else 0.0
    return within, cross


def _label_correlation_importance(X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    y_std = np.std(y_idx)
    if y_std <= EPS:
        return np.zeros(X.shape[1], dtype=float)
    scores = []
    for feature in range(X.shape[1]):
        x = X[:, feature]
        x_std = np.std(x)
        if x_std <= EPS:
            scores.append(0.0)
            continue
        scores.append(abs(float(np.corrcoef(x, y_idx)[0, 1])))
    return np.asarray(scores, dtype=float)


def _default_skew_weights(n_parties: int) -> np.ndarray:
    weights = np.geomspace(1.0, 0.2, num=n_parties)
    return weights / max(float(np.sum(weights)), EPS)


def _validate_X(X: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be two-dimensional")
    return arr
