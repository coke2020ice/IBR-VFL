from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class _ColumnState:
    kind: str
    fill_value: float = 0.0
    mean: float = 0.0
    scale: float = 1.0
    categories: Dict[object, int] | None = None


class TabularEncoder:
    """Train-fold-only tabular encoder for VFL experiments.

    Numeric columns are median-filled and standardized. Categorical columns are
    ordinal-encoded with an explicit unknown value, then standardized. This is
    intentionally small so each party can be encoded from the same train fold
    without pulling in a heavy preprocessing stack.
    """

    def __init__(self) -> None:
        self.columns_: List[_ColumnState] = []

    def fit(self, X: Sequence[Sequence[object]]) -> "TabularEncoder":
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        self.columns_.clear()
        for feature_index in range(arr.shape[1]):
            col = arr[:, feature_index]
            numeric = self._try_numeric(col)
            if numeric is not None:
                observed = numeric[~np.isnan(numeric)]
                fill = float(np.median(observed)) if observed.size else 0.0
                filled = np.nan_to_num(numeric, nan=fill)
                mean = float(np.mean(filled))
                scale = float(np.std(filled)) or 1.0
                self.columns_.append(_ColumnState(kind="numeric", fill_value=fill, mean=mean, scale=scale))
                continue

            normalized = np.array([self._clean_category(value) for value in col], dtype=object)
            categories = {value: idx for idx, value in enumerate(dict.fromkeys(normalized.tolist()).keys(), start=1)}
            encoded = np.array([categories[value] for value in normalized], dtype=float)
            mean = float(np.mean(encoded))
            scale = float(np.std(encoded)) or 1.0
            self.columns_.append(_ColumnState(kind="categorical", mean=mean, scale=scale, categories=categories))
        return self

    def transform(self, X: Sequence[Sequence[object]]) -> np.ndarray:
        if not self.columns_:
            raise ValueError("The encoder has not been fitted")
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2 or arr.shape[1] != len(self.columns_):
            raise ValueError("X does not match the fitted encoder dimensionality")
        encoded = np.zeros(arr.shape, dtype=float)
        for feature_index, state in enumerate(self.columns_):
            col = arr[:, feature_index]
            if state.kind == "numeric":
                numeric = self._try_numeric(col)
                if numeric is None:
                    numeric = np.full(col.shape[0], state.fill_value, dtype=float)
                numeric = np.nan_to_num(numeric, nan=state.fill_value)
                encoded[:, feature_index] = (numeric - state.mean) / state.scale
            else:
                categories = state.categories or {}
                values = [categories.get(self._clean_category(value), 0) for value in col]
                values_arr = np.asarray(values, dtype=float)
                encoded[:, feature_index] = (values_arr - state.mean) / state.scale
        return encoded

    def fit_transform(self, X: Sequence[Sequence[object]]) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _try_numeric(col: np.ndarray) -> np.ndarray | None:
        try:
            return np.asarray(col, dtype=float)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_category(value: object) -> object:
        if value is None:
            return "__missing__"
        if isinstance(value, float) and np.isnan(value):
            return "__missing__"
        if value == "" or value == "?":
            return "__missing__"
        return value


def make_vertical_partitions(
    n_features: int,
    n_parties: int,
    active_party_id: int = 0,
    strategy: str = "contiguous",
    random_state: int | None = None,
) -> Dict[int, List[int]]:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if n_parties <= 0:
        raise ValueError("n_parties must be positive")
    if active_party_id < 0 or active_party_id >= n_parties:
        raise ValueError("active_party_id must be a valid party index")

    features = np.arange(n_features)
    if strategy == "random":
        rng = np.random.default_rng(random_state)
        features = rng.permutation(features)

    if strategy in {"contiguous", "random"}:
        chunks = np.array_split(features, n_parties)
    elif strategy == "round_robin":
        chunks = [features[idx::n_parties] for idx in range(n_parties)]
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")

    partitions = {party_id: [int(value) for value in chunk.tolist()] for party_id, chunk in enumerate(chunks)}
    if not partitions[active_party_id]:
        raise ValueError("The active party must receive at least one feature")
    return partitions
