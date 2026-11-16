from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_multiclass_credit_score_experiment as mc  # noqa: E402
from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import active_weak_split  # noqa: E402


EPS = 1e-12


class PartyRFClient:
    """A simulated passive client that never exposes its local feature values."""

    def __init__(self, party_id: int, X_block: np.ndarray):
        self.party_id = int(party_id)
        self.X = np.asarray(X_block, dtype=float)
        self._pending: Dict[int, Tuple[int, float]] = {}
        self._splits: Dict[int, Tuple[int, float]] = {}
        self._next_candidate = 0
        self._next_split = 0

    def propose(
        self, indices: np.ndarray, rng: np.random.Generator, max_bins: int,
        local_features: Sequence[int] | None = None,
    ) -> List[Tuple[int, np.ndarray]]:
        if self.X.shape[1] == 0:
            return []
        if local_features is None:
            n_pick = max(1, int(np.sqrt(self.X.shape[1])))
            local_features = rng.choice(np.arange(self.X.shape[1]), size=min(n_pick, self.X.shape[1]), replace=False)
        proposals: List[Tuple[int, np.ndarray]] = []
        for local_feature in local_features:
            values = self.X[indices, local_feature]
            unique = np.unique(values)
            if unique.size <= 1:
                continue
            if unique.size <= max_bins:
                thresholds = (unique[:-1] + unique[1:]) / 2.0
            else:
                qs = np.linspace(0.0, 1.0, max_bins + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, qs))
            for threshold in thresholds:
                candidate_id = self._next_candidate
                self._next_candidate += 1
                self._pending[candidate_id] = (int(local_feature), float(threshold))
                proposals.append((candidate_id, values <= threshold))
        return proposals

    def commit(self, candidate_id: int) -> int:
        split_id = self._next_split
        self._next_split += 1
        self._splits[split_id] = self._pending[int(candidate_id)]
        return split_id

    def branch(self, split_id: int, row_indices: np.ndarray) -> np.ndarray:
        local_feature, threshold = self._splits[int(split_id)]
        return self.X[row_indices, local_feature] <= threshold


@dataclass
class Node:
    proba: np.ndarray
    party_id: int | None = None
    split_id: int | None = None
    left: "Node | None" = None
    right: "Node | None" = None


def class_weights(y_idx: np.ndarray, balanced: bool) -> np.ndarray:
    if not balanced:
        return np.ones(len(y_idx), dtype=float)
    counts = np.bincount(y_idx).astype(float)
    per_class = np.max(counts) / np.maximum(counts, 1.0)
    values = per_class[y_idx]
    return values / np.mean(values)


def weighted_counts(y_idx: np.ndarray, weights: np.ndarray, indices: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y_idx[indices], weights=weights[indices], minlength=n_classes).astype(float)


def impurity(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= EPS:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def build_node(
    clients: Mapping[int, PartyRFClient], y_idx: np.ndarray, weights: np.ndarray,
    indices: np.ndarray, rng: np.random.Generator, depth: int, max_depth: int,
    min_leaf: int, max_bins: int, n_classes: int, max_features_global: int,
) -> Node:
    counts = weighted_counts(y_idx, weights, indices, n_classes) + 1.0
    node = Node(proba=counts / np.sum(counts))
    if depth >= max_depth or indices.size < 2 * min_leaf:
        return node
    parent_counts = weighted_counts(y_idx, weights, indices, n_classes)
    parent_total = float(np.sum(parent_counts))
    parent_impurity = impurity(parent_counts)
    best = None
    feature_catalog = [
        (party_id, local_feature)
        for party_id, client in clients.items()
        for local_feature in range(client.X.shape[1])
    ]
    selected = rng.choice(
        np.arange(len(feature_catalog)),
        size=min(max_features_global, len(feature_catalog)),
        replace=False,
    )
    selected_by_party: Dict[int, List[int]] = {}
    for catalog_idx in selected:
        party_id, local_feature = feature_catalog[int(catalog_idx)]
        selected_by_party.setdefault(int(party_id), []).append(int(local_feature))
    for party_id, local_features in selected_by_party.items():
        client = clients[party_id]
        for candidate_id, left_mask in client.propose(indices, rng, max_bins, local_features):
            left = indices[left_mask]
            right = indices[~left_mask]
            if left.size < min_leaf or right.size < min_leaf:
                continue
            left_counts = weighted_counts(y_idx, weights, left, n_classes)
            right_counts = weighted_counts(y_idx, weights, right, n_classes)
            gain = parent_impurity
            gain -= (np.sum(left_counts) / max(parent_total, EPS)) * impurity(left_counts)
            gain -= (np.sum(right_counts) / max(parent_total, EPS)) * impurity(right_counts)
            if best is None or gain > best[0]:
                best = (float(gain), int(party_id), int(candidate_id), left, right)
    if best is None or best[0] <= 0.0:
        return node
    _, party_id, candidate_id, left, right = best
    node.party_id = party_id
    node.split_id = clients[party_id].commit(candidate_id)
    node.left = build_node(clients, y_idx, weights, left, rng, depth + 1, max_depth, min_leaf, max_bins, n_classes, max_features_global)
    node.right = build_node(clients, y_idx, weights, right, rng, depth + 1, max_depth, min_leaf, max_bins, n_classes, max_features_global)
    return node


class StrictVFLRF:
    def __init__(self, n_trees: int, max_depth: int, min_leaf_ratio: float, max_bins: int, balanced: bool, seed: int):
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_leaf_ratio = float(min_leaf_ratio)
        self.max_bins = int(max_bins)
        self.balanced = bool(balanced)
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray, party_features: Mapping[int, Sequence[int]]):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.clients_ = {
            int(pid): PartyRFClient(int(pid), np.nan_to_num(X[:, list(features)], nan=0.0, posinf=0.0, neginf=0.0))
            for pid, features in party_features.items()
        }
        weights = class_weights(y_idx, self.balanced)
        rng = np.random.default_rng(self.seed)
        all_indices = np.arange(len(y_idx))
        min_leaf = max(8, int(self.min_leaf_ratio * len(y_idx)))
        max_features_global = max(1, int(np.sqrt(sum(client.X.shape[1] for client in self.clients_.values()))))
        self.trees_ = []
        for _ in range(self.n_trees):
            bootstrap = rng.choice(all_indices, size=len(all_indices), replace=True)
            self.trees_.append(build_node(
                self.clients_, y_idx, weights, bootstrap, rng, 0, self.max_depth,
                min_leaf, self.max_bins, len(self.classes_), max_features_global,
            ))
        return self

    def _predict_node(self, node: Node, row: int) -> np.ndarray:
        current = node
        row_array = np.asarray([row], dtype=int)
        while current.party_id is not None and current.split_id is not None:
            go_left = bool(self.clients_[current.party_id].branch(current.split_id, row_array)[0])
            current = current.left if go_left else current.right
        return current.proba

    def predict(self, X: np.ndarray, party_features: Mapping[int, Sequence[int]]) -> np.ndarray:
        # Replace each client's training block by its local test block. The server still sees only branch decisions.
        for pid, features in party_features.items():
            self.clients_[int(pid)].X = np.nan_to_num(X[:, list(features)], nan=0.0, posinf=0.0, neginf=0.0)
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        for tree in self.trees_:
            for row in range(len(X)):
                proba[row] += self._predict_node(tree, row)
        proba /= len(self.trees_)
        return self.classes_[np.argmax(proba, axis=1)]


def summarize(rows: List[dict]) -> dict:
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    return {f"{key}_{suffix}": float(func([row[key] for row in rows])) for key in keys for suffix, func in (("mean", np.mean), ("std", np.std))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/strict_vfl_rf_credit_score_audit.json")
    parser.add_argument("--target-tail-proportions", default="0.03,0.02,0.01")
    args = parser.parse_args()
    X_raw, y_raw, metadata = mc.load_credit_score("data/credit_score_classification/train.csv")
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, 6000, 13)
    ratios = [float(v) for v in args.target_tail_proportions.split(",")]
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
    payload = {"dataset": metadata, "selection": "inner validation macro-F1, then accuracy", "results": {}}
    # Capacity-matched to the original RF baseline: 15 trees, depth 5,
    # min_leaf=max(8, 0.006*N), 16 bins, balanced weights, and sqrt(d)
    # globally sampled feature identifiers per node. Only the data interface changes.
    candidates = [(15, 5, 0.006, True)]
    for rho in ratios:
        rows = []
        for fold, (train_idx, test_idx) in enumerate(outer.split(X_raw, y_raw), start=1):
            fold_labels, fold_counts = np.unique(y_raw[train_idx], return_counts=True)
            tail_label = fold_labels[np.argmin(fold_counts)]
            X_train_raw, y_train = mc.multiclass_target_tail_resample(X_raw[train_idx], y_raw[train_idx], tail_label, rho, 13 + 100 * fold)
            X_test_raw, y_test = mc.multiclass_target_tail_resample(X_raw[test_idx], y_raw[test_idx], tail_label, rho, 13 + 1000 * fold)
            encoder = mc.TabularEncoder()
            X_train = encoder.fit_transform(X_train_raw)
            X_test = encoder.transform(X_test_raw)
            fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.2, random_state=13 + fold, stratify=y_train)
            inner_split = active_weak_split(X_train[fit_idx], y_train[fit_idx], n_parties=10, rank_strength=3.0, seed=13 + fold)
            best = None
            for n_trees, depth, leaf_ratio, balanced in candidates:
                model = StrictVFLRF(n_trees, depth, leaf_ratio, 16, balanced, 13 + fold)
                model.fit(X_train[fit_idx], y_train[fit_idx], inner_split.party_features)
                pred = model.predict(X_train[val_idx], inner_split.party_features)
                score = (f1_score(y_train[val_idx], pred, average="macro", zero_division=0), accuracy_score(y_train[val_idx], pred))
                if best is None or score > best[0]:
                    best = (score, (n_trees, depth, leaf_ratio, balanced))
            outer_split = active_weak_split(X_train, y_train, n_parties=10, rank_strength=3.0, seed=13 + fold)
            n_trees, depth, leaf_ratio, balanced = best[1]
            model = StrictVFLRF(n_trees, depth, leaf_ratio, 16, balanced, 13 + fold)
            model.fit(X_train, y_train, outer_split.party_features)
            pred = model.predict(X_test, outer_split.party_features)
            row = mc.metrics(y_test, pred)
            row.update({"fold": fold, "n_trees": n_trees, "max_depth": depth, "min_leaf_ratio": leaf_ratio, "balanced": float(balanced), "validation_macro_f1": best[0][0]})
            rows.append(row)
            print(f"rho={rho:.2f} fold={fold} Acc={row['accuracy']:.4f} TailF1={row['tail_f1']:.4f} cfg={best[1]}", flush=True)
        payload["results"][f"rho={rho:.2f}"] = {"folds": rows, "mean": summarize(rows)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
