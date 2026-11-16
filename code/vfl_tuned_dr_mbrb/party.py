from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.cluster import KMeans

from .belief import EPS, er_fuse, normalize_rows, sigmoid, softmax
from .config import BRBModuleConfig, PassiveLabelMode


@dataclass
class LocalRule:
    center: np.ndarray
    scale: np.ndarray
    theta: float
    beta: np.ndarray
    support: float
    source: str


@dataclass
class PartyBelief:
    party_id: int
    belief: np.ndarray
    module_beliefs: np.ndarray
    raw_module_beliefs: np.ndarray
    la_module_beliefs: np.ndarray
    module_weights: np.ndarray
    feature_groups: List[List[int]]
    rule_counts: List[int]


class LocalBRBModule:
    """Local BRB submodule inside a client-level MBRB.

    Active-party modules can calibrate rule consequents with labels. Passive
    modules default to unsupervised consequents based on class priors and local
    prototype orientation, so labels never enter passive-party fitting unless
    simulation_supervised mode is explicitly enabled.
    """

    def __init__(self, config: BRBModuleConfig, random_state: int | None = None) -> None:
        self.config = config
        self.random_state = random_state
        self.rules_: List[LocalRule] = []
        self.classes_: np.ndarray | None = None
        self.train_mean_belief_: np.ndarray | None = None
        self.global_center_: np.ndarray | None = None
        self.global_scale_: np.ndarray | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[object] | None,
        classes: np.ndarray,
        class_prior: np.ndarray,
        supervised: bool,
    ) -> "LocalBRBModule":
        arr = self._validate_X(X)
        self.classes_ = np.asarray(classes, dtype=object)
        self.global_center_ = np.mean(arr, axis=0)
        self.global_scale_ = np.std(arr, axis=0)
        self.global_scale_[self.global_scale_ <= EPS] = 1.0

        if supervised and y is not None:
            self.rules_ = self._fit_supervised_rules(arr, np.asarray(y, dtype=object), class_prior)
        else:
            self.rules_ = self._fit_unsupervised_rules(arr, class_prior)

        raw = self.predict_raw_belief(arr)
        self.train_mean_belief_ = np.mean(raw, axis=0)
        self.train_mean_belief_ = self.train_mean_belief_ / max(float(np.sum(self.train_mean_belief_)), EPS)
        return self

    def predict_raw_belief(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        return self._predict_raw(self._validate_X(X))

    def apply_logit_la(self, raw_belief: np.ndarray) -> np.ndarray:
        if self.train_mean_belief_ is None:
            raise ValueError("The local BRB module has not been fitted")
        raw = np.asarray(raw_belief, dtype=float)
        if raw.ndim != 2:
            raise ValueError("raw_belief must be a two-dimensional belief matrix")
        logits = (
            np.log(raw + EPS)
            - self.config.gamma_la * np.log(self.train_mean_belief_[None, :] + EPS)
        ) / max(self.config.temperature_la, EPS)
        return softmax(logits)

    def predict_belief(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if self.train_mean_belief_ is None:
            raise ValueError("The local BRB module has not been fitted")
        return self.apply_logit_la(self.predict_raw_belief(X))

    def count_activated_rules(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        activations = self._rule_activations(self._validate_X(X))
        return np.sum(activations > self.config.activation_threshold, axis=1).astype(int)

    def _fit_supervised_rules(self, X: np.ndarray, y: np.ndarray, class_prior: np.ndarray) -> List[LocalRule]:
        rules: List[LocalRule] = []
        class_counts = np.array([np.sum(y == cls) for cls in self.classes_], dtype=float)
        if self.config.class_balance_power > 0:
            class_weights = np.power(np.max(class_counts) / np.maximum(class_counts, EPS), self.config.class_balance_power)
        else:
            class_weights = np.ones_like(class_counts, dtype=float)
        for class_index, label in enumerate(self.classes_):
            class_X = X[y == label]
            if class_X.size == 0:
                continue
            centers = self._select_prototypes(class_X, self.config.n_rules_per_class)
            for center in centers:
                scale = self._rule_scale(class_X, center)
                membership = self._single_membership(X, center, scale)
                mass = np.array([np.sum(membership[y == cls]) for cls in self.classes_], dtype=float)
                mass = mass * class_weights
                mass = mass + self.config.beta_smoothing * class_prior
                beta = mass / max(float(np.sum(mass)), EPS)
                theta = float(np.power(np.sum(membership) + EPS, 1.0 / max(self.config.theta_temperature, EPS)))
                rules.append(LocalRule(center=center, scale=scale, theta=theta, beta=beta, support=float(np.sum(membership)), source=f"class:{label}"))
        return rules or self._fit_unsupervised_rules(X, class_prior)

    def _fit_unsupervised_rules(self, X: np.ndarray, class_prior: np.ndarray) -> List[LocalRule]:
        centers = self._select_prototypes(X, self.config.n_rules_per_module)
        rules = []
        for idx, center in enumerate(centers):
            scale = self._rule_scale(X, center)
            beta = np.asarray(class_prior, dtype=float).copy()
            if len(beta) == 2 and self.global_center_ is not None and self.global_scale_ is not None:
                orientation = float(np.mean((center - self.global_center_) / self.global_scale_))
                p_pos = float(sigmoid(orientation))
                oriented = np.array([1.0 - p_pos, p_pos], dtype=float)
                strength = float(np.clip(self.config.unsupervised_orientation_strength, 0.0, 1.0))
                beta = normalize_rows(((1.0 - strength) * beta + strength * oriented)[None, :])[0]
            support = float(np.sum(self._single_membership(X, center, scale)))
            theta = float(np.power(support + EPS, 1.0 / max(self.config.theta_temperature, EPS)))
            rules.append(LocalRule(center=center, scale=scale, theta=theta, beta=beta, support=support, source=f"prototype:{idx}"))
        return rules

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        activations = self._rule_activations(X)
        betas = np.stack([rule.beta for rule in self.rules_], axis=0)
        evidence = np.broadcast_to(betas[None, :, :], (X.shape[0], betas.shape[0], betas.shape[1]))
        return er_fuse(evidence, activations)

    def _rule_activations(self, X: np.ndarray) -> np.ndarray:
        if not self.rules_:
            raise ValueError("The local BRB module has no rules")
        memberships = []
        for rule in self.rules_:
            memberships.append(self._single_membership(X, rule.center, rule.scale) * rule.theta)
        raw = np.stack(memberships, axis=1)
        return normalize_rows(raw)

    @staticmethod
    def _single_membership(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        dist = ((X - center[None, :]) / np.maximum(scale[None, :], EPS)) ** 2
        return np.exp(-0.5 * np.mean(dist, axis=1))

    def _select_prototypes(self, X: np.ndarray, requested: int) -> np.ndarray:
        n_rules = max(1, min(int(requested), X.shape[0]))
        if X.shape[0] == 1:
            return X.copy()
        if self.config.prototype_method == "quantile":
            return self._select_quantile(X, n_rules)
        if self.config.prototype_method == "centroid":
            return self._select_centroid(X, n_rules)
        unique_X = np.unique(X, axis=0)
        if unique_X.shape[0] <= n_rules:
            return self._fill_prototypes(unique_X, n_rules)
        if self.config.prototype_method == "kmeans" and unique_X.shape[0] > n_rules:
            model = KMeans(n_clusters=n_rules, n_init=5, random_state=self.random_state)
            model.fit(unique_X)
            return np.asarray(model.cluster_centers_, dtype=float)
        if self.config.prototype_method == "farthest":
            return self._select_farthest(unique_X, n_rules)
        if self.config.prototype_method == "hybrid":
            return self._select_hybrid(unique_X, n_rules)
        raise ValueError(f"Unknown prototype_method: {self.config.prototype_method}")

    def _select_quantile(self, X: np.ndarray, n_rules: int) -> np.ndarray:
        z = (X - np.mean(X, axis=0, keepdims=True)) / np.maximum(np.std(X, axis=0, keepdims=True), EPS)
        scores = np.mean(z, axis=1)
        order = np.argsort(scores)
        chosen = np.linspace(0, len(order) - 1, n_rules).round().astype(int)
        return X[order[chosen]].copy()

    def _select_centroid(self, X: np.ndarray, n_rules: int) -> np.ndarray:
        z = (X - np.mean(X, axis=0, keepdims=True)) / np.maximum(np.std(X, axis=0, keepdims=True), EPS)
        scores = np.sum(z**2, axis=1)
        order = np.argsort(scores)
        chosen = np.linspace(0, len(order) - 1, n_rules).round().astype(int)
        return X[order[chosen]].copy()

    def _select_hybrid(self, X: np.ndarray, n_rules: int) -> np.ndarray:
        n_quantile = max(1, int(np.ceil(n_rules / 2)))
        quantile = self._select_quantile(X, n_quantile)
        remaining = n_rules - quantile.shape[0]
        if remaining <= 0:
            return quantile[:n_rules].copy()
        farthest = self._select_farthest(X, n_rules)
        centers = np.vstack([quantile, farthest])
        centers = np.unique(centers, axis=0)
        return self._fill_prototypes(centers, n_rules)

    def _select_farthest(self, X: np.ndarray, n_rules: int) -> np.ndarray:
        z = (X - np.mean(X, axis=0, keepdims=True)) / np.maximum(np.std(X, axis=0, keepdims=True), EPS)
        rng = np.random.default_rng(self.random_state)
        center = np.mean(z, axis=0, keepdims=True)
        first = int(np.argmin(np.sum((z - center) ** 2, axis=1)))
        selected = [first]
        min_dist = np.sum((z - z[first][None, :]) ** 2, axis=1)
        while len(selected) < n_rules:
            jitter = rng.uniform(0.0, EPS, size=min_dist.shape)
            next_idx = int(np.argmax(min_dist + jitter))
            if next_idx in selected:
                break
            selected.append(next_idx)
            min_dist = np.minimum(min_dist, np.sum((z - z[next_idx][None, :]) ** 2, axis=1))
        return self._fill_prototypes(X[np.asarray(selected, dtype=int)], n_rules)

    def _rule_scale(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        if self.config.rule_scale_method == "global":
            scale = self._local_scale(X)
        elif self.config.rule_scale_method == "local_knn":
            scale = self._local_knn_scale(X, center)
        else:
            raise ValueError(f"Unknown rule_scale_method: {self.config.rule_scale_method}")
        return scale * max(float(self.config.rule_scale_multiplier), EPS)

    def _local_knn_scale(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        if X.shape[0] <= 1:
            return self._local_scale(X)
        z_scale = self._local_scale(X)
        dist = np.sum(((X - center[None, :]) / np.maximum(z_scale[None, :], EPS)) ** 2, axis=1)
        k = max(2, min(int(self.config.rule_scale_neighbors), X.shape[0]))
        neighbors = X[np.argpartition(dist, k - 1)[:k]]
        scale = np.std(neighbors, axis=0)
        fallback = self._local_scale(X)
        scale[scale <= EPS] = fallback[scale <= EPS]
        scale[scale <= EPS] = 1.0
        return scale

    @staticmethod
    def _local_scale(X: np.ndarray) -> np.ndarray:
        scale = np.std(X, axis=0)
        scale[scale <= EPS] = 1.0
        return scale

    @staticmethod
    def _fill_prototypes(centers: np.ndarray, requested: int) -> np.ndarray:
        arr = np.asarray(centers, dtype=float)
        if arr.shape[0] == 0:
            raise ValueError("At least one prototype center is required")
        n_rules = max(1, int(requested))
        if arr.shape[0] >= n_rules:
            return arr[:n_rules].copy()
        repeats = int(np.ceil(n_rules / arr.shape[0]))
        return np.tile(arr, (repeats, 1))[:n_rules].copy()

    @staticmethod
    def _validate_X(X: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if arr.shape[1] == 0:
            raise ValueError("A BRB module needs at least one feature")
        return arr


class VFLParty:
    """One VFL client that owns a feature slice and trains a local MBRB.

    Each client first splits its local feature view into several BRB modules.
    Module raw belief is adjusted by logit-level LA, then the client performs
    client-level MBRB integration and returns a compact evidence package.
    """

    def __init__(
        self,
        party_id: int,
        feature_indices: Sequence[int],
        is_active: bool,
        config: BRBModuleConfig,
        passive_label_mode: PassiveLabelMode,
        random_state: int | None = None,
    ) -> None:
        self.party_id = party_id
        self.feature_indices = list(feature_indices)
        self.is_active = is_active
        self.config = config
        self.passive_label_mode = passive_label_mode
        self.random_state = random_state
        self.modules_: List[LocalBRBModule] = []
        self.module_feature_groups_: List[List[int]] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Sequence[object], classes: np.ndarray, class_prior: np.ndarray) -> "VFLParty":
        local_X = np.asarray(X[:, self.feature_indices], dtype=float)
        self.classes_ = np.asarray(classes, dtype=object)
        self.module_feature_groups_ = self._split_local_features(local_X.shape[1], self.config.n_modules_per_party)
        self.modules_.clear()
        supervised = self.is_active or self.passive_label_mode == "simulation_supervised"
        y_for_fit = y if supervised else None
        for module_index, local_features in enumerate(self.module_feature_groups_):
            module = LocalBRBModule(self.config, random_state=None if self.random_state is None else self.random_state + module_index)
            module.fit(local_X[:, local_features], y_for_fit, self.classes_, class_prior, supervised=supervised)
            self.modules_.append(module)
        return self

    def predict_belief(self, X: np.ndarray) -> PartyBelief:
        if not self.modules_:
            raise ValueError("The party has not been fitted")
        local_X = np.asarray(X[:, self.feature_indices], dtype=float)
        raw_module_beliefs = []
        la_module_beliefs = []
        rule_counts = []
        for module, local_features in zip(self.modules_, self.module_feature_groups_):
            module_X = local_X[:, local_features]
            raw_belief = module.predict_raw_belief(module_X)
            raw_module_beliefs.append(raw_belief)
            la_module_beliefs.append(module.apply_logit_la(raw_belief))
            rule_counts.append(len(module.rules_))
        raw_stacked = np.stack(raw_module_beliefs, axis=1)
        la_stacked = np.stack(la_module_beliefs, axis=1)

        # Eq. (14): sample-wise module integration weights from LA belief confidence.
        # Eq. (15): client-level MBRB integration in belief space.
        confidence = np.max(la_stacked, axis=2)
        module_weights = confidence / np.maximum(np.sum(confidence, axis=1, keepdims=True), EPS)
        belief = er_fuse(la_stacked, module_weights)
        global_feature_groups = [
            [self.feature_indices[local_idx] for local_idx in group]
            for group in self.module_feature_groups_
        ]
        return PartyBelief(
            party_id=self.party_id,
            belief=belief,
            module_beliefs=la_stacked,
            raw_module_beliefs=raw_stacked,
            la_module_beliefs=la_stacked,
            module_weights=module_weights,
            feature_groups=global_feature_groups,
            rule_counts=rule_counts,
        )

    def count_activated_rules(self, X: np.ndarray) -> np.ndarray:
        if not self.modules_:
            raise ValueError("The party has not been fitted")
        local_X = np.asarray(X[:, self.feature_indices], dtype=float)
        counts = []
        for module, local_features in zip(self.modules_, self.module_feature_groups_):
            counts.append(module.count_activated_rules(local_X[:, local_features]))
        return np.sum(np.vstack(counts), axis=0)

    @staticmethod
    def _split_local_features(n_features: int, n_modules: int) -> List[List[int]]:
        n_modules = max(1, min(n_modules, n_features))
        return [[int(value) for value in chunk.tolist()] for chunk in np.array_split(np.arange(n_features), n_modules) if len(chunk)]
