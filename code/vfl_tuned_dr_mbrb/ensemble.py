from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .belief import EPS, er_fuse, normalize_rows, softmax
from .config import VFLTunedDRMBRBConfig
from .data import make_vertical_partitions
from .model import VFLTunedDRMBRB


@dataclass
class BaggedMBRBConfig:
    n_estimators: int = 5
    sample_fraction: float = 0.9
    feature_fraction: float = 0.8
    bootstrap: bool = True
    fusion_method: str = "mean"
    random_state: int = 13


class VFLBaggedDRMBRB:
    """Random-subspace bagging ensemble for VFL-Tuned DR-MBRB.

    Each base learner still obeys the vertical feature ownership boundary:
    row sampling is shared across parties, and feature sub-sampling is performed
    inside each party's owned feature slice.
    """

    def __init__(
        self,
        base_config: VFLTunedDRMBRBConfig | None = None,
        ensemble_config: BaggedMBRBConfig | None = None,
    ) -> None:
        self.base_config = base_config or VFLTunedDRMBRBConfig()
        self.ensemble_config = ensemble_config or BaggedMBRBConfig(random_state=self.base_config.random_state)
        self.estimators_: List[VFLTunedDRMBRB] = []
        self.estimator_party_features_: List[Dict[int, List[int]]] = []
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[object],
        party_features: Mapping[int, Sequence[int]] | None = None,
    ) -> "VFLBaggedDRMBRB":
        arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        if arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if party_features is None:
            party_features = make_vertical_partitions(
                n_features=arr.shape[1],
                n_parties=self.base_config.n_parties,
                active_party_id=self.base_config.active_party_id,
                strategy="contiguous",
                random_state=self.base_config.random_state,
            )
        base_party_features = {int(key): [int(value) for value in values] for key, values in party_features.items()}

        self.classes_ = np.unique(y_arr)
        self.estimators_.clear()
        self.estimator_party_features_.clear()
        rng = np.random.default_rng(self.ensemble_config.random_state)
        for estimator_idx in range(max(1, int(self.ensemble_config.n_estimators))):
            sample_idx = self._sample_rows(y_arr, rng)
            subspace = self._sample_party_features(base_party_features, rng)
            config = copy.deepcopy(self.base_config)
            config.random_state = int(self.base_config.random_state + 1009 * estimator_idx)
            model = VFLTunedDRMBRB(config)
            model.fit(arr[sample_idx], y_arr[sample_idx], party_features=subspace)
            self.estimators_.append(model)
            self.estimator_party_features_.append(subspace)
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if not self.estimators_ or self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        arr = np.asarray(X, dtype=float)
        aligned = []
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        for model in self.estimators_:
            proba = model.predict_proba(arr)
            model_aligned = np.zeros((arr.shape[0], len(self.classes_)), dtype=float)
            for source_idx, label in enumerate(model.classes_):
                if label in class_to_idx:
                    model_aligned[:, class_to_idx[label]] = proba[:, source_idx]
            aligned.append(normalize_rows(model_aligned))
        stack = np.stack(aligned, axis=1)
        if self.ensemble_config.fusion_method == "mean":
            return normalize_rows(np.mean(stack, axis=1))
        if self.ensemble_config.fusion_method == "er":
            weights = np.full(stack.shape[:2], 1.0 / max(stack.shape[1], EPS), dtype=float)
            return er_fuse(stack, weights)
        raise ValueError(f"Unknown ensemble fusion method: {self.ensemble_config.fusion_method}")

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def diagnostics(self) -> Dict[str, float]:
        if not self.estimators_:
            raise ValueError("The ensemble has not been fitted")
        feature_counts = [sum(len(values) for values in features.values()) for features in self.estimator_party_features_]
        return {
            "ensemble_estimators": float(len(self.estimators_)),
            "ensemble_mean_feature_count": float(np.mean(feature_counts)),
            "ensemble_min_feature_count": float(np.min(feature_counts)),
            "ensemble_max_feature_count": float(np.max(feature_counts)),
        }

    def _sample_rows(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        indices = []
        fraction = float(np.clip(self.ensemble_config.sample_fraction, EPS, 1.0))
        for label in np.unique(y):
            label_idx = np.flatnonzero(y == label)
            n_draw = max(1, int(round(len(label_idx) * fraction)))
            if self.ensemble_config.bootstrap:
                chosen = rng.choice(label_idx, size=n_draw, replace=True)
            else:
                chosen = rng.choice(label_idx, size=n_draw, replace=False)
            indices.extend(chosen.tolist())
        sampled = np.asarray(indices, dtype=int)
        rng.shuffle(sampled)
        return sampled

    def _sample_party_features(
        self,
        party_features: Mapping[int, Sequence[int]],
        rng: np.random.Generator,
    ) -> Dict[int, List[int]]:
        fraction = float(np.clip(self.ensemble_config.feature_fraction, EPS, 1.0))
        sampled: Dict[int, List[int]] = {}
        for party_id, features in party_features.items():
            feature_arr = np.asarray(list(features), dtype=int)
            n_keep = max(1, int(np.ceil(len(feature_arr) * fraction)))
            if n_keep >= len(feature_arr):
                sampled[int(party_id)] = feature_arr.tolist()
            else:
                sampled[int(party_id)] = sorted(rng.choice(feature_arr, size=n_keep, replace=False).tolist())
        return sampled


@dataclass
class BoostedMBRBConfig:
    n_estimators: int = 5
    sample_fraction: float = 1.0
    learning_rate: float = 0.35
    fusion_method: str = "log"
    weight_update_method: str = "hard"
    sampling_class_balance_power: float = 0.0
    class_conditioned_fusion: bool = False
    class_weight_floor: float = 0.05
    decision_threshold_method: str = "argmax"
    decision_threshold_prior_multiplier: float = 1.0
    random_state: int = 13


class VFLBoostedDRMBRB:
    """Training-only boosted ensemble for VFL-Tuned DR-MBRB.

    The ensemble follows the tree-boosting spirit without adding a validation
    split: each estimator is trained on a weighted bootstrap sample, then the
    next sample distribution emphasizes examples misclassified by the current
    estimator. Fusion uses the training weighted error of each estimator.
    """

    def __init__(
        self,
        base_config: VFLTunedDRMBRBConfig | None = None,
        ensemble_config: BoostedMBRBConfig | None = None,
    ) -> None:
        self.base_config = base_config or VFLTunedDRMBRBConfig()
        self.ensemble_config = ensemble_config or BoostedMBRBConfig(random_state=self.base_config.random_state)
        self.estimators_: List[VFLTunedDRMBRB] = []
        self.estimator_weights_: List[float] = []
        self.estimator_class_weights_: List[np.ndarray] = []
        self.estimator_errors_: List[float] = []
        self.classes_: np.ndarray | None = None
        self.decision_threshold_class_: int | None = None
        self.decision_threshold_: float | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[object],
        party_features: Mapping[int, Sequence[int]] | None = None,
    ) -> "VFLBoostedDRMBRB":
        arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        if arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if party_features is None:
            party_features = make_vertical_partitions(
                n_features=arr.shape[1],
                n_parties=self.base_config.n_parties,
                active_party_id=self.base_config.active_party_id,
                strategy="contiguous",
                random_state=self.base_config.random_state,
            )
        base_party_features = {int(key): [int(value) for value in values] for key, values in party_features.items()}

        self.classes_ = np.unique(y_arr)
        self.estimators_.clear()
        self.estimator_weights_.clear()
        self.estimator_class_weights_.clear()
        self.estimator_errors_.clear()
        self.decision_threshold_class_ = None
        self.decision_threshold_ = None
        rng = np.random.default_rng(self.ensemble_config.random_state)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[label] for label in y_arr], dtype=int)
        sample_weights = self._initial_sample_weights(y_idx, len(self.classes_))
        for estimator_idx in range(max(1, int(self.ensemble_config.n_estimators))):
            sample_idx = self._sample_rows(y_idx, sample_weights, rng)
            config = copy.deepcopy(self.base_config)
            config.random_state = int(self.base_config.random_state + 1009 * estimator_idx)
            model = VFLTunedDRMBRB(config)
            model.fit(arr[sample_idx], y_arr[sample_idx], party_features=base_party_features)

            proba = self._align_model_proba(model, arr)
            pred_idx = np.argmax(proba, axis=1)
            incorrect = np.asarray(pred_idx != y_idx, dtype=float)
            weighted_error = float(np.sum(sample_weights * incorrect))
            estimator_weight = self._estimator_weight(weighted_error, len(self.classes_))
            class_weights = self._class_estimator_weights(y_idx, proba, sample_weights, estimator_weight)

            self.estimators_.append(model)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_class_weights_.append(class_weights)
            self.estimator_errors_.append(weighted_error)
            sample_weights = self._updated_sample_weights(sample_weights, incorrect, proba, y_idx, estimator_weight)
            sample_weights = sample_weights / max(float(np.sum(sample_weights)), EPS)
        self._fit_decision_threshold(arr, y_idx)
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if not self.estimators_ or self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        arr = np.asarray(X, dtype=float)
        aligned = [self._align_model_proba(model, arr) for model in self.estimators_]
        stack = np.stack(aligned, axis=1)
        weights = self._normalized_class_weights() if self.ensemble_config.class_conditioned_fusion else self._normalized_estimator_weights()
        if self.ensemble_config.fusion_method == "mean":
            if weights.ndim == 1:
                return normalize_rows(np.sum(stack * weights[None, :, None], axis=1))
            return normalize_rows(np.sum(stack * weights[None, :, :], axis=1))
        if self.ensemble_config.fusion_method == "log":
            if weights.ndim == 1:
                logits = np.sum(weights[None, :, None] * np.log(stack + EPS), axis=1)
            else:
                logits = np.sum(weights[None, :, :] * np.log(stack + EPS), axis=1)
            return softmax(logits)
        raise ValueError(f"Unknown boosted ensemble fusion method: {self.ensemble_config.fusion_method}")

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        proba = self.predict_proba(X)
        if self.decision_threshold_class_ is not None and self.decision_threshold_ is not None:
            target = int(self.decision_threshold_class_)
            other = 1 - target
            pred_idx = np.full(proba.shape[0], other, dtype=int)
            pred_idx[proba[:, target] >= float(self.decision_threshold_)] = target
            return self.classes_[pred_idx]
        return self.classes_[np.argmax(proba, axis=1)]

    def diagnostics(self) -> Dict[str, float]:
        if not self.estimators_:
            raise ValueError("The ensemble has not been fitted")
        weights = np.asarray(self.estimator_weights_, dtype=float)
        class_weights = np.asarray(self.estimator_class_weights_, dtype=float)
        errors = np.asarray(self.estimator_errors_, dtype=float)
        normalized = self._normalized_estimator_weights()
        report = {
            "boost_estimators": float(len(self.estimators_)),
            "boost_weight_mean": float(np.mean(weights)),
            "boost_weight_max": float(np.max(weights)),
            "boost_weight_entropy": float(
                -np.sum(normalized * np.log(normalized + EPS)) / max(np.log(len(normalized)), EPS)
            ),
            "boost_error_mean": float(np.mean(errors)),
            "boost_error_min": float(np.min(errors)),
            "boost_error_max": float(np.max(errors)),
            "boost_class_conditioned_fusion": float(self.ensemble_config.class_conditioned_fusion),
            "boost_sampling_class_balance_power": float(self.ensemble_config.sampling_class_balance_power),
            "boost_decision_threshold_enabled": float(self.decision_threshold_ is not None),
        }
        if self.decision_threshold_ is not None and self.decision_threshold_class_ is not None:
            report["boost_decision_threshold"] = float(self.decision_threshold_)
            report["boost_decision_threshold_class"] = float(self.decision_threshold_class_)
        if class_weights.ndim == 2 and class_weights.size:
            report["boost_class_weight_min"] = float(np.min(class_weights))
            report["boost_class_weight_max"] = float(np.max(class_weights))
            normalized_by_class = self._normalized_class_weights()
            report["boost_class_weight_entropy"] = float(
                np.mean(
                    [
                        -np.sum(normalized_by_class[:, idx] * np.log(normalized_by_class[:, idx] + EPS))
                        / max(np.log(normalized_by_class.shape[0]), EPS)
                        for idx in range(normalized_by_class.shape[1])
                    ]
                )
            )
        return report

    def _align_model_proba(self, model: VFLTunedDRMBRB, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        proba = model.predict_proba(X)
        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        for source_idx, label in enumerate(model.classes_):
            if label in class_to_idx:
                aligned[:, class_to_idx[label]] = proba[:, source_idx]
        return normalize_rows(aligned)

    def _fit_decision_threshold(self, X: np.ndarray, y_idx: np.ndarray) -> None:
        if self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        if self.ensemble_config.decision_threshold_method == "argmax":
            return
        if self.ensemble_config.decision_threshold_method != "train_prior":
            raise ValueError(f"Unknown decision threshold method: {self.ensemble_config.decision_threshold_method}")
        if len(self.classes_) != 2:
            return
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        minority_idx = int(np.argmin(counts))
        minority_prior = float(counts[minority_idx] / max(float(np.sum(counts)), EPS))
        target_rate = float(
            np.clip(
                minority_prior * float(self.ensemble_config.decision_threshold_prior_multiplier),
                EPS,
                1.0 - EPS,
            )
        )
        proba = self.predict_proba(X)
        threshold = float(np.quantile(proba[:, minority_idx], 1.0 - target_rate))
        self.decision_threshold_class_ = minority_idx
        self.decision_threshold_ = threshold

    def _initial_sample_weights(self, y_idx: np.ndarray, n_classes: int) -> np.ndarray:
        weights = np.full(y_idx.shape[0], 1.0 / max(y_idx.shape[0], 1), dtype=float)
        balance_power = float(np.clip(self.ensemble_config.sampling_class_balance_power, 0.0, 1.0))
        if balance_power <= 0:
            return weights
        counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        class_weights = np.power(np.max(counts) / np.maximum(counts, EPS), balance_power)
        weights = weights * class_weights[y_idx]
        return weights / max(float(np.sum(weights)), EPS)

    def _sample_rows(self, y_idx: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fraction = float(np.clip(self.ensemble_config.sample_fraction, EPS, 1.0))
        n_draw = max(len(np.unique(y_idx)), int(round(len(y_idx) * fraction)))
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), EPS)
        sampled = rng.choice(np.arange(len(y_idx)), size=n_draw, replace=True, p=weights)
        sampled_labels = set(y_idx[sampled].tolist())
        missing_labels = [label for label in np.unique(y_idx) if label not in sampled_labels]
        for offset, label in enumerate(missing_labels):
            label_idx = np.flatnonzero(y_idx == label)
            label_weights = weights[label_idx]
            label_weights = label_weights / max(float(np.sum(label_weights)), EPS)
            replacement = rng.choice(label_idx, size=1, replace=True, p=label_weights)[0]
            sampled[offset % len(sampled)] = replacement
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
        method = self.ensemble_config.weight_update_method
        if estimator_weight <= EPS:
            return current
        if method == "hard":
            return current * np.exp(estimator_weight * incorrect)
        if method == "residual":
            true_proba = np.clip(proba[np.arange(proba.shape[0]), y_idx], EPS, 1.0)
            residual = 1.0 - true_proba
            scaled = residual / max(float(np.average(residual, weights=current)), EPS)
            return current * np.exp(float(self.ensemble_config.learning_rate) * scaled)
        raise ValueError(f"Unknown boosted weight update method: {method}")

    def _class_estimator_weights(
        self,
        y_idx: np.ndarray,
        proba: np.ndarray,
        sample_weights: np.ndarray,
        estimator_weight: float,
    ) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("The ensemble has not been fitted")
        if estimator_weight <= EPS:
            return np.zeros(len(self.classes_), dtype=float)
        floor = float(np.clip(self.ensemble_config.class_weight_floor, EPS, 0.49))
        class_weights = np.zeros(len(self.classes_), dtype=float)
        for class_idx in range(len(self.classes_)):
            mask = y_idx == class_idx
            if not np.any(mask):
                class_weights[class_idx] = estimator_weight * floor
                continue
            class_mass = float(np.sum(sample_weights[mask]))
            avg_true_proba = float(
                np.sum(sample_weights[mask] * proba[mask, class_idx]) / max(class_mass, EPS)
            )
            class_weights[class_idx] = estimator_weight * max(avg_true_proba, floor)
        return class_weights

    def _estimator_weight(self, weighted_error: float, n_classes: int) -> float:
        error = float(np.clip(weighted_error, EPS, 1.0 - EPS))
        random_guess_error = 1.0 - 1.0 / max(float(n_classes), 2.0)
        if error >= random_guess_error:
            return 0.0
        class_term = 0.0 if n_classes <= 2 else np.log(float(n_classes - 1))
        return float(self.ensemble_config.learning_rate * max(0.0, np.log((1.0 - error) / error) + class_term))

    def _normalized_estimator_weights(self) -> np.ndarray:
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if weights.size == 0:
            raise ValueError("The ensemble has not been fitted")
        if float(np.sum(weights)) <= EPS:
            return np.full(weights.shape[0], 1.0 / weights.shape[0], dtype=float)
        return weights / np.sum(weights)

    def _normalized_class_weights(self) -> np.ndarray:
        class_weights = np.asarray(self.estimator_class_weights_, dtype=float)
        if class_weights.ndim != 2 or class_weights.shape[0] == 0:
            scalar = self._normalized_estimator_weights()
            if self.classes_ is None:
                raise ValueError("The ensemble has not been fitted")
            return np.tile(scalar[:, None], (1, len(self.classes_)))
        scalar_fallback = self._normalized_estimator_weights()
        normalized = np.zeros_like(class_weights, dtype=float)
        for class_idx in range(class_weights.shape[1]):
            total = float(np.sum(class_weights[:, class_idx]))
            if total <= EPS:
                normalized[:, class_idx] = scalar_fallback
            else:
                normalized[:, class_idx] = class_weights[:, class_idx] / total
        return normalized
