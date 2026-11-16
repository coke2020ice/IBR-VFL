from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .belief import EPS, accuracy_score, macro_f1_score, smooth_prior, softmax
from .config import VFLTunedDRMBRBConfig
from .data import make_vertical_partitions
from .party import PartyBelief, VFLParty
from .server import ServerReasoningState, VFLBeliefServer


@dataclass
class _FittedState:
    classes: np.ndarray
    class_prior: np.ndarray
    parties: List[VFLParty]
    server: VFLBeliefServer
    party_features: Dict[int, List[int]]
    train_party_beliefs: List[PartyBelief]
    final_la_mean_belief: np.ndarray | None
    final_la_enabled: bool
    final_la_gamma: float
    final_la_temperature: float
    final_la_blend: float
    final_la_apply_on_difficult_only: bool
    stacking_calibrator: Any | None = None


class VFLTunedDRMBRB:
    """VFL-Tuned DR-MBRB, implemented in a VFLAIR-like modular style.

    The model is deliberately scikit-style at the public API while preserving
    VFL roles internally:
    - parties own feature slices;
    - only the active party is allowed to use labels by default;
    - the server receives belief evidence, not raw features;
    - the active party performs outer validation-driven tuning.
    """

    def __init__(self, config: VFLTunedDRMBRBConfig | None = None) -> None:
        self.config = config or VFLTunedDRMBRBConfig()
        self.state_: _FittedState | None = None
        self.best_params_: Dict[str, Any] = {}
        self.best_validation_score_: float | None = None
        self.last_reasoning_state_: ServerReasoningState | None = None

    @property
    def classes_(self) -> np.ndarray:
        if self.state_ is None:
            raise ValueError("The model has not been fitted")
        return self.state_.classes

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[object],
        party_features: Mapping[int, Sequence[int]] | None = None,
    ) -> "VFLTunedDRMBRB":
        arr = self._validate_X(X)
        y_arr = np.asarray(y, dtype=object)
        if arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if party_features is None:
            party_features = make_vertical_partitions(
                n_features=arr.shape[1],
                n_parties=self.config.n_parties,
                active_party_id=self.config.active_party_id,
                strategy="contiguous",
                random_state=self.config.random_state,
            )
        party_features = {int(key): [int(value) for value in values] for key, values in party_features.items()}

        train_idx, val_idx = self._stratified_validation_split(y_arr)
        best_config = copy.deepcopy(self.config)
        best_params: Dict[str, Any] = {}
        best_score = -np.inf
        candidate_params = self._candidate_params()
        if val_idx.size > 0 and any(candidate_params):
            for params in candidate_params:
                candidate_config = self._config_with_overrides(self.config, params)
                state = self._fit_once(arr[train_idx], y_arr[train_idx], party_features, candidate_config)
                proba = self._predict_proba_from_state(arr[val_idx], state)
                pred = state.classes[np.argmax(proba, axis=1)]
                score = self._validation_score(y_arr[val_idx], pred, candidate_config.tuning.primary_metric)
                if score > best_score:
                    best_score = score
                    best_config = candidate_config
                    best_params = params
        else:
            best_score = float("nan")

        self.best_params_ = best_params
        self.best_validation_score_ = None if np.isnan(best_score) else float(best_score)
        self.config = best_config
        calibrator = None
        if best_config.stacking.enabled and val_idx.size > 0:
            stack_state = self._fit_once(arr[train_idx], y_arr[train_idx], party_features, best_config)
            stack_features = self._stacking_features(arr[val_idx], stack_state)
            if len(np.unique(y_arr[val_idx])) > 1:
                class_to_idx = {label: idx for idx, label in enumerate(stack_state.classes)}
                y_val_idx = np.asarray([class_to_idx[label] for label in y_arr[val_idx]], dtype=int)
                calibrator = LogisticRegression(
                    C=float(best_config.stacking.C),
                    class_weight=best_config.stacking.class_weight,
                    max_iter=int(best_config.stacking.max_iter),
                    random_state=best_config.random_state,
                )
                calibrator.fit(stack_features, y_val_idx)
        self.state_ = self._fit_once(arr, y_arr, party_features, best_config)
        self.state_.stacking_calibrator = calibrator
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        if self.state_ is None:
            raise ValueError("The model has not been fitted")
        return self._predict_proba_from_state(self._validate_X(X), self.state_)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def client_evidence(self, X: Sequence[Sequence[float]]) -> List[PartyBelief]:
        """Return client-level MBRB evidence packages for inspection/reporting."""

        if self.state_ is None:
            raise ValueError("The model has not been fitted")
        arr = self._validate_X(X)
        return [party.predict_belief(arr) for party in self.state_.parties]

    def score_report(self, X: Sequence[Sequence[float]], y: Sequence[object]) -> Dict[str, float]:
        pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": macro_f1_score(y, pred),
            **self.diagnostics(X),
        }

    def diagnostics(self, X: Sequence[Sequence[float]] | None = None) -> Dict[str, float]:
        if self.state_ is None:
            raise ValueError("The model has not been fitted")
        rule_count = 0
        module_count = 0
        for party in self.state_.parties:
            module_count += len(party.modules_)
            for module in party.modules_:
                rule_count += len(module.rules_)
        report = {
            "rule_count": float(rule_count),
            "client_mbrb_module_count": float(module_count),
            "client_weight_min": float(np.min(self.state_.server.client_weights_)),
            "client_weight_max": float(np.max(self.state_.server.client_weights_)),
            "server_n_reasoning_steps": float(self.state_.server.config.n_reasoning_steps),
            "server_client_weight_method": float(
                {"confidence": 0.0, "label_accuracy": 1.0, "label_log_loss": 2.0}.get(
                    self.state_.server.config.client_weight_method,
                    -1.0,
                )
            ),
            "final_la_enabled": float(self.state_.final_la_enabled),
            "final_la_gamma": float(self.state_.final_la_gamma),
            "final_la_temperature": float(self.state_.final_la_temperature),
            "final_la_blend": float(self.state_.final_la_blend),
            "final_la_apply_on_difficult_only": float(self.state_.final_la_apply_on_difficult_only),
            "stacking_enabled": float(self.state_.stacking_calibrator is not None),
        }
        if self.state_.final_la_mean_belief is not None:
            report["final_la_mean_min"] = float(np.min(self.state_.final_la_mean_belief))
            report["final_la_mean_max"] = float(np.max(self.state_.final_la_mean_belief))
        if X is not None:
            _ = self.predict_proba(X)
            if self.last_reasoning_state_ is not None:
                report["high_conflict_samples"] = float(np.sum(self.last_reasoning_state_.difficult_mask))
                report["mean_disagreement"] = float(np.mean(self.last_reasoning_state_.disagreement))
                report["mean_difficulty"] = float(np.mean(self.last_reasoning_state_.difficulty))
            active_counts = np.zeros(self._validate_X(X).shape[0], dtype=float)
            for party in self.state_.parties:
                active_counts += party.count_activated_rules(self._validate_X(X))
            report["avg_active_rules"] = float(np.mean(active_counts))
        return report

    def _fit_once(
        self,
        X: np.ndarray,
        y: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        config: VFLTunedDRMBRBConfig,
    ) -> _FittedState:
        classes = np.unique(y)
        prior = smooth_prior(y, classes, alpha=config.brb.beta_smoothing)
        parties: List[VFLParty] = []
        for party_id in sorted(party_features):
            party = VFLParty(
                party_id=party_id,
                feature_indices=party_features[party_id],
                is_active=party_id == config.active_party_id,
                config=config.brb,
                passive_label_mode=config.passive_label_mode,
                random_state=config.random_state + party_id,
            )
            party.fit(X, y, classes, prior)
            parties.append(party)
        train_beliefs = [party.predict_belief(X) for party in parties]
        server = VFLBeliefServer(config.server)
        server.fit([belief.belief for belief in train_beliefs], y=y, classes=classes)
        train_reasoning = server.reason([belief.belief for belief in train_beliefs], update_last=False)
        train_base = self._apply_decision_prior_correction(
            train_reasoning.final_belief,
            prior,
            config.server.decision_prior_correction,
        )
        final_la_enabled = bool(config.final_la.enabled)
        if config.final_la.prior_gate_enabled:
            final_la_enabled = final_la_enabled and float(np.min(prior)) >= float(config.final_la.min_class_prior_for_la)
        final_la_mean_belief = None
        if final_la_enabled:
            final_la_mean_belief = np.mean(train_base, axis=0)
            final_la_mean_belief = final_la_mean_belief / max(float(np.sum(final_la_mean_belief)), EPS)
        return _FittedState(
            classes=classes,
            class_prior=prior,
            parties=parties,
            server=server,
            party_features={int(key): [int(value) for value in values] for key, values in party_features.items()},
            train_party_beliefs=train_beliefs,
            final_la_mean_belief=final_la_mean_belief,
            final_la_enabled=final_la_enabled,
            final_la_gamma=float(config.final_la.gamma),
            final_la_temperature=float(config.final_la.temperature),
            final_la_blend=float(config.final_la.blend),
            final_la_apply_on_difficult_only=bool(config.final_la.apply_on_difficult_only),
        )

    def _predict_proba_from_state(self, X: np.ndarray, state: _FittedState) -> np.ndarray:
        proba, reasoning = self._base_predict_proba_from_state(X, state)
        if state is self.state_:
            self.last_reasoning_state_ = reasoning
        if state.stacking_calibrator is None:
            return proba
        return self._apply_stacking_calibrator(proba, reasoning, state)

    def _base_predict_proba_from_state(self, X: np.ndarray, state: _FittedState) -> tuple[np.ndarray, ServerReasoningState]:
        party_evidence = [party.predict_belief(X) for party in state.parties]
        reasoning = state.server.reason([evidence.belief for evidence in party_evidence])
        base = self._apply_decision_prior_correction(
            reasoning.final_belief,
            state.class_prior,
            state.server.config.decision_prior_correction,
        )
        return self._apply_final_la(base, state, reasoning.difficult_mask), reasoning

    def _stacking_features(self, X: np.ndarray, state: _FittedState) -> np.ndarray:
        proba, reasoning = self._base_predict_proba_from_state(X, state)
        return self._stacking_feature_matrix(proba, reasoning)

    @staticmethod
    def _stacking_feature_matrix(proba: np.ndarray, reasoning: ServerReasoningState) -> np.ndarray:
        clipped = np.clip(proba, EPS, 1.0 - EPS)
        logits = np.log(clipped) - np.log1p(-clipped)
        sample_weights = np.asarray(reasoning.sample_client_weights, dtype=float)
        scalar_features = np.column_stack(
            [
                reasoning.confidence,
                reasoning.disagreement,
                reasoning.difficulty,
                reasoning.difficult_mask.astype(float),
            ]
        )
        return np.column_stack([proba, logits, scalar_features, sample_weights])

    def _apply_stacking_calibrator(
        self,
        proba: np.ndarray,
        reasoning: ServerReasoningState,
        state: _FittedState,
    ) -> np.ndarray:
        if state.stacking_calibrator is None:
            return proba
        features = self._stacking_feature_matrix(proba, reasoning)
        calibrated = state.stacking_calibrator.predict_proba(features)
        aligned = np.zeros((calibrated.shape[0], len(state.classes)), dtype=float)
        class_to_index = {label: idx for idx, label in enumerate(state.classes)}
        for source_idx, label in enumerate(state.stacking_calibrator.classes_):
            if isinstance(label, (int, np.integer)) and 0 <= int(label) < len(state.classes):
                aligned[:, int(label)] = calibrated[:, source_idx]
            elif label in class_to_index:
                aligned[:, class_to_index[label]] = calibrated[:, source_idx]
        return self._normalize_probability(aligned)

    def _apply_final_la(
        self,
        proba: np.ndarray,
        state: _FittedState,
        difficult_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            not state.final_la_enabled
            or state.final_la_mean_belief is None
            or float(state.final_la_gamma) <= 0.0
        ):
            return proba
        logits = (
            np.log(proba + EPS)
            - float(state.final_la_gamma) * np.log(state.final_la_mean_belief[None, :] + EPS)
        ) / max(float(state.final_la_temperature), EPS)
        adjusted = softmax(logits)
        blend = float(np.clip(state.final_la_blend, 0.0, 1.0))
        mixed = self._normalize_probability((1.0 - blend) * proba + blend * adjusted)
        if state.final_la_apply_on_difficult_only:
            if difficult_mask is None:
                return proba
            gated = proba.copy()
            gated[np.asarray(difficult_mask, dtype=bool)] = mixed[np.asarray(difficult_mask, dtype=bool)]
            return gated
        return mixed

    @staticmethod
    def _apply_decision_prior_correction(proba: np.ndarray, prior: np.ndarray, correction: float) -> np.ndarray:
        correction = float(correction)
        if correction <= 0:
            return proba
        logits = np.log(proba + EPS) - correction * np.log(prior[None, :] + EPS)
        return softmax(logits)

    @staticmethod
    def _normalize_probability(proba: np.ndarray) -> np.ndarray:
        total = np.sum(proba, axis=1, keepdims=True)
        return proba / np.maximum(total, EPS)

    def _candidate_params(self) -> List[Dict[str, Any]]:
        grid = copy.deepcopy(self.config.tuning.grid)
        if self.config.final_la.enabled and self.config.final_la.tune_on_validation:
            for key, values in self.config.final_la.grid.items():
                grid[f"final_la_{key}"] = values
        names = list(grid.keys())
        values = [grid[name] for name in names]
        combos = [dict(zip(names, combo)) for combo in itertools.product(*values)]
        max_trials = self.config.tuning.max_trials
        if max_trials is not None:
            combos = combos[: max(1, int(max_trials))]
        return combos

    @staticmethod
    def _config_with_overrides(base: VFLTunedDRMBRBConfig, params: Mapping[str, Any]) -> VFLTunedDRMBRBConfig:
        config = copy.deepcopy(base)
        for key, value in params.items():
            if hasattr(config.brb, key):
                setattr(config.brb, key, value)
            elif hasattr(config.server, key):
                setattr(config.server, key, value)
            elif key.startswith("final_la_") and hasattr(config.final_la, key.removeprefix("final_la_")):
                setattr(config.final_la, key.removeprefix("final_la_"), value)
            elif hasattr(config.final_la, key):
                setattr(config.final_la, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown tuning parameter: {key}")
        return config

    def _stratified_validation_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ratio = float(self.config.tuning.validation_ratio)
        if ratio <= 0.0 or len(y) < 5:
            return np.arange(len(y)), np.array([], dtype=int)
        rng = np.random.default_rng(self.config.random_state)
        train_idx = []
        val_idx = []
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * ratio)))
            if n_val >= len(indices):
                n_val = max(0, len(indices) - 1)
            val_idx.extend(indices[:n_val].tolist())
            train_idx.extend(indices[n_val:].tolist())
        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    @staticmethod
    def _validation_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        y_true_arr = np.asarray(y_true, dtype=object)
        y_pred_arr = np.asarray(y_pred, dtype=object)
        metric_name = metric.lower()
        if metric_name in {"macro_f1", "f1_macro"}:
            return macro_f1_score(y_true_arr, y_pred_arr)
        if metric_name in {"accuracy", "acc"}:
            return accuracy_score(y_true_arr, y_pred_arr)
        labels, counts = np.unique(y_true_arr, return_counts=True)
        if labels.size == 0:
            return 0.0
        if metric_name in {"balanced_accuracy", "bal_acc"}:
            recalls = []
            for label in labels:
                positives = y_true_arr == label
                recalls.append(float(np.mean(y_pred_arr[positives] == label)) if np.any(positives) else 0.0)
            return float(np.mean(recalls))
        minority_label = labels[int(np.argmin(counts))]
        tp = float(np.sum((y_true_arr == minority_label) & (y_pred_arr == minority_label)))
        fp = float(np.sum((y_true_arr != minority_label) & (y_pred_arr == minority_label)))
        fn = float(np.sum((y_true_arr == minority_label) & (y_pred_arr != minority_label)))
        if metric_name in {"minority_recall", "minority_rec"}:
            return tp / max(tp + fn, EPS)
        if metric_name in {"minority_precision", "minority_prec"}:
            return tp / max(tp + fp, EPS)
        if metric_name in {"minority_f1", "minority_f1_score"}:
            precision = tp / max(tp + fp, EPS)
            recall = tp / max(tp + fn, EPS)
            return 2.0 * precision * recall / max(precision + recall, EPS)
        raise ValueError(f"Unknown tuning primary_metric: {metric}")

    @staticmethod
    def _validate_X(X: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        return arr
