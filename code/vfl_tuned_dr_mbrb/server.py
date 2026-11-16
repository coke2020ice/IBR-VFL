from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .belief import EPS, er_fuse, normalize_rows
from .config import ServerReasoningConfig


@dataclass
class ServerReasoningState:
    base_belief: np.ndarray
    final_belief: np.ndarray
    client_weights: np.ndarray
    sample_client_weights: np.ndarray
    confidence: np.ndarray
    disagreement: np.ndarray
    difficulty: np.ndarray
    difficult_mask: np.ndarray


class VFLBeliefServer:
    def __init__(self, config: ServerReasoningConfig) -> None:
        self.config = config
        self.client_weights_: np.ndarray | None = None
        self.train_difficulty_mean_: float = 0.0
        self.train_difficulty_std_: float = 1.0
        self.last_state_: ServerReasoningState | None = None

    def fit(
        self,
        client_beliefs: Sequence[np.ndarray],
        y: Sequence[object] | None = None,
        classes: np.ndarray | None = None,
    ) -> "VFLBeliefServer":
        stack = self._stack(client_beliefs)
        self.client_weights_ = self._fit_client_weights(stack, y=y, classes=classes)
        base = self._fuse(stack, np.broadcast_to(self.client_weights_[None, :], stack.shape[:2]))
        difficulty = self._difficulty_from_reference(stack, base)[3]
        self.train_difficulty_mean_ = float(np.mean(difficulty))
        self.train_difficulty_std_ = float(np.std(difficulty)) or 1.0
        return self

    def reason(self, client_beliefs: Sequence[np.ndarray], update_last: bool = True) -> ServerReasoningState:
        if self.client_weights_ is None:
            raise ValueError("The server has not been fitted")
        stack = self._stack(client_beliefs)
        base = self._fuse(stack, np.broadcast_to(self.client_weights_[None, :], stack.shape[:2]))
        reference = base
        final = base.copy()
        sample_weights = np.broadcast_to(self.client_weights_[None, :], stack.shape[:2]).copy()
        confidence, per_client_gap, disagreement, difficulty = self._difficulty_from_reference(stack, reference)
        difficult = self._difficult_mask(difficulty)

        for _ in range(max(1, int(self.config.n_reasoning_steps))):
            confidence, per_client_gap, disagreement, difficulty = self._difficulty_from_reference(stack, reference)
            difficult = self._difficult_mask(difficulty)
            sample_weights = self._sample_weights(stack, per_client_gap)
            corrected = self._fuse(stack, sample_weights)
            final = base.copy()
            final[difficult] = corrected[difficult]
            reference = corrected

        state = ServerReasoningState(
            base_belief=base,
            final_belief=final,
            client_weights=self.client_weights_.copy(),
            sample_client_weights=sample_weights,
            confidence=confidence,
            disagreement=disagreement,
            difficulty=difficulty,
            difficult_mask=difficult,
        )
        if update_last:
            self.last_state_ = state
        return state

    def _fit_client_weights(
        self,
        stack: np.ndarray,
        y: Sequence[object] | None = None,
        classes: np.ndarray | None = None,
    ) -> np.ndarray:
        method = self.config.client_weight_method
        if method == "confidence":
            raw = np.mean(np.max(stack, axis=2), axis=0)
        elif method in {"label_accuracy", "label_log_loss"}:
            if y is None or classes is None:
                raise ValueError(f"{method} client weighting requires y and classes")
            y_arr = np.asarray(y, dtype=object)
            class_to_idx = {label: idx for idx, label in enumerate(np.asarray(classes, dtype=object))}
            if any(label not in class_to_idx for label in y_arr):
                raise ValueError("All labels must be present in classes")
            target = np.asarray([class_to_idx[label] for label in y_arr], dtype=int)
            if method == "label_accuracy":
                raw = np.mean(np.argmax(stack, axis=2) == target[:, None], axis=0)
            else:
                true_prob = np.take_along_axis(stack, target[:, None, None], axis=2)[:, :, 0]
                raw = np.exp(-np.mean(-np.log(true_prob + EPS), axis=0))
        else:
            raise ValueError(f"Unknown client_weight_method: {method}")
        raw = np.power(np.maximum(raw, EPS), self.config.confidence_power)
        return raw / max(float(np.sum(raw)), EPS)

    def _difficulty_from_reference(self, stack: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        confidence = np.max(reference, axis=1)
        per_client_gap = np.sum(np.abs(stack - reference[:, None, :]), axis=2)
        disagreement = np.mean(per_client_gap, axis=1)
        difficulty = (1.0 - confidence) + self.config.lambda_d * disagreement / 2.0
        return confidence, per_client_gap, disagreement, difficulty

    def _difficult_mask(self, difficulty: np.ndarray) -> np.ndarray:
        if self.config.use_train_difficulty_stats:
            threshold = self.train_difficulty_mean_ + self.config.kappa * self.train_difficulty_std_
        else:
            threshold = float(np.mean(difficulty) + self.config.kappa * (np.std(difficulty) or 1.0))
        return difficulty > threshold

    def _sample_weights(self, stack: np.ndarray, per_client_gap: np.ndarray) -> np.ndarray:
        local_confidence = np.max(stack, axis=2)
        logits = (
            np.log(self.client_weights_[None, :] + EPS)
            + self.config.eta_c * local_confidence
            - self.config.eta_d * per_client_gap
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        sample_weights = np.exp(logits)
        return sample_weights / np.maximum(np.sum(sample_weights, axis=1, keepdims=True), EPS)

    @staticmethod
    def _stack(client_beliefs: Sequence[np.ndarray]) -> np.ndarray:
        if not client_beliefs:
            raise ValueError("At least one client belief array is required")
        stack = np.stack([np.asarray(item, dtype=float) for item in client_beliefs], axis=1)
        if stack.ndim != 3:
            raise ValueError("Client beliefs must stack into shape (n_samples, n_clients, n_classes)")
        return stack

    def _fuse(self, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self.config.client_fusion_method == "mean":
            return normalize_rows(np.sum(stack * weights[:, :, None], axis=1))
        if self.config.client_fusion_method == "er":
            return er_fuse(stack, weights)
        raise ValueError(f"Unknown client_fusion_method: {self.config.client_fusion_method}")
