from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_k10_active_weak_ablation import active_weak_split  # noqa: E402
from scripts.run_long_tail_ebrb_experiment import (  # noqa: E402
    long_tail_metrics,
    long_tail_resample,
    parse_list,
    summarize,
)
from vfl_tuned_dr_mbrb.belief import EPS, er_fuse, normalize_rows, softmax  # noqa: E402


@dataclass(frozen=True)
class SCERRobustnessConfig:
    name: str
    group: str
    n_parties: int = 10
    total_rule_budget: int = 72
    n_estimators: int = 9
    learning_rate: float = 0.25
    sample_fraction: float = 0.85
    top_rules_per_party: int = 4
    combo_budget: int = 64
    combo_gamma: float = 0.75
    combo_min_support: float = 0.001
    parameter_iterations: int = 2
    parameter_update_blend: float = 0.01
    threshold_prior_multiplier: float = 1.0
    class_balanced_selection: bool = False
    class_balance_power: float = 1.0
    sampling_class_balance_power: float = 0.75
    multiclass_min_support_fraction: float = 0.005
    combo_quota_prior_power: float = 0.0


@dataclass
class _SCERComboRule:
    estimator_idx: int
    left_party_id: int
    left_rule_idx: int
    right_party_id: int
    right_rule_idx: int
    beta: np.ndarray
    theta: float
    reliability: float
    support: float
    score: float
    dominant_class: int


class StrictVFLEBRBLTIterSCER:
    """Script-local server-composed evidence rule wrapper.

    The base learner is the existing Strict VFL-EBRB boost-LT-Iter. SCER adds
    server-side pairwise combinations of high-scoring local rules from distinct
    clients, then fuses the composed belief with the base ensemble belief.
    """

    def __init__(self, config: SCERRobustnessConfig, random_state: int) -> None:
        self.config = config
        self.random_state = int(random_state)
        self.rules_per_party = max(4, int(math.ceil(config.total_rule_budget / max(config.n_parties, 1))))
        self.model: Any | None = None
        self.combo_rules_: List[_SCERComboRule] = []
        self.classes_: np.ndarray | None = None
        self.decision_threshold_class_: int | None = None
        self.decision_threshold_: float | None = None
        self.combo_support_mean_: float = 0.0
        self.combo_score_mean_: float = 0.0
        self.combo_class_counts_: np.ndarray = np.asarray([], dtype=int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        party_features: Mapping[int, Sequence[int]],
        active_party_id: int,
    ) -> "StrictVFLEBRBLTIterSCER":
        is_multiclass = len(np.unique(np.asarray(y, dtype=object))) > 2
        self.model = exp._StrictVFLBoostedEBRB(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            sample_fraction=self.config.sample_fraction,
            n_rules_per_party=self.rules_per_party,
            bins=6,
            max_antecedents=2,
            beam_width=2 * self.rules_per_party,
            beta_smoothing=0.5,
            min_support_fraction=self.config.multiclass_min_support_fraction if is_multiclass else 0.02,
            class_balance_power=self.config.class_balance_power if is_multiclass else 0.5,
            parameter_iterations=self.config.parameter_iterations,
            parameter_update_blend=self.config.parameter_update_blend,
            sampling_class_balance_power=self.config.sampling_class_balance_power,
            client_weight_method="label_accuracy",
            fusion_method="log",
            weight_update_method="residual",
            decision_threshold_method="argmax",
            decision_threshold_prior_multiplier=1.0,
            server_logit_la_gamma=0.0,
            server_logit_la_temperature=1.0,
            server_logit_la_blend=0.0,
            server_class_conditioned_fusion=False,
            server_class_weight_power=1.0,
            client_fit_weight_power=0.0,
            random_state=self.random_state,
        )
        self.model.fit(X, y, party_features=party_features, active_party_id=active_party_id)
        self.classes_ = np.asarray(self.model.classes_, dtype=object)
        self._fit_combos(np.asarray(X, dtype=float), np.asarray(y, dtype=object))
        train_proba = self.predict_proba(X)
        self._fit_decision_threshold(train_proba, np.asarray(y, dtype=object))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("SCER model has not been fitted")
        base = self.model.predict_proba(X)
        if not self.combo_rules_ or self.config.combo_gamma <= 0:
            return base
        combo = self._combo_predict_proba(np.asarray(X, dtype=float))
        gamma = float(np.clip(self.config.combo_gamma, 0.0, 1.0))
        return softmax((1.0 - gamma) * np.log(base + EPS) + gamma * np.log(combo + EPS))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("SCER model has not been fitted")
        proba = self.predict_proba(X)
        if self.decision_threshold_class_ is not None and self.decision_threshold_ is not None and len(self.classes_) == 2:
            target = int(self.decision_threshold_class_)
            other = 1 - target
            pred_idx = np.full(proba.shape[0], other, dtype=int)
            pred_idx[proba[:, target] >= float(self.decision_threshold_)] = target
            return self.classes_[pred_idx]
        return self.classes_[np.argmax(proba, axis=1)]

    def diagnostics(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("SCER model has not been fitted")
        out = self.model.diagnostics()
        out.update(
            {
                "scer_combo_rule_count": float(len(self.combo_rules_)),
                "scer_top_rules_per_party": float(self.config.top_rules_per_party),
                "scer_combo_budget": float(self.config.combo_budget),
                "scer_combo_gamma": float(self.config.combo_gamma),
                "scer_combo_support_mean": float(self.combo_support_mean_),
                "scer_combo_score_mean": float(self.combo_score_mean_),
                "scer_n_parties": float(self.config.n_parties),
                "scer_total_rule_budget": float(self.config.total_rule_budget),
                "scer_rules_per_party": float(self.rules_per_party),
                "scer_threshold_prior_multiplier": float(self.config.threshold_prior_multiplier),
                "scer_class_balanced_selection": float(self.config.class_balanced_selection),
                "scer_class_balance_power": float(self.config.class_balance_power),
                "scer_sampling_class_balance_power": float(self.config.sampling_class_balance_power),
                "scer_multiclass_min_support_fraction": float(self.config.multiclass_min_support_fraction),
                "scer_combo_quota_prior_power": float(self.config.combo_quota_prior_power),
            }
        )
        for class_idx, count in enumerate(self.combo_class_counts_.tolist()):
            out[f"scer_combo_class_{class_idx}_count"] = float(count)
        if self.decision_threshold_ is not None:
            out["scer_decision_threshold"] = float(self.decision_threshold_)
            out["scer_decision_threshold_class"] = float(self.decision_threshold_class_ or 0)
        return out

    def _fit_combos(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.model is None or self.classes_ is None:
            raise ValueError("Base model has not been fitted")
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[label] for label in y], dtype=int)
        n_classes = len(self.classes_)
        counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        class_prior = (counts + 1.0) / max(float(np.sum(counts) + n_classes), EPS)
        balance_power = 1.0 if n_classes > 2 else 0.5
        class_weights = np.power(np.max(counts) / np.maximum(counts, EPS), balance_power)
        candidates: List[_SCERComboRule] = []
        for estimator_idx, estimator in enumerate(self._combo_estimators()):
            party_matches: Dict[int, np.ndarray] = {}
            party_top_rules: Dict[int, List[int]] = {}
            for party_id in sorted(estimator.parties_):
                party = estimator.parties_[party_id]
                features = estimator.party_features_[party_id]
                match = party._rule_match_matrix(X[:, features])
                party_matches[party_id] = match
                scored = [
                    (rule_idx, party.rules_[rule_idx].score, party.rules_[rule_idx].support)
                    for rule_idx in range(1, len(party.rules_))
                ]
                scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
                local_budget = max(1, self.config.top_rules_per_party)
                if self.config.class_balanced_selection and n_classes > 1:
                    local_groups: Dict[int, List[Tuple[int, float, float]]] = {
                        class_idx: [] for class_idx in range(n_classes)
                    }
                    for item in scored:
                        rule_idx = item[0]
                        dominant_class = int(np.argmax(party.rules_[rule_idx].beta))
                        local_groups[dominant_class].append(item)
                    local_base, local_remainder = divmod(local_budget, n_classes)
                    local_selected: List[Tuple[int, float, float]] = []
                    local_leftovers: List[Tuple[int, float, float]] = []
                    for class_idx in range(n_classes):
                        quota = local_base + (1 if class_idx < local_remainder else 0)
                        local_selected.extend(local_groups[class_idx][:quota])
                        local_leftovers.extend(local_groups[class_idx][quota:])
                    if len(local_selected) < local_budget:
                        local_leftovers.sort(key=lambda item: (item[1], item[2]), reverse=True)
                        local_selected.extend(local_leftovers[: local_budget - len(local_selected)])
                    party_top_rules[party_id] = [idx for idx, _, _ in local_selected[:local_budget]]
                else:
                    party_top_rules[party_id] = [idx for idx, _, _ in scored[:local_budget]]
            for left_party, right_party in itertools.combinations(sorted(party_top_rules), 2):
                for left_rule_idx in party_top_rules[left_party]:
                    for right_rule_idx in party_top_rules[right_party]:
                        activation = party_matches[left_party][:, left_rule_idx] * party_matches[right_party][:, right_rule_idx]
                        combo = self._candidate_from_activation(
                            estimator_idx=estimator_idx,
                            left_party_id=left_party,
                            left_rule_idx=left_rule_idx,
                            right_party_id=right_party,
                            right_rule_idx=right_rule_idx,
                            activation=activation,
                            y_idx=y_idx,
                            n_classes=n_classes,
                            class_prior=class_prior,
                            class_weights=class_weights,
                        )
                        if combo is not None:
                            candidates.append(combo)
        candidates.sort(key=lambda item: item.score, reverse=True)
        budget = max(0, int(self.config.combo_budget))
        if self.config.class_balanced_selection and n_classes > 1:
            # Allocate the finite SCER budget across consequent classes before
            # ranking within each class. power=0 gives equal quotas; increasing
            # the power smoothly restores more capacity to frequent classes
            # without allowing them to exhaust the entire budget.
            grouped = {
                class_idx: [rule for rule in candidates if rule.dominant_class == class_idx]
                for class_idx in range(n_classes)
            }
            quota_power = max(0.0, float(self.config.combo_quota_prior_power))
            quota_weights = np.power(np.maximum(class_prior, EPS), quota_power)
            quota_weights = quota_weights / max(float(np.sum(quota_weights)), EPS)
            raw_quotas = budget * quota_weights
            quotas = np.floor(raw_quotas).astype(int)
            remainder = budget - int(np.sum(quotas))
            if remainder > 0:
                order = np.argsort(-(raw_quotas - quotas))
                for class_idx in order[:remainder]:
                    quotas[int(class_idx)] += 1
            selected: List[_SCERComboRule] = []
            leftovers: List[_SCERComboRule] = []
            for class_idx in range(n_classes):
                quota = int(quotas[class_idx])
                selected.extend(grouped[class_idx][:quota])
                leftovers.extend(grouped[class_idx][quota:])
            if len(selected) < budget:
                leftovers.sort(key=lambda item: item.score, reverse=True)
                selected.extend(leftovers[: budget - len(selected)])
            self.combo_rules_ = selected[:budget]
        else:
            self.combo_rules_ = candidates[:budget]
        self.combo_class_counts_ = np.bincount(
            [rule.dominant_class for rule in self.combo_rules_], minlength=n_classes
        ).astype(int)
        if self.combo_rules_:
            self.combo_support_mean_ = float(np.mean([rule.support for rule in self.combo_rules_]))
            self.combo_score_mean_ = float(np.mean([rule.score for rule in self.combo_rules_]))
        else:
            self.combo_support_mean_ = 0.0
            self.combo_score_mean_ = 0.0

    def _candidate_from_activation(
        self,
        *,
        estimator_idx: int,
        left_party_id: int,
        left_rule_idx: int,
        right_party_id: int,
        right_rule_idx: int,
        activation: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        class_prior: np.ndarray,
        class_weights: np.ndarray,
    ) -> _SCERComboRule | None:
        support = float(np.mean(activation))
        if support < max(float(self.config.combo_min_support), EPS):
            return None
        mass = np.bincount(y_idx, weights=activation, minlength=n_classes).astype(float)
        observed = mass / max(float(np.sum(mass)), EPS)
        beta = mass * class_weights + 0.5 * class_prior
        beta = beta / max(float(np.sum(beta)), EPS)
        # The rule's consequent class must be defined from the same
        # class-balanced belief distribution that is used during inference.
        # Using the raw activation distribution here made multiclass SCER
        # allocate no rule budget to the tail class even when beta was
        # tail-dominant after long-tail correction.
        dominant = int(np.argmax(beta))
        purity = float(beta[dominant])
        lift = max(0.0, float(beta[dominant] - class_prior[dominant]))
        gini_prior = 1.0 - float(np.sum(class_prior**2))
        gini_combo = 1.0 - float(np.sum(beta**2))
        gain = max(0.0, gini_prior - gini_combo) * support
        score = float(gain + 0.45 * lift * math.sqrt(max(support, EPS)) + 0.05 * purity)
        theta = float(np.clip(np.power(max(support, EPS), 0.25), 0.05, 1.0))
        reliability = float(np.clip(0.15 + 0.85 * purity * math.sqrt(max(support, EPS)), 0.05, 1.0))
        return _SCERComboRule(
            estimator_idx=estimator_idx,
            left_party_id=left_party_id,
            left_rule_idx=left_rule_idx,
            right_party_id=right_party_id,
            right_rule_idx=right_rule_idx,
            beta=beta,
            theta=theta,
            reliability=reliability,
            support=support,
            score=score,
            dominant_class=dominant,
        )

    def _combo_predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise ValueError("SCER model has not been fitted")
        if not self.combo_rules_:
            return self.model.predict_proba(X)
        activation = np.zeros((X.shape[0], len(self.combo_rules_)), dtype=float)
        betas = np.stack([rule.beta for rule in self.combo_rules_], axis=0)
        estimators = self._combo_estimators()
        for combo_idx, combo in enumerate(self.combo_rules_):
            estimator = estimators[combo.estimator_idx]
            left_party = estimator.parties_[combo.left_party_id]
            right_party = estimator.parties_[combo.right_party_id]
            left_features = estimator.party_features_[combo.left_party_id]
            right_features = estimator.party_features_[combo.right_party_id]
            left_match = left_party._rule_match_matrix(X[:, left_features])[:, combo.left_rule_idx]
            right_match = right_party._rule_match_matrix(X[:, right_features])[:, combo.right_rule_idx]
            activation[:, combo_idx] = (
                left_match
                * right_match
                * max(combo.theta, EPS)
                * max(combo.reliability, EPS)
            )
        row_sum = np.sum(activation, axis=1, keepdims=True)
        inactive = row_sum[:, 0] <= EPS
        weights = activation / np.maximum(row_sum, EPS)
        evidence = np.broadcast_to(betas[None, :, :], (X.shape[0], betas.shape[0], betas.shape[1]))
        combo_proba = er_fuse(evidence, weights)
        if np.any(inactive):
            combo_proba[inactive] = self.model.predict_proba(X[inactive])
        return normalize_rows(combo_proba)

    def _combo_estimators(self) -> List[Any]:
        if self.model is None:
            raise ValueError("SCER model has not been fitted")
        if hasattr(self.model, "estimators_"):
            return list(self.model.estimators_)
        return [self.model]

    def _fit_decision_threshold(self, train_proba: np.ndarray, y: np.ndarray) -> None:
        if self.classes_ is None or len(self.classes_) != 2:
            return
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.asarray([class_to_idx[label] for label in y], dtype=int)
        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(float)
        minority_idx = int(np.argmin(counts))
        minority_prior = float(counts[minority_idx] / max(float(np.sum(counts)), EPS))
        target_rate = float(
            np.clip(
                minority_prior * float(self.config.threshold_prior_multiplier),
                EPS,
                1.0 - EPS,
            )
        )
        self.decision_threshold_class_ = minority_idx
        self.decision_threshold_ = float(np.quantile(train_proba[:, minority_idx], 1.0 - target_rate))


def build_configurations(groups: Sequence[str], client_values: Sequence[int]) -> List[SCERRobustnessConfig]:
    base = SCERRobustnessConfig(name="default", group="default")
    configs: List[SCERRobustnessConfig] = []
    if "manuscript_parties" in groups:
        for n_parties in [2, 4, 6, 8, 10, 12]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"P{n_parties}", group="manuscript_parties", n_parties=n_parties
                )
            )
    if "manuscript_rules" in groups:
        for rules_per_party in [4, 8, 12, 16, 24, 32]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"L{rules_per_party}", group="manuscript_rules",
                    total_rule_budget=rules_per_party * 10,
                )
            )
    if "manuscript_gamma" in groups:
        for gamma in [0.0, 0.25, 0.50, 0.75, 1.0]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"G{gamma:.2f}", group="manuscript_gamma", combo_gamma=gamma
                )
            )
    if "clients" in groups:
        for n_parties in client_values:
            configs.append(
                SCERRobustnessConfig(
                    name=f"K{n_parties}",
                    group="clients",
                    n_parties=n_parties,
                )
            )
    if "clients_aligned" in groups:
        for n_parties in client_values:
            configs.append(
                SCERRobustnessConfig(
                    name=f"K{n_parties}",
                    group="clients_aligned",
                    n_parties=n_parties,
                    total_rule_budget=8 * n_parties,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                )
            )
    if "rules" in groups:
        for total_rule_budget in [36, 54, 72, 96, 144, 180]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"L{total_rule_budget}",
                    group="rules",
                    total_rule_budget=total_rule_budget,
                )
            )
    if "rules_aligned" in groups:
        for rules_per_party in [4, 6, 8, 10, 15, 18]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"R{rules_per_party}",
                    group="rules_aligned",
                    total_rule_budget=rules_per_party * 10,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                )
            )
    if "scer" in groups:
        configs.extend(
            [
                SCERRobustnessConfig(name="top4-G64-g0.75", group="scer"),
                SCERRobustnessConfig(name="top2-G64-g0.75", group="scer", top_rules_per_party=2),
                SCERRobustnessConfig(name="top6-G64-g0.75", group="scer", top_rules_per_party=6),
                SCERRobustnessConfig(name="top4-G32-g0.75", group="scer", combo_budget=32),
                SCERRobustnessConfig(name="top4-G128-g0.75", group="scer", combo_budget=128),
                SCERRobustnessConfig(name="top4-G64-g0.25", group="scer", combo_gamma=0.25),
                SCERRobustnessConfig(name="top4-G64-g0.50", group="scer", combo_gamma=0.50),
            ]
        )
    if "ltiter" in groups:
        configs.extend(
            [
                SCERRobustnessConfig(name="iter0", group="ltiter", parameter_iterations=0, parameter_update_blend=0.0),
                SCERRobustnessConfig(name="iter1-b0.01", group="ltiter", parameter_iterations=1),
                SCERRobustnessConfig(name="iter2-b0.01", group="ltiter"),
                SCERRobustnessConfig(name="iter3-b0.01", group="ltiter", parameter_iterations=3),
                SCERRobustnessConfig(name="iter5-b0.01", group="ltiter", parameter_iterations=5),
                SCERRobustnessConfig(name="iter2-b0.005", group="ltiter", parameter_update_blend=0.005),
                SCERRobustnessConfig(name="iter2-b0.05", group="ltiter", parameter_update_blend=0.05),
                SCERRobustnessConfig(name="iter2-b0.10", group="ltiter", parameter_update_blend=0.10),
            ]
        )
    if "ltiter_aligned" in groups:
        configs.extend(
            [
                SCERRobustnessConfig(
                    name="iter0",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_iterations=0,
                    parameter_update_blend=0.0,
                ),
                SCERRobustnessConfig(
                    name="iter1-b0.01",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_iterations=1,
                ),
                SCERRobustnessConfig(
                    name="iter2-b0.01",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                ),
                SCERRobustnessConfig(
                    name="iter3-b0.01",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_iterations=3,
                ),
                SCERRobustnessConfig(
                    name="iter5-b0.01",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_iterations=5,
                ),
                SCERRobustnessConfig(
                    name="iter2-b0.005",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_update_blend=0.005,
                ),
                SCERRobustnessConfig(
                    name="iter2-b0.05",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_update_blend=0.05,
                ),
                SCERRobustnessConfig(
                    name="iter2-b0.10",
                    group="ltiter_aligned",
                    total_rule_budget=80,
                    n_estimators=1,
                    learning_rate=0.20,
                    sample_fraction=1.0,
                    parameter_update_blend=0.10,
                ),
            ]
        )
    if "threshold" in groups:
        for tau in [0.75, 1.0, 1.25, 1.5, 2.0]:
            configs.append(
                SCERRobustnessConfig(
                    name=f"tau{tau:g}",
                    group="threshold",
                    threshold_prior_multiplier=tau,
                )
            )
    if not configs:
        configs.append(base)
    return configs


def evaluate_config(
    config: SCERRobustnessConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    minority_label: object,
    seed: int,
) -> Dict[str, float]:
    start = time.perf_counter()
    model = StrictVFLEBRBLTIterSCER(config, random_state=seed)
    model.fit(X_train, y_train, party_features=party_features, active_party_id=0)
    pred = model.predict(X_test)
    row = long_tail_metrics(y_test, pred, minority_label)
    row["wall_time_sec"] = float(time.perf_counter() - start)
    for key, value in model.diagnostics().items():
        row[key] = float(value)
    return row


def aggregate_across_datasets(dataset_payload: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for dataset_result in dataset_payload.values():
        for config_name, payload in dataset_result["results"].items():
            grouped.setdefault(config_name, []).append(payload["mean"])
    out: Dict[str, Dict[str, float]] = {}
    keys = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "minority_precision_mean",
        "minority_recall_mean",
        "minority_f1_mean",
        "pred_minority_ratio_mean",
        "wall_time_sec_mean",
        "scer_combo_rule_count_mean",
        "scer_rules_per_party_mean",
        "ebrb_boost_parameter_update_shift_mean_mean",
    ]
    for config_name, rows in grouped.items():
        summary: Dict[str, float] = {}
        for key in keys:
            vals = [float(row[key]) for row in rows if key in row]
            if vals:
                summary[key.replace("_mean", "") if key.endswith("_mean") else key] = float(np.mean(vals))
        minf1_vals = [float(row["minority_f1_mean"]) for row in rows if "minority_f1_mean" in row]
        if minf1_vals:
            summary["minority_f1_worst_dataset"] = float(np.min(minf1_vals))
        out[config_name] = summary
    return out


def write_summary_csv(path: str, result: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    config_lookup = {item["name"]: item for item in result["configs"]}
    for config_name, metrics in result["aggregate"].items():
        config = config_lookup.get(config_name, {})
        row = {"config": config_name, "group": config.get("group", "")}
        row.update(config)
        row.update(metrics)
        rows.append(row)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness sweeps for Strict VFL-EBRB-LT-Iter+SCER.")
    parser.add_argument("--datasets", default="default_credit,adult,bank_marketing")
    parser.add_argument("--groups", default="clients,rules,scer,ltiter,threshold")
    parser.add_argument("--client-values", default="3,5,10")
    parser.add_argument("--rho", type=float, default=0.03)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "scer_robustness", "scer_robustness_experiment.json"))
    args = parser.parse_args()

    specs = {
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    groups = parse_list(args.groups)
    client_values = [int(item) for item in parse_list(args.client_values)]
    configs = build_configurations(groups, client_values=client_values)
    result: Dict[str, Any] = {
        "experiment": "strict_vfl_ebrb_lt_iter_scer_robustness",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rho": float(args.rho),
        "rank_strength": float(args.rank_strength),
        "splits": int(args.splits),
        "max_samples": int(args.max_samples),
        "groups": groups,
        "configs": [asdict(config) for config in configs],
        "datasets": {},
    }
    start_all = time.perf_counter()
    for dataset_name in parse_list(args.datasets):
        spec = specs[dataset_name]
        X_raw, y_raw = exp.load_dataset(spec)
        y_raw = np.asarray(y_raw, dtype=object).astype(str)
        X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=13)
        splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=13)
        rows: Dict[str, List[Dict[str, float]]] = {config.name: [] for config in configs}
        fold_meta: List[Dict[str, Any]] = []
        print(f"\nDataset {dataset_name}", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, exp.stratify_labels(y_raw)), start=1):
            y_train_base = y_raw[train_idx]
            labels, counts = np.unique(y_train_base, return_counts=True)
            minority_label = labels[int(np.argmin(counts))]
            X_train_raw, y_train = long_tail_resample(
                X_raw[train_idx],
                y_train_base,
                minority_label=minority_label,
                target_minority_ratio=args.rho,
                seed=1000 * fold + int(round(args.rho * 1000)),
            )
            X_test_raw, y_test = long_tail_resample(
                X_raw[test_idx],
                y_raw[test_idx],
                minority_label=minority_label,
                target_minority_ratio=args.rho,
                seed=2000 * fold + int(round(args.rho * 1000)),
            )
            encoder = exp.TabularEncoder()
            X_train = encoder.fit_transform(X_train_raw)
            X_test = encoder.transform(X_test_raw)
            split_cache: Dict[int, Any] = {}
            fold_meta.append(
                {
                    "fold": int(fold),
                    "minority_label": str(minority_label),
                    "train_n": int(len(y_train)),
                    "test_n": int(len(y_test)),
                    "train_minority_ratio": float(np.mean(y_train == minority_label)),
                    "test_minority_ratio": float(np.mean(y_test == minority_label)),
                }
            )
            print(f"  Fold {fold}: n_train={len(y_train)} n_test={len(y_test)}", flush=True)
            for config in configs:
                if config.n_parties not in split_cache:
                    split_cache[config.n_parties] = active_weak_split(
                        X_train,
                        y_train,
                        n_parties=config.n_parties,
                        rank_strength=args.rank_strength,
                        seed=(13 + fold if any(group.startswith("manuscript_") for group in groups) else 13 + 100 * fold),
                    )
                split = split_cache[config.n_parties]
                metric = evaluate_config(
                    config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    split.party_features,
                    minority_label=minority_label,
                    seed=13 + fold,
                )
                rows[config.name].append(metric)
                print(
                    f"    {config.group}/{config.name}: "
                    f"acc={metric['accuracy']:.4f} "
                    f"minf1={metric['minority_f1']:.4f} "
                    f"minrec={metric['minority_recall']:.4f}",
                    flush=True,
                )
        result["datasets"][dataset_name] = {
            "fold_meta": fold_meta,
            "results": {
                config_name: {
                    "folds": config_rows,
                    "mean": summarize(config_rows),
                }
                for config_name, config_rows in rows.items()
            },
        }
    result["aggregate"] = aggregate_across_datasets(result["datasets"])
    result["wall_time_sec"] = float(time.perf_counter() - start_all)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    csv_path = os.path.splitext(args.output)[0] + "_summary.csv"
    write_summary_csv(csv_path, result)

    print(f"\nSaved JSON: {args.output}")
    print(f"Saved CSV: {csv_path}")
    print("| Group | Config | Acc | MinF1 | MinRecall | Worst Dataset MinF1 | ComboRules |")
    print("|---|---|---:|---:|---:|---:|---:|")
    config_lookup = {config.name: config for config in configs}
    for name, metrics in result["aggregate"].items():
        config = config_lookup[name]
        print(
            f"| {config.group} | {name} | "
            f"{metrics.get('accuracy', float('nan')):.4f} | "
            f"{metrics.get('minority_f1', float('nan')):.4f} | "
            f"{metrics.get('minority_recall', float('nan')):.4f} | "
            f"{metrics.get('minority_f1_worst_dataset', float('nan')):.4f} | "
            f"{metrics.get('scer_combo_rule_count', float('nan')):.1f} |"
        )


if __name__ == "__main__":
    main()
