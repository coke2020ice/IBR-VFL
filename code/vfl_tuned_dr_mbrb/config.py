from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping


PassiveLabelMode = Literal["unsupervised", "simulation_supervised"]
ClientFusionMethod = Literal["mean", "er"]
PrototypeMethod = Literal["quantile", "centroid", "kmeans", "farthest", "hybrid"]
ClientWeightMethod = Literal["confidence", "label_accuracy", "label_log_loss"]
RuleScaleMethod = Literal["global", "local_knn"]


@dataclass
class BRBModuleConfig:
    n_modules_per_party: int = 2
    n_rules_per_module: int = 8
    n_rules_per_class: int = 4
    prototype_method: PrototypeMethod = "quantile"
    rule_scale_method: RuleScaleMethod = "global"
    rule_scale_neighbors: int = 48
    rule_scale_multiplier: float = 1.0
    beta_smoothing: float = 0.5
    theta_temperature: float = 1.0
    gamma_la: float = 0.35
    temperature_la: float = 1.0
    unsupervised_orientation_strength: float = 0.25
    activation_threshold: float = 1e-4
    class_balance_power: float = 0.0


@dataclass
class ServerReasoningConfig:
    client_fusion_method: ClientFusionMethod = "er"
    client_weight_method: ClientWeightMethod = "confidence"
    confidence_power: float = 1.0
    n_reasoning_steps: int = 1
    lambda_d: float = 1.0
    kappa: float = 0.0
    eta_c: float = 1.0
    eta_d: float = 0.5
    use_train_difficulty_stats: bool = True
    decision_prior_correction: float = 0.0


@dataclass
class FinalLAConfig:
    enabled: bool = False
    gamma: float = 0.0
    temperature: float = 1.0
    blend: float = 1.0
    apply_on_difficult_only: bool = False
    prior_gate_enabled: bool = False
    min_class_prior_for_la: float = 0.0
    tune_on_validation: bool = False
    grid: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "gamma": [0.0, 0.8, 1.0, 1.2],
            "temperature": [0.8, 1.0],
            "blend": [0.5, 0.75, 1.0],
        }
    )


@dataclass
class StackingCalibrationConfig:
    enabled: bool = False
    C: float = 1.0
    class_weight: str | None = None
    max_iter: int = 1000


@dataclass
class TuningConfig:
    validation_ratio: float = 0.2
    max_trials: int | None = 24
    primary_metric: str = "macro_f1"
    grid: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "gamma_la": [0.15, 0.35],
            "temperature_la": [0.8, 1.0],
            "lambda_d": [0.5, 1.0],
            "kappa": [0.0, 0.5],
            "eta_c": [0.5, 1.0],
            "eta_d": [0.25, 0.5],
        }
    )


@dataclass
class VFLTunedDRMBRBConfig:
    n_parties: int = 3
    active_party_id: int = 0
    passive_label_mode: PassiveLabelMode = "unsupervised"
    random_state: int = 13
    brb: BRBModuleConfig = field(default_factory=BRBModuleConfig)
    server: ServerReasoningConfig = field(default_factory=ServerReasoningConfig)
    final_la: FinalLAConfig = field(default_factory=FinalLAConfig)
    stacking: StackingCalibrationConfig = field(default_factory=StackingCalibrationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_dict(payload: Mapping[str, Any]) -> VFLTunedDRMBRBConfig:
    brb = BRBModuleConfig(**payload.get("brb", {}))
    server = ServerReasoningConfig(**payload.get("server", {}))
    final_la = FinalLAConfig(**payload.get("final_la", {}))
    stacking = StackingCalibrationConfig(**payload.get("stacking", {}))
    tuning = TuningConfig(**payload.get("tuning", {}))
    top_level = {key: value for key, value in payload.items() if key not in {"brb", "server", "final_la", "stacking", "tuning"}}
    return VFLTunedDRMBRBConfig(**top_level, brb=brb, server=server, final_la=final_la, stacking=stacking, tuning=tuning)
