from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from scripts.run_k10_active_weak_ablation import active_weak_split  # noqa: E402
from scripts.run_multiclass_credit_score_experiment import (  # noqa: E402
    load_credit_score,
    metrics,
    multiclass_long_tail_resample,
    summarize,
)
from scripts.run_scer_robustness_experiment import (  # noqa: E402
    SCERRobustnessConfig,
    StrictVFLEBRBLTIterSCER,
)
from vfl_tuned_dr_mbrb import TabularEncoder  # noqa: E402


DATA = "data/credit_score_classification/train.csv"
SEED = 13
MAX_SAMPLES = 6000
IMBALANCE_FACTOR = 3.0


def configurations() -> list[SCERRobustnessConfig]:
    common = dict(
        class_balanced_selection=True,
        class_balance_power=1.0,
        sampling_class_balance_power=0.75,
    )
    rows: list[SCERRobustnessConfig] = []
    for parties in (2, 4, 6, 8, 10, 12):
        rows.append(SCERRobustnessConfig(name=f"P{parties}", group="parties", n_parties=parties, **common))
    for rules in (4, 8, 12, 16, 24, 32):
        rows.append(
            SCERRobustnessConfig(
                name=f"L{rules}", group="rules", n_parties=10,
                total_rule_budget=rules * 10, **common,
            )
        )
    base = SCERRobustnessConfig(name="G0.00", group="gamma", combo_gamma=0.0, **common)
    for gamma in (0.0, 0.25, 0.50, 0.75, 1.0):
        rows.append(replace(base, name=f"G{gamma:.2f}", combo_gamma=gamma))
    return rows


def main() -> None:
    X_raw, y_raw, metadata = load_credit_score(DATA)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, MAX_SAMPLES, SEED)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, y_raw), start=1):
        labels, counts = np.unique(y_raw[train_idx], return_counts=True)
        class_order = labels[np.argsort(-counts)]
        X_train_raw, y_train = multiclass_long_tail_resample(
            X_raw[train_idx], y_raw[train_idx], class_order, IMBALANCE_FACTOR, SEED + 100 * fold
        )
        X_test_raw, y_test = multiclass_long_tail_resample(
            X_raw[test_idx], y_raw[test_idx], class_order, IMBALANCE_FACTOR, SEED + 1000 * fold
        )
        encoder = TabularEncoder()
        folds.append((encoder.fit_transform(X_train_raw), y_train, encoder.transform(X_test_raw), y_test))

    output = {
        "experiment": "credit_score_if3_sensitivity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "folds": 3,
        "sample_cap": MAX_SAMPLES,
        "imbalance_factor": IMBALANCE_FACTOR,
        "dataset": metadata,
        "results": {},
    }
    for config in configurations():
        rows = []
        for fold, (X_train, y_train, X_test, y_test) in enumerate(folds, start=1):
            split = active_weak_split(X_train, y_train, n_parties=config.n_parties, rank_strength=3.0, seed=SEED + fold)
            started = time.perf_counter()
            model = StrictVFLEBRBLTIterSCER(config, random_state=SEED + fold)
            model.fit(X_train, y_train, party_features=split.party_features, active_party_id=0)
            row = metrics(y_test, np.asarray(model.predict(X_test), dtype=object))
            row["wall_time_sec"] = time.perf_counter() - started
            rows.append(row)
            print(config.group, config.name, fold, f"Acc={row['accuracy']:.4f}", f"TailF1={row['tail_f1']:.4f}", flush=True)
        output["results"][f"{config.group}:{config.name}"] = {
            "config": asdict(config), "fold_metrics": rows, "summary": summarize(rows)
        }
    os.makedirs("results", exist_ok=True)
    with open("results/credit_score_if3_sensitivity.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
