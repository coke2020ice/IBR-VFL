from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

PRIMARY_DEFAULT = ROOT / "results" / "strict_vfl_all_ratios_all_methods_ebrb_lt_iter_3datasets.json"
ACTIVE_WEAK_DEFAULT = ROOT / "results" / "k10_active_weak_ablation_3datasets_optimized_boost.json"
BOOST_GRID_DEFAULT = ROOT / "results" / "current_boost_optimization_grid.json"
ITER_SWEEP_DEFAULT = ROOT / "results" / "strict_vfl_ebrb_ratio001_parameter_iteration_sweep.tsv"
BLEND_SWEEP_DEFAULT = ROOT / "results" / "strict_vfl_ebrb_ratio001_parameter_blend_sweep.tsv"


METHOD_ORDER = [
    "Strict VFL-XGBoost Hist 9",
    "Strict VFL-XGBoost Hist 9 LT",
    "FedTree Vertical GBDT 9",
    "Strict VFL-Tuned DR-MBRB boost",
    "Strict VFL-Tuned DR-MBRB boost LT",
    "Strict VFL-EBRB boost",
    "Strict VFL-EBRB boost LT-Iter",
    "Strict VFL-EBRB boost LT-Iter-p0.75",
    "Strict VFL-EBRB boost LT-Iter-p0.5",
]

DISPLAY = {
    "Strict VFL-XGBoost Hist 9": "Strict VFL-XGB Hist",
    "Strict VFL-XGBoost Hist 9 LT": "Strict VFL-XGB Hist-LT",
    "FedTree Vertical GBDT 9": "Vertical FedTree-GBDT",
    "Strict VFL-Tuned DR-MBRB boost": "Strict VFL-MBRB boost",
    "Strict VFL-Tuned DR-MBRB boost LT": "Strict VFL-MBRB boost-LT",
    "Strict VFL-EBRB boost": "Strict VFL-EBRB boost",
    "Strict VFL-EBRB boost LT-Iter": "Strict VFL-EBRB boost-LT-Iter",
    "Strict VFL-EBRB boost LT-Iter-p0.75": "Strict VFL-EBRB boost-LT-Iter-p0.75",
    "Strict VFL-EBRB boost LT-Iter-p0.5": "Strict VFL-EBRB boost-LT-Iter-p0.5",
}

THRESHOLD_METHODS = {
    "Strict VFL-EBRB boost LT-Iter-p0.5": 0.50,
    "Strict VFL-EBRB boost LT-Iter-p0.75": 0.75,
    "Strict VFL-EBRB boost LT-Iter": 1.00,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def fmt(value: Any, digits: int = 4) -> str:
    value = as_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def collect_primary_rows(primary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, dataset_payload in primary["datasets"].items():
        for ratio, ratio_payload in dataset_payload["results"].items():
            for heterogeneity, mode_payload in ratio_payload.items():
                for method, method_payload in mode_payload.items():
                    for fold_id, fold_row in enumerate(method_payload["folds"], start=1):
                        row = {
                            "dataset": dataset,
                            "ratio": float(ratio),
                            "heterogeneity": heterogeneity,
                            "method": method,
                            "method_display": DISPLAY.get(method, method),
                            "fold": fold_id,
                        }
                        row.update({key: as_float(value) for key, value in fold_row.items()})
                        rows.append(row)
    return rows


def group_rows(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> Dict[tuple, List[Mapping[str, Any]]]:
    grouped: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    minf1 = [as_float(row.get("minority_f1")) for row in rows]
    acc = [as_float(row.get("accuracy")) for row in rows]
    recall = [as_float(row.get("minority_recall")) for row in rows]
    pred_min = [as_float(row.get("pred_minority_ratio")) for row in rows]
    true_min = [as_float(row.get("true_minority_ratio")) for row in rows]
    zero_f1 = sum(1 for value in minf1 if value <= 1e-12)
    collapse = 0
    for pred, true in zip(pred_min, true_min):
        if not math.isnan(pred) and not math.isnan(true) and pred < 0.1 * true:
            collapse += 1
    return {
        "scenario_count": len(rows),
        "accuracy_mean": mean(acc),
        "accuracy_std": std(acc),
        "minority_f1_mean": mean(minf1),
        "minority_f1_std": std(minf1),
        "minority_f1_worst": min(minf1) if minf1 else math.nan,
        "minority_recall_mean": mean(recall),
        "pred_minority_ratio_mean": mean(pred_min),
        "zero_f1_count": zero_f1,
        "collapse_count": collapse,
    }


def build_long_tail_tables(primary_rows: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chosen = [row for row in primary_rows if row["method"] in METHOD_ORDER]
    summary_rows: List[Dict[str, Any]] = []
    for (method,), rows in group_rows(chosen, ["method"]).items():
        summary = summarize_metric_rows(rows)
        summary_rows.append(
            {
                "method": DISPLAY.get(method, method),
                "method_key": method,
                **summary,
            }
        )
    summary_rows.sort(key=lambda row: METHOD_ORDER.index(row["method_key"]))

    ratio_rows: List[Dict[str, Any]] = []
    for (method, ratio), rows in group_rows(chosen, ["method", "ratio"]).items():
        summary = summarize_metric_rows(rows)
        ratio_rows.append(
            {
                "method": DISPLAY.get(method, method),
                "method_key": method,
                "ratio": ratio,
                **summary,
            }
        )
    ratio_rows.sort(key=lambda row: (METHOD_ORDER.index(row["method_key"]), row["ratio"]))
    return summary_rows, ratio_rows


def build_heterogeneity_tables(primary_rows: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chosen = [row for row in primary_rows if row["method"] in METHOD_ORDER]
    mode_rows: List[Dict[str, Any]] = []
    mode_lookup: Dict[tuple, Dict[str, Any]] = {}
    for (method, heterogeneity), rows in group_rows(chosen, ["method", "heterogeneity"]).items():
        summary = summarize_metric_rows(rows)
        out = {
            "method": DISPLAY.get(method, method),
            "method_key": method,
            "heterogeneity": heterogeneity,
            **summary,
        }
        mode_rows.append(out)
        mode_lookup[(method, heterogeneity)] = out
    mode_rows.sort(key=lambda row: (METHOD_ORDER.index(row["method_key"]), row["heterogeneity"]))

    delta_rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        balanced = mode_lookup.get((method, "balanced"))
        skew = mode_lookup.get((method, "importance_skew"))
        if not balanced or not skew:
            continue
        delta_rows.append(
            {
                "method": DISPLAY.get(method, method),
                "method_key": method,
                "balanced_minority_f1": balanced["minority_f1_mean"],
                "importance_skew_minority_f1": skew["minority_f1_mean"],
                "delta_minority_f1": skew["minority_f1_mean"] - balanced["minority_f1_mean"],
                "balanced_accuracy": balanced["accuracy_mean"],
                "importance_skew_accuracy": skew["accuracy_mean"],
                "delta_accuracy": skew["accuracy_mean"] - balanced["accuracy_mean"],
                "balanced_zero_f1_count": balanced["zero_f1_count"],
                "importance_skew_zero_f1_count": skew["zero_f1_count"],
            }
        )
    return mode_rows, delta_rows


def build_threshold_table(primary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, tau in THRESHOLD_METHODS.items():
        method_rows = [row for row in primary_rows if row["method"] == method]
        summary = summarize_metric_rows(method_rows)
        rows.append(
            {
                "tau_pri": tau,
                "method": DISPLAY.get(method, method),
                **summary,
            }
        )
    rows.sort(key=lambda row: row["tau_pri"])
    return rows


def build_active_weak_table(active_weak: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_rows: List[Dict[str, Any]] = []
    for dataset, dataset_payload in active_weak["datasets"].items():
        for method, payload in dataset_payload["results"].items():
            mean_payload = payload["mean"]
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "accuracy_mean": as_float(mean_payload.get("accuracy_mean")),
                    "accuracy_std": as_float(mean_payload.get("accuracy_std")),
                    "minority_f1_mean": as_float(mean_payload.get("minority_f1_mean")),
                    "minority_f1_std": as_float(mean_payload.get("minority_f1_std")),
                    "minority_recall_mean": as_float(mean_payload.get("minority_recall_mean")),
                    "minority_precision_mean": as_float(mean_payload.get("minority_precision_mean")),
                    "pred_minority_ratio_mean": as_float(mean_payload.get("pred_minority_ratio_mean")),
                }
            )

    method_groups = group_rows(dataset_rows, ["method"])
    local_key = "Local-P0 EBRB boost-LT-Iter"
    local_mean = mean(row["minority_f1_mean"] for row in method_groups.get((local_key,), []))
    aggregate_rows: List[Dict[str, Any]] = []
    for (method,), rows in method_groups.items():
        minf1_values = [as_float(row["minority_f1_mean"]) for row in rows]
        aggregate_rows.append(
            {
                "method": method,
                "dataset_count": len(rows),
                "accuracy_mean": mean(row["accuracy_mean"] for row in rows),
                "minority_f1_mean": mean(minf1_values),
                "minority_f1_worst_dataset": min(minf1_values) if minf1_values else math.nan,
                "minority_recall_mean": mean(row["minority_recall_mean"] for row in rows),
                "gain_minority_f1_vs_local_boost": mean(minf1_values) - local_mean,
            }
        )
    preferred = [
        "Majority-only",
        "Local-P0 EBRB-LT-Iter",
        "Local-P0 EBRB boost-LT-Iter",
        "Strict VFL-EBRB-LT-Iter",
        "Strict VFL-EBRB boost-LT-Iter",
        "Centralized EBRB-LT-Iter",
        "Centralized EBRB boost-LT-Iter",
    ]
    aggregate_rows.sort(key=lambda row: preferred.index(row["method"]) if row["method"] in preferred else 999)
    return dataset_rows, aggregate_rows


def build_boost_grid_table(boost_grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    configs = boost_grid.get("configs", {})
    aggregate = boost_grid.get("aggregate", {})
    for name, config in configs.items():
        stats = aggregate.get(name, {})
        rows.append(
            {
                "config": name,
                "n_estimators": config.get("n_estimators"),
                "learning_rate": config.get("learning_rate"),
                "sample_fraction": config.get("sample_fraction"),
                "tau": config.get("tau"),
                "mean_accuracy": as_float(stats.get("mean_accuracy")),
                "mean_minority_f1": as_float(stats.get("mean_minority_f1")),
                "mean_minority_recall": as_float(stats.get("mean_minority_recall")),
                "min_dataset_minority_f1": as_float(stats.get("min_dataset_minority_f1")),
            }
        )
    rows.sort(key=lambda row: (row["config"] != "current", str(row["config"])))
    return rows


def normalize_sweep_rows(rows: Sequence[Mapping[str, Any]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": source,
                "variant": row.get("variant", ""),
                "accuracy": as_float(row.get("accuracy")),
                "balanced_accuracy": as_float(row.get("balanced_accuracy")),
                "minority_precision": as_float(row.get("minority_precision")),
                "minority_recall": as_float(row.get("minority_recall")),
                "minority_f1": as_float(row.get("minority_f1")),
                "pred_minority_ratio": as_float(row.get("pred_minority_ratio")),
                "parameter_update_shift": as_float(row.get("ebrb_boost_parameter_update_shift_mean")),
                "wall_time_sec": as_float(row.get("wall_time_sec")),
            }
        )
    return out


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], limit: int | None = None) -> str:
    shown = list(rows[:limit] if limit is not None else rows)
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(fmt(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    path: Path,
    *,
    long_tail_summary: Sequence[Mapping[str, Any]],
    heterogeneity_delta: Sequence[Mapping[str, Any]],
    active_weak_aggregate: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    boost_grid_rows: Sequence[Mapping[str, Any]],
    iter_sweep_rows: Sequence[Mapping[str, Any]],
) -> None:
    selected_long_tail = [
        row
        for row in long_tail_summary
        if row["method_key"]
        in {
            "Strict VFL-XGBoost Hist 9",
            "Strict VFL-XGBoost Hist 9 LT",
            "FedTree Vertical GBDT 9",
            "Strict VFL-EBRB boost",
            "Strict VFL-EBRB boost LT-Iter",
        }
    ]
    selected_delta = [
        row
        for row in heterogeneity_delta
        if row["method_key"]
        in {
            "Strict VFL-XGBoost Hist 9",
            "Strict VFL-XGBoost Hist 9 LT",
            "Strict VFL-EBRB boost",
            "Strict VFL-EBRB boost LT-Iter",
        }
    ]
    selected_boost = [
        row
        for row in boost_grid_rows
        if row["config"] in {"single", "current", "sample0.85", "lr0.15", "lr0.25", "tau0.75", "tau1.25", "T12", "T15"}
    ]
    best_iter = sorted(iter_sweep_rows, key=lambda row: row.get("minority_f1", 0), reverse=True)[:8]

    text = f"""# Robustness Analysis Report

This report is generated from existing experiment outputs under `results/`.

## 1. Long-Tail Robustness

The table aggregates all folds across three datasets, three long-tail ratios, and two vertical heterogeneity modes.
`Zero-F1` counts cases where minority F1 collapsed to zero at the fold level.

{md_table(
    selected_long_tail,
    [
        ("method", "Method"),
        ("accuracy_mean", "Acc"),
        ("minority_f1_mean", "MinF1"),
        ("minority_f1_worst", "Worst MinF1"),
        ("minority_f1_std", "MinF1 Std"),
        ("zero_f1_count", "Zero-F1"),
        ("collapse_count", "Pred Collapse"),
    ],
)}

## 2. Vertical Heterogeneity Robustness

`delta_minority_f1` is computed as `importance_skew - balanced`.

{md_table(
    selected_delta,
    [
        ("method", "Method"),
        ("balanced_minority_f1", "Balanced MinF1"),
        ("importance_skew_minority_f1", "Skew MinF1"),
        ("delta_minority_f1", "Delta MinF1"),
        ("delta_accuracy", "Delta Acc"),
        ("balanced_zero_f1_count", "Zero-F1 Balanced"),
        ("importance_skew_zero_f1_count", "Zero-F1 Skew"),
    ],
)}

## 3. Active-Weak Robustness

The table aggregates the P=10 active-weak experiment across the three datasets.

{md_table(
    active_weak_aggregate,
    [
        ("method", "Method"),
        ("accuracy_mean", "Acc"),
        ("minority_f1_mean", "MinF1"),
        ("minority_f1_worst_dataset", "Worst Dataset MinF1"),
        ("minority_recall_mean", "MinRecall"),
        ("gain_minority_f1_vs_local_boost", "Gain vs Local Boost"),
    ],
)}

## 4. Threshold Sensitivity

This table uses the three train-prior threshold variants already included in the main experiment.

{md_table(
    threshold_rows,
    [
        ("tau_pri", "tau_pri"),
        ("accuracy_mean", "Acc"),
        ("minority_f1_mean", "MinF1"),
        ("minority_recall_mean", "MinRecall"),
        ("pred_minority_ratio_mean", "PredMin"),
    ],
)}

## 5. Boosting / Sampling Sensitivity

This table summarizes the existing boost optimization grid.

{md_table(
    selected_boost,
    [
        ("config", "Config"),
        ("n_estimators", "T"),
        ("learning_rate", "LR"),
        ("sample_fraction", "Sample"),
        ("tau", "Tau"),
        ("mean_accuracy", "Acc"),
        ("mean_minority_f1", "MinF1"),
        ("min_dataset_minority_f1", "Worst Dataset MinF1"),
    ],
)}

## 6. LT-Iter Parameter Sensitivity

Top variants from the existing iteration/blend sweeps, ranked by minority F1.

{md_table(
    best_iter,
    [
        ("source", "Source"),
        ("variant", "Variant"),
        ("accuracy", "Acc"),
        ("minority_f1", "MinF1"),
        ("minority_recall", "MinRecall"),
        ("pred_minority_ratio", "PredMin"),
        ("parameter_update_shift", "Update Shift"),
    ],
)}

## Manuscript Takeaway

The most defensible robustness claim is not that the proposed method is always the best at every single point.
The stronger claim is that it reduces minority-class collapse under strict VFL long-tail settings,
especially compared with uncalibrated tree baselines, while keeping the accuracy--minority-F1 trade-off controllable through the train-prior threshold and LT-Iter settings.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def try_write_plots(out_dir: Path, ratio_rows: Sequence[Mapping[str, Any]], threshold_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_methods = [
        "Strict VFL-XGB Hist",
        "Strict VFL-XGB Hist-LT",
        "Vertical FedTree-GBDT",
        "Strict VFL-EBRB boost",
        "Strict VFL-EBRB boost-LT-Iter",
    ]
    by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for row in ratio_rows:
        if row["method"] in plot_methods:
            by_method.setdefault(row["method"], []).append(row)

    plt.figure(figsize=(7.2, 4.2))
    for method in plot_methods:
        rows = sorted(by_method.get(method, []), key=lambda item: item["ratio"])
        if not rows:
            continue
        plt.plot(
            [row["ratio"] for row in rows],
            [row["minority_f1_mean"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=method,
        )
    plt.gca().invert_xaxis()
    plt.xlabel("Minority ratio rho")
    plt.ylabel("Minority F1")
    plt.title("Long-tail robustness across minority ratios")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "long_tail_minf1_by_ratio.png", dpi=240)
    plt.close()

    rows = sorted(threshold_rows, key=lambda item: item["tau_pri"])
    plt.figure(figsize=(5.8, 3.8))
    plt.plot([row["tau_pri"] for row in rows], [row["accuracy_mean"] for row in rows], marker="o", label="Accuracy")
    plt.plot([row["tau_pri"] for row in rows], [row["minority_f1_mean"] for row in rows], marker="o", label="Minority F1")
    plt.plot([row["tau_pri"] for row in rows], [row["minority_recall_mean"] for row in rows], marker="o", label="Minority Recall")
    plt.xlabel("Train-prior threshold multiplier")
    plt.ylabel("Score")
    plt.title("Threshold sensitivity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sensitivity.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness analysis tables from existing Strict VFL experiment outputs.")
    parser.add_argument("--primary", type=Path, default=PRIMARY_DEFAULT)
    parser.add_argument("--active-weak", type=Path, default=ACTIVE_WEAK_DEFAULT)
    parser.add_argument("--boost-grid", type=Path, default=BOOST_GRID_DEFAULT)
    parser.add_argument("--iter-sweep", type=Path, default=ITER_SWEEP_DEFAULT)
    parser.add_argument("--blend-sweep", type=Path, default=BLEND_SWEEP_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "robustness_analysis")
    args = parser.parse_args()

    primary = load_json(args.primary)
    active_weak = load_json(args.active_weak)
    boost_grid = load_json(args.boost_grid)

    primary_rows = collect_primary_rows(primary)
    long_tail_summary, long_tail_by_ratio = build_long_tail_tables(primary_rows)
    heterogeneity_by_mode, heterogeneity_delta = build_heterogeneity_tables(primary_rows)
    threshold_rows = build_threshold_table(primary_rows)
    active_weak_dataset, active_weak_aggregate = build_active_weak_table(active_weak)
    boost_grid_rows = build_boost_grid_table(boost_grid)
    iter_rows = normalize_sweep_rows(read_tsv(args.iter_sweep), "iteration_sweep")
    blend_rows = normalize_sweep_rows(read_tsv(args.blend_sweep), "blend_sweep")
    ltiter_rows = iter_rows + blend_rows

    out_dir = args.output_dir
    write_csv(out_dir / "long_tail_robustness_summary.csv", long_tail_summary)
    write_csv(out_dir / "long_tail_by_ratio.csv", long_tail_by_ratio)
    write_csv(out_dir / "heterogeneity_by_mode.csv", heterogeneity_by_mode)
    write_csv(out_dir / "heterogeneity_delta.csv", heterogeneity_delta)
    write_csv(out_dir / "threshold_sensitivity.csv", threshold_rows)
    write_csv(out_dir / "active_weak_by_dataset.csv", active_weak_dataset)
    write_csv(out_dir / "active_weak_aggregate.csv", active_weak_aggregate)
    write_csv(out_dir / "boost_grid_sensitivity.csv", boost_grid_rows)
    write_csv(out_dir / "ltiter_parameter_sensitivity.csv", ltiter_rows)

    write_report(
        out_dir / "robustness_report.md",
        long_tail_summary=long_tail_summary,
        heterogeneity_delta=heterogeneity_delta,
        active_weak_aggregate=active_weak_aggregate,
        threshold_rows=threshold_rows,
        boost_grid_rows=boost_grid_rows,
        iter_sweep_rows=ltiter_rows,
    )
    try_write_plots(out_dir, long_tail_by_ratio, threshold_rows)

    print(f"Saved robustness analysis to: {out_dir}")
    print(f"- {out_dir / 'robustness_report.md'}")
    print(f"- {out_dir / 'long_tail_robustness_summary.csv'}")
    print(f"- {out_dir / 'heterogeneity_delta.csv'}")
    print(f"- {out_dir / 'active_weak_aggregate.csv'}")
    print(f"- {out_dir / 'threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
