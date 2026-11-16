from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import run_long_tail_ebrb_experiment as lt  # noqa: E402
from scripts import run_vertibench_heterogeneity_experiment as exp  # noqa: E402
from vfl_tuned_dr_mbrb.belief import EPS, smooth_prior  # noqa: E402
from vfl_tuned_dr_mbrb.config import BRBModuleConfig  # noqa: E402
from vfl_tuned_dr_mbrb.party import LocalBRBModule, VFLParty  # noqa: E402


COLORS = {
    "blue": "#2E86AB",
    "teal": "#33BBEE",
    "green": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create method-agnostic diagnostic figures for motivating VFL+EBRB: "
            "centralized rule evidence fragmentation and local belief conflict."
        )
    )
    parser.add_argument("--dataset", default="bank_marketing", choices=["bank_marketing", "adult", "default_credit"])
    parser.add_argument("--ratio", type=float, default=0.03)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--n-parties", type=int, default=10)
    parser.add_argument("--rank-strength", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--rules-per-class", type=int, default=12)
    parser.add_argument("--local-beta-smoothing", type=float, default=0.05)
    parser.add_argument("--local-class-balance-power", type=float, default=1.0)
    parser.add_argument("--local-gamma-la", type=float, default=0.0)
    parser.add_argument("--sample-heatmap-rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        default="results/intro_vfl_ebrb_diagnostic",
        help="Directory for CSV/JSON summaries and publication figures.",
    )
    parser.add_argument(
        "--paper-figures-dir",
        default="/Users/lin/Desktop/福州大学第三篇论文/figures",
        help="Optional paper figures directory. Leave empty to skip copying.",
    )
    return parser.parse_args()


def dataset_spec(name: str) -> exp.DatasetSpec:
    specs = {
        "default_credit": exp.DatasetSpec(
            "default_credit",
            ("default-of-credit-card-clients", "default_credit_card_clients", "default-credit-card-clients"),
        ),
        "adult": exp.DatasetSpec("adult", ("adult",)),
        "bank_marketing": exp.DatasetSpec("bank_marketing", ("bank-marketing",)),
    }
    return specs[name]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def make_analysis_split(args: argparse.Namespace) -> Dict[str, object]:
    X_raw, y_raw = exp.load_dataset(dataset_spec(args.dataset))
    y_raw = np.asarray(y_raw, dtype=object).astype(str)
    X_raw, y_raw = exp.stratified_limit(X_raw, y_raw, max_samples=args.max_samples, seed=args.seed)

    splitter = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(X_raw, exp.stratify_labels(y_raw)))
    train_idx, test_idx = folds[args.fold - 1]
    y_train_base = y_raw[train_idx]
    labels, counts = np.unique(y_train_base, return_counts=True)
    minority_label = labels[int(np.argmin(counts))]

    X_train_tail_raw, y_train_tail = lt.long_tail_resample(
        X_raw[train_idx],
        y_train_base,
        minority_label=minority_label,
        target_minority_ratio=args.ratio,
        seed=1000 * args.fold + int(round(args.ratio * 1000)),
    )
    X_test_tail_raw, y_test_tail = lt.long_tail_resample(
        X_raw[test_idx],
        y_raw[test_idx],
        minority_label=minority_label,
        target_minority_ratio=args.ratio,
        seed=2000 * args.fold + int(round(args.ratio * 1000)),
    )

    encoder = exp.TabularEncoder()
    X_train = encoder.fit_transform(X_train_tail_raw)
    X_test = encoder.transform(X_test_tail_raw)
    split = lt.active_weak_split(
        X_train,
        y_train_tail,
        n_parties=args.n_parties,
        rank_strength=args.rank_strength,
        seed=args.seed + 100 * args.fold,
    )

    return {
        "X_train": X_train,
        "y_train": np.asarray(y_train_tail, dtype=object),
        "X_test": X_test,
        "y_test": np.asarray(y_test_tail, dtype=object),
        "party_features": split.party_features,
        "minority_label": minority_label,
        "split": split,
    }


def fit_centralized_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rules_per_class: int,
    seed: int,
) -> LocalBRBModule:
    classes = np.unique(y_train)
    prior = smooth_prior(y_train, classes, alpha=0.5)
    config = BRBModuleConfig(
        n_modules_per_party=1,
        n_rules_per_module=max(2 * rules_per_class, 1),
        n_rules_per_class=rules_per_class,
        prototype_method="hybrid",
        rule_scale_method="local_knn",
        rule_scale_neighbors=48,
        rule_scale_multiplier=1.0,
        beta_smoothing=0.5,
        theta_temperature=1.0,
        gamma_la=0.0,
        temperature_la=1.0,
        class_balance_power=0.25,
    )
    return LocalBRBModule(config, random_state=seed).fit(X_train, y_train, classes, prior, supervised=True)


def party_partial_memberships(
    module: LocalBRBModule,
    X: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
) -> np.ndarray:
    rules = module.rules_
    n_samples = X.shape[0]
    party_ids = sorted(party_features)
    partial = np.zeros((n_samples, len(party_ids), len(rules)), dtype=float)
    for rule_index, rule in enumerate(rules):
        sq_dist = ((X - rule.center[None, :]) / np.maximum(rule.scale[None, :], EPS)) ** 2
        for party_pos, party_id in enumerate(party_ids):
            features = list(party_features[party_id])
            if features:
                partial[:, party_pos, rule_index] = np.exp(-0.5 * np.mean(sq_dist[:, features], axis=1))
    return partial


def centralized_evidence_diagnostics(
    module: LocalBRBModule,
    X_test: np.ndarray,
    y_test: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    tau: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activations = module._rule_activations(X_test)
    partial = party_partial_memberships(module, X_test, party_features)
    contribution = np.sum(partial * activations[:, None, :], axis=2)
    contribution = contribution / np.maximum(np.sum(contribution, axis=1, keepdims=True), EPS)
    sorted_contribution = -np.sort(-contribution, axis=1)
    cumulative = np.cumsum(sorted_contribution, axis=1)
    span = np.argmax(cumulative >= tau, axis=1) + 1
    active_share = contribution[:, sorted(party_features).index(0)]
    top1_share = sorted_contribution[:, 0]
    top2_share = np.sum(sorted_contribution[:, :2], axis=1)
    entropy = -np.sum(contribution * np.log(contribution + EPS), axis=1)
    effective_parties = np.exp(entropy)

    rows = pd.DataFrame(
        {
            "sample_index": np.arange(X_test.shape[0], dtype=int),
            "label": y_test.astype(str),
            "span_tau": span,
            "active_share": active_share,
            "top1_share": top1_share,
            "top2_share": top2_share,
            "effective_parties": effective_parties,
        }
    )
    cumulative_rows = pd.DataFrame(
        {
            "top_k": np.arange(1, contribution.shape[1] + 1, dtype=int),
            "mean_cumulative_share": np.mean(cumulative, axis=0),
            "q25_cumulative_share": np.quantile(cumulative, 0.25, axis=0),
            "q75_cumulative_share": np.quantile(cumulative, 0.75, axis=0),
        }
    )
    for party_pos, party_id in enumerate(sorted(party_features)):
        rows[f"party_{party_id}_share"] = contribution[:, party_pos]
    return rows, cumulative_rows


def fit_local_party_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    party_features: Mapping[int, Sequence[int]],
    rules_per_class: int,
    seed: int,
    beta_smoothing: float,
    class_balance_power: float,
    gamma_la: float,
) -> list[VFLParty]:
    classes = np.unique(y_train)
    prior = smooth_prior(y_train, classes, alpha=0.5)
    config = BRBModuleConfig(
        n_modules_per_party=1,
        n_rules_per_module=max(2 * rules_per_class, 1),
        n_rules_per_class=max(2, rules_per_class // 2),
        prototype_method="hybrid",
        rule_scale_method="global",
        beta_smoothing=beta_smoothing,
        theta_temperature=1.0,
        gamma_la=gamma_la,
        temperature_la=1.0,
        class_balance_power=class_balance_power,
    )
    parties: list[VFLParty] = []
    for party_id in sorted(party_features):
        party = VFLParty(
            party_id=party_id,
            feature_indices=party_features[party_id],
            is_active=True,
            config=config,
            passive_label_mode="simulation_supervised",
            random_state=seed + party_id,
        )
        party.fit(X_train, y_train, classes, prior)
        parties.append(party)
    return parties


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + EPS) - np.log(m + EPS)), axis=1)
    kl_qm = np.sum(q * (np.log(q + EPS) - np.log(m + EPS)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def local_belief_conflict_diagnostics(
    parties: Sequence[VFLParty],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    classes = np.asarray(parties[0].classes_, dtype=object)
    class_to_idx = {str(label): idx for idx, label in enumerate(classes.astype(str))}
    beliefs = np.stack([party.predict_belief(X_test).belief for party in parties], axis=1)
    n_samples, n_parties, _ = beliefs.shape
    pair_values = []
    for left in range(n_parties):
        for right in range(left + 1, n_parties):
            pair_values.append(js_divergence(beliefs[:, left, :], beliefs[:, right, :]))
    conflict = np.mean(np.stack(pair_values, axis=1), axis=1)
    mean_belief = np.mean(beliefs, axis=1)
    pred = classes[np.argmax(mean_belief, axis=1)].astype(str)
    y_str = y_test.astype(str)
    correct = pred == y_str
    true_indices = np.asarray([class_to_idx[value] for value in y_str], dtype=int)
    true_class_belief = beliefs[np.arange(n_samples)[:, None], np.arange(n_parties)[None, :], true_indices[:, None]]
    party_pred_indices = np.argmax(beliefs, axis=2)
    vote_counts = np.zeros((n_samples, len(classes)), dtype=int)
    for sample_idx in range(n_samples):
        vote_counts[sample_idx] = np.bincount(party_pred_indices[sample_idx], minlength=len(classes))
    vote_disagreement = 1.0 - np.max(vote_counts, axis=1) / float(n_parties)
    confidence_spread = np.max(np.max(beliefs, axis=2), axis=1) - np.min(np.max(beliefs, axis=2), axis=1)

    rows = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples, dtype=int),
            "label": y_str,
            "mean_fusion_pred": pred,
            "mean_fusion_correct": correct.astype(int),
            "belief_js_conflict": conflict,
            "vote_disagreement": vote_disagreement,
            "confidence_spread": confidence_spread,
        }
    )
    for party_index, party in enumerate(parties):
        rows[f"party_{party.party_id}_true_belief"] = true_class_belief[:, party_index]
        rows[f"party_{party.party_id}_pred"] = classes[np.argmax(beliefs[:, party_index, :], axis=1)].astype(str)

    labels = np.array(["0", "<=0.2", "0.2-0.4", ">0.4"], dtype=object)
    group_values = np.full(n_samples, labels[0], dtype=object)
    group_values[(vote_disagreement > 0.0) & (vote_disagreement <= 0.2)] = labels[1]
    group_values[(vote_disagreement > 0.2) & (vote_disagreement <= 0.4)] = labels[2]
    group_values[vote_disagreement > 0.4] = labels[3]
    rows["disagreement_group"] = group_values
    group_rows = []
    for group_name in labels:
        group = rows[rows["disagreement_group"] == group_name]
        if group.empty:
            continue
        idx = group.index.to_numpy()
        group_rows.append(
            {
                "disagreement_group": group_name,
                "n": int(len(idx)),
                "mean_conflict": float(np.mean(conflict[idx])),
                "mean_vote_disagreement": float(np.mean(vote_disagreement[idx])),
                "mean_fusion_error_rate": float(1.0 - np.mean(correct[idx])),
                "mean_true_class_belief": float(np.mean(true_class_belief[idx])),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    return rows, group_summary, beliefs, true_class_belief


def plot_centralized_fragmentation(
    rows: pd.DataFrame,
    cumulative_rows: pd.DataFrame,
    tau: float,
    output_dir: Path,
    paper_dir: Path | None,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax = axes[0]
    min_span = int(rows["span_tau"].min())
    max_span = int(rows["span_tau"].max())
    bins = list(range(min_span, max_span + 1))
    counts = [float(np.mean(rows["span_tau"] == value) * 100.0) for value in bins]
    labels = [str(value) for value in bins]
    ax.bar(labels, counts, color=COLORS["blue"], edgecolor="white", linewidth=0.8)
    ax.set_xlabel(f"Parties needed for {int(tau * 100)}% evidence")
    ax.set_ylabel("Test samples (%)")
    ax.set_title("(a) Evidence span of centralized EBRB rules", loc="left")
    ax.set_ylim(0, max(5.0, max(counts) * 1.18))

    ax = axes[1]
    x = cumulative_rows["top_k"].to_numpy()
    y = cumulative_rows["mean_cumulative_share"].to_numpy()
    q25 = cumulative_rows["q25_cumulative_share"].to_numpy()
    q75 = cumulative_rows["q75_cumulative_share"].to_numpy()
    ax.plot(x, y, color=COLORS["orange"], linewidth=2.0, marker="o", markersize=3.5)
    ax.fill_between(x, q25, q75, color=COLORS["orange"], alpha=0.18, linewidth=0)
    ax.axhline(tau, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Top-k parties by partial rule match")
    ax.set_ylabel("Cumulative evidence share")
    ax.set_title("(b) Cross-party accumulation", loc="left")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.04)
    ax.text(x[-1], tau + 0.025, f"$\\tau={tau:.2f}$", ha="right", va="bottom", color=COLORS["gray"])
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output_dir, paper_dir, "intro_diag_fig1_centralized_ebrb_fragmentation")


def plot_local_conflict(
    rows: pd.DataFrame,
    group_summary: pd.DataFrame,
    true_class_belief: np.ndarray,
    output_dir: Path,
    paper_dir: Path | None,
    heatmap_rows: int,
) -> None:
    setup_style()
    order = np.lexsort(
        (
            -rows["confidence_spread"].to_numpy(),
            -rows["belief_js_conflict"].to_numpy(),
            -rows["vote_disagreement"].to_numpy(),
        )
    )[:heatmap_rows]
    heatmap = true_class_belief[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("(a) Local EBRB belief assigned to true class", loc="left")
    ax.set_xlabel("Client")
    ax.set_ylabel("High-conflict test samples")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_xticklabels([f"P{i}" for i in range(heatmap.shape[1])], rotation=45, ha="right")
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Belief")

    ax = axes[1]
    labels = group_summary["disagreement_group"].tolist()
    errors = group_summary["mean_fusion_error_rate"].to_numpy() * 100.0
    ns = group_summary["n"].to_numpy()
    x = np.arange(len(labels))
    ax.bar(x, errors, color=COLORS["red"], edgecolor="white", linewidth=0.8)
    ax.set_title("(b) Simple averaging under local disagreement", loc="left")
    ax.set_ylabel("Mean-fusion error rate (%)")
    ax.set_xlabel("Local EBRB vote disagreement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, max(5.0, float(np.max(errors)) * 1.22))
    for idx, (err, n_value) in enumerate(zip(errors, ns)):
        ax.text(idx, err + max(errors) * 0.035, f"n={n_value}", ha="center", va="bottom", fontsize=7, color=COLORS["gray"])
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, output_dir, paper_dir, "intro_diag_fig2_local_ebrb_belief_conflict")


def save_figure(fig: plt.Figure, output_dir: Path, paper_dir: Path | None, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=450)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(paper_dir / f"{stem}.{suffix}", dpi=450)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_figures_dir) if args.paper_figures_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = make_analysis_split(args)
    X_train = payload["X_train"]
    y_train = payload["y_train"]
    X_test = payload["X_test"]
    y_test = payload["y_test"]
    party_features = payload["party_features"]
    split = payload["split"]

    centralized = fit_centralized_probe(X_train, y_train, args.rules_per_class, args.seed)
    span_rows, cumulative_rows = centralized_evidence_diagnostics(
        centralized,
        X_test,
        y_test,
        party_features,
        tau=args.tau,
    )
    local_parties = fit_local_party_probes(
        X_train,
        y_train,
        party_features,
        args.rules_per_class,
        args.seed,
        beta_smoothing=args.local_beta_smoothing,
        class_balance_power=args.local_class_balance_power,
        gamma_la=args.local_gamma_la,
    )
    conflict_rows, conflict_summary, _, true_class_belief = local_belief_conflict_diagnostics(
        local_parties,
        X_test,
        y_test,
    )

    span_rows.to_csv(output_dir / "centralized_rule_evidence_span.csv", index=False)
    cumulative_rows.to_csv(output_dir / "centralized_rule_evidence_cumulative.csv", index=False)
    conflict_rows.to_csv(output_dir / "local_ebrb_belief_conflict.csv", index=False)
    conflict_summary.to_csv(output_dir / "local_ebrb_conflict_groups.csv", index=False)

    plot_centralized_fragmentation(span_rows, cumulative_rows, args.tau, output_dir, paper_dir)
    plot_local_conflict(
        conflict_rows,
        conflict_summary,
        true_class_belief,
        output_dir,
        paper_dir,
        heatmap_rows=args.sample_heatmap_rows,
    )

    classes = np.unique(y_train).astype(str)
    mean_pred = conflict_rows["mean_fusion_pred"].to_numpy(dtype=str)
    y_test_str = y_test.astype(str)
    minority_label = str(payload["minority_label"])
    summary = {
        "dataset": args.dataset,
        "fold": int(args.fold),
        "splits": int(args.splits),
        "long_tail_ratio": float(args.ratio),
        "n_parties": int(args.n_parties),
        "rank_strength": float(args.rank_strength),
        "local_beta_smoothing": float(args.local_beta_smoothing),
        "local_class_balance_power": float(args.local_class_balance_power),
        "local_gamma_la": float(args.local_gamma_la),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "minority_label": minority_label,
        "train_minority_ratio": float(np.mean(y_train.astype(str) == minority_label)),
        "test_minority_ratio": float(np.mean(y_test_str == minority_label)),
        "centralized_rule_count": int(len(centralized.rules_)),
        "mean_evidence_span_tau": float(span_rows["span_tau"].mean()),
        "median_evidence_span_tau": float(span_rows["span_tau"].median()),
        "share_span_ge_3": float(np.mean(span_rows["span_tau"] >= 3)),
        "mean_active_party_evidence_share": float(span_rows["active_share"].mean()),
        "mean_effective_parties": float(span_rows["effective_parties"].mean()),
        "mean_js_belief_conflict": float(conflict_rows["belief_js_conflict"].mean()),
        "median_js_belief_conflict": float(conflict_rows["belief_js_conflict"].median()),
        "mean_vote_disagreement": float(conflict_rows["vote_disagreement"].mean()),
        "share_any_vote_disagreement": float(np.mean(conflict_rows["vote_disagreement"] > 0.0)),
        "share_high_vote_disagreement": float(np.mean(conflict_rows["vote_disagreement"] > 0.4)),
        "mean_confidence_spread": float(conflict_rows["confidence_spread"].mean()),
        "mean_fusion_accuracy": float(accuracy_score(y_test_str, mean_pred)),
        "mean_fusion_macro_f1": float(f1_score(y_test_str, mean_pred, average="macro", zero_division=0)),
        "mean_fusion_minority_f1": float(
            f1_score(y_test_str, mean_pred, labels=[minority_label], average="macro", zero_division=0)
        ),
        "party_feature_counts": {str(key): int(len(value)) for key, value in sorted(party_features.items())},
        "party_importance": {str(key): float(value) for key, value in sorted(split.party_importance.items())},
        "classes": classes.tolist(),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
