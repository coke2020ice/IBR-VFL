#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 code/scripts/run_long_tail_ebrb_experiment.py \
  --datasets default_credit,adult,bank_marketing \
  --ratios 0.03,0.02,0.01 \
  --heterogeneity active_weak \
  --n-parties 10 \
  --split-strategy active_weak \
  --rank-strength 3.0 \
  --total-rule-budget 72 \
  --use-ablation-boost-params \
  --splits 3 \
  --max-samples 6000 \
  --output results/strict_vfl_all_ratios_all_methods_ablation_params_active_weak_k10.json \
  --table-output results/strict_vfl_all_ratios_all_methods_ablation_params_active_weak_k10_table.tsv
