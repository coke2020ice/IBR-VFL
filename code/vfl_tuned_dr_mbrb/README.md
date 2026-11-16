# VFL-Tuned DR-MBRB

This package is a lightweight VFLAIR-style implementation of the paper method
`VFL-Tuned DR-MBRB`.

The code is organized around four roles:

- `config.py`: experiment and method configuration.
- `data.py`: train-fold tabular encoding and vertical feature partitioning.
- `party.py`: active/passive client-level MBRB, built from several local BRB modules.
- `server.py`: unlabeled belief fusion, hard-sample detection, and sample-level weight correction.
- `model.py`: scikit-style `fit`, `predict_proba`, `predict`, validation-driven tuning, and diagnostics.

Client-side method chain:

1. Split each client's local feature view into multiple BRB modules.
2. Produce raw module belief with rule-level ER-style fusion.
3. Apply logit-level LA to each module belief.
4. Build sample-wise module weights from the maximum LA belief.
5. Run client-level MBRB integration in belief space.
6. Return an evidence package containing raw module belief, LA module belief,
   module weights, global feature groups, rule counts, and client belief.

Run a smoke experiment:

```bash
python scripts/run_vfl_tuned_dr_mbrb_demo.py \
  --config configs/vfl_tuned_dr_mbrb_demo.json
```

By default, passive parties do not use labels. For centralized simulation or
ablation runs, set:

```bash
python scripts/run_vfl_tuned_dr_mbrb_demo.py \
  --passive-label-mode simulation_supervised
```
