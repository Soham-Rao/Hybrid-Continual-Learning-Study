# Phase 1 Porting Audit

## Goal
Phase 1 in v2 is about rebuilding the reusable research scaffold without
pretending that all v1 implementations are already revalidated. The point
is to keep stable infrastructure, adapt it to the v2 workspace layout, and
leave dataset/method correctness verification to later phases.

## Imported From v1 With Minimal Logic Changes
- `Project/src/datasets/`
- `Project/src/methods/`
- `Project/src/metrics/`
- `Project/src/models/`
- `Project/src/trainers/`
- `Project/src/visualization/`
- `Project/requirements.txt`
- `Project/experiments/run_experiment.py`
- `Project/experiments/run_all.py`
- `Project/experiments/configs/`

These were copied forward because they already provide the working
continual-learning backbone that v2 will refine and revalidate.

## Rewritten / Adapted For v2
- `Project/src/utils/paths.py`
  - new canonical v2 path builder
  - maps runs into `results/runs/...`
  - maps figures into `results/figures/...`
  - maps ablations into `results/ablations/...`
- `Project/src/utils/logger.py`
  - now supports separate `logs/` and `metrics/` directories
- `Project/src/models/__init__.py`
  - added `vit_small_patch16_224` alias for the planned optional backbone
- `Project/experiments/configs/base_config.yaml`
  - switched to v2 defaults (`data_local`, workspace-level `results`)
  - added `result_group` / `ablation_family` fields
- `Project/experiments/run_experiment.py`
  - now routes outputs through the v2 path layer
- `Project/experiments/run_all.py`
  - now writes aggregate outputs into the v2 analysis/figure tree

## Added Fresh In v2
- `Project/tests/test_phase1_infrastructure.py`
  - infrastructure-only pytest coverage for:
    - imports
    - deterministic seeding
    - metric math
    - v2 run-path layout
    - logger separation
    - checkpoint round-trip
    - base-config merge behavior

## Explicitly Not Claimed By Phase 1
Phase 1 does **not** claim:
- that every dataset loader is already validated on this machine
- that every model backbone is already stable
- that every method implementation is fully audited
- that smoke runs or real experiments are already complete

Those belong to later phases:
- Phase 2: dataset + model preparation
- Phase 3: method verification and smoke runs

## Why This Split Matters
The v1 codebase already contains substantial working logic, but the v2
project has different artifact-layout requirements and a stronger
verification standard. Reusing infrastructure now saves time, while still
keeping the scientific validation gates explicit and honest.
