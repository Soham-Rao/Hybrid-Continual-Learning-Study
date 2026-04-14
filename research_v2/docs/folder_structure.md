# Folder Structure v2

## Top Level
### `docs/`
Broad project documentation, planning, scope, novelty framing, experiment matrix, and task tracking.

### `Project/`
The actual code workspace.
Expected contents:
- `src/`
- `experiments/`
- `notebooks/`
- `tests/`
- `app/`
- `data/`
- `data_local/`

### `results/`
All v2 outputs, intentionally separated by artifact type.

## Results Structure
### Primary roots
- `results/runs/`
- `results/figures/`
- `results/analysis/`
- `results/ablations/`

### Epoch partitioning
Each root is partitioned by epoch schedule where relevant:
- `epoch_1/`
- `epoch_5/`
- `epoch_10/`

## Run Artifact Structure
For non-image outputs, the canonical structure is:
`results/runs/<epoch>/<dataset>/<method_family>/<method>/seed_<seed>/`

Inside each seed folder, keep separate subfolders:
- `logs/`
- `metrics/`
- `checkpoints/`
- `artifacts/`

## Figure Structure
For image outputs, keep figures separate from logs/metrics:
`results/figures/<epoch>/<dataset>/<method_family>/<method>/seed_<seed>/`

For analysis-wide figures:
- `results/figures/<epoch>/analysis/`

For ablation figures:
- `results/figures/<epoch>/ablations/<dataset>/<ablation_family>/<method_variant>/`

## Ablation Structure
Ablations must not be mixed into primary run trees.
Use:
`results/ablations/<epoch>/<dataset>/<ablation_family>/<method_variant>/`

Inside that tree, keep:
- per-seed run outputs
- aggregated summaries
- ablation-specific notes if needed

## Analysis Structure
Use `results/analysis/<epoch>/` for:
- aggregated master results
- paper-ready summary tables
- statistical test outputs
- recommendation-engine inputs/outputs
- final report-ready markdown summaries

## Checkpoint Policy
- Keep checkpoints during active experiments.
- Keep them through aggregation and verification.
- Delete them only after the corresponding run family has been verified and summarized.

## Dataset Policy
- Local dataset mirrors live under `Project/data_local/`.
- `Project/data/` can be used for temporary/debug download paths if needed.
- Datasets should remain ignored in git.
