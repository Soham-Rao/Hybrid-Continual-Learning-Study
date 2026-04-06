# Project Context (Source of Truth)

## What This Project Is
- A comparative continual learning (CL) study focused on hybrid methods that combine replay, distillation, and regularization to mitigate catastrophic forgetting.
- Primary scenario is Class-IL (no task ID at test time). Permuted MNIST is Domain-IL.
- Outputs are: reproducible experiments, metrics and plots, ablations, statistical analysis, and a decision-support dashboard.

## Key Docs (Reference Points)
- `Docs/tasklist.md`: master checklist and phase status.
- `Docs/summary.md`: consolidated tables, ablation notes, feature extraction, and detailed implementation notes.
- `Docs/report.md`: short project abstract and aims.
- `Docs/CatastrophicForgetting.md`: survey + research proposal framing.

## Compute-Bound Decisions (Why We Chose Them)
- Dataset selection differences are compute-bound, not conceptual inconsistency.
- Local runs for MNIST/CIFAR; Colab for Mini-ImageNet and any ViT-Small runs.
- Tiny-ImageNet is reserved for top-method subsets only (cost heavy).
- 1 epoch per task is used for all current experiments to keep compute consistent and comparable across methods.
- A Phase 7 has been added to revisit higher-epoch training and stronger backbones for publishable convergence stability.

## Project Structure (Current)
- Core code: `Project/src/`
- Experiments: `Project/experiments/`
- Colab scaffolding: `colab experiments/mini_imagenet/`
- Results (reorganized): `results/epoch_1/` (with empty `epoch_5/`, `epoch_10/`).

## Results Reorganization (Important)
- All results were restructured under `results/epoch_1/` with hierarchy:
  - `baselines/` (split into `fwt` and `no_fwt`)
  - `hybrids/` (including `fwt` and `fwt_sider_fix`)
  - `ablations/` (interactions, interactions_plots, param_sweeps)
  - `analysis/phase3/`
- Old result roots moved to `results_legacy_sources/` (no deletions performed).

## Phase 1 Status (Complete)
- Dataset loaders, model wrappers, training harness, metrics, plots, method implementations, registries, CLI runners, configs, notebooks, and tests are complete.
- FWT computation fixed in trainer (zero-shot evaluation before training task).

## Phase 2 Status (Complete)
- Baselines for Permuted MNIST, Split CIFAR-10, Split CIFAR-100 completed.
- FWT-corrected baseline reruns stored under `results/epoch_1/baselines/fwt/`.

## Phase 3 Status (Complete, with one deferred item)
- All 7 hybrids completed on MNIST/CIFAR10/CIFAR100 with FWT enabled.
- SI-DER stability fix applied; CIFAR-100 SI-DER rerun stored under `fwt_sider_fix`.
- Interaction ablations done for Permuted MNIST and CIFAR-10 (3 seeds each).
- Buffer/lambda/task-length ablations completed (notably ER+EWC lambda sweep; DER buffer sweep; task-length on MNIST for DER and SI-DER).
- Statistical analysis outputs (paired t-tests, Bonferroni, Cohen’s d, Friedman) in `results/epoch_1/analysis/phase3/`.
- Paper-ready summary tables generated in `results/epoch_1/analysis/phase3/`.
- Deferred: Progress & Compress interaction ablations (explicitly marked in tasklist).

## Phase 4 (In Progress) - Mini-ImageNet on Colab
- Colab packaging created under `colab experiments/mini_imagenet/`:
  - `code/` (full runnable code tree)
  - `colab_cells.txt` (cell-by-cell commands)
  - `run.txt` (concise run commands)
  - `download.txt` (what to download after runs)
- Two zips provided:
  - `content_full.zip` (contains `content/` prefix)
  - `drive_full.zip` (recommended for Drive, contains `code/` and `data/` at Drive root)
- Resume support added in Colab code:
  - `code/experiments/run_experiment.py` loads latest checkpoint if `resume=True`.
  - `code/colab_run_mini_imagenet.py` supports `--resume`.
- GPU usage: default `--device cuda`; falls back to CPU if no GPU.

## Mini-ImageNet Dataset (Local and Drive)
- Dataset downloaded from Kaggle into `Project/data_local/mini-imagenet/`.
- Kaggle download required a valid `kaggle.json` under `%USERPROFILE%\.kaggle\kaggle.json`.
- Dataset extracted into `Project/data_local/mini-imagenet/train/` with class folders.
- Drive-based workflow updated to avoid manual upload failures.

## Colab Run Status (Latest)
- ResNet18 full run was interrupted mid-run.
- `fine_tune` completed all 5 seeds.
- `joint_training` started at seed 42 and was interrupted; resume command is available.
- Resume command example (for joint_training seed 42):
  - In `colab_cells.txt` and `run.txt`, use `--resume` with `--methods joint_training --seeds 42`.

## Storage Constraints (Drive)
- Drive storage ran out; safe deletions recommended:
  - `data/mini-imagenet/` in Drive (dataset is large)
  - `checkpoints/` once runs are complete
  - Keep `master_results.csv` and `figures/` as primary outputs

## Method Implementation Notes (Where to Look)
- Detailed per-method implementation notes are in `Docs/summary.md`.
- Hybrid details include DER, X-DER, iCaRL, ER+EWC, Progress & Compress, A-GEM+Distill, SI-DER.
- Baselines include Fine-Tune, Joint Training, EWC, A-GEM, LwF.

## Why 1 Epoch (Current Results)
- Used to keep compute consistent across methods and ensure comparability.
- 1 epoch per task is standard in CL for budgeted evaluation.
- Phase 7 will revisit with 5–10 epochs for convergence stability.

## Immediate Next Steps (Recommended)
- Resume Mini-ImageNet ResNet18 `joint_training` seed 42 from checkpoint.
- Continue remaining seeds for `joint_training`, then proceed with remaining methods.
- Ensure Drive storage by deleting dataset duplicates and old checkpoints after verifying outputs.
- After ResNet18, run ViT-Small (batch 16, image size 224) if compute allows.
- Download outputs per `colab experiments/mini_imagenet/download.txt`.

## Phase 7 (Planned)
- Rerun key baselines and hybrids with higher epochs (5/10) for convergence.
- Evaluate stronger backbones (wider ResNet or ViT-Small where feasible).
- Additional ablation: pretrain on half tasks jointly, then continue sequentially on remaining tasks to measure forgetting (baseline vs hybrid).

## Known Decisions and Rejections (So We Don’t Repeat)
- Do not store datasets in git; use local `data_local` and Drive.
- Do not rely on manual Drive upload for Mini-ImageNet; use Drive zips.
- Use `disable_checkpoints: false` and `disable_plots: false` for all runs.
- Full Tiny-ImageNet sweep deferred due to compute constraints; only top methods later.

## Where to Check Progress
- `Docs/tasklist.md` for phase status.
- `results/epoch_1/analysis/phase3/summary_metrics.csv` for aggregated metrics.
- `results/epoch_1/analysis/phase3/paper_ready_summary_pretty.csv` for mean ± std tables.
- Colab results in Drive: `/content/drive/MyDrive/colab_results/mini_imagenet/`.
