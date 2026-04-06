# CL Comparative Study — Master Task Checklist

**Scope Notes (Compute-Bound)**
Local Mini-ImageNet ResNet18 runs are now prioritized on the RTX 4050 laptop. Tiny-ImageNet remains top-methods-only, and ViT-Small stays optional / fallback depending on local feasibility.
**Always run with `disable_plots: false` and `disable_checkpoints: false`.**

## Phase 1 — Core Infrastructure ✅ COMPLETE (verified)
- [x] Project scaffold (`src/`, `experiments/`, `results/`, `notebooks/`)
- [x] Config system (YAML per experiment: dataset, method, hyperparams, seeds)
- [x] Seeding & reproducibility utility ([utils/seed.py](file:///d:/Catastrophic%20Forgetting/Project/src/utils/seed.py))
- [x] Logging & checkpointing utility ([utils/logger.py](file:///d:/Catastrophic%20Forgetting/Project/src/utils/logger.py), [utils/checkpoint.py](file:///d:/Catastrophic%20Forgetting/Project/src/utils/checkpoint.py))
- [x] **Dataset loaders**
  - [x] Permuted MNIST (Domain-IL, 10 tasks)
  - [x] Split CIFAR-10 (Class-IL, 5 tasks × 2 classes)
  - [x] Split CIFAR-100 (Class-IL, 20 tasks × 5 classes)
  - [x] Split Mini-ImageNet (Class-IL, 20 tasks × 5 classes) — local runner ready
  - [x] Sequential Tiny-ImageNet (Class-IL, 10 tasks × 20 classes) — local runner ready
- [x] **Model wrappers**
  - [x] Slim ResNet-18 (main local model, ~2M params)
  - [x] ResNet-8 (fast debug model)
  - [x] ViT-Small (high-memory optional, gradient checkpointing + FP16)
  - [x] Expandable classifier head (grows as new classes arrive)
- [x] **CL Training harness** ([trainers/cl_trainer.py](file:///d:/Catastrophic%20Forgetting/Project/src/trainers/cl_trainer.py))
  - [x] Task-sequential training loop
  - [x] Task boundary handling (head expansion, buffer updates)
  - [x] Evaluation after every task (full accuracy matrix)
- [x] **Metrics engine** ([metrics/continual_metrics.py](file:///d:/Catastrophic%20Forgetting/Project/src/metrics/continual_metrics.py))
  - [x] Average Accuracy (AA)
  - [x] Forgetting Measure (F)
  - [x] Backward Transfer (BWT)
  - [x] Forward Transfer (FWT)
  - [x] Time-per-task tracking in trainer
- [x] **Visualization pipeline** ([visualization/plots.py](file:///d:/Catastrophic%20Forgetting/Project/src/visualization/plots.py))
  - [x] Per-task accuracy heatmap
  - [x] Forgetting / BWT / FWT bar charts
  - [x] Sequential training accuracy curves
  - [x] Pareto frontier plot (accuracy vs. memory)
  - [x] Compute cost chart (time per task)
  - [x] Summary grid (heatmap + curves + metric header)
- [x] **All method files written**
  - [x] Baselines: FineTune, JointTraining, EWC, A-GEM, LwF
  - [x] Hybrids: DER, X-DER, iCaRL, ER+EWC, Progress & Compress, A-GEM+Distill, SI+DER
- [x] Method & dataset registries ([get_method](file:///d:/Catastrophic%20Forgetting/Project/src/methods/__init__.py#35-49), [get_dataset](file:///d:/Catastrophic%20Forgetting/Project/src/datasets/__init__.py#19-37), [get_model](file:///d:/Catastrophic%20Forgetting/Project/src/models/__init__.py#15-36))
- [x] Experiment runner CLI ([experiments/run_experiment.py](file:///d:/Catastrophic%20Forgetting/Project/experiments/run_experiment.py))
- [x] Batch runner for all seeds ([experiments/run_all.py](file:///d:/Catastrophic%20Forgetting/Project/experiments/run_all.py))
- [x] 10+ YAML configs created (all dataset × method combos)
- [x] Local Phase 4 runners: `Project/notebooks/local_mini_imagenet.py`, `Project/notebooks/local_tiny_imagenet.py`
- [x] Test suite: [tests/test_phase1.py](file:///d:/Catastrophic%20Forgetting/Project/tests/test_phase1.py) — **10/10 tests PASSED**

## Phase 2 — Run Baselines on Local Datasets
- [x] Run all 5 baselines on Permuted MNIST (5 seeds each, validate AA/F vs. published)
- [x] Run all 5 baselines on Split CIFAR-10
- [x] Run all 5 baselines on Split CIFAR-100 (FP16 enabled)
- [x] Export results to `results/raw/` CSVs and generate comparison plots
- [x] FWT-corrected baseline rerun saved under `results/epoch_1/baselines/fwt/` (Permuted MNIST, Split CIFAR-10, Split CIFAR-100)

## Phase 3 — Run Hybrid Methods
- [x] Run all 7 hybrids on Permuted MNIST (FWT-enabled) → `results/epoch_1/hybrids/fwt/`
- [x] Backfill missing Permuted MNIST `agem_distill` seeds 789/1024 (final metrics missing)
- [x] Run all 7 hybrids on Split CIFAR-10 (FWT-enabled)
- [x] Run all 7 hybrids on Split CIFAR-100 (FWT-enabled)
- [x] Progress & Compress fixes (KB device + Fisher grads) for stable runs
- [x] SI-DER stability fix (unscale grads for SI accumulation) + Split CIFAR-100 full 5-seed rerun; results copied into `results/epoch_1/hybrids/fwt_sider_fix/`
- [x] Hybrid interaction ablations (component-level toggles where applicable)
- [ ] Progress & Compress interaction ablations deferred (may revisit later; not dropped)
- [x] Ablation config set generated (buffer size, λ, task-length) under `experiments/configs/ablations/`
- [x] Buffer-size ablation complete for Split CIFAR-10 DER (buf100/200/500) → `results/epoch_1/ablations/`
- [x] Ablation studies: buffer size {100, 200, 500}
- [x] Ablation studies: regularization strength λ {1, 10, 100, 1000}
- [x] Ablation studies: task sequence length {5, 10} where applicable
- [x] Statistical analysis (paired t-tests, Bonferroni, Cohen's d, confidence intervals) → `results/epoch_1/analysis/phase3/`
- [x] Generate Pareto frontier plots per dataset (see `results/epoch_1/.../figures/`)
- [x] Ensure configs exist for all dataset × method combos used in the study (incl. ablations)

## Phase 4 — Local Large-Dataset Experiments
- [ ] Mini-ImageNet: all methods, Slim ResNet-18, FP16, resume-enabled local runs
- [ ] Mini-ImageNet: optional ViT-Small subset if local memory permits
- [ ] Tiny-ImageNet: top 3-4 hybrid methods, local-first feasibility pass
- [ ] Merge local Phase 4 CSVs + PNGs into master results

## Phase 5 — Analysis & Decision Framework
- [ ] Aggregate all results into `results/master_results.csv`
- [ ] Statistical significance notebook
- [ ] Pareto optimality analysis
- [ ] Decision tree design + visualization (inputs: memory, compute, forgetting threshold, task similarity)
- [x] Paper-ready summary table (mean ± std) → `results/epoch_1/analysis/phase3/paper_ready_summary_pretty.csv`

## Phase 6 — Frontend (Streamlit Dashboard)
- [ ] Dashboard scaffold (`Project/app/main.py`, to be created)
- [ ] Hardware constraint input form (memory, compute, forgetting threshold, task similarity)
- [ ] Algorithm recommendation engine
- [ ] Embedded Pareto charts, heatmaps, metric tables
- [ ] Interactive method comparison view

## Phase 7 — Higher-Epoch Revisit + Stronger Models
- [ ] Re-run key baselines/hybrids at higher epochs (e.g., 5/10) for convergence stability
- [ ] Evaluate stronger backbones (e.g., wider ResNet-18 or ViT-Small where feasible)
- [ ] Pretrain on half the tasks jointly, then continue sequentially on remaining tasks to measure forgetting (baseline vs hybrid)
