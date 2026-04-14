# Continual Learning Research v2 — Master Task Checklist

**Working Assumption**
This is the real study workspace. All baselines, hybrids, and planned ablations for the four core datasets are to be conducted locally on this machine first.

**Core Datasets**
- Permuted MNIST
- Split CIFAR-10
- Split CIFAR-100
- Split Mini-ImageNet

**Method Families**
- Baselines: Fine-Tune, Joint Training, EWC, A-GEM, LwF
- Hybrids: DER, X-DER, iCaRL, ER+EWC, Progress & Compress, A-GEM+Distill, SI-DER

**Backbones**
- Required: `slim_resnet18`
- Secondary attempt: `vit_small_patch16_224`
- Additional stretch attempt: `convnext_tiny`

## Phase 0 — Reset, Archive, and Workspace Setup ✅ COMPLETE
- [x] Archive prototype work under `../v1_deadline_prototype/`
- [x] Create clean v2 workspace structure
- [x] Create fresh v2 docs set
- [x] Lock v2 folder conventions for code, configs, results, and figures
- [x] Lock naming / artifact policy decisions for v2
- [x] Move code-facing folders under `research_v2/Project/`
- [x] Create separated result roots for runs, figures, analysis, and ablations
- [x] Lock checkpoint retention policy: keep until aggregation/verification, then delete
- [x] Lock backbone policy for required and optional backbone attempts

## Phase 1 — Rebuild / Revalidate Core Infrastructure ✅ COMPLETE
- [x] Decide what code is imported from v1 versus rewritten cleanly in v2
- [x] Recreate project scaffold inside `Project/` (`src/`, `experiments/`, `notebooks/`, `tests/`, `app/`)
- [x] Recreate config system for dataset × method × epoch × seed runs
- [x] Recreate deterministic seeding utilities
- [x] Recreate logging utilities with clean raw/summary separation
- [x] Recreate checkpoint/resume utilities with cleanup-on-success behavior
- [x] Recreate dataset registry
- [x] Recreate method registry
- [x] Recreate model registry
- [x] Recreate trainer with task-sequential evaluation and accuracy matrix handling
- [x] Recreate metrics module (AA, Forgetting, BWT, FWT, runtime)
- [x] Recreate plotting layer with publication-oriented outputs
- [x] Build v2 unit and smoke test suite before large reruns begin
- [x] Record the v1→v2 porting split in `docs/phase1_porting_audit.md`

## Phase 2 — Dataset And Model Preparation ✅ COMPLETE
- [x] Rebuild Permuted MNIST loader and validate task permutations
- [x] Rebuild Split CIFAR-10 loader and validate class partitions
- [x] Rebuild Split CIFAR-100 loader and validate class partitions
- [x] Rebuild Split Mini-ImageNet loader and validate local data path usage
- [x] Validate all four datasets on this machine with small loader smoke tests
- [x] Rebuild `slim_resnet18`
- [x] Validate expandable head / class handling for Class-IL
- [x] Integrate `vit_small_patch16_224`
- [x] Integrate `convnext_tiny`
- [x] Document exact fallback / skip conditions for optional backbones
- [x] Replace legacy prototype-style Mini-ImageNet configs with canonical v2 configs
- [x] Follow `docs/phase2_plan.md` as the detailed execution guide
- [x] Record final Phase 2 status in `docs/phase2_status.md`

## Phase 3 — Method Reimplementation And Verification ✅ COMPLETE
### Baselines
- [x] Reimplement Fine-Tune cleanly in v2
- [x] Reimplement Joint Training cleanly in v2
- [x] Reimplement EWC cleanly in v2
- [x] Reimplement A-GEM cleanly in v2
- [x] Reimplement LwF cleanly in v2
- [x] Add method-level tests for all five baselines

### Hybrids
- [x] Reimplement DER cleanly in v2
- [x] Reimplement X-DER cleanly in v2
- [x] Reimplement iCaRL cleanly in v2
- [x] Reimplement ER+EWC cleanly in v2
- [x] Reimplement Progress & Compress cleanly in v2
- [x] Reimplement A-GEM+Distill cleanly in v2
- [x] Reimplement SI-DER cleanly in v2
- [x] Add method-level tests for all seven hybrids

### Verification gate
- [x] Perform code-level audit of every method implementation
- [x] Run tiny smoke passes for all 12 methods using very small batches / very few images
- [x] Mark each method as runnable only after it passes the smoke check
- [x] Fix broken or unstable methods before primary runs begin
- [x] Record final Phase 3 status in `docs/phase3_status.md`

## Phase 4 — Primary Epoch-1 Study On All Four Datasets (`slim_resnet18` compulsory)
### Permuted MNIST
- [ ] Run all 5 baselines for 5 seeds at epoch 1
- [ ] Run all 7 hybrids for 5 seeds at epoch 1
- [ ] Aggregate and sanity-check results

### Split CIFAR-10
- [ ] Run all 5 baselines for 5 seeds at epoch 1
- [ ] Run all 7 hybrids for 5 seeds at epoch 1
- [ ] Aggregate and sanity-check results

### Split CIFAR-100
- [ ] Run all 5 baselines for 5 seeds at epoch 1
- [ ] Run all 7 hybrids for 5 seeds at epoch 1
- [ ] Aggregate and sanity-check results

### Split Mini-ImageNet
- [ ] Run all 5 baselines for 5 seeds at epoch 1
- [ ] Run all 7 hybrids for 5 seeds at epoch 1
- [ ] Aggregate and sanity-check results

### Cross-cutting checks
- [ ] Confirm all expected raw logs exist
- [ ] Confirm all expected seeds completed for each run family
- [ ] Confirm reruns are labeled clearly and stale runs are isolated
- [ ] Confirm primary epoch-1 matrix is complete before ablations begin

## Phase 5 — Optional Backbone Attempts
### ViT-Small
- [ ] Attempt the planned study workflow on `vit_small_patch16_224`
- [ ] Keep results only if runs are stable and usable
- [ ] Document reasons if skipped or abandoned

### ConvNeXt-Tiny
- [ ] Attempt the planned study workflow on `convnext_tiny`
- [ ] Keep results only if runs are stable and usable
- [ ] Document reasons if skipped or abandoned

## Phase 6 — Ablation Studies v2
### Interaction ablations
- [ ] Define the exact interaction ablation matrix for each hybrid
- [ ] Run interaction ablations on the selected datasets under the final v2 policy
- [ ] Aggregate component-level results and interpret mechanism contributions

### Hyperparameter ablations
- [ ] Buffer size sweeps
- [ ] Regularization / distillation weight sweeps
- [ ] Method-specific trade-off parameter sweeps

### Stress / sensitivity ablations
- [ ] Task-count or task-length variants where meaningful
- [ ] Resume/restart robustness checks on representative methods
- [ ] Runtime / memory sensitivity summaries

## Phase 7 — Statistical Analysis And Study Outputs (v2 version of old Phase 5)
- [ ] Aggregate v2 raw results into a clean master results table
- [ ] Generate paper-ready summary tables for the full v2 epoch-1 matrix
- [ ] Run pairwise significance testing with correction
- [ ] Run effect size reporting
- [ ] Run method ranking tests across datasets
- [ ] Generate accuracy / forgetting / runtime / memory trade-off plots
- [ ] Generate dataset leader summaries with caveat notes
- [ ] Produce a final v2 analysis report sourced only from v2 evidence

## Phase 8 — Recommendation Engine v2
- [ ] Redesign recommendation inputs around the final v2 result space
- [ ] Rebuild recommendation scoring / policy from v2 evidence only
- [ ] Validate recommendation outputs against the final summary tables
- [ ] Create representative recommendation case studies
- [ ] Document what the recommendation engine can and cannot claim

## Phase 9 — Dashboard v2 (v2 version of old Phase 6)
- [ ] Design a cleaner research-facing dashboard structure for v2
- [ ] Connect dashboard data loading to v2 analysis artifacts only
- [ ] Rebuild recommendation panel using the v2 engine
- [ ] Rebuild method comparison workspace using v2 tables
- [ ] Rebuild visual panels using v2 figures
- [ ] Rebuild report/about panel using v2 study outputs
- [ ] Validate local runtime and presentation flow on this machine

## Phase 10 — Higher-Epoch Campaign
- [ ] Rerun the primary matrix at epoch 5
- [ ] Aggregate and compare against epoch-1 conclusions
- [ ] Rerun the primary matrix at epoch 10
- [ ] Aggregate and compare against epoch-5 conclusions
- [ ] Revisit claims that change materially under longer training

## Phase 11 — Final Research Packaging
- [ ] Freeze the final v2 evidence set
- [ ] Freeze the final claim set
- [ ] Finalize plots, tables, and appendix-style details
- [ ] Prepare paper/report structure from v2 artifacts
- [ ] Mark any deferred extensions explicitly instead of leaving them ambiguous

## Completion Rule
v2 should be considered complete only when:
- the four-dataset primary matrix is rerun locally
- baselines, hybrids, and selected ablations are complete
- statistics and recommendation outputs are regenerated from v2 only
- dashboard v2 consumes v2 artifacts only
- claims are backed by the final rerun evidence rather than prototype results
