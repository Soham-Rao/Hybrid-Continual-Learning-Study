# Pre-Phase 4 V2 Preflight

Date: 2026-04-14

This note records the final implementation check performed before starting
the v2 research runs.

## Scope Checked

- Phase 0 workspace structure and results layout
- Phase 1 infrastructure, config loading, path resolution, logging, checkpoints
- Phase 2 dataset availability and backbone validation
- Phase 3 method reimplementation checks and method-level smoke tests
- CUDA-only execution policy for v2 training and test execution
- Known v1 implementation issues that had to be fixed before v2

## Verified Results

- Combined pytest run passed:
  - `research_v2/Project/tests/test_phase1_infrastructure.py`
  - `research_v2/Project/tests/test_phase2_datasets_models.py`
  - `research_v2/Project/tests/test_phase3_methods.py`
- Result: `65 passed`
- `compileall` passed on:
  - `research_v2/Project/src`
  - `research_v2/Project/experiments`
  - `research_v2/Project/tests`
- End-to-end runner sanity check passed:
  - `run_experiment.py` with `smoke_test.yaml`
  - CUDA device used correctly
  - logs, metrics, figures, checkpoints, and cleanup behaved correctly

## Dataset Status

The v2 dataset root is:

- `research_v2/Project/data_local/`

Current entries are Windows junctions pointing to the archived prototype
dataset mirrors:

- `MNIST`
- `cifar-10-batches-py`
- `cifar-100-python`
- `mini-imagenet`
- `tiny-imagenet-200`

This is acceptable for v2 startup because the paths exposed to the code are
clean v2-local paths, while storage reuse avoids duplication.

## CUDA / Device Policy

The v2 execution path is CUDA-only for training and test execution:

- `run_experiment.py` rejects non-CUDA devices
- `BaseCLMethod` rejects non-CUDA devices
- Phase 2 and Phase 3 tests now build models and run forward passes on CUDA
- legacy `torch.cuda.amp` calls were migrated to `torch.amp`

Intentional CPU storage still exists only for memory-management and checkpoint
storage paths:

- replay buffer storage
- iCaRL exemplar storage
- joint-training cached samples

These are kept on CPU intentionally to avoid wasting VRAM and do not represent
CPU execution fallback.

## V1 Issues Rechecked in V2

The major v1 issues that mattered before v2 runs were rechecked:

- A-GEM gradient handling uses a separate memory-gradient path and no longer
  overwrites the current gradient incorrectly
- AGEM+Distill uses the same corrected projection logic
- replay-buffer checkpoint restore normalizes stored tensors back to CPU,
  preventing mixed CPU/CUDA resume bugs
- iCaRL prunes exemplars per class and recomputes class means from the actual
  stored exemplar set
- resume/checkpoint flow restores model/method state at task boundaries
- v2 output paths separate:
  - run artifacts
  - figures
  - ablations
  - analysis

## Readiness Decision

V2 is ready to start Phase 4 runs.

The only caveat is operational rather than structural:

- optional larger backbones (`vit_small_patch16_224`, `convnext_tiny`) should
  still be attempted methodically and skipped only if they prove unstable or
  infeasible during actual run execution

For the required core study, the scaffold is ready.
