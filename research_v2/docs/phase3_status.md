# Phase 3 Status

## Outcome
Phase 3 is complete.

Verified on April 14, 2026 in the `genai` environment with CUDA:
- `conda run -n genai python -m pytest research_v2/Project/tests/test_phase3_methods.py -q`
  - result: `47 passed`

## What Phase 3 Covered
The v2 method gate now includes:
- all 5 baselines
- all 7 hybrids
- Class-IL tiny smoke passes
- Domain-IL tiny smoke passes
- state round-trip checks for stateful methods

## Methods Verified
### Baselines
- `fine_tune`
- `joint_training`
- `ewc`
- `agem`
- `lwf`

### Hybrids
- `der`
- `xder`
- `icarl`
- `er_ewc`
- `progress_compress`
- `agem_distill`
- `si_der`

## Smoke-Test Scenarios
### Class-IL smoke path
- dataset: `split_cifar10`
- backbone: `slim_resnet18`
- device: `cuda`
- procedure:
  - task 0 tiny observe pass
  - tiny `after_task()` path
  - task 1 tiny observe pass

### Domain-IL smoke path
- dataset: `permuted_mnist`
- backbone: `slim_resnet18`
- device: `cuda`
- procedure:
  - task 0 tiny observe pass only
  - resized 32x32 PMNIST path

### Stateful method verification
Checked for methods with replay buffers or extra consolidation state:
- `joint_training`
- `ewc`
- `agem`
- `der`
- `xder`
- `icarl`
- `er_ewc`
- `progress_compress`
- `agem_distill`
- `si_der`

These methods were required to:
- save state without error
- load state into a fresh instance without error
- preserve the expected core internal structures

## Implementation Notes
- The Phase 3 test suite is now CUDA-first rather than CPU-forced.
- Tiny subset loaders are used intentionally so `after_task()` logic is exercised without accidentally running full dataset passes.
- This fixes the earlier pseudo-hang behavior, which came from smoke tests feeding full task loaders into expensive end-of-task logic.

## Runnable Status
All 12 methods are currently marked runnable for the v2 primary matrix gate.

## Known Caveat
Replay buffers and checkpoint payloads still intentionally store tensors on
CPU for memory efficiency and safer persistence. That is a storage policy,
not an execution-device fallback.

## Main Artifact
- `research_v2/Project/tests/test_phase3_methods.py`
