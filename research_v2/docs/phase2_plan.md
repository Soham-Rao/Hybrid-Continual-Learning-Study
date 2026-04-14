# Phase 2 Plan — Dataset and Model Preparation

## Purpose
Phase 2 converts the Phase 1 scaffold into a machine-validated input/model
stack. By the end of Phase 2, every mandatory dataset loader and backbone
path should be locally testable, and every optional backbone should be
either runnable or explicitly skipped with a documented reason.

## Phase 1 Verification Summary
Verified on April 12, 2026:
- `pytest research_v2/Project/tests/test_phase1_infrastructure.py -q`
  - result: `8 passed`
- `python -m compileall research_v2/Project/src research_v2/Project/experiments`
  - result: passed

Phase 1 can be treated as complete for the **core scaffold**:
- v2 pathing exists
- logging / metrics separation works
- checkpoints round-trip correctly
- config loading works
- registries import cleanly

Important carry-forward findings from verification:
- `research_v2/Project/data_local/` is currently empty
- regular root configs exist for:
  - Permuted MNIST
  - Split CIFAR-10
  - Split CIFAR-100
- regular root configs do **not** yet exist for Split Mini-ImageNet
- legacy imported configs under `Project/experiments/configs/phase4/` still use prototype-era output paths and should not be treated as final v2 configs
- `vit_small_patch16_224` is available only as an alias to the existing ViT wrapper
- `convnext_tiny` is not implemented yet

## Phase 2 Success Criteria
Phase 2 is complete only when all of the following are true:
- all four mandatory dataset loaders work locally on this machine
- `slim_resnet18` is validated for the dataset scenarios we will actually run
- expandable classifier behavior is verified for Class-IL use
- `vit_small_patch16_224` is either runnable end-to-end on at least one smoke case or explicitly marked as blocked with reason
- `convnext_tiny` is either runnable end-to-end on at least one smoke case or explicitly marked as blocked with reason
- v2 dataset/model tests exist and pass
- v2 Mini-ImageNet configs exist outside the legacy `phase4/` folder

## Execution Order

### Step 1 — Populate / verify local dataset roots
Expected dataset availability under `research_v2/Project/data_local/`:
- `MNIST/` or torchvision-managed MNIST cache
- `cifar-10-batches-py/` or torchvision-managed CIFAR-10 cache
- `cifar-100-python/` or torchvision-managed CIFAR-100 cache
- `mini-imagenet/`
  - `train/<class>/*.jpg`
  - optional `test/<class>/*.jpg`

Actions:
- inspect what is already available in `v1_deadline_prototype/Project/data_local/`
- copy or reuse local mirrors into `research_v2/Project/data_local/`
- keep the canonical v2 path policy under `research_v2/Project/data_local/`

Deliverable:
- machine-readable note of which datasets are already present and which were populated during Phase 2

### Step 2 — Validate all four mandatory dataset loaders
Datasets in scope:
- `permuted_mnist`
- `split_cifar10`
- `split_cifar100`
- `split_mini_imagenet`

Validation checks for each loader:
- constructor works locally
- reported `n_tasks`, `n_classes_per_task`, `n_classes_total`, and `scenario` are correct
- first task train/test loaders produce batches successfully
- tensor shapes and labels are correct
- per-task class partitions are correct for Class-IL datasets
- permutation differences are correct for Permuted MNIST
- Mini-ImageNet local root logic works with the actual on-disk layout

Deliverable:
- a dedicated dataset smoke test module in `research_v2/Project/tests/`

### Step 3 — Validate `slim_resnet18`
Checks:
- model builds for RGB input
- feature dimension is correct
- classifier head expansion preserves old weights
- forward pass works for CIFAR-sized inputs
- forward pass works for Mini-ImageNet-sized inputs
- parameter count remains consistent with the intended lightweight design

Deliverable:
- Phase 2 model tests covering `slim_resnet18`

### Step 4 — Validate expandable-head behavior
Checks:
- Class-IL head expands correctly across tasks
- old logits remain addressable after expansion
- Domain-IL datasets do not require unnecessary expansion

Deliverable:
- explicit tests for expandable head behavior under both Class-IL and Domain-IL assumptions

### Step 5 — Make `vit_small_patch16_224` a real v2 option
Current state:
- registry alias exists
- no dedicated v2 validation yet

Required work:
- verify `timm` import works in the active env
- decide how image resolution will be handled for each dataset
  - Mini-ImageNet loader currently produces `84x84`
  - Tiny-ImageNet loader currently produces `64x64`
  - ViT name implies `224x224`
- either:
  - resize inputs to `224x224` in a ViT-specific data path, or
  - prove a smaller image-size path is valid and stable
- run a tiny forward/backward smoke pass

Deliverable:
- documented ViT data-policy decision
- smoke-tested ViT path or explicit skip note

### Step 6 — Integrate `convnext_tiny`
Current state:
- planned in policy only
- not implemented in `Project/src/models/`

Required work:
- add a `convnext_tiny` backbone wrapper
- register it in `Project/src/models/__init__.py`
- document feature dimension
- validate a tiny forward/backward pass
- note memory/runtime implications on this machine

Deliverable:
- working `convnext_tiny` registry entry or explicit blocked/skip decision

### Step 7 — Rewrite stale v2 config coverage
Current state:
- Mini-ImageNet still depends on imported legacy `phase4/` configs
- those configs contain prototype-era output paths like `results/phase4/...`

Required work:
- add standard v2 configs for Split Mini-ImageNet in `Project/experiments/configs/`
- ensure they rely on base-config + v2 path resolution
- keep legacy imported `phase4/` configs clearly non-canonical or remove them later
- plan optional ViT / ConvNeXt configs only after their paths are validated

Deliverable:
- standard root-level v2 Mini-ImageNet configs

### Step 8 — Build Phase 2 pytest coverage
Recommended test file:
- `research_v2/Project/tests/test_phase2_datasets_models.py`

Coverage should include:
- all mandatory dataset loader smoke checks
- `slim_resnet18` checks
- expandable head checks
- optional backbone import/build checks

Deliverable:
- passing Phase 2 pytest suite

## Recommended Definition of Done
Mark Phase 2 complete only after:
1. mandatory datasets load locally
2. `slim_resnet18` is verified
3. Mini-ImageNet has canonical v2 configs
4. optional backbone status is explicitly recorded
5. Phase 2 tests pass in `genai`

## Risks To Watch
- Mini-ImageNet path/layout drift
- ViT input-size mismatch versus dataset transforms
- ConvNeXt memory/runtime cost on the RTX 4050 laptop GPU
- silent reliance on legacy prototype configs
- dataset download assumptions leaking into tests instead of using the local mirrors

## Immediate Next Actions
1. inspect `v1_deadline_prototype/Project/data_local/` and map what can be reused
2. write `test_phase2_datasets_models.py`
3. validate mandatory dataset loaders one by one
4. confirm `slim_resnet18`
5. then move to ViT and ConvNeXt attempts
