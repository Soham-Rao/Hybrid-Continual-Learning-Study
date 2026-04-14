# Continual Learning Research Project v2

This folder is the clean workspace for the real study after the deadline-driven prototype pass.

## Workspace Layout
- `docs/` holds broad project documentation, planning, scope, and research-facing notes.
- `Project/` holds code, configs, tests, notebooks, app code, and local dataset paths.
- `results/` holds all v2 outputs, separated into run artifacts, figures, analysis, and ablations.

## Current Research Position
- `v1_deadline_prototype/` is the archived learning/prototype version.
- `research_v2/` is the real study workspace.
- All mandatory baselines, hybrids, and ablations are to be executed locally on this machine first.

## Core Study Targets
- Datasets: Permuted MNIST, Split CIFAR-10, Split CIFAR-100, Split Mini-ImageNet
- Methods: 5 baselines + 7 hybrids
- Epoch schedules: 1, then 5, then 10
- Primary required backbone: `slim_resnet18`
- Additional backbone attempts: `vit_small_patch16_224`, then `convnext_tiny` if feasible

## Start Here
- `docs/context.md`
- `docs/project_scope.md`
- `docs/folder_structure.md`
- `docs/experiment_matrix.md`
- `docs/tasklist.md`
