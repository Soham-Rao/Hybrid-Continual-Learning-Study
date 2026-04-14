# Research Context v2

## Purpose
This workspace exists to convert the earlier deadline-driven prototype into a proper research project. The prototype taught us what works, what breaks, what claims are safe, and what parts of the pipeline need to be rebuilt more rigorously. This v2 folder is where the clean execution should happen.

## Source of Truth
- v1 archive path: `../v1_deadline_prototype/`
- v1 docs path: `../v1_deadline_prototype/Docs/`
- v1 code path: `../v1_deadline_prototype/Project/`
- v1 results path: `../v1_deadline_prototype/results/`

Use v1 as reference for:
- implemented methods
- prior bugs and fixes
- prior runtime constraints
- prototype analysis outputs
- dashboard concepts

Do not assume v1 outputs are final publication evidence.

## Study Goal
Run a full comparative continual-learning study on this machine across the four core datasets:
- Permuted MNIST
- Split CIFAR-10
- Split CIFAR-100
- Split Mini-ImageNet

The final v2 study should include:
- baselines
- hybrids
- ablations
- statistical analysis
- recommendation engine v2
- dashboard v2
- paper-ready evidence and claims

## Machine / Execution Assumption
- Primary execution target: this machine
- GPU assumption: RTX 4050 laptop GPU
- Local-first execution is the default plan for all four core datasets
- Tiny-ImageNet is not part of the mandatory core matrix for v2 unless explicitly added later

## Locked Phase 0 Decisions
- Broad workspace remains at `research_v2/`
- Code, configs, tests, datasets, notebooks, and app code live under `research_v2/Project/`
- Results are separated into `runs/`, `figures/`, `analysis/`, and `ablations/`
- Image outputs stay separate from logs, metrics, checkpoints, and other run artifacts
- Checkpoints are kept until aggregation/verification passes, then deleted
- All 12 methods remain in scope
- Every method must be verified first at implementation level, then by tiny smoke runs on very small batches/images before full runs begin

## Current Progress
- Phase 0 complete
- Phase 1 complete
- Phase 2 complete
- Phase 3 complete
- Current next step: Phase 4 primary epoch-1 study

## Backbone Policy
### Required backbone
- `slim_resnet18`

### Additional backbone attempts
- `vit_small_patch16_224`
- `convnext_tiny`

Policy:
- `slim_resnet18` is compulsory for the real study
- `vit_small_patch16_224` should be attempted after the compulsory path is stable
- `convnext_tiny` should also be attempted after the compulsory path is stable
- If either secondary backbone runs successfully and produces usable results, keep it in the study
- If a secondary backbone is unstable or infeasible, skip it explicitly rather than forcing it

## What v1 Taught Us
Important prototype lessons to carry forward:
- Resume support matters and must remain reliable
- Checkpoint cleanup after successful completion is useful and should remain part of the workflow
- AGEM-family methods previously had implementation/runtime issues and must be treated carefully in v2 validation
- Numerical stability and mixed-device restore paths need dedicated tests, not only ad hoc fixes
- Phase 5 and Phase 6 should consume structured outputs, not rely on scattered logs
- Claims must be driven by rerun, verified evidence instead of prototype-only observations

## v2 Research Standard
v2 is only complete when:
- the experiment matrix is explicitly defined
- all primary runs are executed under the intended settings
- all major claims are backed by rerun results
- analysis and recommendation outputs are regenerated from v2 evidence
- the dashboard reflects v2 outputs, not the archived prototype
