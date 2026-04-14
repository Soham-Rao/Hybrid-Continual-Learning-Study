# Project Scope v2

## Core Question
Which continual-learning methods and hybrids best balance accuracy, forgetting, memory, and compute under a controlled local training budget across four datasets of increasing difficulty?

## Mandatory Study Scope
### Datasets
- Permuted MNIST
- Split CIFAR-10
- Split CIFAR-100
- Split Mini-ImageNet

### Baselines
- Fine-Tune
- Joint Training
- EWC
- A-GEM
- LwF

### Hybrid Methods
- DER
- X-DER
- iCaRL
- ER+EWC
- Progress & Compress
- A-GEM+Distill
- SI-DER

### Ablations
- interaction ablations for hybrid components
- replay buffer sweeps where relevant
- regularization/distillation weight sweeps where relevant
- task-length or task-count stress tests where meaningful
- epoch schedule reruns after the primary epoch-1 study is complete

## Mandatory Outputs
- reproducible codebase
- clean experiment configs
- raw and aggregated result tables
- plots for comparison and trade-offs
- statistical analysis
- recommendation engine v2
- dashboard v2
- paper-ready claim set

## Out of Scope For The Mandatory Core
- Tiny-ImageNet as a required dataset
- cloud-first execution plans
- feature creep in the dashboard beyond research communication value
- unsupported claims of SOTA unless v2 evidence justifies them

## Quality Bar
The v2 project should be treated as complete only when the study is coherent end to end:
- methods are revalidated
- experiments are rerun under the final matrix
- results are aggregated from v2 outputs only
- recommendation logic is justified by v2 evidence
- dashboard is connected to v2 artifacts
