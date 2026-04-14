# Experiment Matrix v2

## Core Execution Matrix
### Datasets
- Permuted MNIST
- Split CIFAR-10
- Split CIFAR-100
- Split Mini-ImageNet

### Methods
#### Baselines
- fine_tune
- joint_training
- ewc
- agem
- lwf

#### Hybrids
- der
- xder
- icarl
- er_ewc
- progress_compress
- agem_distill
- si_der

### Seeds
Primary seed set:
- 42
- 123
- 456
- 789
- 1024

### Epoch Schedules
Planned order:
- epoch 1 first for the full matrix
- epoch 5 after epoch 1 is complete and aggregated
- epoch 10 after epoch 5 sanity checks pass

## Backbone Matrix
### Required backbone
- `slim_resnet18`

### Secondary planned backbone
- `vit_small_patch16_224`

### Additional stretch backbone
- `convnext_tiny`

## Backbone Execution Policy
- Run the full mandatory study on `slim_resnet18`
- Attempt the same study workflow on `vit_small_patch16_224`
- Attempt the same study workflow on `convnext_tiny`
- If a secondary backbone works and produces reliable runs, keep its results
- If a secondary backbone fails, is too unstable, or is too expensive on this machine, skip it explicitly and document the reason

## Verification Before Primary Runs
Before any real epoch-1 sweep:
- verify every method implementation
- run a tiny functional smoke pass for every method
- use very small batches / very few images, not a full epoch
- only mark a method as runnable once it completes the smoke pass cleanly
- fix unstable or broken methods before entering the primary matrix

## Execution Priority
### Pass 1
- implementation verification and tiny smoke runs for all methods

### Pass 2
- all baselines and hybrids on all four datasets at epoch 1 with `slim_resnet18`

### Pass 3
- ablations at epoch 1
- only after the primary epoch-1 matrix is complete and stable

### Pass 4
- epoch 5 reruns for the primary matrix

### Pass 5
- epoch 10 reruns for the primary matrix

### Parallel / optional track
- attempt `vit_small_patch16_224`
- attempt `convnext_tiny`
- retain results only if stable and usable

## Ablation Families
### Interaction ablations
- remove distillation where applicable
- remove regularization where applicable
- remove revision / classifier variants where applicable

### Hyperparameter ablations
- buffer size
- EWC / SI / distillation lambda values
- replay ratio or method-specific trade-off weights

### Stress / scaling ablations
- task-count or task-length changes where meaningful
- higher-epoch sensitivity
- resume/restart robustness on representative methods

## Required Aggregation Levels
For each dataset × backbone × method × epoch setting:
- per-seed raw logs
- per-seed metric outputs
- separated figure outputs
- aggregated mean and std
- runtime summary
- memory/buffer summary
- note of any caveats or reruns
