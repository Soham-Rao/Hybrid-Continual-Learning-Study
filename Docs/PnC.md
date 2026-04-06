# PnC

Post-report continuation checklist. This is the master hierarchical run plan so later reruns do not get mixed up with the immediate Phase 4 + report deadline work.

## Phase A: Immediate Deadline Track

- [ ] Phase 4: Mini-ImageNet local runs
  - [ ] Dataset: `split_mini_imagenet`
    - [ ] Model: `slim_resnet18`
      - [ ] Methods: baselines
        - [ ] `fine_tune`
        - [ ] `joint_training`
        - [ ] `ewc`
        - [ ] `agem`
        - [ ] `lwf`
      - [ ] Methods: hybrids
        - [ ] `der`
        - [ ] `xder`
        - [ ] `icarl`
        - [ ] `er_ewc`
        - [ ] `progress_compress`
        - [ ] `agem_distill`
        - [ ] `si_der`
      - [ ] Seeds
        - [ ] `42`
        - [ ] `123`
        - [ ] `456`
        - [ ] `789`
        - [ ] `1024`
      - [ ] Epochs
        - [ ] `1`
  - [ ] Optional if time permits
    - [ ] Dataset: `split_mini_imagenet`
      - [ ] Model: `vit_small`
        - [ ] Methods: top subset only
          - [ ] `fine_tune`
          - [ ] `der`
          - [ ] `xder`
          - [ ] `icarl`
          - [ ] `si_der`
        - [ ] Seeds
          - [ ] `42`
          - [ ] `123`
          - [ ] `456`
          - [ ] `789`
          - [ ] `1024`
        - [ ] Epochs
          - [ ] `1`

- [ ] Phase 5: analysis + recommendation layer
  - [ ] Aggregate Phase 4 metrics into summary tables
  - [ ] Update plots
  - [ ] Add report-ready comparisons
  - [ ] Add recommendation logic draft

- [ ] Phase 6: lightweight dashboard / presentation layer
  - [ ] Minimal dashboard scaffold
  - [ ] Load summary tables
  - [ ] Show method comparison
  - [ ] Show dataset/method filters

## Phase B: Post-Report Full Rerun Campaign

- [ ] Epoch-1 full rerun campaign
  - [ ] Dataset: `permuted_mnist`
    - [ ] Methods: baselines
      - [ ] `fine_tune`
      - [ ] `joint_training`
      - [ ] `ewc`
      - [ ] `agem`
      - [ ] `lwf`
    - [ ] Methods: hybrids
      - [ ] `der`
      - [ ] `xder`
      - [ ] `icarl`
      - [ ] `er_ewc`
      - [ ] `progress_compress`
      - [ ] `agem_distill`
      - [ ] `si_der`
    - [ ] Seeds
      - [ ] `42`
      - [ ] `123`
      - [ ] `456`
      - [ ] `789`
      - [ ] `1024`
  - [ ] Dataset: `split_cifar10`
    - [ ] Methods: baselines
      - [ ] `fine_tune`
      - [ ] `joint_training`
      - [ ] `ewc`
      - [ ] `agem`
      - [ ] `lwf`
    - [ ] Methods: hybrids
      - [ ] `der`
      - [ ] `xder`
      - [ ] `icarl`
      - [ ] `er_ewc`
      - [ ] `progress_compress`
      - [ ] `agem_distill`
      - [ ] `si_der`
    - [ ] Seeds
      - [ ] `42`
      - [ ] `123`
      - [ ] `456`
      - [ ] `789`
      - [ ] `1024`
  - [ ] Dataset: `split_cifar100`
    - [ ] Methods: baselines
      - [ ] `fine_tune`
      - [ ] `joint_training`
      - [ ] `ewc`
      - [ ] `agem`
      - [ ] `lwf`
    - [ ] Methods: hybrids
      - [ ] `der`
      - [ ] `xder`
      - [ ] `icarl`
      - [ ] `er_ewc`
      - [ ] `progress_compress`
      - [ ] `agem_distill`
      - [ ] `si_der`
    - [ ] Seeds
      - [ ] `42`
      - [ ] `123`
      - [ ] `456`
      - [ ] `789`
      - [ ] `1024`
  - [ ] Dataset: `split_mini_imagenet`
    - [ ] Model: `slim_resnet18`
    - [ ] Methods: full baseline + hybrid set
    - [ ] Seeds: `42,123,456,789,1024`
  - [ ] Dataset: `seq_tiny_imagenet`
    - [ ] Model: top feasible backbone
    - [ ] Methods: top subset first
    - [ ] Seeds: `42,123,456,789,1024`

- [ ] Epoch-5 rerun campaign
  - [ ] `permuted_mnist`
  - [ ] `split_cifar10`
  - [ ] `split_cifar100`
  - [ ] `split_mini_imagenet`
  - [ ] `seq_tiny_imagenet` if feasible

- [ ] Epoch-10 rerun campaign
  - [ ] `permuted_mnist`
  - [ ] `split_cifar10`
  - [ ] `split_cifar100`
  - [ ] `split_mini_imagenet`
  - [ ] `seq_tiny_imagenet` if feasible

## Phase C: Ablation Completion / Cleanup

- [ ] Buffer-size ablations
  - [ ] `der`
  - [ ] `xder`
  - [ ] `icarl`
  - [ ] `er_ewc`
  - [ ] `agem_distill`
  - [ ] `si_der`

- [ ] Lambda ablations
  - [ ] `ewc`
  - [ ] `er_ewc`
  - [ ] `si_der`
  - [ ] `agem_distill` distillation weight sweep

- [ ] Interaction ablations
  - [ ] `der_nodistill`
  - [ ] `xder_norevision`
  - [ ] `icarl_nonmc`
  - [ ] `er_ewc_noewc`
  - [ ] `agem_distill_nodistill`
  - [ ] `si_der_nosi`
  - [ ] `progress_compress` interaction ablation completion

- [ ] Task-length / sequence-length ablations
  - [ ] `der`
  - [ ] `si_der`

## Phase D: Final Paper Package

- [ ] Rebuild all master result tables
- [ ] Recompute statistics after full reruns
- [ ] Rebuild paper-ready figures
- [ ] Freeze final claims to only rerun-backed findings
- [ ] Archive stale pre-fix result folders separately
