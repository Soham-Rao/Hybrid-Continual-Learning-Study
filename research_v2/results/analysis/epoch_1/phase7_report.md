# Phase 7 Analysis Report

This report is generated from the finalized v2 epoch-1 primary matrix only.
Primary inferential claims use matched-seed per-dataset tests on average accuracy and forgetting.

## Dataset Leaders

- `permuted_mnist`: leader `joint_training` with AA 95.3738, forgetting 1.1907, runtime 0.1777 h, memory proxy 2048.00 MB. Top cluster: joint_training|xder|der|si_der|agem_distill|er_ewc|agem|fine_tune|lwf|ewc|progress_compress|icarl.
- `split_cifar10`: leader `joint_training` with AA 63.7080, forgetting 10.9575, runtime 0.0152 h, memory proxy 2048.00 MB. Top cluster: joint_training|icarl|er_ewc|xder|progress_compress|si_der|agem_distill|ewc|der|agem|lwf|fine_tune.
- `split_cifar100`: leader `joint_training` with AA 46.7460, forgetting 6.1010, runtime 0.0184 h, memory proxy 2048.00 MB. Top cluster: joint_training|er_ewc|icarl|progress_compress|si_der|agem_distill|der|lwf|xder|agem|fine_tune|ewc.
- `split_mini_imagenet`: leader `joint_training` with AA 45.1883, forgetting 3.5473, runtime 0.2234 h, memory proxy 2048.00 MB. Top cluster: joint_training|er_ewc|icarl|progress_compress|xder|agem_distill|si_der|ewc|der|fine_tune|agem|lwf.

## Statistical Caveats

- `avg_accuracy` is the primary claim metric; `forgetting` is the main secondary claim metric.
- Wilcoxon signed-rank tests are matched by seed within each dataset and corrected with Holm at alpha 0.05.
- `backward_transfer`, `forward_transfer`, runtime, and memory are descriptive only in Phase 7.
- The cross-dataset Friedman result is secondary because it is based on only four datasets.

- Friedman average-rank summary: statistic=28.1154, p=0.003108 across 4 datasets.

## Trade-Off Highlights

- `permuted_mnist` non-joint Pareto candidates: xder, agem, fine_tune, lwf.
- `split_cifar10` non-joint Pareto candidates: icarl, xder, progress_compress, der, agem, lwf, fine_tune.
- `split_cifar100` non-joint Pareto candidates: er_ewc, icarl, progress_compress, agem_distill, der, lwf.
- `split_mini_imagenet` non-joint Pareto candidates: er_ewc, icarl, xder, der, fine_tune.

## Phase 6 Context

- Completed ablation families: buffer=90, interaction=36, lambda=40, robustness=9, tasklen=20.
- Resume/restart robustness checks completed: 9 representative runs.
