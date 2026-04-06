# Project Summary

Note: the current repository computes FWT from pre-training zero-shot accuracy minus a baseline term `b_i`, and the current trainer uses chance-level `b_i`. In Class-IL this often drives strongly negative FWT values, so treat FWT here as a zero-shot transfer proxy rather than an especially strong standalone ranking signal.

## Tables

### Permuted MNIST (10 tasks × 10 classes) — 10 fixed pixel permutations of MNIST digits (Domain‑IL).

| Method | Model | Epochs/Task | Batch | LR | Optimizer | Loss | Buffer | Distill λ | EWC λ | SI λ | FP16 | Other | Avg Acc | Forgetting | BWT | FWT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agem (baseline) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | - | - | - | false | agem_mem_batch=64 | 39.55 ± 1.55 | 2.21 ± 0.25 | 15.89 ± 1.67 | 6.19 ± 0.49 |
| ewc (baseline) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 100.0 | - | false | - | 26.58 ± 2.60 | 75.42 ± 2.98 | -75.42 ± 2.98 | 1.19 ± 0.34 |
| fine_tune (baseline) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE | - | - | - | - | false | - | 25.38 ± 1.67 | 77.49 ± 1.78 | -77.49 ± 1.78 | 2.26 ± 0.64 |
| joint_training (baseline) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (joint replay) | full (all tasks) | - | - | - | false | joint_replay_epochs=1 | 93.41 ± 0.04 | 0.70 ± 0.21 | -0.18 ± 0.20 | 2.45 ± 0.48 |
| lwf (baseline) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD (LwF) | - | 1.0 | - | - | false | lwf_lambda=1.0, lwf_temp=2.0 | 33.84 ± 1.82 | 65.07 ± 1.91 | -65.07 ± 1.91 | 0.50 ± 0.74 |
| agem_distill (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 200 | 1.0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 41.78 ± 1.48 | 12.81 ± 1.25 | -12.64 ± 1.09 | 6.69 ± 1.22 |
| der (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | false | der_alpha=0.5 | 60.77 ± 2.39 | 39.30 ± 2.70 | -39.30 ± 2.70 | 1.93 ± 0.77 |
| er_ewc (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | false | er_replay_ratio=0.5 | 42.75 ± 8.12 | 55.89 ± 8.17 | -55.89 ± 8.17 | 2.73 ± 0.42 |
| icarl (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 200 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=True | 48.63 ± 0.49 | 50.25 ± 0.77 | -50.25 ± 0.77 | 5.83 ± 2.72 |
| progress_compress (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + distill + EWC (compress) | - | 1.0 | 100.0 | - | false | pc_distill_w=1.0, pc_compress_epochs=3 | 26.94 ± 1.40 | 76.43 ± 1.50 | -76.43 ± 1.50 | 1.13 ± 0.42 |
| si_der (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | false | - | 65.13 ± 2.07 | 33.91 ± 2.31 | -33.91 ± 2.31 | 1.90 ± 1.00 |
| xder (hybrid) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 200 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 61.67 ± 2.02 | 38.26 ± 2.28 | -38.26 ± 2.28 | 2.19 ± 0.68 |
| agem_distill_nodistill (ablation: no distill) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | 0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 42.56 ± 0.67 | 12.17 ± 0.38 | -12.06 ± 0.46 | 6.03 ± 1.01 |
| der_nodistill (ablation: no distill) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (DER) | 200 | 0 | - | - | false | der_alpha=0.5 | 27.38 ± 1.05 | 75.63 ± 1.14 | -75.63 ± 1.14 | 2.61 ± 0.60 |
| der_ntasks10 (ablation: n_tasks=10) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | false | der_alpha=0.5, n_tasks=10 | 60.77 ± 2.39 | 39.30 ± 2.70 | -39.30 ± 2.70 | 1.93 ± 0.77 |
| der_ntasks5 (ablation: n_tasks=5) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | false | der_alpha=0.5, n_tasks=5 | 78.51 ± 0.86 | 21.93 ± 1.17 | -21.93 ± 1.17 | 1.53 ± 1.49 |
| er_ewc_noewc (ablation: no ewc) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay | 200 | - | 0 | - | false | er_replay_ratio=0.5 | 48.21 ± 1.71 | 51.49 ± 0.90 | -51.49 ± 0.90 | 3.05 ± 0.58 |
| icarl_nonmc (ablation: no nmc) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD | 200 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=False | 45.87 ± 2.82 | 53.80 ± 3.24 | -53.80 ± 3.24 | 1.96 ± 0.86 |
| si_der_nosi (ablation: no si) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE | 200 | 1.0 | - | 0 | false | - | 61.89 ± 1.28 | 38.01 ± 1.43 | -38.01 ± 1.43 | 2.01 ± 1.07 |
| si_der_ntasks10 (ablation: n_tasks=10) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | false | n_tasks=10 | 65.13 ± 2.07 | 33.91 ± 2.31 | -33.91 ± 2.31 | 1.90 ± 1.00 |
| si_der_ntasks5 (ablation: n_tasks=5) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | false | n_tasks=5 | 80.27 ± 0.64 | 19.40 ± 0.88 | -19.40 ± 0.88 | 1.50 ± 1.27 |
| xder_norevision (ablation: no revision) | ResNet‑8 (8 layers) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) (no revision) | 200 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 61.37 ± 0.65 | 38.59 ± 0.80 | -38.59 ± 0.80 | 1.30 ± 0.60 |

### Split CIFAR‑10 (5 tasks × 2 classes) — CIFAR‑10 split into sequential class pairs (Class‑IL).

| Method | Model | Epochs/Task | Batch | LR | Optimizer | Loss | Buffer | Distill λ | EWC λ | SI λ | FP16 | Other | Avg Acc | Forgetting | BWT | FWT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agem (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | - | - | - | false | agem_mem_batch=64 | 24.44 ± 1.98 | 13.49 ± 3.35 | 9.35 ± 3.29 | -15.58 ± 0.37 |
| ewc (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 100.0 | - | false | - | 15.98 ± 1.09 | 80.40 ± 1.19 | -80.40 ± 1.19 | -15.58 ± 0.37 |
| ewc_lambda1 (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 1.0 | - | false | - | 16.00 ± 1.52 | 79.21 ± 1.28 | -79.21 ± 1.28 | -15.58 ± 0.37 |
| ewc_lambda10 (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 10.0 | - | false | - | 16.57 ± 0.49 | 80.24 ± 1.14 | -80.24 ± 1.14 | -15.58 ± 0.37 |
| ewc_lambda100 (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 100.0 | - | false | - | 15.98 ± 1.09 | 80.40 ± 1.19 | -80.40 ± 1.19 | -15.58 ± 0.37 |
| ewc_lambda1000 (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 1000.0 | - | false | - | 16.72 ± 0.23 | 79.97 ± 1.05 | -79.97 ± 1.05 | -15.58 ± 0.37 |
| fine_tune (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE | - | - | - | - | false | - | 15.98 ± 1.19 | 79.50 ± 1.13 | -79.50 ± 1.13 | -15.58 ± 0.37 |
| joint_training (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (joint replay) | full (all tasks) | - | - | - | false | joint_replay_epochs=1 | 64.51 ± 3.37 | 10.14 ± 1.31 | -8.71 ± 1.22 | -15.79 ± 0.29 |
| lwf (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD (LwF) | - | 1.0 | - | - | false | lwf_lambda=1.0, lwf_temp=2.0 | 16.16 ± 1.17 | 80.12 ± 0.95 | -80.12 ± 0.95 | -15.58 ± 0.37 |
| agem_distill (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 200 | 1.0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 27.28 ± 1.65 | 30.95 ± 2.33 | -30.95 ± 2.33 | -15.73 ± 0.51 |
| der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | false | der_alpha=0.5 | 16.50 ± 0.67 | 79.61 ± 1.65 | -79.61 ± 1.65 | -15.44 ± 0.89 |
| er_ewc (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | false | er_replay_ratio=0.5 | 28.16 ± 2.32 | 69.05 ± 3.00 | -69.05 ± 3.00 | -15.75 ± 0.35 |
| icarl (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 200 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=True | 16.95 ± 0.76 | 76.71 ± 1.68 | -76.71 ± 1.68 | -16.04 ± 0.00 |
| progress_compress (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + distill + EWC (compress) | - | 1.0 | 100.0 | - | false | pc_distill_w=1.0, pc_compress_epochs=3 | 18.51 ± 1.23 | 78.56 ± 2.12 | -78.56 ± 2.12 | -15.41 ± 0.91 |
| si_der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | false | - | 16.33 ± 0.58 | 79.74 ± 1.10 | -79.74 ± 1.10 | -15.44 ± 0.89 |
| xder (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 200 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 19.45 ± 1.05 | 76.85 ± 1.91 | -76.85 ± 1.91 | -15.15 ± 1.09 |
| agem_distill_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 100 | 1.0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 21.80 ± 1.28 | 33.05 ± 2.85 | -33.05 ± 2.85 | -15.82 ± 0.31 |
| agem_distill_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 200 | 1.0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 27.28 ± 1.65 | 30.95 ± 2.33 | -30.95 ± 2.33 | -15.73 ± 0.51 |
| agem_distill_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 500 | 1.0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 30.65 ± 1.63 | 29.69 ± 2.90 | -29.56 ± 3.12 | -15.80 ± 0.24 |
| agem_distill_nodistill (ablation: no distill) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | 0 | - | - | false | agem_mem_batch=64, distill_temp=2.0 | 26.30 ± 1.39 | 32.03 ± 2.49 | -32.03 ± 2.49 | -15.55 ± 0.63 |
| der_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 100 | 1.0 | - | - | false | der_alpha=0.5 | 16.66 ± 0.43 | 81.35 ± 1.13 | -81.35 ± 1.13 | -15.24 ± 1.22 |
| der_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | false | der_alpha=0.5 | 16.50 ± 0.67 | 79.61 ± 1.65 | -79.61 ± 1.65 | -15.44 ± 0.89 |
| der_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 500 | 1.0 | - | - | false | der_alpha=0.5 | 15.81 ± 0.58 | 79.44 ± 1.59 | -79.44 ± 1.59 | -15.08 ± 1.93 |
| der_nodistill (ablation: no distill) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (DER) | 200 | 0 | - | - | false | der_alpha=0.5 | 16.23 ± 0.28 | 80.06 ± 0.22 | -80.06 ± 0.22 | -15.27 ± 0.06 |
| er_ewc_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 100 | - | 100.0 | - | false | er_replay_ratio=0.5 | 22.39 ± 1.06 | 76.84 ± 2.15 | -76.84 ± 2.15 | -15.54 ± 0.36 |
| er_ewc_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | false | er_replay_ratio=0.5 | 28.16 ± 2.32 | 69.05 ± 3.00 | -69.05 ± 3.00 | -15.75 ± 0.35 |
| er_ewc_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 500 | - | 100.0 | - | false | er_replay_ratio=0.5 | 34.72 ± 3.59 | 60.45 ± 4.12 | -60.45 ± 4.12 | -15.71 ± 0.36 |
| er_ewc_lambda1 (ablation: lambda=1) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 1.0 | - | false | er_replay_ratio=0.5 | 26.34 ± 2.07 | 72.35 ± 2.35 | -72.35 ± 2.35 | -15.74 ± 0.35 |
| er_ewc_lambda10 (ablation: lambda=10) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 10.0 | - | false | er_replay_ratio=0.5 | 27.60 ± 4.01 | 69.65 ± 5.96 | -69.65 ± 5.96 | -15.70 ± 0.38 |
| er_ewc_lambda100 (ablation: lambda=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | false | er_replay_ratio=0.5 | 28.16 ± 2.32 | 69.05 ± 3.00 | -69.05 ± 3.00 | -15.75 ± 0.35 |
| er_ewc_lambda1000 (ablation: lambda=1000) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 1000.0 | - | false | er_replay_ratio=0.5 | 25.13 ± 1.34 | 72.41 ± 2.95 | -72.41 ± 2.95 | -15.75 ± 0.35 |
| er_ewc_noewc (ablation: no ewc) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay | 200 | - | 0 | - | false | er_replay_ratio=0.5 | 25.78 ± 2.96 | 72.73 ± 4.90 | -72.73 ± 4.90 | -15.68 ± 0.44 |
| icarl_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 100 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=True | 17.39 ± 1.19 | 75.45 ± 3.72 | -75.45 ± 3.72 | -16.04 ± 0.00 |
| icarl_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 200 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=True | 17.67 ± 1.11 | 75.55 ± 3.00 | -75.55 ± 3.00 | -16.04 ± 0.00 |
| icarl_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 500 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=True | 16.95 ± 0.76 | 76.71 ± 1.68 | -76.71 ± 1.68 | -16.04 ± 0.00 |
| icarl_nonmc (ablation: no nmc) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD | 200 | 1.0 | - | - | false | icarl_temp=2.0, use_nmc=False | 17.46 ± 0.19 | 80.96 ± 0.60 | -80.96 ± 0.60 | -15.37 ± 0.29 |
| si_der_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 100 | 1.0 | - | 1.0 | false | - | 16.78 ± 0.29 | 80.91 ± 1.26 | -80.91 ± 1.26 | -15.24 ± 1.22 |
| si_der_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | false | - | 16.33 ± 0.58 | 79.74 ± 1.10 | -79.74 ± 1.10 | -15.44 ± 0.89 |
| si_der_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 500 | 1.0 | - | 1.0 | false | - | 16.07 ± 0.43 | 79.58 ± 1.36 | -79.58 ± 1.36 | -15.08 ± 1.93 |
| si_der_nosi (ablation: no si) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE | 200 | 1.0 | - | 0 | false | - | 16.55 ± 0.87 | 79.67 ± 1.68 | -79.67 ± 1.68 | -15.17 ± 1.13 |
| xder_buf100 (ablation: buffer=100) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 100 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 17.98 ± 1.05 | 77.23 ± 2.91 | -77.23 ± 2.91 | -15.58 ± 0.86 |
| xder_buf200 (ablation: buffer=200) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 200 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 19.45 ± 1.05 | 76.85 ± 1.91 | -76.85 ± 1.91 | -15.15 ± 1.09 |
| xder_buf500 (ablation: buffer=500) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 500 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 20.22 ± 2.25 | 75.18 ± 2.29 | -75.18 ± 2.29 | -15.46 ± 0.92 |
| xder_norevision (ablation: no revision) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) (no revision) | 200 | 1.0 | - | - | false | der_alpha=0.5, xder_beta=0.5 | 16.20 ± 0.80 | 79.92 ± 0.18 | -79.92 ± 0.18 | -15.17 ± 1.13 |

### Split CIFAR‑100 (20 tasks × 5 classes) — CIFAR‑100 split into sequential class groups (Class‑IL).

| Method | Model | Epochs/Task | Batch | LR | Optimizer | Loss | Buffer | Distill λ | EWC λ | SI λ | FP16 | Other | Avg Acc | Forgetting | BWT | FWT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agem (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | - | - | - | true | agem_mem_batch=64 | 4.32 ± 0.37 | 7.09 ± 0.65 | 1.99 ± 0.48 | -2.73 ± 0.01 |
| ewc (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 100.0 | - | true | - | 2.06 ± 0.52 | 35.33 ± 3.09 | -35.33 ± 3.09 | -2.73 ± 0.00 |
| fine_tune (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE | - | - | - | - | true | - | 2.12 ± 0.68 | 37.39 ± 3.47 | -37.39 ± 3.47 | -2.73 ± 0.00 |
| joint_training (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (joint replay) | full (all tasks) | - | - | - | true | joint_replay_epochs=1 | 47.04 ± 1.56 | 5.32 ± 0.90 | 12.12 ± 2.13 | -2.73 ± 0.00 |
| lwf (baseline) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD (LwF) | - | 1.0 | - | - | true | lwf_lambda=1.0, lwf_temp=2.0 | 2.47 ± 0.21 | 40.73 ± 4.45 | -40.73 ± 4.45 | -2.73 ± 0.00 |
| agem_distill (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 200 | 1.0 | - | - | true | agem_mem_batch=64, distill_temp=2.0 | 4.12 ± 0.65 | 20.30 ± 1.31 | -20.28 ± 1.31 | -2.73 ± 0.01 |
| der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | true | der_alpha=0.5 | 2.96 ± 0.25 | 45.81 ± 4.13 | -45.81 ± 4.13 | -2.72 ± 0.03 |
| er_ewc (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | true | er_replay_ratio=0.5 | 4.07 ± 0.34 | 59.05 ± 1.58 | -59.05 ± 1.58 | -2.73 ± 0.00 |
| icarl (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 200 | 1.0 | - | - | true | icarl_temp=2.0, use_nmc=True | 2.94 ± 0.20 | 44.83 ± 1.12 | -44.83 ± 1.12 | -2.73 ± 0.00 |
| progress_compress (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + distill + EWC (compress) | - | 1.0 | 100.0 | - | true | pc_distill_w=1.0, pc_compress_epochs=3 | 3.65 ± 0.23 | 52.64 ± 1.59 | -52.64 ± 1.59 | -2.72 ± 0.02 |
| si_der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | true | - | 2.77 ± 0.23 | 45.14 ± 2.19 | -45.14 ± 2.19 | -2.71 ± 0.03 |
| xder (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 200 | 1.0 | - | - | true | der_alpha=0.5, xder_beta=0.5 | 1.65 ± 0.90 | 36.11 ± 9.19 | -36.11 ± 9.19 | -2.72 ± 0.02 |

### Split Mini‑ImageNet (20 tasks × 5 classes) — 100‑class subset; local Phase 4 runs pending.

| Method | Model | Epochs/Task | Batch | LR | Optimizer | Loss | Buffer | Distill λ | EWC λ | SI λ | FP16 | Other | Avg Acc | Forgetting | BWT | FWT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fine_tune (baseline) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE | - | - | - | - | true | - | - | - | - | - |
| joint_training (baseline) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE (joint replay) | full (all tasks) | - | - | - | true | joint_replay_epochs=1 | - | - | - | - |
| ewc (baseline) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + EWC | - | - | 100.0 | - | true | - | - | - | - | - |
| agem (baseline) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection | 200 | - | - | - | true | agem_mem_batch=64 | - | - | - | - |
| lwf (baseline) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD (LwF) | - | 1.0 | - | - | true | lwf_lambda=1.0, lwf_temp=2.0 | - | - | - | - |
| der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (DER) | 200 | 1.0 | - | - | true | der_alpha=0.5 | - | - | - | - |
| xder (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + logit MSE (X‑DER) | 200 | 1.0 | - | - | true | der_alpha=0.5, xder_beta=0.5 | - | - | - | - |
| icarl (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + KD + NMC | 200 | 1.0 | - | - | true | icarl_temp=2.0, use_nmc=True | - | - | - | - |
| er_ewc (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + replay + EWC | 200 | - | 100.0 | - | true | er_replay_ratio=0.5 | - | - | - | - |
| progress_compress (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + distill + EWC (compress) | - | 1.0 | 100.0 | - | true | pc_distill_w=1.0, pc_compress_epochs=3 | - | - | - | - |
| agem_distill (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + GEM projection + distill | 200 | 1.0 | - | - | true | agem_mem_batch=64, distill_temp=2.0 | - | - | - | - |
| si_der (hybrid) | Slim ResNet‑18 (18 layers, ~2M params) / ViT‑Small (optional) | 1 | 32 | 0.03 | SGD (m=0.9, wd=0.0001) | CE + SI + logit MSE | 200 | 1.0 | - | 1.0 | true | - | - | - | - | - |

## Tools

- **PyTorch** — Core training and autograd for all models and CL methods.
- **torchvision** — Dataset utilities and transforms for MNIST/CIFAR.
- **timm** — ViT‑Small model for optional higher-memory runs.
- **NumPy** — Vectorized metric aggregation and statistics.
- **Pandas** — Result table assembly and CSV outputs.
- **Matplotlib** — Primary plotting backend for curves/heatmaps.
- **Seaborn** — Styling and metric bar charts.
- **scikit‑learn** — Decision tree for Phase 5 recommendation engine.
- **PyYAML** — Config loading for experiment YAMLs.
- **tqdm** — Progress bars during training.
- **pytest** — Sanity tests for trainer, metrics, and plotting.
- **Streamlit** — Planned dashboard UI for Phase 6.
- **Local RTX 4050 + optional cloud fallback** — resumable larger-dataset runs.

## Ablation Notes

- **Buffer size (`buf`)** — number of stored replay samples; larger buffers typically reduce forgetting but increase memory cost.
- **EWC/SI λ (`lambda`)** — regularization strength; higher values prioritize stability over plasticity.
- **`n_tasks`** — number of sequential tasks in the run (e.g., 5 vs 10), used to probe task‑length sensitivity.
- **`nodistill`** — distillation term removed to isolate the effect of replay/regularization alone.
- **`noewc` / `nosi`** — regularization term removed to isolate replay/distillation effects.
- **`nonmc`** — iCaRL uses softmax classifier only (no Nearest‑Mean‑of‑Exemplars).
- **`norevision`** — X‑DER buffer revision disabled to isolate the benefit of revision.

## Feature Extraction

Our ResNet backbones learn hierarchical visual features from images: early layers capture edges and strokes (especially for MNIST digits), mid‑layers capture textures and simple shapes, and deeper layers capture class‑specific parts and object structure. For CIFAR‑10/100, the model combines color/texture cues with spatial composition to form discriminative embeddings. The expandable classifier head then maps these embeddings to logits over all classes seen so far; in planned ViT‑Small runs, patch embeddings and self‑attention are used to build global, context‑aware features before classification.

## Implementation

### Baselines

**Fine‑Tune (baseline) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We use the same backbone for all baselines (ResNet‑8 for Permuted MNIST, Slim ResNet‑18 for Split CIFAR‑10/100) and the expandable classifier head for Class‑IL. In `FineTune.observe` the model runs a forward pass on the current task batch only, computes cross‑entropy, and updates weights via AMP‑scaled backprop (`autocast` + `GradScaler`). There is **no replay buffer and no regularization term**, so as the head expands, older class logits are overwritten by new‑task gradients. This is the intended “forgetting‑heavy” reference: the same architecture is trained, but it only optimizes on the most recent task’s data.

**Joint Training (baseline) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We keep the architecture unchanged, but modify the training data flow: `JointTraining.after_task` concatenates all seen task data into `_all_x/_all_y`, builds a `TensorDataset`, and **replays the entire cumulative dataset** for `joint_replay_epochs` after each task. The core SGD step is still CE on a standard batch, but the *batch source* is now the full joint dataset. This turns the baseline into an upper bound that simulates “full memory” continual learning without modifying the network layers.

**EWC (baseline) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** The network is identical to fine‑tune, but we add a **Fisher‑weighted quadratic penalty** to the loss in `EWC.observe`: `loss = CE + (λ/2) Σ F_i (θ_i − θ*_i)^2`. At the end of each task, `after_task` snapshots current parameters and estimates the diagonal Fisher using model predictions (pseudo‑labels) on `fisher_samples`. The penalty code explicitly handles head expansion by **slicing only the overlapping parameter region**, preventing shape errors and protecting only the old weights.

**A‑GEM (baseline) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We keep the same backbone/head but insert **gradient projection** before the optimizer step. `AGEM.observe` computes CE on the current batch, backpropagates to get `g_cur`, then samples memory from a reservoir buffer, computes `g_mem`, and projects `g_cur` if `g_cur·g_mem < 0`. The projected gradient is written back into parameter `.grad` tensors before `optimizer.step()`. The replay buffer itself is populated per task in `after_task` by taking a fixed number of examples per task.

**LwF (baseline) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** LwF adds **distillation from a frozen snapshot** without storing any data. In `before_task`, we deep‑copy the current model **before head expansion** to create a teacher with the old output size. In `observe`, we compute CE on the current batch plus a KL‑divergence term between the student’s old‑class logits and the teacher’s outputs (temperature‑scaled), weighted by `lwf_lambda`. This keeps the architecture fixed while adding a soft‑target constraint on old classes.

### Hybrid Methods

**DER (Dark Experience Replay) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We add a reservoir `ReplayBuffer` that stores `(x, y, logits)` and, in `DER.observe`, compute **two losses per step**: CE on the current batch and MSE between current logits on replayed samples and the **stored logits**. The stored logit dimension is masked (`logit_sizes`) to handle head growth. The total loss is `CE + der_alpha * dark_loss`. After the optimizer step, we **store the current batch and its logits** in the buffer, which is the only structural change relative to the baseline.

**X‑DER (Extended Dark Experience Replay) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** X‑DER builds on DER by adding a **hard‑label calibration loss** on replay samples and a **buffer revision pass**. In `XDER.observe`, we compute CE on new data, MSE to stored logits (dark replay), and **CE on replay labels**, combined as `CE + der_alpha*dark + xder_beta*cal`. In `after_task`, `buffer.update_logits()` refreshes all stored logits with the **current model**, ensuring replay targets stay synchronized as the head expands.

**iCaRL (incremental Classifier and Representation Learning) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** iCaRL combines **exemplar herding**, **distillation**, and optional **Nearest‑Mean Classifier (NMC)**. In `before_task`, we snapshot the teacher pre‑expansion. In `observe`, we **concatenate** exemplars from the herding buffer with the current batch, compute CE on the current task samples, and add KL‑distillation on *all* samples using the teacher’s old‑class outputs. In `after_task`, we extract features with `model.get_features`, select exemplars via herding (`buffer.add_task_exemplars`), and compute per‑class prototype means used by `predict_nmc` at evaluation time.

**ER+EWC (Experience Replay + Elastic Weight Consolidation) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We merge replay and regularization in the loss. `ER_EWC.observe` samples a replay fraction (`er_replay_ratio`), concatenates buffer samples with the current batch, computes CE on the combined batch, and adds the EWC penalty from stored Fisher/weights. `after_task` estimates Fisher on the current task data and snapshots parameters; the penalty logic handles **head expansion by slicing** the overlapping tensor region so old parameters are protected while new class weights are unconstrained.

**Progress & Compress (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We introduce a **knowledge‑base (KB) model** that shadows the main model. During the **progress phase**, `observe` trains the active model with CE plus **distillation from the KB** (`pc_distill_w`) on old classes. During the **compress phase**, `after_task` unfreezes the KB, runs `pc_compress_epochs` of distillation from the active model into the KB, and adds an **EWC‑style penalty** on KB weights (`pc_ewc_lambda`) using a Fisher estimate computed on the KB. `before_task` expands the KB head alongside the active model so their output dimensions stay aligned.

**A‑GEM + Distill (Average Gradient Episodic Memory) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We combine A‑GEM’s gradient projection with a **teacher‑student distillation term**. `before_task` snapshots a frozen teacher (pre‑expansion). In `observe`, we compute CE + `distill_lambda` * KD loss (temperature‑scaled), backpropagate, then compute the memory gradient on a buffer batch and **project the current gradient** if it violates the A‑GEM constraint. The only architectural change is the addition of the replay buffer and the extra KD loss; the backbone/head remain unchanged.

**SI + DER (Synaptic Intelligence + Dark Experience Replay) (ResNet‑8: 8 layers / Slim ResNet‑18: 18 layers).** We add **SI importance tracking** alongside DER replay. At task start we snapshot weights and initialize per‑parameter running integrals; after each backward pass we unscale grads (for FP16) and accumulate the **path integral** `−grad * Δw`. The loss combines CE, DER MSE on replayed logits, and an **SI penalty** `Σ ω_i (θ_i − θ_i^start)^2`. At task end, we convert the running integral into `ω` using `(Δw^2 + ξ)` and clamp it to non‑negative values. Head expansion is handled by slicing/expanding ω to match new parameter shapes, while the replay buffer continues to store `(x, y, logits)` like DER.
## One‑Epoch Justification

We train **one epoch per task** to align with standard continual‑learning protocol where each task is a single pass over its data. This keeps compute bounded on free‑tier hardware, ensures *every method sees the same number of updates*, and makes the forgetting signal measurable without confounding from long training schedules. The datasets here are small‑to‑medium (MNIST 60k, CIFAR‑10/100 50k), and our backbones are compact (ResNet‑8, Slim ResNet‑18 ~2M params), so a single epoch already produces a meaningful learning signal while avoiding multi‑epoch overfitting in later tasks. For publication, the key is **relative comparison under a consistent budget**; we explicitly report the 1‑epoch setting and can extend to multi‑epoch schedules in a follow‑up if reviewers request it.
