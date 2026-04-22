# Epoch-1 Dataset Summary Tables

Metrics below are means across the completed seed runs listed in `current_results.csv`.

## Permuted MNIST

| Method | Avg Accuracy | Forgetting | BWT | FWT | Backbone | Epochs | Batch | LR | FP16 | Buffer | Key Hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| Fine-Tune | 35.5605 | 70.4846 | -70.4846 | 0.8061 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | baseline only |
| Joint Training | 95.3738 | 1.1907 | -0.5540 | 1.2156 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | joint_replay_epochs=1 |
| EWC | 28.8238 | 75.0224 | -75.0224 | 0.7102 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | ewc_lambda=100.0, fisher_samples=200 |
| A-GEM | 36.1646 | 66.9020 | -66.9020 | 1.5660 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | agem_mem_batch=64 |
| LwF | 32.9686 | 70.4687 | -70.4687 | 0.8973 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | lwf_lambda=1.0, lwf_temp=2.0 |
| DER | 70.9626 | 29.2447 | -29.2447 | 1.1191 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | der_alpha=0.5 |
| X-DER | 71.1744 | 28.9718 | -28.9718 | 0.8918 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | xder_beta=0.5 |
| iCaRL | 20.3134 | 5.6791 | -4.5738 | -0.0489 | slim_resnet18 | 1 | 32 | 0.03 | False | 500 | icarl_distill_w=1.0, icarl_temp=2.0, use_nmc=True |
| ER+EWC | 56.3174 | 44.3907 | -44.3907 | 1.1593 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | ewc_lambda=100.0, er_replay_ratio=0.5 |
| Progress & Compress | 28.3760 | 76.0035 | -76.0035 | 1.2342 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | pc_distill_w=1.0, pc_ewc_lambda=100.0, pc_compress_epochs=3 |
| A-GEM+Distill | 62.5618 | 37.0129 | -36.5342 | 2.4484 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | agem_mem_batch=64, distill_lambda=1.0, distill_temp=2.0 |
| SI-DER | 70.8658 | 29.2642 | -29.2642 | 0.9929 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | si_lambda=1.0, si_xi=0.1, der_alpha=0.5 |

## Split CIFAR-10

| Method | Avg Accuracy | Forgetting | BWT | FWT | Backbone | Epochs | Batch | LR | FP16 | Buffer | Key Hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| Fine-Tune | 15.8620 | 78.6075 | -78.6075 | -15.4642 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | baseline only |
| Joint Training | 63.7080 | 10.9575 | -7.4275 | -15.5467 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | joint_replay_epochs=1 |
| EWC | 16.8940 | 79.0250 | -79.0250 | -15.4642 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | ewc_lambda=100.0, fisher_samples=200 |
| A-GEM | 15.9780 | 79.3500 | -79.3500 | -15.4642 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | agem_mem_batch=64 |
| LwF | 15.9320 | 78.2675 | -78.2675 | -15.4642 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | lwf_lambda=1.0, lwf_temp=2.0 |
| DER | 16.6240 | 80.2650 | -80.2650 | -13.1367 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | der_alpha=0.5 |
| X-DER | 21.2160 | 74.1875 | -74.1875 | -14.1567 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | xder_beta=0.5 |
| iCaRL | 27.2480 | 39.7550 | -39.7550 | -16.0417 | slim_resnet18 | 1 | 32 | 0.03 | False | 500 | icarl_distill_w=1.0, icarl_temp=2.0, use_nmc=True |
| ER+EWC | 23.5220 | 73.1625 | -73.1625 | -15.3042 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | ewc_lambda=100.0, er_replay_ratio=0.5 |
| Progress & Compress | 17.7580 | 80.1675 | -80.1675 | -13.9442 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | pc_distill_w=1.0, pc_ewc_lambda=100.0, pc_compress_epochs=3 |
| A-GEM+Distill | 16.9820 | 80.0500 | -80.0500 | -15.4642 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | agem_mem_batch=64, distill_lambda=1.0, distill_temp=2.0 |
| SI-DER | 17.0040 | 80.1225 | -80.1225 | -13.1367 | slim_resnet18 | 1 | 32 | 0.03 | False | 200 | si_lambda=1.0, si_xi=0.1, der_alpha=0.5 |

## Split CIFAR-100

| Method | Avg Accuracy | Forgetting | BWT | FWT | Backbone | Epochs | Batch | LR | FP16 | Buffer | Key Hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| Fine-Tune | 2.1300 | 38.9200 | -38.9200 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | baseline only |
| Joint Training | 46.7460 | 6.1010 | 13.3516 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | joint_replay_epochs=1 |
| EWC | 2.0500 | 34.4295 | -34.4295 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | ewc_lambda=100.0, fisher_samples=200 |
| A-GEM | 2.1960 | 40.4716 | -40.4716 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | agem_mem_batch=64 |
| LwF | 2.6080 | 42.0905 | -42.0905 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | lwf_lambda=1.0, lwf_temp=2.0 |
| DER | 2.6800 | 44.7495 | -44.7495 | -2.6966 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | der_alpha=0.5 |
| X-DER | 2.3900 | 45.1263 | -45.1263 | -2.7282 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | xder_beta=0.5 |
| iCaRL | 4.3740 | 11.2295 | -9.2505 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 500 | icarl_distill_w=1.0, icarl_temp=2.0, use_nmc=True |
| ER+EWC | 4.4800 | 58.2758 | -58.2758 | -2.7303 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | ewc_lambda=100.0, er_replay_ratio=0.5 |
| Progress & Compress | 3.5520 | 52.7516 | -52.7516 | -2.7219 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | pc_distill_w=1.0, pc_ewc_lambda=100.0, pc_compress_epochs=3 |
| A-GEM+Distill | 2.7220 | 46.5347 | -46.5347 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | agem_mem_batch=64, distill_lambda=1.0, distill_temp=2.0 |
| SI-DER | 2.8620 | 44.9811 | -44.9811 | -2.6966 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | si_lambda=1.0, si_xi=0.1, der_alpha=0.5 |

## Split Mini-ImageNet

| Method | Avg Accuracy | Forgetting | BWT | FWT | Backbone | Epochs | Batch | LR | FP16 | Buffer | Key Hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| Fine-Tune | 1.9400 | 29.6403 | -29.6403 | -2.7292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | baseline only |
| Joint Training | 45.1883 | 3.5473 | 19.8456 | -2.7310 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | joint_replay_epochs=1 |
| EWC | 2.0750 | 28.9631 | -28.9631 | -2.7292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | ewc_lambda=100.0, fisher_samples=200 |
| A-GEM | 1.9350 | 32.2052 | -32.2052 | -2.7292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | agem_mem_batch=64 |
| LwF | 1.7650 | 28.4667 | -28.4667 | -2.7257 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | lwf_lambda=1.0, lwf_temp=2.0 |
| DER | 1.9600 | 34.3439 | -34.3439 | -2.6292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | der_alpha=0.5 |
| X-DER | 2.6567 | 39.8018 | -39.8018 | -2.6520 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | xder_beta=0.5 |
| iCaRL | 3.1567 | 7.8070 | -6.2316 | -2.7345 | slim_resnet18 | 1 | 32 | 0.03 | True | 500 | icarl_distill_w=1.0, icarl_temp=2.0, use_nmc=True |
| ER+EWC | 3.3367 | 44.6158 | -44.6158 | -2.7292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | ewc_lambda=100.0, er_replay_ratio=0.5 |
| Progress & Compress | 2.6767 | 43.2158 | -43.2158 | -2.6310 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | pc_distill_w=1.0, pc_ewc_lambda=100.0, pc_compress_epochs=3 |
| A-GEM+Distill | 2.3016 | 34.7983 | -34.7983 | -2.7292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | agem_mem_batch=64, distill_lambda=1.0, distill_temp=2.0 |
| SI-DER | 2.2450 | 34.7807 | -34.7807 | -2.6292 | slim_resnet18 | 1 | 32 | 0.03 | True | 200 | si_lambda=1.0, si_xi=0.1, der_alpha=0.5 |
