$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions/datasets/permuted_mnist/methods/er_ewc_noewc_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/agem_distill' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/er_ewc' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/er_ewc_noewc_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed123' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed42' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed456' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/permuted_mnist/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/permuted_mnist/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/split_cifar100/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/split_cifar100/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/split_cifar10/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/fwt/datasets/split_cifar10/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10/methods/fine_tune' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10/methods/joint_training' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/agem_distill' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/er_ewc' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/progress_compress' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/si_der' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/agem_distill' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/er_ewc' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/progress_compress' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/si_der' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/agem_distill' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/er_ewc' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/progress_compress' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/si_der' | Out-Null
New-Item -ItemType Directory -Force -Path 'results/epoch_1/hybrids/fwt_sider_fix/datasets/split_cifar100/methods/si_der' | Out-Null

Move-Item -LiteralPath 'results/epoch_1/ablations/interactions/datasets/permuted_mnist_er_ewc_noewc/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions/datasets/permuted_mnist/methods/er_ewc_noewc_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_agem/methods/distill/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/agem_distill/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_er/methods/ewc/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/er_ewc/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_er_ewc_noewc/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/er_ewc_noewc_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_icarl_nonmc/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_icarl_nonmc/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_icarl_nonmc/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/icarl_nonmc_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_si_der_nosi/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_si_der_nosi/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_si_der_nosi/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/si_der_nosi_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_xder_norevision/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_xder_norevision/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist_xder_norevision/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/permuted_mnist/methods/xder_norevision_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_agem_distill_nodistill/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_agem_distill_nodistill/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_agem_distill_nodistill/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/agem_distill_nodistill_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_der_nodistill/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_der_nodistill/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_der_nodistill/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/der_nodistill_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_er_ewc_noewc/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_er_ewc_noewc/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_er_ewc_noewc/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/er_ewc_noewc_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_icarl_nonmc/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_icarl_nonmc/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_icarl_nonmc/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/icarl_nonmc_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_si_der_nosi/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_si_der_nosi/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_si_der_nosi/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/si_der_nosi_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_xder_norevision/methods/seed123/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed123/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_xder_norevision/methods/seed42/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed42/figures'
Move-Item -LiteralPath 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10_xder_norevision/methods/seed456/figures' -Destination 'results/epoch_1/ablations/interactions_plots/datasets/split_cifar10/methods/xder_norevision_seed456/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/permuted_mnist_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/permuted_mnist/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/permuted_mnist_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/permuted_mnist/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/split_cifar100_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/split_cifar100/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/split_cifar100_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/split_cifar100/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/split_cifar10_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/split_cifar10/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/fwt/datasets/split_cifar10_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/fwt/datasets/split_cifar10/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/permuted_mnist/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/split_cifar100/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10_fine/methods/tune/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10/methods/fine_tune/figures'
Move-Item -LiteralPath 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10_joint/methods/training/figures' -Destination 'results/epoch_1/baselines/no_fwt/datasets/split_cifar10/methods/joint_training/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist_agem/methods/distill/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/agem_distill/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist_er/methods/ewc/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/er_ewc/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist_progress/methods/compress/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/progress_compress/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist_si/methods/der/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/permuted_mnist/methods/si_der/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar100_agem/methods/distill/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/agem_distill/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar100_er/methods/ewc/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/er_ewc/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar100_progress/methods/compress/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/progress_compress/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar100_si/methods/der/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar100/methods/si_der/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar10_agem/methods/distill/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/agem_distill/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar10_er/methods/ewc/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/er_ewc/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar10_progress/methods/compress/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/progress_compress/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt/datasets/split_cifar10_si/methods/der/figures' -Destination 'results/epoch_1/hybrids/fwt/datasets/split_cifar10/methods/si_der/figures'
Move-Item -LiteralPath 'results/epoch_1/hybrids/fwt_sider_fix/datasets/split_cifar100_si/methods/der/figures' -Destination 'results/epoch_1/hybrids/fwt_sider_fix/datasets/split_cifar100/methods/si_der/figures'