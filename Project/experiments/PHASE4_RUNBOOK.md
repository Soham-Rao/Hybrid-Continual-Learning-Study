# Phase 4 Runbook

## Main Execution Files

- `Project/notebooks/local_mini_imagenet.py`
- `Project/notebooks/local_tiny_imagenet.py`
- `Project/notebooks/verify_phase4_setup.py`
- `Project/experiments/run_phase4_local.ps1`

## Preset Config Files

- `Project/experiments/configs/phase4/mini_imagenet_resnet18_local.yaml`
- `Project/experiments/configs/phase4/mini_imagenet_vit_small_local.yaml`
- `Project/experiments/configs/phase4/tiny_imagenet_resnet18_local.yaml`
- `Project/experiments/configs/phase4/tiny_imagenet_vit_small_local.yaml`

## Recommended Sequence

1. Run setup validation:

   `conda run -n genai python Project/notebooks/verify_phase4_setup.py`

2. Run Mini-ImageNet locally first:

   `conda run -n genai python Project/notebooks/local_mini_imagenet.py --device cuda --model slim_resnet18 --methods all --seeds 42,123,456,789,1024`

3. Run Tiny-ImageNet only after Mini-ImageNet is complete and the dataset exists locally:

   `conda run -n genai python Project/notebooks/local_tiny_imagenet.py --device cuda --model vit_small --methods fine_tune,der,xder,si_der,icarl --seeds 42,123,456,789,1024`

## PowerShell Convenience Wrapper

Mini-ImageNet all methods:

`pwsh -File Project/experiments/run_phase4_local.ps1 -RunMiniAll`

Mini-ImageNet top subset:

`pwsh -File Project/experiments/run_phase4_local.ps1 -RunMiniTop`

Tiny-ImageNet top subset:

`pwsh -File Project/experiments/run_phase4_local.ps1 -RunTinyTop -UseViT`
