# Phase 2 Status

## Outcome
Phase 2 is complete.

Verified on April 12, 2026 in the `genai` environment:
- `pytest research_v2/Project/tests/test_phase2_datasets_models.py -q`
  - result: `10 passed`
- `python -m compileall research_v2/Project/src research_v2/Project/experiments research_v2/Project/notebooks`
  - result: passed

## Dataset Setup
The v2 local dataset root is:
- `research_v2/Project/data_local/`

Datasets were made available there via local links to the archived v1 mirrors:
- `MNIST`
- `cifar-10-batches-py`
- `cifar-100-python`
- `mini-imagenet`
- `tiny-imagenet-200`

Bootstrap utility:
- `research_v2/Project/notebooks/bootstrap_phase2_data.py`

This script:
- prefers reusing archived local mirrors
- falls back to torchvision downloads for MNIST / CIFAR if needed
- does not silently fetch Mini-ImageNet or Tiny-ImageNet

## Mandatory Dataset Validation
Validated:
- `permuted_mnist`
- `split_cifar10`
- `split_cifar100`
- `split_mini_imagenet`

Checks performed:
- constructors work locally
- metadata is correct
- first task loaders yield valid batches
- class/task partitions are consistent
- Permuted MNIST task permutations differ as expected
- Mini-ImageNet local layout works from the v2 root

## Model Validation
Validated:
- `slim_resnet18`
- expandable classifier head behavior

Checks performed:
- forward pass on CIFAR-sized inputs
- forward pass on Mini-ImageNet-sized inputs
- head expansion preserves previously learned weights

## Optional Backbone Status
### `vit_small_patch16_224`
Status:
- integrated
- `timm` installed and importable
- smoke-tested successfully

Current data policy:
- use `dataset_image_size: 224`
- use `model_image_size: 224`
- for RGB datasets, resizing is now supported in the v2 loaders

### `convnext_tiny`
Status:
- implemented in `Project/src/models/convnext_tiny.py`
- registered in the v2 model registry
- smoke-tested successfully

Current data policy:
- recommended with `dataset_image_size: 224` for image datasets when used as an optional comparison backbone

### Permuted MNIST optional-backbone bridge
Added:
- `dataset_out_channels`
- resized / channel-expanded Permuted MNIST path

Meaning:
- optional RGB backbones are no longer blocked by the default 1-channel MNIST representation

## Config Status
Canonical root-level v2 configs now exist for Split Mini-ImageNet:
- `split_mini_imagenet_finetune.yaml`
- `split_mini_imagenet_joint.yaml`
- `split_mini_imagenet_ewc.yaml`
- `split_mini_imagenet_agem.yaml`
- `split_mini_imagenet_lwf.yaml`
- `split_mini_imagenet_der.yaml`
- `split_mini_imagenet_xder.yaml`
- `split_mini_imagenet_icarl.yaml`
- `split_mini_imagenet_er_ewc.yaml`
- `split_mini_imagenet_progress_compress.yaml`
- `split_mini_imagenet_agem_distill.yaml`
- `split_mini_imagenet_si_der.yaml`

Legacy imported prototype configs still remain under:
- `Project/experiments/configs/phase4/`

Current rule:
- root-level Split Mini-ImageNet configs are the canonical v2 configs
- legacy `phase4/` configs are reference material only and should not drive new v2 runs

## Skip / Fallback Rules
- If `timm` is missing, optional backbones must be skipped explicitly rather than silently failing.
- If ViT or ConvNeXt show repeated OOM or unstable behavior during later method smoke runs, keep `slim_resnet18` as the mandatory backbone and document the optional-backbone skip reason.
- If a dataset-specific resized path proves unstable later, keep native-size `slim_resnet18` as the canonical study path.
