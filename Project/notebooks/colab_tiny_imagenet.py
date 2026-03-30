"""Colab notebook for Sequential Tiny-ImageNet + ViT-Small experiments.

⚡ Tiny-ImageNet with ViT-Small is the most compute-intensive experiment.
Run only top hybrid methods here (select 3-4 based on CIFAR-100 results).

SETUP:
1. Runtime → Change runtime type → GPU (T4), High RAM
2. Mount Drive, run setup cell, download Tiny-ImageNet
3. Run only 1-2 methods per Colab session to stay within free tier limits
   (each method on Tiny-ImageNet takes ~2-3 hours on T4)

Download Tiny-ImageNet:
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip tiny-imagenet-200.zip -d /content/data/
"""

# ============================================================
# CELL 1: Environment Setup
# ============================================================
# !pip install timm seaborn pyyaml tqdm -q
# from google.colab import drive
# drive.mount('/content/drive')

# import subprocess, sys, os, shutil
# DRIVE_ROOT = "/content/drive/MyDrive/CL_Project"
# if not os.path.exists("/content/src"):
#     shutil.copytree(f"{DRIVE_ROOT}/src", "/content/src")
# sys.path.insert(0, "/content")

# ============================================================
# CELL 2: Download Tiny-ImageNet
# ============================================================
# import os
# DATA_DIR = "/content/data"
# os.makedirs(DATA_DIR, exist_ok=True)
# if not os.path.exists(f"{DATA_DIR}/tiny-imagenet-200"):
#     !wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -P {DATA_DIR}
#     !unzip -q {DATA_DIR}/tiny-imagenet-200.zip -d {DATA_DIR}/

# ============================================================
# CELL 3: ViT-Small experiment runner
# ============================================================

import sys, os
sys.path.insert(0, "/content")   # adjust if not on Colab

import torch
import numpy as np

from src.datasets    import get_dataset
from src.models      import get_model, get_vit_small
from src.models.classifier_head import CLModel
from src.methods     import get_method
from src.trainers.cl_trainer import CLTrainer
from src.visualization.plots import plot_summary_grid, plot_training_curves
from src.utils import seed_everything, RunLogger

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/content/data/tiny-imagenet-200"
LOG_DIR   = "/content/results/raw"
FIG_DIR   = "/content/results/figures"
CKPT_DIR  = "/content/results/checkpoints"
SEED      = 42

for d in [LOG_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# Select top methods to evaluate (from CIFAR-100 results)
METHODS_TO_RUN = [
    {"method": "fine_tune",   "buffer_size": 0},
    {"method": "der",         "buffer_size": 500,  "der_alpha": 0.5},
    {"method": "xder",        "buffer_size": 500,  "der_alpha": 0.5, "xder_beta": 0.5},
    {"method": "si_der",      "buffer_size": 500,  "si_lambda": 1.0, "der_alpha": 0.5},
    {"method": "icarl",       "buffer_size": 2000, "use_nmc": True},
]

VIT_CFG = {
    "dataset":           "seq_tiny_imagenet",
    "model":             "vit_small",
    "pretrained":        True,          # ImageNet-21k pretrained
    "n_epochs":          1,
    "batch_size":        16,            # small batch for ViT on T4
    "lr":                0.001,         # ViT prefers lower LR
    "momentum":          0.9,
    "weight_decay":      1e-4,
    "fp16":              True,
    "num_workers":       4,
    "checkpoint_dir":    CKPT_DIR,
    "figure_dir":        FIG_DIR,
    "log_dir":           LOG_DIR,
    "data_root":         DATA_ROOT,
    # Gradient accumulation to simulate batch=64 with batch=16
    "grad_accum_steps":  4,
}

all_results = {}

for method_overrides in METHODS_TO_RUN:
    cfg = {**VIT_CFG, **method_overrides}
    seed_everything(SEED)

    method_name = cfg["method"]
    run_name    = f"tiny_imagenet_vit_{method_name}_seed{SEED}"
    cfg["run_name"] = run_name

    print(f"\n{'='*60}")
    print(f" Running: ViT-Small + {method_name} on Tiny-ImageNet")
    print(f"{'='*60}")

    try:
        dataset = get_dataset(
            "seq_tiny_imagenet",
            root        = DATA_ROOT,
            batch_size  = cfg["batch_size"],
            num_workers = cfg["num_workers"],
        )

        # Build ViT-Small model
        backbone = get_vit_small(
            pretrained          = cfg.get("pretrained", True),
            grad_checkpointing  = True,    # saves ~50% activation memory
            image_size          = 64,      # Tiny-ImageNet native resolution
        )
        model = CLModel(backbone, feature_dim=384).to(DEVICE)

        # Enable gradient accumulation via monkey-patching the trainer
        cfg["batch_size"] = 16

        method = get_method(method_name, model, cfg, torch.device(DEVICE))
        logger = RunLogger(LOG_DIR, run_name)
        trainer = CLTrainer(method, dataset, cfg, logger, torch.device(DEVICE))

        # Clear GPU memory before start
        torch.cuda.empty_cache()

        result = trainer.train()
        all_results[method_name] = result

        plot_summary_grid(
            result["acc_matrix"], result["metrics"],
            method_name  = f"vit_{method_name}",
            dataset_name = "tiny_imagenet",
            fig_dir      = FIG_DIR,
        )
        print(f"\n✓ ViT-{method_name}: {result['metrics']}")

    except torch.cuda.OutOfMemoryError:
        print(f"\n✗ {method_name}: CUDA OOM — try reducing buffer_size or batch_size")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n✗ {method_name} FAILED: {e}")
        import traceback; traceback.print_exc()

# ============================================================
# CELL 4: Save results to Drive
# ============================================================
# import shutil
# shutil.copytree("/content/results", f"{DRIVE_ROOT}/results_vit", dirs_exist_ok=True)
# print("ViT results saved to Drive ✓")
