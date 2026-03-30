"""Colab notebook for Split Mini-ImageNet experiments.

Copy this file to Google Colab (or open via the Colab extension).

SETUP:
1. Mount Google Drive and set DRIVE_ROOT to your project folder
2. Run the setup cell to install dependencies and download data  
3. Run experiments — each task checkpoints to Drive so sessions can resume

Recommended runtime: GPU (T4), High RAM
"""

# ============================================================
# CELL 1: Mount Drive & Setup
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# DRIVE_ROOT = "/content/drive/MyDrive/CL_Project"
# import subprocess, sys, os

# subprocess.run(["pip", "install", "timm", "seaborn", "pyyaml", "tqdm"], check=True)

# # Clone / copy project src into Colab
# import shutil
# if not os.path.exists("/content/src"):
#     shutil.copytree(f"{DRIVE_ROOT}/src", "/content/src")

# sys.path.insert(0, "/content")
# os.makedirs("/content/results/raw", exist_ok=True)
# os.makedirs("/content/results/figures", exist_ok=True)
# os.makedirs("/content/results/checkpoints", exist_ok=True)

# ============================================================
# CELL 2: Download Mini-ImageNet to Drive
# ============================================================
# Option A: from Kaggle (requires kaggle API key in Drive)
# import subprocess
# subprocess.run([
#     "kaggle", "datasets", "download",
#     "-d", "arjunashok33/miniimagenet",
#     "-p", f"{DRIVE_ROOT}/data/"
# ])
# subprocess.run(["unzip", f"{DRIVE_ROOT}/data/miniimagenet.zip",
#                 "-d", f"{DRIVE_ROOT}/data/mini-imagenet/"])
#
# Option B: manual upload to Drive and set DATA_ROOT below

# ============================================================
# CELL 3: Run experiments (all methods on Mini-ImageNet)
# ============================================================

import sys, os
sys.path.insert(0, "/content")  # adjust if running locally

import torch
import yaml
import numpy as np

from src.datasets    import get_dataset
from src.models      import get_model
from src.methods     import get_method
from src.trainers.cl_trainer import CLTrainer
from src.visualization.plots import plot_summary_grid, plot_training_curves
from src.utils import seed_everything, RunLogger

# ── Configuration ──────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/content/drive/MyDrive/CL_Project/data/mini-imagenet"
LOG_DIR   = "/content/results/raw"
FIG_DIR   = "/content/results/figures"
CKPT_DIR  = "/content/results/checkpoints"
SEED      = 42

# Methods to evaluate on Mini-ImageNet (run one block at a time per session)
METHODS_TO_RUN = [
    {"method": "fine_tune",   "buffer_size": 0,   "fp16": True},
    {"method": "joint_training","buffer_size":0,   "fp16": True, "joint_replay_epochs": 1},
    {"method": "ewc",         "buffer_size": 0,   "fp16": True, "ewc_lambda": 100.0},
    {"method": "der",         "buffer_size": 500, "fp16": True, "der_alpha": 0.5},
    {"method": "xder",        "buffer_size": 500, "fp16": True, "der_alpha": 0.5, "xder_beta": 0.5},
    {"method": "icarl",       "buffer_size": 2000,"fp16": True, "use_nmc": True},
    {"method": "er_ewc",      "buffer_size": 500, "fp16": True, "ewc_lambda": 50.0},
    {"method": "agem_distill","buffer_size": 500, "fp16": True, "distill_lambda": 1.0},
    {"method": "si_der",      "buffer_size": 500, "fp16": True, "si_lambda": 1.0},
]

BASE_CFG = {
    "dataset":        "split_mini_imagenet",
    "model":          "slim_resnet18",     # swap to "vit_small" for transformer runs
    "n_epochs":       1,
    "batch_size":     32,
    "lr":             0.03,
    "momentum":       0.9,
    "weight_decay":   1e-4,
    "num_workers":    4,
    "checkpoint_dir": CKPT_DIR,
    "figure_dir":     FIG_DIR,
    "log_dir":        LOG_DIR,
    "data_root":      DATA_ROOT,
}

all_results = {}

for method_overrides in METHODS_TO_RUN:
    cfg = {**BASE_CFG, **method_overrides}
    seed_everything(SEED)

    method_name = cfg["method"]
    run_name    = f"mini_imagenet_{method_name}_seed{SEED}"
    cfg["run_name"] = run_name

    print(f"\n{'='*60}")
    print(f" Running: {method_name} on Mini-ImageNet (seed={SEED})")
    print(f"{'='*60}")

    try:
        dataset = get_dataset(
            cfg["dataset"],
            root        = DATA_ROOT,
            batch_size  = cfg["batch_size"],
            num_workers = cfg["num_workers"],
        )

        in_ch = dataset.input_size[0]
        model = get_model(cfg["model"], in_channels=in_ch).to(DEVICE)
        method = get_method(method_name, model, cfg, torch.device(DEVICE))
        logger = RunLogger(LOG_DIR, run_name)
        trainer = CLTrainer(method, dataset, cfg, logger, torch.device(DEVICE))
        result  = trainer.train()

        all_results[method_name] = result

        # Generate individual plots
        plot_summary_grid(
            result["acc_matrix"], result["metrics"],
            method_name  = method_name,
            dataset_name = "mini_imagenet",
            fig_dir      = FIG_DIR,
        )
        plot_training_curves(
            result["acc_matrix"],
            method_name  = method_name,
            dataset_name = "mini_imagenet",
            fig_dir      = FIG_DIR,
        )
        print(f"\n✓ {method_name}: {result['metrics']}")

    except Exception as e:
        print(f"\n✗ {method_name} FAILED: {e}")
        import traceback; traceback.print_exc()

# ============================================================
# CELL 4: Generate comparison plots (run after all methods done)
# ============================================================

from src.visualization.plots import plot_all_metric_bars, plot_pareto_frontier

comparison_results = {
    name: {
        **res["metrics"],
        "buffer_size": BASE_CFG.get("buffer_size", 0)
    }
    for name, res in all_results.items()
}

plot_all_metric_bars(comparison_results, "mini_imagenet", fig_dir=FIG_DIR)
plot_pareto_frontier(comparison_results, "mini_imagenet", fig_dir=FIG_DIR)
print("Comparison plots saved to", FIG_DIR)

# Copy results back to Drive
# import shutil
# shutil.copytree("/content/results", f"{DRIVE_ROOT}/results", dirs_exist_ok=True)
# print("Results saved to Drive ✓")
