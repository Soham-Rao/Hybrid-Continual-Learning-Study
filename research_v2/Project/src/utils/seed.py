"""Deterministic seeding for full reproducibility across torch, numpy, and random."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for deterministic behaviour.

    Args:
        seed: Integer seed value. Use the same seed across train/eval to
              guarantee identical results on re-runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable non-deterministic CUDA kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
