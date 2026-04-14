"""Save and restore training state (model weights, optimizer, replay buffer)."""

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Persist a state dict to disk.

    Args:
        state: Arbitrary dictionary—typically contains keys like
               ``model_state``, ``optimizer_state``, ``task_id``,
               ``acc_matrix``, and ``buffer``.
        path:  File path (parent directories are created automatically).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] Saved -> {path}")


def load_checkpoint(path: str, device: str = "cuda") -> Dict[str, Any]:
    """Load a checkpoint from *path* onto *device*.

    Args:
        path:   Path to the ``.pt`` file written by :func:`save_checkpoint`.
        device: Target device string, e.g. ``"cuda"`` or ``"cuda:0"``.

    Returns:
        The state dictionary that was originally passed to
        :func:`save_checkpoint`.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    print(f"[checkpoint] Loaded <- {path}")
    return state


def latest_checkpoint(checkpoint_dir: str, prefix: str) -> str | None:
    """Return the path of the most-recently saved checkpoint or *None*.

    Checkpoints are expected to follow the naming convention::

        {prefix}_task{N}.pt

    The one with the highest *N* is returned.
    """
    ckpt_dir = Path(checkpoint_dir)
    candidates = sorted(ckpt_dir.glob(f"{prefix}_task*.pt"))
    return str(candidates[-1]) if candidates else None


def cleanup_checkpoints(checkpoint_dir: str, prefix: str) -> int:
    """Delete all checkpoints matching ``{prefix}_task*.pt``.

    Returns the number of deleted checkpoint files.
    """
    ckpt_dir = Path(checkpoint_dir)
    deleted = 0
    for path in ckpt_dir.glob(f"{prefix}_task*.pt"):
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted
