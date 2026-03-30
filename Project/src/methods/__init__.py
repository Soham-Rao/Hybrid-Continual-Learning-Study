"""Method registries for baselines and hybrid methods."""

from .base_method import BaseCLMethod, ReplayBuffer
from .baselines.fine_tune   import FineTune
from .baselines.joint_train import JointTraining
from .baselines.ewc         import EWC
from .baselines.agem        import AGEM
from .baselines.lwf         import LwF
from .hybrid.der            import DER
from .hybrid.xder           import XDER
from .hybrid.icarl          import iCaRL
from .hybrid.er_ewc         import ER_EWC
from .hybrid.progress_compress import ProgressCompress
from .hybrid.agem_distill   import AGEM_Distill
from .hybrid.si_der         import SI_DER

_REGISTRY = {
    # Baselines
    "fine_tune":         FineTune,
    "joint_training":    JointTraining,
    "ewc":               EWC,
    "agem":              AGEM,
    "lwf":               LwF,
    # Hybrids
    "der":               DER,
    "xder":              XDER,
    "icarl":             iCaRL,
    "er_ewc":            ER_EWC,
    "progress_compress": ProgressCompress,
    "agem_distill":      AGEM_Distill,
    "si_der":            SI_DER,
}


def get_method(name: str, model, cfg: dict, device) -> BaseCLMethod:
    """Return a CL method instance by registry key.

    Args:
        name:   One of the registry keys listed above.
        model:  A :class:`~src.models.CLModel` instance.
        cfg:    Hyperparameter dict (from YAML config).
        device: ``torch.device``.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](model, cfg, device)


__all__ = list(_REGISTRY.keys()) + ["get_method", "BaseCLMethod", "ReplayBuffer"]
