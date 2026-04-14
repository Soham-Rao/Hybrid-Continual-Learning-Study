# src/methods/hybrid/__init__.py
from .der              import DER
from .xder             import XDER
from .icarl            import iCaRL
from .er_ewc           import ER_EWC
from .progress_compress import ProgressCompress
from .agem_distill     import AGEM_Distill
from .si_der           import SI_DER
__all__ = ["DER", "XDER", "iCaRL", "ER_EWC", "ProgressCompress", "AGEM_Distill", "SI_DER"]
