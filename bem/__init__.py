"""
BEM solver classes for MNPBEM.

This module provides boundary element method solvers:
- BEMStat: Quasistatic BEM solver
- BEMRet: Retarded (full electromagnetic) BEM solver
- bemsolver: Factory function to create appropriate solver
- PlasmonMode: Eigenmode analysis for plasmonic structures
"""

from .bem_base import BEMBase
from .bem_stat import BEMStat
from .bem_ret import BEMRet
from .factory import bemsolver
from .plasmonmode import PlasmonMode, plasmonmode

__all__ = [
    "BEMBase",
    "BEMStat",
    "BEMRet",
    "bemsolver",
    "PlasmonMode",
    "plasmonmode",
]
