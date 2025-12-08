"""
BEM solver classes for MNPBEM.

This module provides boundary element method solvers:

Basic:
- BEMStat: Quasistatic BEM solver
- BEMRet: Retarded (full electromagnetic) BEM solver
- bemsolver: Factory function to create appropriate solver

Layer substrate:
- BEMStatLayer: Quasistatic BEM with layer substrate
- BEMRetLayer: Retarded BEM with layer substrate

Mirror symmetry:
- BEMStatMirror: Quasistatic BEM with mirror symmetry
- BEMRetMirror: Retarded BEM with mirror symmetry

Iterative solvers:
- BEMIter: Base iterative BEM solver
- BEMStatIter: Iterative quasistatic BEM solver
- BEMRetIter: Iterative retarded BEM solver

Eigenvalue solvers:
- BEMStatEig: Eigenvalue BEM for plasmon modes
- BEMStatEigMirror: Eigenvalue BEM with mirror symmetry

Analysis:
- PlasmonMode: Eigenmode analysis for plasmonic structures
"""

from .bem_base import BEMBase
from .bem_stat import BEMStat
from .bem_ret import BEMRet
from .factory import bemsolver
from .plasmonmode import PlasmonMode, plasmonmode
from .bem_stat_layer import BEMStatLayer
from .bem_stat_mirror import BEMStatMirror
from .bem_ret_layer import BEMRetLayer, BEMRetMirror
from .bem_iter import BEMIter, BEMStatIter, BEMRetIter
from .bem_stat_eig import BEMStatEig, BEMStatEigMirror

__all__ = [
    # Basic
    "BEMBase",
    "BEMStat",
    "BEMRet",
    "bemsolver",
    # Layer substrate
    "BEMStatLayer",
    "BEMRetLayer",
    # Mirror symmetry
    "BEMStatMirror",
    "BEMRetMirror",
    # Iterative
    "BEMIter",
    "BEMStatIter",
    "BEMRetIter",
    # Eigenvalue
    "BEMStatEig",
    "BEMStatEigMirror",
    # Analysis
    "PlasmonMode",
    "plasmonmode",
]
