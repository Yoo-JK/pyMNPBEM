"""
Green function classes for MNPBEM.

This module provides Green function calculations:

Basic:
- GreenStat: Quasistatic (Coulomb) Green function
- CompGreenStat: Composite Green function for compound particles
- GreenRet: Retarded (full electromagnetic) Green function
- CompGreenRet: Composite retarded Green function

Layer effects:
- GreenRetLayer: Retarded Green function with layer effects
- CompGreenRetLayer: Composite retarded Green function with layer effects
- CoverLayer: Coating layer for core-shell particles
- GreenStatCover: Green function with cover layer
- GreenRetCover: Retarded Green function with cover layer

Acceleration:
- ACAMatrix: Low-rank matrix from ACA decomposition
- ACAGreen: ACA-compressed Green function
- HMatrix: Hierarchical matrix representation
- HMatrixGreen: H-matrix representation of Green function
"""

from .green_stat import GreenStat
from .comp_green_stat import CompGreenStat
from .green_ret import GreenRet, GreenRetLayer
from .comp_green_ret import CompGreenRet, CompGreenRetLayer
from .coverlayer import CoverLayer, GreenStatCover, GreenRetCover, coverlayer
from .aca import ACAMatrix, ACAGreen, aca, aca_full, aca_partial, CompressedGreenMatrix
from .hmatrix import ClusterTree, HMatrix, HMatrixBlock, HMatrixGreen, hmatrix_solve

__all__ = [
    # Basic
    "GreenStat",
    "CompGreenStat",
    "GreenRet",
    "GreenRetLayer",
    "CompGreenRet",
    "CompGreenRetLayer",
    # Cover layer
    "CoverLayer",
    "GreenStatCover",
    "GreenRetCover",
    "coverlayer",
    # ACA
    "ACAMatrix",
    "ACAGreen",
    "aca",
    "aca_full",
    "aca_partial",
    "CompressedGreenMatrix",
    # H-matrices
    "ClusterTree",
    "HMatrix",
    "HMatrixBlock",
    "HMatrixGreen",
    "hmatrix_solve",
]
