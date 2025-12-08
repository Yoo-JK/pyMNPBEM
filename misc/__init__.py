"""
Miscellaneous utilities for MNPBEM.
"""

from .units import eV2nm, nm2eV, HARTREE, TUNIT
from .options import BEMOptions, bemoptions, getbemoptions
from .helpers import inner, outer, matcross, matmul, spdiag, vecnorm, vecnormalize
from .plotting import (
    plot_particle, plot_spectrum, plot_field_slice,
    arrow_plot, plot_eels_map, create_colormap
)

__all__ = [
    "eV2nm",
    "nm2eV",
    "HARTREE",
    "TUNIT",
    "BEMOptions",
    "bemoptions",
    "getbemoptions",
    "inner",
    "outer",
    "matcross",
    "matmul",
    "spdiag",
    "vecnorm",
    "vecnormalize",
    "plot_particle",
    "plot_spectrum",
    "plot_field_slice",
    "arrow_plot",
    "plot_eels_map",
    "create_colormap",
]
