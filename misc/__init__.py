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
from .geometry import (
    distmin3, distmin_particle, point_in_particle,
    nearest_face, project_to_surface, surface_distance,
    gap_distance, compute_solid_angle, mesh_quality
)

__all__ = [
    # Units
    "eV2nm",
    "nm2eV",
    "HARTREE",
    "TUNIT",
    # Options
    "BEMOptions",
    "bemoptions",
    "getbemoptions",
    # Helpers
    "inner",
    "outer",
    "matcross",
    "matmul",
    "spdiag",
    "vecnorm",
    "vecnormalize",
    # Plotting
    "plot_particle",
    "plot_spectrum",
    "plot_field_slice",
    "arrow_plot",
    "plot_eels_map",
    "create_colormap",
    # Geometry
    "distmin3",
    "distmin_particle",
    "point_in_particle",
    "nearest_face",
    "project_to_surface",
    "surface_distance",
    "gap_distance",
    "compute_solid_angle",
    "mesh_quality",
]
