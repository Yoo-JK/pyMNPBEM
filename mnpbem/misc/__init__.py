"""
Miscellaneous utilities for MNPBEM.
"""

from .units import (
    eV2nm, nm2eV, HARTREE, TUNIT, SPEED_OF_LIGHT,
    BOHR, bohr, FINE, fine, hartree, c,
    wavelength_to_energy, energy_to_wavelength
)
from .options import BEMOptions, bemoptions, getbemoptions
from .helpers import (
    inner, outer, matcross, matmul, spdiag, vecnorm, vecnormalize,
    bdist2, pdist2, bradius, refinematrix, refinematrixlayer,
    progress_bar, memsize, memsize_str
)
from .plotting import (
    plot_particle, plot_spectrum, plot_field_slice,
    arrow_plot, plot_eels_map, create_colormap,
    coneplot, coneplot2, patchcurvature, plot_curvature, streamplot3d
)
from .geometry import (
    distmin3, distmin_particle, point_in_particle,
    nearest_face, project_to_surface, surface_distance,
    gap_distance, compute_solid_angle, mesh_quality
)
from .meshfield import (
    MeshField, meshfield, interpolate_field, field_at_points,
    GridField, gridfield, GridFieldPoints
)
from .arrays import (
    ValArray, VecArray, valarray, vecarray,
    igrid, meshgrid3d, linspace_grid, sphere_grid, cylinder_grid
)
from .integration import (
    lglnodes, lgwt, triangle_quadrature, quad_quadrature,
    QuadFace, quadface
)

__all__ = [
    # Units and physical constants
    "eV2nm",
    "nm2eV",
    "HARTREE",
    "hartree",
    "TUNIT",
    "SPEED_OF_LIGHT",
    "c",
    "BOHR",
    "bohr",
    "FINE",
    "fine",
    "wavelength_to_energy",
    "energy_to_wavelength",
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
    "bdist2",
    "pdist2",
    "bradius",
    "refinematrix",
    "refinematrixlayer",
    "progress_bar",
    "memsize",
    "memsize_str",
    # Plotting
    "plot_particle",
    "plot_spectrum",
    "plot_field_slice",
    "arrow_plot",
    "plot_eels_map",
    "create_colormap",
    "coneplot",
    "coneplot2",
    "patchcurvature",
    "plot_curvature",
    "streamplot3d",
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
    # Mesh field
    "MeshField",
    "meshfield",
    "GridField",
    "gridfield",
    "GridFieldPoints",
    "interpolate_field",
    "field_at_points",
    # Arrays
    "ValArray",
    "VecArray",
    "valarray",
    "vecarray",
    # Grids
    "igrid",
    "meshgrid3d",
    "linspace_grid",
    "sphere_grid",
    "cylinder_grid",
    # Integration
    "lglnodes",
    "lgwt",
    "triangle_quadrature",
    "quad_quadrature",
    "QuadFace",
    "quadface",
]
