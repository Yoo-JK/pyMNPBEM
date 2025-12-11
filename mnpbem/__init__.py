"""
MNPBEM - Metallic Nanoparticle Boundary Element Method

A Python library for simulating the optical properties of metallic nanoparticles
using the boundary element method (BEM).

This is a Python port of the original MATLAB MNPBEM toolbox.

Modules:
--------
- material: Dielectric functions (constant, Drude, tabulated)
- particles: Particle geometry classes
- particles.shapes: Shape generation functions
- greenfun: Green function classes (quasistatic and retarded)
- bem: BEM solvers (quasistatic and retarded)
- simulation: Excitation and spectrum classes
- mie: Mie theory for spheres and ellipsoids
- mesh2d: 2D mesh generation utilities
- misc: Options, units, plotting utilities
"""

__version__ = "1.0.0"

# Core imports
from .misc.options import BEMOptions, bemoptions
from .misc.units import eV2nm, nm2eV

# Material classes
from .material import EpsConst, EpsDrude, EpsTable, EpsFun

# Particle classes
from .particles import Particle, Compound, ComParticle, ComPoint

# Particle shapes
from .particles.shapes import (
    trisphere, tricube, trirod, tritorus,
    triellipsoid, tricone, trinanodisk, triplate, tribiconical,
    tricylinder, triprism
)

# Green functions
from .greenfun import GreenStat, CompGreenStat, GreenRet, CompGreenRet

# BEM solvers
from .bem import BEMStat, BEMRet, bemsolver

# Simulation classes
from .simulation import (
    PlaneWaveStat, DipoleStat, SpectrumStat, planewave, dipole,
    PlaneWaveStatLayer, PlaneWaveStatMirror, planewave_layer,
    DipoleStatLayer, DipoleStatMirror, dipole_layer,
    EELSStat, eels
)

# Mie theory
from .mie import MieStat, MieRet, MieGans, miesolver

# Mesh2d module
from . import mesh2d

# Visualization utilities
from .misc.plotting import (
    plot_particle, plot_spectrum, plot_field_slice,
    arrow_plot, plot_eels_map, create_colormap
)

__all__ = [
    # Version
    "__version__",
    # Options
    "BEMOptions",
    "bemoptions",
    # Units
    "eV2nm",
    "nm2eV",
    # Materials
    "EpsConst",
    "EpsDrude",
    "EpsTable",
    "EpsFun",
    # Particles
    "Particle",
    "Compound",
    "ComParticle",
    "ComPoint",
    # Shapes
    "trisphere",
    "tricube",
    "trirod",
    "tritorus",
    "triellipsoid",
    "tricone",
    "trinanodisk",
    "triplate",
    "tribiconical",
    "tricylinder",
    "triprism",
    # Green functions
    "GreenStat",
    "CompGreenStat",
    "GreenRet",
    "CompGreenRet",
    # BEM
    "BEMStat",
    "BEMRet",
    "bemsolver",
    # Simulation
    "PlaneWaveStat",
    "DipoleStat",
    "SpectrumStat",
    "planewave",
    "dipole",
    "PlaneWaveStatLayer",
    "PlaneWaveStatMirror",
    "planewave_layer",
    "DipoleStatLayer",
    "DipoleStatMirror",
    "dipole_layer",
    "EELSStat",
    "eels",
    # Mie
    "MieStat",
    "MieRet",
    "MieGans",
    "miesolver",
    # Mesh2d
    "mesh2d",
    # Visualization
    "plot_particle",
    "plot_spectrum",
    "plot_field_slice",
    "arrow_plot",
    "plot_eels_map",
    "create_colormap",
]
