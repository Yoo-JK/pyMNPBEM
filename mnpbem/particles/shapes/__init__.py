"""
Particle shape generators for MNPBEM.

This module provides functions to create common particle geometries:
- trisphere: Triangulated sphere
- trispherescale: Scaled sphere (non-uniform scaling)
- tricube: Cube with rounded edges
- trirod: Rod-shaped particle (cylinder with hemispherical caps)
- tritorus: Torus (donut shape)
- trispheresegment: Spherical segment
- tripolygon: Particle from polygon extrusion
- triellipsoid: Ellipsoid (prolate/oblate spheroid)
- tricone: Cone (with optional truncated tip)
- tribiconical: Bicone (two cones joined at base)
- trinanodisk: Nanodisk (flat cylinder)
- tricylinder: Cylinder with caps
- triplate: Rectangular plate
- triprism: Polygonal prism
"""

from .trisphere import trisphere, sphtriangulate, trispherescale
from .tricube import tricube
from .trirod import trirod
from .tritorus import tritorus
from .trispheresegment import trispheresegment
from .tripolygon import tripolygon
from .triellipsoid import triellipsoid, triellipsoid_uv
from .tricone import tricone, tribiconical
from .trinanodisk import trinanodisk, tricylinder
from .triplate import triplate, triprism
from .utils import fvgrid

__all__ = [
    "trisphere",
    "trispherescale",
    "sphtriangulate",
    "tricube",
    "trirod",
    "tritorus",
    "trispheresegment",
    "tripolygon",
    "triellipsoid",
    "triellipsoid_uv",
    "tricone",
    "tribiconical",
    "trinanodisk",
    "tricylinder",
    "triplate",
    "triprism",
    "fvgrid",
]
