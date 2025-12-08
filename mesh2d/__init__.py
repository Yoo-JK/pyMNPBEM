"""
2D mesh generation module for MNPBEM.

This module provides functions for generating 2D triangular meshes
that can be used to create particle surfaces.

Functions
---------
mesh2d : Main mesh generation function
meshpoly : Mesh a polygon region
refine : Mesh refinement
smoothmesh : Mesh smoothing
quality : Mesh quality metrics
"""

from .mesh2d import mesh2d, meshpoly
from .refine import refine
from .smoothmesh import smoothmesh
from .quality import quality, triarea
from .delaunay import delaunay_triangulate
from .inpoly import inpoly
from .quadtree import QuadTree

__all__ = [
    "mesh2d",
    "meshpoly",
    "refine",
    "smoothmesh",
    "quality",
    "triarea",
    "delaunay_triangulate",
    "inpoly",
    "QuadTree",
]
