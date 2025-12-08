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

Classes
-------
Polygon : 2D polygon class for boundary definition
EdgeProfile : Edge profile for rounded corners
QuadTree : Spatial indexing for mesh generation

Shapes
------
circle : Create circular polygon
ellipse : Create elliptical polygon
rectangle : Create rectangular polygon
rounded_rectangle : Create rectangle with rounded corners
regular_polygon : Create regular polygon
"""

from .mesh2d import mesh2d, meshpoly
from .refine import refine
from .smoothmesh import smoothmesh
from .quality import quality, triarea
from .delaunay import delaunay_triangulate
from .inpoly import inpoly
from .quadtree import QuadTree
from .polygon import (
    Polygon, EdgeProfile, polygon_from_function,
    circle, ellipse, rectangle, rounded_rectangle, regular_polygon
)

__all__ = [
    # Meshing functions
    "mesh2d",
    "meshpoly",
    "refine",
    "smoothmesh",
    "quality",
    "triarea",
    "delaunay_triangulate",
    "inpoly",
    # Classes
    "Polygon",
    "EdgeProfile",
    "QuadTree",
    # Shape functions
    "polygon_from_function",
    "circle",
    "ellipse",
    "rectangle",
    "rounded_rectangle",
    "regular_polygon",
]
