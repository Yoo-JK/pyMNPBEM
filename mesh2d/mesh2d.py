"""
Main 2D mesh generation functions.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.spatial import Delaunay

from .inpoly import inpoly
from .refine import refine
from .smoothmesh import smoothmesh


def mesh2d(
    polygon: np.ndarray,
    h: float = None,
    n_refine: int = 0,
    n_smooth: int = 3,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D triangular mesh inside a polygon.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices, shape (n_verts, 2). Must be closed (first = last).
    h : float, optional
        Target edge length. If None, uses automatic sizing.
    n_refine : int
        Number of refinement iterations.
    n_smooth : int
        Number of smoothing iterations.

    Returns
    -------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2).
    faces : ndarray
        Mesh triangles, shape (n_triangles, 3).

    Examples
    --------
    >>> # Create mesh inside a square
    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    >>> verts, faces = mesh2d(square, h=0.1)
    """
    polygon = np.asarray(polygon)

    # Close polygon if needed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    # Determine bounding box
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)

    # Determine target edge length
    if h is None:
        perimeter = np.sum(np.linalg.norm(np.diff(polygon, axis=0), axis=1))
        area = polygon_area(polygon)
        h = np.sqrt(area / 50)  # Heuristic: ~50 triangles

    # Generate initial grid of points
    n_x = int((x_max - x_min) / h) + 1
    n_y = int((y_max - y_min) / h) + 1

    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)

    # Create point grid
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Keep only points inside polygon
    mask = inpoly(points, polygon)
    interior_points = points[mask]

    # Add boundary points
    boundary_points = sample_polygon_boundary(polygon, h)

    # Combine points
    all_points = np.vstack([boundary_points, interior_points])

    # Remove duplicates
    all_points = remove_duplicate_points(all_points, tol=h / 10)

    # Delaunay triangulation
    tri = Delaunay(all_points)
    faces = tri.simplices

    # Remove triangles outside polygon
    centroids = all_points[faces].mean(axis=1)
    inside = inpoly(centroids, polygon)
    faces = faces[inside]

    # Refine mesh
    verts = all_points
    for _ in range(n_refine):
        verts, faces = refine(verts, faces, polygon=polygon)

    # Smooth mesh
    verts = smoothmesh(verts, faces, polygon=polygon, n_iter=n_smooth)

    return verts, faces


def meshpoly(
    polygon: np.ndarray,
    holes: List[np.ndarray] = None,
    h: float = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mesh inside polygon with optional holes.

    Parameters
    ----------
    polygon : ndarray
        Outer boundary polygon.
    holes : list of ndarray, optional
        List of hole polygons.
    h : float, optional
        Target edge length.

    Returns
    -------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.
    """
    # Generate base mesh
    verts, faces = mesh2d(polygon, h=h, **kwargs)

    if holes is None:
        return verts, faces

    # Remove triangles inside holes
    centroids = verts[faces].mean(axis=1)

    for hole in holes:
        hole = np.asarray(hole)
        if not np.allclose(hole[0], hole[-1]):
            hole = np.vstack([hole, hole[0]])

        inside_hole = inpoly(centroids, hole)
        faces = faces[~inside_hole]
        centroids = verts[faces].mean(axis=1)

    # Clean up unused vertices
    used_verts = np.unique(faces)
    vert_map = np.zeros(len(verts), dtype=int)
    vert_map[used_verts] = np.arange(len(used_verts))

    verts = verts[used_verts]
    faces = vert_map[faces]

    return verts, faces


def polygon_area(polygon: np.ndarray) -> float:
    """
    Compute area of a polygon using shoelace formula.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices (closed).

    Returns
    -------
    float
        Polygon area.
    """
    x = polygon[:-1, 0]
    y = polygon[:-1, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sample_polygon_boundary(polygon: np.ndarray, h: float) -> np.ndarray:
    """
    Sample points along polygon boundary with spacing h.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices (closed).
    h : float
        Target spacing.

    Returns
    -------
    ndarray
        Boundary sample points.
    """
    points = []

    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        edge_len = np.linalg.norm(p2 - p1)
        n_pts = max(int(edge_len / h), 1)

        t = np.linspace(0, 1, n_pts, endpoint=False)
        for ti in t:
            points.append(p1 + ti * (p2 - p1))

    return np.array(points)


def remove_duplicate_points(points: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Remove duplicate points within tolerance.

    Parameters
    ----------
    points : ndarray
        Point array.
    tol : float
        Tolerance for considering points as duplicates.

    Returns
    -------
    ndarray
        Points with duplicates removed.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, tol)

    # Keep first point in each group
    keep = []
    seen = set()
    for i, group in enumerate(groups):
        rep = min(group)
        if rep not in seen:
            keep.append(rep)
            seen.update(group)

    return points[sorted(keep)]
