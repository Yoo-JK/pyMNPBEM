"""
Delaunay triangulation utilities.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import Delaunay


def delaunay_triangulate(points: np.ndarray) -> np.ndarray:
    """
    Perform Delaunay triangulation on a set of 2D points.

    Parameters
    ----------
    points : ndarray
        Point coordinates, shape (n_points, 2).

    Returns
    -------
    ndarray
        Triangle indices, shape (n_triangles, 3).
    """
    tri = Delaunay(points)
    return tri.simplices


def circumcircle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute circumcircle of a triangle.

    Parameters
    ----------
    p1, p2, p3 : ndarray
        Triangle vertices.

    Returns
    -------
    center : ndarray
        Circumcircle center.
    radius : float
        Circumcircle radius.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(d) < 1e-10:
        # Degenerate triangle
        center = (p1 + p2 + p3) / 3
        radius = 0
        return center, radius

    ux = ((ax * ax + ay * ay) * (by - cy) +
          (bx * bx + by * by) * (cy - ay) +
          (cx * cx + cy * cy) * (ay - by)) / d

    uy = ((ax * ax + ay * ay) * (cx - bx) +
          (bx * bx + by * by) * (ax - cx) +
          (cx * cx + cy * cy) * (bx - ax)) / d

    center = np.array([ux, uy])
    radius = np.linalg.norm(p1 - center)

    return center, radius


def in_circumcircle(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    """
    Check if point is inside circumcircle of triangle.

    Parameters
    ----------
    point : ndarray
        Point to test.
    p1, p2, p3 : ndarray
        Triangle vertices.

    Returns
    -------
    bool
        True if point is inside circumcircle.
    """
    center, radius = circumcircle(p1, p2, p3)
    return np.linalg.norm(point - center) < radius


def find_edge(faces: np.ndarray, v1: int, v2: int) -> list:
    """
    Find triangles sharing an edge.

    Parameters
    ----------
    faces : ndarray
        Triangle indices.
    v1, v2 : int
        Edge vertex indices.

    Returns
    -------
    list
        Indices of triangles containing the edge.
    """
    edge = {v1, v2}
    result = []

    for i, face in enumerate(faces):
        face_verts = set(face)
        if edge.issubset(face_verts):
            result.append(i)

    return result


def edge_flip(
    verts: np.ndarray,
    faces: np.ndarray,
    edge_idx: Tuple[int, int]
) -> np.ndarray:
    """
    Flip an edge to improve mesh quality.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.
    edge_idx : tuple
        Edge vertex indices (v1, v2).

    Returns
    -------
    ndarray
        Updated faces array.
    """
    v1, v2 = edge_idx
    tris = find_edge(faces, v1, v2)

    if len(tris) != 2:
        return faces  # Can only flip internal edges

    # Get the four vertices
    tri1 = faces[tris[0]]
    tri2 = faces[tris[1]]

    v3 = [v for v in tri1 if v not in [v1, v2]][0]
    v4 = [v for v in tri2 if v not in [v1, v2]][0]

    # Check if flip improves quality
    # (edge v1-v2 to edge v3-v4)
    p1, p2, p3, p4 = verts[v1], verts[v2], verts[v3], verts[v4]

    # Check Delaunay criterion
    if in_circumcircle(p4, p1, p2, p3) or in_circumcircle(p3, p1, p2, p4):
        # Flip the edge
        new_faces = faces.copy()
        new_faces[tris[0]] = [v1, v3, v4]
        new_faces[tris[1]] = [v2, v4, v3]
        return new_faces

    return faces
