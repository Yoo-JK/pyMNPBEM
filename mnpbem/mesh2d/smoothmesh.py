"""
Mesh smoothing functions.
"""

import numpy as np
from typing import Optional

from .inpoly import inpoly


def smoothmesh(
    verts: np.ndarray,
    faces: np.ndarray,
    polygon: np.ndarray = None,
    n_iter: int = 3,
    method: str = 'laplacian'
) -> np.ndarray:
    """
    Smooth a triangular mesh.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2).
    faces : ndarray
        Mesh triangles, shape (n_triangles, 3).
    polygon : ndarray, optional
        Boundary polygon (boundary points are not moved).
    n_iter : int
        Number of smoothing iterations.
    method : str
        Smoothing method: 'laplacian' or 'taubin'.

    Returns
    -------
    ndarray
        Smoothed vertex positions.
    """
    verts = verts.copy()
    n_verts = len(verts)

    # Find boundary vertices
    if polygon is not None:
        boundary_mask = is_boundary_vertex(verts, polygon)
    else:
        boundary_mask = find_mesh_boundary(verts, faces)

    # Build adjacency
    adjacency = build_adjacency(faces, n_verts)

    if method == 'laplacian':
        for _ in range(n_iter):
            verts = laplacian_smooth(verts, adjacency, boundary_mask)
    elif method == 'taubin':
        for _ in range(n_iter):
            verts = taubin_smooth(verts, adjacency, boundary_mask)

    return verts


def laplacian_smooth(
    verts: np.ndarray,
    adjacency: list,
    boundary_mask: np.ndarray,
    weight: float = 0.5
) -> np.ndarray:
    """
    One iteration of Laplacian smoothing.

    Parameters
    ----------
    verts : ndarray
        Vertex positions.
    adjacency : list
        Adjacency list for each vertex.
    boundary_mask : ndarray
        Boolean mask for boundary vertices.
    weight : float
        Smoothing weight (0-1).

    Returns
    -------
    ndarray
        Smoothed positions.
    """
    new_verts = verts.copy()

    for i in range(len(verts)):
        if boundary_mask[i]:
            continue

        neighbors = adjacency[i]
        if len(neighbors) > 0:
            centroid = verts[neighbors].mean(axis=0)
            new_verts[i] = (1 - weight) * verts[i] + weight * centroid

    return new_verts


def taubin_smooth(
    verts: np.ndarray,
    adjacency: list,
    boundary_mask: np.ndarray,
    lambda_factor: float = 0.5,
    mu_factor: float = -0.53
) -> np.ndarray:
    """
    Taubin smoothing (shrinkage-free).

    Parameters
    ----------
    verts : ndarray
        Vertex positions.
    adjacency : list
        Adjacency list.
    boundary_mask : ndarray
        Boundary mask.
    lambda_factor : float
        Shrink factor.
    mu_factor : float
        Inflate factor (should be negative, |mu| > lambda).

    Returns
    -------
    ndarray
        Smoothed positions.
    """
    # Shrink step
    verts = laplacian_smooth(verts, adjacency, boundary_mask, lambda_factor)
    # Inflate step
    verts = laplacian_smooth(verts, adjacency, boundary_mask, -mu_factor)

    return verts


def build_adjacency(faces: np.ndarray, n_verts: int) -> list:
    """
    Build vertex adjacency list from triangulation.

    Parameters
    ----------
    faces : ndarray
        Triangle indices.
    n_verts : int
        Number of vertices.

    Returns
    -------
    list
        Adjacency list for each vertex.
    """
    adjacency = [set() for _ in range(n_verts)]

    for face in faces:
        v0, v1, v2 = face
        adjacency[v0].update([v1, v2])
        adjacency[v1].update([v0, v2])
        adjacency[v2].update([v0, v1])

    return [list(adj) for adj in adjacency]


def is_boundary_vertex(verts: np.ndarray, polygon: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Check if vertices are on the boundary polygon.

    Parameters
    ----------
    verts : ndarray
        Vertex positions.
    polygon : ndarray
        Boundary polygon.
    tol : float
        Distance tolerance.

    Returns
    -------
    ndarray
        Boolean mask for boundary vertices.
    """
    from .inpoly import point_to_segment_distance

    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    boundary_mask = np.zeros(len(verts), dtype=bool)

    for i in range(len(polygon) - 1):
        p1, p2 = polygon[i], polygon[i + 1]
        dist = point_to_segment_distance(verts, p1, p2)
        boundary_mask |= (dist < tol)

    return boundary_mask


def find_mesh_boundary(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Find boundary vertices of a mesh (edges with only one adjacent face).

    Parameters
    ----------
    verts : ndarray
        Vertex positions.
    faces : ndarray
        Triangle indices.

    Returns
    -------
    ndarray
        Boolean mask for boundary vertices.
    """
    from collections import Counter

    edge_count = Counter()

    for face in faces:
        v0, v1, v2 = face
        edge_count[tuple(sorted([v0, v1]))] += 1
        edge_count[tuple(sorted([v1, v2]))] += 1
        edge_count[tuple(sorted([v2, v0]))] += 1

    boundary_verts = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_verts.update(edge)

    boundary_mask = np.zeros(len(verts), dtype=bool)
    boundary_mask[list(boundary_verts)] = True

    return boundary_mask
