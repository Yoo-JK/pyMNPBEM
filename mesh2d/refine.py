"""
Mesh refinement functions.
"""

import numpy as np
from typing import Tuple, Optional

from .inpoly import inpoly


def refine(
    verts: np.ndarray,
    faces: np.ndarray,
    polygon: np.ndarray = None,
    max_area: float = None,
    max_edge: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine a triangular mesh by subdivision.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2).
    faces : ndarray
        Mesh triangles, shape (n_triangles, 3).
    polygon : ndarray, optional
        Boundary polygon (to keep new points inside).
    max_area : float, optional
        Maximum triangle area.
    max_edge : float, optional
        Maximum edge length.

    Returns
    -------
    new_verts : ndarray
        Refined mesh vertices.
    new_faces : ndarray
        Refined mesh triangles.
    """
    # Calculate triangle areas and edge lengths
    areas = triarea(verts, faces)
    edges = get_edges(faces)
    edge_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)

    # Determine which triangles to refine
    if max_area is None:
        max_area = np.median(areas) * 2

    if max_edge is None:
        max_edge = np.median(edge_lengths) * 1.5

    # Find edges to split
    long_edges = edges[edge_lengths > max_edge]

    if len(long_edges) == 0:
        return verts, faces

    # Add midpoints
    new_verts_list = [verts]
    edge_midpoint_idx = {}

    for edge in long_edges:
        v1, v2 = sorted(edge)
        edge_key = (v1, v2)

        if edge_key not in edge_midpoint_idx:
            midpoint = (verts[v1] + verts[v2]) / 2

            # Check if inside polygon
            if polygon is not None:
                if not inpoly(midpoint.reshape(1, -1), polygon)[0]:
                    continue

            new_idx = len(verts) + len(edge_midpoint_idx)
            edge_midpoint_idx[edge_key] = new_idx
            new_verts_list.append(midpoint.reshape(1, -1))

    if len(new_verts_list) == 1:
        return verts, faces

    new_verts = np.vstack(new_verts_list)

    # Update faces
    new_faces = []

    for face in faces:
        v0, v1, v2 = face

        # Check which edges were split
        e01 = tuple(sorted([v0, v1]))
        e12 = tuple(sorted([v1, v2]))
        e20 = tuple(sorted([v2, v0]))

        m01 = edge_midpoint_idx.get(e01)
        m12 = edge_midpoint_idx.get(e12)
        m20 = edge_midpoint_idx.get(e20)

        splits = [m01, m12, m20]
        n_splits = sum(m is not None for m in splits)

        if n_splits == 0:
            # No splits, keep original
            new_faces.append(face)
        elif n_splits == 1:
            # One edge split - create two triangles
            if m01 is not None:
                new_faces.append([v0, m01, v2])
                new_faces.append([m01, v1, v2])
            elif m12 is not None:
                new_faces.append([v0, v1, m12])
                new_faces.append([v0, m12, v2])
            else:  # m20
                new_faces.append([v0, v1, m20])
                new_faces.append([m20, v1, v2])
        elif n_splits == 2:
            # Two edges split - create three triangles
            if m01 is None:
                new_faces.append([v0, v1, m12])
                new_faces.append([v0, m12, m20])
                new_faces.append([m12, v2, m20])
            elif m12 is None:
                new_faces.append([v0, m01, m20])
                new_faces.append([m01, v1, m20])
                new_faces.append([m20, v1, v2])
            else:  # m20 is None
                new_faces.append([v0, m01, v2])
                new_faces.append([m01, v1, m12])
                new_faces.append([m01, m12, v2])
        else:
            # All three edges split - create four triangles
            new_faces.append([v0, m01, m20])
            new_faces.append([m01, v1, m12])
            new_faces.append([m20, m12, v2])
            new_faces.append([m01, m12, m20])

    new_faces = np.array(new_faces)

    return new_verts, new_faces


def get_edges(faces: np.ndarray) -> np.ndarray:
    """
    Extract unique edges from triangulation.

    Parameters
    ----------
    faces : ndarray
        Triangle indices.

    Returns
    -------
    ndarray
        Edge array, shape (n_edges, 2).
    """
    edges = set()

    for face in faces:
        v0, v1, v2 = face
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))

    return np.array(list(edges))


def triarea(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute areas of triangles.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    ndarray
        Triangle areas.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Cross product for 2D
    area = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
                        (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))

    return area
