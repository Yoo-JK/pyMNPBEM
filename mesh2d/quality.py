"""
Mesh quality metrics.
"""

import numpy as np
from typing import Tuple


def quality(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute mesh quality for each triangle.

    Quality is measured as the ratio of inscribed to circumscribed circle
    radii, normalized to [0, 1] where 1 is a perfect equilateral triangle.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2).
    faces : ndarray
        Mesh triangles, shape (n_triangles, 3).

    Returns
    -------
    ndarray
        Quality metric for each triangle.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Edge vectors
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # Edge lengths
    a = np.linalg.norm(e0, axis=1)
    b = np.linalg.norm(e1, axis=1)
    c = np.linalg.norm(e2, axis=1)

    # Semi-perimeter
    s = (a + b + c) / 2

    # Area (Heron's formula)
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0)  # Handle numerical issues
    area = np.sqrt(area_sq)

    # Circumradius
    R = (a * b * c) / (4 * area + 1e-10)

    # Inradius
    r = area / (s + 1e-10)

    # Quality: 2 * r / R (normalized so equilateral = 1)
    q = 2 * r / (R + 1e-10)

    return q


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


def aspect_ratio(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute aspect ratio of triangles.

    Aspect ratio = longest edge / shortest edge.
    Ideal value is 1 for equilateral triangles.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    ndarray
        Aspect ratio for each triangle.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)

    edges = np.column_stack([a, b, c])
    longest = edges.max(axis=1)
    shortest = edges.min(axis=1)

    return longest / (shortest + 1e-10)


def edge_lengths(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute edge length statistics.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    lengths : ndarray
        All edge lengths.
    mean_length : float
        Mean edge length.
    std_length : float
        Standard deviation of edge lengths.
    """
    edges = set()

    for face in faces:
        v0, v1, v2 = face
        edges.add(tuple(sorted([v0, v1])))
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v0])))

    edges = np.array(list(edges))
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)

    return lengths, lengths.mean(), lengths.std()


def mesh_stats(verts: np.ndarray, faces: np.ndarray) -> dict:
    """
    Compute comprehensive mesh statistics.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    dict
        Dictionary of mesh statistics.
    """
    areas = triarea(verts, faces)
    qualities = quality(verts, faces)
    ratios = aspect_ratio(verts, faces)
    lengths, mean_len, std_len = edge_lengths(verts, faces)

    return {
        'n_vertices': len(verts),
        'n_triangles': len(faces),
        'total_area': areas.sum(),
        'mean_area': areas.mean(),
        'min_area': areas.min(),
        'max_area': areas.max(),
        'mean_quality': qualities.mean(),
        'min_quality': qualities.min(),
        'mean_aspect_ratio': ratios.mean(),
        'max_aspect_ratio': ratios.max(),
        'mean_edge_length': mean_len,
        'std_edge_length': std_len,
        'min_edge_length': lengths.min(),
        'max_edge_length': lengths.max(),
    }
