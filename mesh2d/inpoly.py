"""
Point-in-polygon tests.
"""

import numpy as np
from typing import Union


def inpoly(
    points: np.ndarray,
    polygon: np.ndarray
) -> np.ndarray:
    """
    Test if points are inside a polygon.

    Uses ray casting algorithm.

    Parameters
    ----------
    points : ndarray
        Test points, shape (n_points, 2).
    polygon : ndarray
        Polygon vertices, shape (n_verts, 2). Should be closed.

    Returns
    -------
    ndarray
        Boolean array, True if inside polygon.

    Examples
    --------
    >>> polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    >>> points = np.array([[0.5, 0.5], [2, 2]])
    >>> inpoly(points, polygon)
    array([ True, False])
    """
    points = np.asarray(points)
    polygon = np.asarray(polygon)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Close polygon if needed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    n_points = len(points)
    n_verts = len(polygon) - 1

    inside = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        x, y = points[i]
        crossings = 0

        for j in range(n_verts):
            x1, y1 = polygon[j]
            x2, y2 = polygon[j + 1]

            # Check if ray from (x, y) going right crosses edge
            if ((y1 <= y < y2) or (y2 <= y < y1)):
                # Compute x-coordinate of intersection
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    crossings += 1

        inside[i] = (crossings % 2) == 1

    return inside


def inpoly_fast(
    points: np.ndarray,
    polygon: np.ndarray
) -> np.ndarray:
    """
    Vectorized point-in-polygon test.

    Parameters
    ----------
    points : ndarray
        Test points, shape (n_points, 2).
    polygon : ndarray
        Polygon vertices, shape (n_verts, 2).

    Returns
    -------
    ndarray
        Boolean array, True if inside polygon.
    """
    from matplotlib.path import Path

    path = Path(polygon)
    return path.contains_points(points)


def dist2poly(
    points: np.ndarray,
    polygon: np.ndarray
) -> np.ndarray:
    """
    Compute distance from points to polygon boundary.

    Parameters
    ----------
    points : ndarray
        Test points, shape (n_points, 2).
    polygon : ndarray
        Polygon vertices.

    Returns
    -------
    ndarray
        Distance to nearest boundary edge.
    """
    points = np.asarray(points)
    polygon = np.asarray(polygon)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    n_points = len(points)
    n_edges = len(polygon) - 1

    min_dist = np.full(n_points, np.inf)

    for j in range(n_edges):
        p1 = polygon[j]
        p2 = polygon[j + 1]

        # Distance from each point to this edge
        dist = point_to_segment_distance(points, p1, p2)
        min_dist = np.minimum(min_dist, dist)

    # Negative for points inside
    inside = inpoly(points, polygon)
    min_dist[inside] *= -1

    return min_dist


def point_to_segment_distance(
    points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray
) -> np.ndarray:
    """
    Compute distance from points to a line segment.

    Parameters
    ----------
    points : ndarray
        Test points, shape (n_points, 2).
    p1, p2 : ndarray
        Segment endpoints.

    Returns
    -------
    ndarray
        Distance to segment.
    """
    v = p2 - p1
    w = points - p1

    c1 = np.sum(w * v, axis=1)
    c2 = np.dot(v, v)

    if c2 == 0:
        return np.linalg.norm(w, axis=1)

    t = np.clip(c1 / c2, 0, 1)

    projection = p1 + t[:, np.newaxis] * v
    return np.linalg.norm(points - projection, axis=1)
