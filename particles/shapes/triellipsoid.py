"""
Triangulated ellipsoid generation.
"""

import numpy as np
from typing import Tuple
from ..particle import Particle


def triellipsoid(
    n: int,
    axes: Tuple[float, float, float],
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated ellipsoid.

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    axes : tuple
        Semi-axes (a, b, c) in nm. Ellipsoid: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    interp : str
        Interpolation type: 'flat' or 'curv'.

    Returns
    -------
    Particle
        Triangulated ellipsoid.

    Examples
    --------
    >>> # Prolate spheroid (cigar-shaped)
    >>> p = triellipsoid(144, (50, 10, 10))
    >>> # Oblate spheroid (disk-shaped)
    >>> p = triellipsoid(144, (20, 20, 5))
    """
    a, b, c = axes

    # Generate points on unit sphere using Fibonacci spiral
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    # Spherical to Cartesian (unit sphere)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    # Scale to ellipsoid
    verts = np.column_stack([a * x, b * y, c * z])

    # Triangulate using convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    faces = hull.simplices

    return Particle(verts, faces, interp)


def triellipsoid_uv(
    n_theta: int,
    n_phi: int,
    axes: Tuple[float, float, float],
    interp: str = 'curv'
) -> Particle:
    """
    Create ellipsoid using UV parameterization.

    Parameters
    ----------
    n_theta : int
        Number of points in azimuthal direction.
    n_phi : int
        Number of points in polar direction.
    axes : tuple
        Semi-axes (a, b, c).
    interp : str
        Interpolation type.

    Returns
    -------
    Particle
        Triangulated ellipsoid.
    """
    a, b, c = axes

    # Create parameter grid
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)

    # Generate vertices
    verts = []
    for p in phi:
        for t in theta:
            x = a * np.sin(p) * np.cos(t)
            y = b * np.sin(p) * np.sin(t)
            z = c * np.cos(p)
            verts.append([x, y, z])

    verts = np.array(verts)

    # Generate faces
    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta

            # Current row indices
            idx0 = i * n_theta + j
            idx1 = i * n_theta + j_next

            # Next row indices
            idx2 = (i + 1) * n_theta + j
            idx3 = (i + 1) * n_theta + j_next

            # Two triangles per quad
            if i > 0:  # Not at north pole
                faces.append([idx0, idx1, idx3])
            if i < n_phi - 2:  # Not at south pole
                faces.append([idx0, idx3, idx2])

    faces = np.array(faces)

    return Particle(verts, faces, interp)
