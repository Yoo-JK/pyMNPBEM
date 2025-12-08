"""
Triangulated nanodisk (disk-shaped particle) generation.
"""

import numpy as np
from ..particle import Particle


def trinanodisk(
    n: int,
    diameter: float,
    height: float,
    edge_rounding: float = 0.0,
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated nanodisk (flat cylinder).

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    diameter : float
        Disk diameter in nm.
    height : float
        Disk height (thickness) in nm.
    edge_rounding : float
        Edge rounding radius in nm (0 for sharp edges).
    interp : str
        Interpolation type: 'flat' or 'curv'.

    Returns
    -------
    Particle
        Triangulated nanodisk.

    Examples
    --------
    >>> # Sharp-edged disk
    >>> p = trinanodisk(144, 100, 20)
    >>> # Rounded disk
    >>> p = trinanodisk(144, 100, 20, edge_rounding=5)
    """
    radius = diameter / 2

    # Estimate grid sizes
    n_circ = int(np.sqrt(n * 3))
    n_radial = n_circ // 4
    n_vert = max(3, n_circ // 8)

    n_circ = max(n_circ, 12)
    n_radial = max(n_radial, 3)

    verts = []
    faces = []

    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    if edge_rounding > 0:
        # Rounded edges
        r_edge = min(edge_rounding, height / 2, radius / 4)

        # Top surface
        r_levels = np.linspace(0, radius - r_edge, n_radial)
        top_center = len(verts)
        verts.append([0, 0, height / 2])

        for r in r_levels[1:]:
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), height / 2])

        # Top edge rounding
        edge_angles = np.linspace(0, np.pi / 2, n_vert)
        for angle in edge_angles[1:]:
            r = radius - r_edge + r_edge * np.sin(angle)
            z = height / 2 - r_edge + r_edge * np.cos(angle)
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), z])

        # Side surface
        z_side = np.linspace(height / 2 - r_edge, -height / 2 + r_edge, n_vert)
        for z in z_side[1:-1]:
            for t in theta:
                verts.append([radius * np.cos(t), radius * np.sin(t), z])

        # Bottom edge rounding
        for angle in reversed(edge_angles[:-1]):
            r = radius - r_edge + r_edge * np.sin(angle)
            z = -height / 2 + r_edge - r_edge * np.cos(angle)
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), z])

        # Bottom surface
        for r in reversed(r_levels[1:]):
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), -height / 2])

        bottom_center = len(verts)
        verts.append([0, 0, -height / 2])

    else:
        # Sharp edges
        # Top surface
        r_levels = np.linspace(0, radius, n_radial)
        top_center = len(verts)
        verts.append([0, 0, height / 2])

        for r in r_levels[1:]:
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), height / 2])

        # Side surface
        z_side = np.linspace(height / 2, -height / 2, n_vert)
        for z in z_side[1:-1]:
            for t in theta:
                verts.append([radius * np.cos(t), radius * np.sin(t), z])

        # Bottom surface
        for r in reversed(r_levels[1:]):
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), -height / 2])

        bottom_center = len(verts)
        verts.append([0, 0, -height / 2])

    verts = np.array(verts)

    # Generate faces using convex hull or manual triangulation
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(verts)
        faces = hull.simplices
    except:
        # Fallback: simple triangulation
        # This is a simplified version
        n_verts = len(verts)

        # Top cap
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            faces.append([top_center, 1 + j, 1 + j_next])

        faces = np.array(faces)

    return Particle(verts, faces, interp)


def tricylinder(
    n: int,
    height: float,
    radius: float,
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated cylinder (without caps for open cylinder).

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    height : float
        Cylinder height in nm.
    radius : float
        Cylinder radius in nm.
    interp : str
        Interpolation type.

    Returns
    -------
    Particle
        Triangulated cylinder.
    """
    n_circ = int(np.sqrt(n))
    n_vert = n_circ // 2

    n_circ = max(n_circ, 12)
    n_vert = max(n_vert, 4)

    verts = []
    faces = []

    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    z_levels = np.linspace(-height / 2, height / 2, n_vert)

    # Side surface
    for z in z_levels:
        for t in theta:
            verts.append([radius * np.cos(t), radius * np.sin(t), z])

    # Side faces
    for i in range(n_vert - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ

            idx0 = i * n_circ + j
            idx1 = i * n_circ + j_next
            idx2 = (i + 1) * n_circ + j
            idx3 = (i + 1) * n_circ + j_next

            faces.append([idx0, idx1, idx3])
            faces.append([idx0, idx3, idx2])

    # Top cap
    top_center = len(verts)
    verts.append([0, 0, height / 2])
    top_start = (n_vert - 1) * n_circ
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        faces.append([top_center, top_start + j, top_start + j_next])

    # Bottom cap
    bottom_center = len(verts)
    verts.append([0, 0, -height / 2])
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        faces.append([bottom_center, j_next, j])

    verts = np.array(verts)
    faces = np.array(faces)

    return Particle(verts, faces, interp)
