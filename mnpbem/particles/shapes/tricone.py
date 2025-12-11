"""
Triangulated cone generation.
"""

import numpy as np
from ..particle import Particle


def tricone(
    n: int,
    height: float,
    radius: float,
    tip_radius: float = 0.0,
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated cone.

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    height : float
        Cone height in nm.
    radius : float
        Base radius in nm.
    tip_radius : float
        Tip radius for truncated cone (0 for sharp tip).
    interp : str
        Interpolation type: 'flat' or 'curv'.

    Returns
    -------
    Particle
        Triangulated cone.

    Examples
    --------
    >>> # Sharp cone
    >>> p = tricone(144, 50, 20)
    >>> # Truncated cone (frustum)
    >>> p = tricone(144, 50, 20, tip_radius=5)
    """
    # Estimate number of circumferential and vertical points
    n_circ = int(np.sqrt(n * 2))
    n_vert = n_circ // 2

    n_circ = max(n_circ, 8)
    n_vert = max(n_vert, 4)

    verts = []
    faces = []

    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    if tip_radius > 0:
        # Truncated cone
        z_levels = np.linspace(0, height, n_vert)
        radii = np.linspace(radius, tip_radius, n_vert)

        # Side surface
        for i, (z, r) in enumerate(zip(z_levels, radii)):
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), z])

        # Generate faces for side surface
        for i in range(n_vert - 1):
            for j in range(n_circ):
                j_next = (j + 1) % n_circ

                idx0 = i * n_circ + j
                idx1 = i * n_circ + j_next
                idx2 = (i + 1) * n_circ + j
                idx3 = (i + 1) * n_circ + j_next

                faces.append([idx0, idx1, idx3])
                faces.append([idx0, idx3, idx2])

        # Base cap (z = 0)
        base_center = len(verts)
        verts.append([0, 0, 0])
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            faces.append([base_center, j_next, j])

        # Top cap (z = height)
        top_center = len(verts)
        verts.append([0, 0, height])
        top_start = (n_vert - 1) * n_circ
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            faces.append([top_center, top_start + j, top_start + j_next])

    else:
        # Sharp cone
        z_levels = np.linspace(0, height * 0.95, n_vert - 1)
        radii = radius * (1 - z_levels / height)

        # Side surface (excluding tip)
        for z, r in zip(z_levels, radii):
            for t in theta:
                verts.append([r * np.cos(t), r * np.sin(t), z])

        # Tip vertex
        tip_idx = len(verts)
        verts.append([0, 0, height])

        # Faces for side surface
        for i in range(n_vert - 2):
            for j in range(n_circ):
                j_next = (j + 1) % n_circ

                idx0 = i * n_circ + j
                idx1 = i * n_circ + j_next
                idx2 = (i + 1) * n_circ + j
                idx3 = (i + 1) * n_circ + j_next

                faces.append([idx0, idx1, idx3])
                faces.append([idx0, idx3, idx2])

        # Tip triangles
        top_ring = (n_vert - 2) * n_circ
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            faces.append([top_ring + j, top_ring + j_next, tip_idx])

        # Base cap
        base_center = len(verts)
        verts.append([0, 0, 0])
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            faces.append([base_center, j_next, j])

    verts = np.array(verts)
    faces = np.array(faces)

    return Particle(verts, faces, interp)


def tribiconical(
    n: int,
    height: float,
    radius: float,
    interp: str = 'curv'
) -> Particle:
    """
    Create a bicone (two cones joined at base).

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    height : float
        Total height in nm.
    radius : float
        Maximum radius (at center).
    interp : str
        Interpolation type.

    Returns
    -------
    Particle
        Triangulated bicone.
    """
    n_circ = int(np.sqrt(n))
    n_vert = n_circ

    n_circ = max(n_circ, 8)
    n_vert = max(n_vert, 6)

    verts = []
    faces = []

    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)

    # Z levels from bottom tip to top tip
    z_levels = np.linspace(-height / 2, height / 2, n_vert)
    # Radius varies: 0 at tips, max at center
    radii = radius * (1 - 2 * np.abs(z_levels) / height)

    # Bottom tip
    bottom_tip = len(verts)
    verts.append([0, 0, -height / 2])

    # Middle vertices
    for z, r in zip(z_levels[1:-1], radii[1:-1]):
        for t in theta:
            verts.append([r * np.cos(t), r * np.sin(t), z])

    # Top tip
    top_tip = len(verts)
    verts.append([0, 0, height / 2])

    # Faces from bottom tip
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        faces.append([bottom_tip, 1 + j_next, 1 + j])

    # Middle faces
    for i in range(n_vert - 3):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ

            idx0 = 1 + i * n_circ + j
            idx1 = 1 + i * n_circ + j_next
            idx2 = 1 + (i + 1) * n_circ + j
            idx3 = 1 + (i + 1) * n_circ + j_next

            faces.append([idx0, idx1, idx3])
            faces.append([idx0, idx3, idx2])

    # Faces to top tip
    top_ring = 1 + (n_vert - 3) * n_circ
    for j in range(n_circ):
        j_next = (j + 1) % n_circ
        faces.append([top_ring + j, top_ring + j_next, top_tip])

    verts = np.array(verts)
    faces = np.array(faces)

    return Particle(verts, faces, interp)
