"""
Triangulated plate (rectangular prism) generation.
"""

import numpy as np
from ..particle import Particle


def triplate(
    n: int,
    dimensions: tuple,
    edge_rounding: float = 0.0,
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated rectangular plate.

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    dimensions : tuple
        (length, width, height) in nm.
    edge_rounding : float
        Edge rounding radius in nm.
    interp : str
        Interpolation type: 'flat' or 'curv'.

    Returns
    -------
    Particle
        Triangulated plate.

    Examples
    --------
    >>> # Flat plate
    >>> p = triplate(144, (100, 50, 10))
    >>> # Rounded plate
    >>> p = triplate(144, (100, 50, 10), edge_rounding=2)
    """
    lx, ly, lz = dimensions

    if edge_rounding > 0:
        # Rounded edges - use supershape approach
        r = min(edge_rounding, lx / 4, ly / 4, lz / 4)

        # Sample points on rounded box surface
        verts = []

        n_per_edge = max(int(np.cbrt(n) / 2), 3)
        n_corner = max(3, n_per_edge // 2)

        # Sample on each face with rounded edges
        # This is a simplified version - full implementation would
        # properly handle all corners and edges

        # Top and bottom faces (z = +/- lz/2)
        x_inner = np.linspace(-lx / 2 + r, lx / 2 - r, n_per_edge)
        y_inner = np.linspace(-ly / 2 + r, ly / 2 - r, n_per_edge)

        for z_sign in [1, -1]:
            z = z_sign * lz / 2
            # Inner region
            for x in x_inner:
                for y in y_inner:
                    verts.append([x, y, z])

        # Front and back faces (y = +/- ly/2)
        z_inner = np.linspace(-lz / 2 + r, lz / 2 - r, n_per_edge)

        for y_sign in [1, -1]:
            y = y_sign * ly / 2
            for x in x_inner:
                for z in z_inner:
                    verts.append([x, y, z])

        # Left and right faces (x = +/- lx/2)
        for x_sign in [1, -1]:
            x = x_sign * lx / 2
            for y in y_inner:
                for z in z_inner:
                    verts.append([x, y, z])

        # Edge rounding - simplified as corner spheres
        corner_signs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]

        for sx, sy, sz in corner_signs:
            cx = sx * (lx / 2 - r)
            cy = sy * (ly / 2 - r)
            cz = sz * (lz / 2 - r)

            # Sample sphere octant
            for i in range(n_corner):
                for j in range(n_corner):
                    theta = sx * np.pi / 2 * i / (n_corner - 1) + (1 - sx) * np.pi / 2
                    phi = sy * np.pi / 2 * j / (n_corner - 1) + (1 - sy) * np.pi / 2

                    x = cx + r * np.cos(theta) * np.cos(phi) * sx
                    y = cy + r * np.sin(theta) * np.cos(phi) * sy
                    z = cz + r * np.sin(phi) * sz

                    verts.append([x, y, z])

        verts = np.array(verts)

        # Triangulate using convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(verts)
        faces = hull.simplices

    else:
        # Sharp edges - simple box
        n_per_dim = max(int(np.cbrt(n)), 2)

        verts = []

        # Sample points on each face
        x_pts = np.linspace(-lx / 2, lx / 2, n_per_dim)
        y_pts = np.linspace(-ly / 2, ly / 2, n_per_dim)
        z_pts = np.linspace(-lz / 2, lz / 2, n_per_dim)

        # Top face (z = lz/2)
        for x in x_pts:
            for y in y_pts:
                verts.append([x, y, lz / 2])

        # Bottom face (z = -lz/2)
        for x in x_pts:
            for y in y_pts:
                verts.append([x, y, -lz / 2])

        # Front face (y = ly/2)
        for x in x_pts:
            for z in z_pts[1:-1]:
                verts.append([x, ly / 2, z])

        # Back face (y = -ly/2)
        for x in x_pts:
            for z in z_pts[1:-1]:
                verts.append([x, -ly / 2, z])

        # Left face (x = -lx/2)
        for y in y_pts[1:-1]:
            for z in z_pts[1:-1]:
                verts.append([-lx / 2, y, z])

        # Right face (x = lx/2)
        for y in y_pts[1:-1]:
            for z in z_pts[1:-1]:
                verts.append([lx / 2, y, z])

        verts = np.array(verts)

        # Remove duplicates
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        groups = tree.query_ball_tree(tree, 1e-10)
        unique_idx = [min(g) for g in groups]
        unique_idx = list(set(unique_idx))
        verts = verts[unique_idx]

        # Triangulate
        from scipy.spatial import ConvexHull
        hull = ConvexHull(verts)
        faces = hull.simplices

    return Particle(verts, faces, interp)


def triprism(
    n: int,
    n_sides: int,
    radius: float,
    height: float,
    interp: str = 'curv'
) -> Particle:
    """
    Create a triangulated prism (polygon extruded).

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    n_sides : int
        Number of sides (3=triangle, 4=square, 6=hexagon, etc.)
    radius : float
        Circumradius of polygon in nm.
    height : float
        Prism height in nm.
    interp : str
        Interpolation type.

    Returns
    -------
    Particle
        Triangulated prism.

    Examples
    --------
    >>> # Triangular prism
    >>> p = triprism(144, 3, 20, 50)
    >>> # Hexagonal prism
    >>> p = triprism(144, 6, 20, 30)
    """
    # Polygon vertices
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x_poly = radius * np.cos(theta)
    y_poly = radius * np.sin(theta)

    verts = []
    faces = []

    # Top face
    top_center = 0
    verts.append([0, 0, height / 2])
    for i in range(n_sides):
        verts.append([x_poly[i], y_poly[i], height / 2])

    # Bottom face
    bottom_center = len(verts)
    verts.append([0, 0, -height / 2])
    for i in range(n_sides):
        verts.append([x_poly[i], y_poly[i], -height / 2])

    # Top face triangles
    for i in range(n_sides):
        i_next = (i + 1) % n_sides
        faces.append([top_center, 1 + i, 1 + i_next])

    # Bottom face triangles
    for i in range(n_sides):
        i_next = (i + 1) % n_sides
        faces.append([bottom_center, bottom_center + 1 + i_next, bottom_center + 1 + i])

    # Side faces
    for i in range(n_sides):
        i_next = (i + 1) % n_sides

        top_i = 1 + i
        top_next = 1 + i_next
        bot_i = bottom_center + 1 + i
        bot_next = bottom_center + 1 + i_next

        faces.append([top_i, bot_i, bot_next])
        faces.append([top_i, bot_next, top_next])

    verts = np.array(verts)
    faces = np.array(faces)

    return Particle(verts, faces, interp)
