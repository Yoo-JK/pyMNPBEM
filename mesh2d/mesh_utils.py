"""
Additional Mesh2D utilities.

Provides mesh operations: circumcircle, connectivity, fix, interpolation, etc.
"""

import numpy as np
from typing import Tuple, Optional, List


def circumcircle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute circumcircle of triangle.

    Parameters
    ----------
    p1, p2, p3 : ndarray
        Triangle vertices (2D).

    Returns
    -------
    center : ndarray
        Circumcenter coordinates.
    radius : float
        Circumradius.
    """
    ax, ay = p1[0], p1[1]
    bx, by = p2[0], p2[1]
    cx, cy = p3[0], p3[1]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if abs(d) < 1e-12:
        # Degenerate triangle
        center = (p1 + p2 + p3) / 3
        radius = 0
        return center, radius

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / d

    center = np.array([ux, uy])
    radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)

    return center, radius


def circumcircle_array(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute circumcircles for all triangles.

    Parameters
    ----------
    verts : ndarray
        Vertex coordinates (n, 2).
    faces : ndarray
        Face connectivity (m, 3).

    Returns
    -------
    centers : ndarray
        Circumcenters (m, 2).
    radii : ndarray
        Circumradii (m,).
    """
    n_faces = len(faces)
    centers = np.zeros((n_faces, 2))
    radii = np.zeros(n_faces)

    for i, face in enumerate(faces):
        p1 = verts[face[0]]
        p2 = verts[face[1]]
        p3 = verts[face[2]]
        centers[i], radii[i] = circumcircle(p1, p2, p3)

    return centers, radii


def connectivity(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mesh connectivity.

    Parameters
    ----------
    faces : ndarray
        Face connectivity (n_faces, 3).

    Returns
    -------
    edges : ndarray
        Edge list (n_edges, 2).
    face_edges : ndarray
        Edges for each face (n_faces, 3).
    edge_faces : list
        Faces sharing each edge.
    """
    n_faces = len(faces)

    # Build edge dictionary
    edge_dict = {}
    edges_list = []
    face_edges = np.zeros((n_faces, 3), dtype=int)

    for i, face in enumerate(faces):
        for j in range(3):
            v1, v2 = face[j], face[(j + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))

            if edge not in edge_dict:
                edge_dict[edge] = len(edges_list)
                edges_list.append(edge)

            face_edges[i, j] = edge_dict[edge]

    edges = np.array(edges_list)

    # Build edge-to-face mapping
    edge_faces = [[] for _ in range(len(edges))]
    for i, face in enumerate(faces):
        for j in range(3):
            edge_idx = face_edges[i, j]
            edge_faces[edge_idx].append(i)

    return edges, face_edges, edge_faces


def findedge(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Find boundary edges.

    Parameters
    ----------
    verts : ndarray
        Vertices.
    faces : ndarray
        Faces.

    Returns
    -------
    boundary_edges : ndarray
        Boundary edge vertex indices (n_boundary, 2).
    """
    edges, face_edges, edge_faces = connectivity(faces)

    # Boundary edges have only one adjacent face
    boundary = []
    for i, ef in enumerate(edge_faces):
        if len(ef) == 1:
            boundary.append(edges[i])

    return np.array(boundary)


def fixmesh(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix mesh by removing duplicate vertices and degenerate faces.

    Parameters
    ----------
    verts : ndarray
        Vertices.
    faces : ndarray
        Faces.
    tol : float
        Tolerance for duplicate detection.

    Returns
    -------
    verts_fixed : ndarray
        Fixed vertices.
    faces_fixed : ndarray
        Fixed faces.
    """
    n_verts = len(verts)

    # Find duplicate vertices
    vert_map = np.arange(n_verts)
    for i in range(n_verts):
        if vert_map[i] != i:
            continue
        for j in range(i + 1, n_verts):
            if vert_map[j] != j:
                continue
            if np.linalg.norm(verts[i] - verts[j]) < tol:
                vert_map[j] = i

    # Create new vertex list
    unique_verts = []
    new_map = np.full(n_verts, -1, dtype=int)

    for i in range(n_verts):
        if vert_map[i] == i:
            new_map[i] = len(unique_verts)
            unique_verts.append(verts[i])
        else:
            new_map[i] = new_map[vert_map[i]]

    verts_fixed = np.array(unique_verts)

    # Update faces
    faces_fixed = []
    for face in faces:
        new_face = [new_map[v] for v in face]

        # Check for degenerate face
        if len(set(new_face)) < 3:
            continue

        faces_fixed.append(new_face)

    return verts_fixed, np.array(faces_fixed)


def dist2poly(points: np.ndarray, poly_verts: np.ndarray) -> np.ndarray:
    """
    Compute distance from points to polygon boundary.

    Parameters
    ----------
    points : ndarray
        Query points (n, 2).
    poly_verts : ndarray
        Polygon vertices (m, 2).

    Returns
    -------
    distances : ndarray
        Distances (n,).
    """
    points = np.atleast_2d(points)
    n_points = len(points)
    n_poly = len(poly_verts)

    distances = np.full(n_points, np.inf)

    for i in range(n_poly):
        j = (i + 1) % n_poly
        p1 = poly_verts[i]
        p2 = poly_verts[j]

        # Vector from p1 to p2
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)

        if edge_len < 1e-10:
            continue

        edge_unit = edge / edge_len

        # Project points onto edge
        for k, pt in enumerate(points):
            v = pt - p1
            t = np.dot(v, edge_unit)
            t = np.clip(t, 0, edge_len)

            closest = p1 + t * edge_unit
            dist = np.linalg.norm(pt - closest)
            distances[k] = min(distances[k], dist)

    return distances


def mytsearch(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Find which triangle contains each point.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices (n, 2).
    faces : ndarray
        Mesh faces (m, 3).
    points : ndarray
        Query points (k, 2).

    Returns
    -------
    face_indices : ndarray
        Face index for each point (-1 if not found).
    """
    from scipy.spatial import Delaunay

    points = np.atleast_2d(points)

    # Use scipy's Delaunay for point location
    try:
        tri = Delaunay(verts)
        indices = tri.find_simplex(points)
        return indices
    except Exception:
        # Fall back to brute force
        return _tsearch_brute(verts, faces, points)


def _tsearch_brute(verts: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Brute force triangle search."""
    n_points = len(points)
    n_faces = len(faces)

    indices = np.full(n_points, -1, dtype=int)

    for i, pt in enumerate(points):
        for j, face in enumerate(faces):
            if _point_in_triangle(pt, verts[face[0]], verts[face[1]], verts[face[2]]):
                indices[i] = j
                break

    return indices


def _point_in_triangle(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
    """Check if point is inside triangle using barycentric coordinates."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def tinterp(
    verts: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Interpolate values on triangular mesh.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices (n, 2).
    faces : ndarray
        Mesh faces (m, 3).
    values : ndarray
        Values at vertices (n,).
    points : ndarray
        Query points (k, 2).

    Returns
    -------
    interp_values : ndarray
        Interpolated values (k,).
    """
    from scipy.interpolate import LinearNDInterpolator

    points = np.atleast_2d(points)

    # Use scipy's interpolator
    interp = LinearNDInterpolator(verts, values)
    return interp(points)


def checkgeometry(verts: np.ndarray, faces: np.ndarray) -> dict:
    """
    Check mesh geometry for issues.

    Parameters
    ----------
    verts : ndarray
        Vertices.
    faces : ndarray
        Faces.

    Returns
    -------
    issues : dict
        Dictionary of detected issues.
    """
    issues = {
        'degenerate_faces': [],
        'duplicate_vertices': [],
        'flipped_normals': [],
        'non_manifold_edges': [],
        'isolated_vertices': []
    }

    n_verts = len(verts)
    n_faces = len(faces)

    # Check for degenerate faces
    for i, face in enumerate(faces):
        v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]

        # Check area
        area = 0.5 * np.abs(
            (v2[0] - v1[0]) * (v3[1] - v1[1]) -
            (v3[0] - v1[0]) * (v2[1] - v1[1])
        )

        if area < 1e-12:
            issues['degenerate_faces'].append(i)

    # Check for duplicate vertices
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            if np.linalg.norm(verts[i] - verts[j]) < 1e-10:
                issues['duplicate_vertices'].append((i, j))

    # Check for non-manifold edges
    edges, face_edges, edge_faces = connectivity(faces)
    for i, ef in enumerate(edge_faces):
        if len(ef) > 2:
            issues['non_manifold_edges'].append(i)

    # Check for isolated vertices
    used_verts = set()
    for face in faces:
        used_verts.update(face)
    for i in range(n_verts):
        if i not in used_verts:
            issues['isolated_vertices'].append(i)

    return issues


def triarea(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute triangle areas.

    Parameters
    ----------
    verts : ndarray
        Vertices.
    faces : ndarray
        Faces.

    Returns
    -------
    areas : ndarray
        Triangle areas.
    """
    areas = np.zeros(len(faces))

    for i, face in enumerate(faces):
        v1 = verts[face[0]]
        v2 = verts[face[1]]
        v3 = verts[face[2]]

        # Cross product formula
        if len(v1) == 2:
            areas[i] = 0.5 * abs(
                (v2[0] - v1[0]) * (v3[1] - v1[1]) -
                (v3[0] - v1[0]) * (v2[1] - v1[1])
            )
        else:
            cross = np.cross(v2 - v1, v3 - v1)
            areas[i] = 0.5 * np.linalg.norm(cross)

    return areas


def mesh_collection() -> dict:
    """
    Get collection of predefined mesh templates.

    Returns
    -------
    dict
        Dictionary of mesh generation functions.
    """
    return {
        'circle': _mesh_circle,
        'rectangle': _mesh_rectangle,
        'annulus': _mesh_annulus,
        'ellipse': _mesh_ellipse
    }


def _mesh_circle(radius: float, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate circular mesh."""
    from .mesh2d import mesh2d

    # Boundary
    theta = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    boundary = np.column_stack([x, y])

    return mesh2d(boundary)


def _mesh_rectangle(width: float, height: float, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate rectangular mesh."""
    from .mesh2d import mesh2d

    x = np.array([-width / 2, width / 2, width / 2, -width / 2])
    y = np.array([-height / 2, -height / 2, height / 2, height / 2])

    boundary = np.column_stack([x, y])

    return mesh2d(boundary)


def _mesh_annulus(r_inner: float, r_outer: float, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate annular mesh."""
    from .mesh2d import mesh2d

    theta = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    # Outer boundary
    x_out = r_outer * np.cos(theta)
    y_out = r_outer * np.sin(theta)

    # Inner boundary (reversed)
    x_in = r_inner * np.cos(theta[::-1])
    y_in = r_inner * np.sin(theta[::-1])

    boundary = np.column_stack([np.concatenate([x_out, x_in]),
                                np.concatenate([y_out, y_in])])

    return mesh2d(boundary)


def _mesh_ellipse(a: float, b: float, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate elliptical mesh."""
    from .mesh2d import mesh2d

    theta = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    boundary = np.column_stack([x, y])

    return mesh2d(boundary)
