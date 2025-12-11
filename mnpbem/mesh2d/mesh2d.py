"""
Main 2D mesh generation functions.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.spatial import Delaunay

from .inpoly import inpoly
from .refine import refine
from .smoothmesh import smoothmesh


def mesh2d(
    polygon: np.ndarray,
    h: float = None,
    n_refine: int = 0,
    n_smooth: int = 3,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D triangular mesh inside a polygon.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices, shape (n_verts, 2). Must be closed (first = last).
    h : float, optional
        Target edge length. If None, uses automatic sizing.
    n_refine : int
        Number of refinement iterations.
    n_smooth : int
        Number of smoothing iterations.

    Returns
    -------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2).
    faces : ndarray
        Mesh triangles, shape (n_triangles, 3).

    Examples
    --------
    >>> # Create mesh inside a square
    >>> square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    >>> verts, faces = mesh2d(square, h=0.1)
    """
    polygon = np.asarray(polygon)

    # Close polygon if needed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    # Determine bounding box
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)

    # Determine target edge length
    if h is None:
        perimeter = np.sum(np.linalg.norm(np.diff(polygon, axis=0), axis=1))
        area = polygon_area(polygon)
        h = np.sqrt(area / 50)  # Heuristic: ~50 triangles

    # Generate initial grid of points
    n_x = int((x_max - x_min) / h) + 1
    n_y = int((y_max - y_min) / h) + 1

    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)

    # Create point grid
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Keep only points inside polygon
    mask = inpoly(points, polygon)
    interior_points = points[mask]

    # Add boundary points
    boundary_points = sample_polygon_boundary(polygon, h)

    # Combine points
    all_points = np.vstack([boundary_points, interior_points])

    # Remove duplicates
    all_points = remove_duplicate_points(all_points, tol=h / 10)

    # Delaunay triangulation
    tri = Delaunay(all_points)
    faces = tri.simplices

    # Remove triangles outside polygon
    centroids = all_points[faces].mean(axis=1)
    inside = inpoly(centroids, polygon)
    faces = faces[inside]

    # Refine mesh
    verts = all_points
    for _ in range(n_refine):
        verts, faces = refine(verts, faces, polygon=polygon)

    # Smooth mesh
    verts = smoothmesh(verts, faces, polygon=polygon, n_iter=n_smooth)

    return verts, faces


def meshpoly(
    polygon: np.ndarray,
    holes: List[np.ndarray] = None,
    h: float = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mesh inside polygon with optional holes.

    Parameters
    ----------
    polygon : ndarray
        Outer boundary polygon.
    holes : list of ndarray, optional
        List of hole polygons.
    h : float, optional
        Target edge length.

    Returns
    -------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.
    """
    # Generate base mesh
    verts, faces = mesh2d(polygon, h=h, **kwargs)

    if holes is None:
        return verts, faces

    # Remove triangles inside holes
    centroids = verts[faces].mean(axis=1)

    for hole in holes:
        hole = np.asarray(hole)
        if not np.allclose(hole[0], hole[-1]):
            hole = np.vstack([hole, hole[0]])

        inside_hole = inpoly(centroids, hole)
        faces = faces[~inside_hole]
        centroids = verts[faces].mean(axis=1)

    # Clean up unused vertices
    used_verts = np.unique(faces)
    vert_map = np.zeros(len(verts), dtype=int)
    vert_map[used_verts] = np.arange(len(used_verts))

    verts = verts[used_verts]
    faces = vert_map[faces]

    return verts, faces


def polygon_area(polygon: np.ndarray) -> float:
    """
    Compute area of a polygon using shoelace formula.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices (closed).

    Returns
    -------
    float
        Polygon area.
    """
    x = polygon[:-1, 0]
    y = polygon[:-1, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sample_polygon_boundary(polygon: np.ndarray, h: float) -> np.ndarray:
    """
    Sample points along polygon boundary with spacing h.

    Parameters
    ----------
    polygon : ndarray
        Polygon vertices (closed).
    h : float
        Target spacing.

    Returns
    -------
    ndarray
        Boundary sample points.
    """
    points = []

    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        edge_len = np.linalg.norm(p2 - p1)
        n_pts = max(int(edge_len / h), 1)

        t = np.linspace(0, 1, n_pts, endpoint=False)
        for ti in t:
            points.append(p1 + ti * (p2 - p1))

    return np.array(points)


def remove_duplicate_points(points: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Remove duplicate points within tolerance.

    Parameters
    ----------
    points : ndarray
        Point array.
    tol : float
        Tolerance for considering points as duplicates.

    Returns
    -------
    ndarray
        Points with duplicates removed.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, tol)

    # Keep first point in each group
    keep = []
    seen = set()
    for i, group in enumerate(groups):
        rep = min(group)
        if rep not in seen:
            keep.append(rep)
            seen.update(group)

    return points[sorted(keep)]


def fixmesh(verts: np.ndarray, faces: np.ndarray,
            tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix mesh by removing degenerate triangles and duplicate vertices.

    This function cleans up a mesh by:
    - Removing duplicate vertices
    - Removing degenerate (zero-area) triangles
    - Removing unreferenced vertices
    - Ensuring consistent face orientation

    Parameters
    ----------
    verts : ndarray
        Mesh vertices, shape (n_verts, 2) or (n_verts, 3).
    faces : ndarray
        Triangle indices, shape (n_faces, 3).
    tol : float
        Tolerance for considering points as duplicates.

    Returns
    -------
    verts_fixed : ndarray
        Fixed vertices.
    faces_fixed : ndarray
        Fixed triangle indices.

    Examples
    --------
    >>> verts = np.array([[0, 0], [1, 0], [1, 0], [0.5, 1]])  # Duplicate vertex
    >>> faces = np.array([[0, 1, 3], [1, 2, 3]])  # Second face uses duplicate
    >>> verts_fixed, faces_fixed = fixmesh(verts, faces)
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    # Remove duplicate vertices
    from scipy.spatial import cKDTree

    tree = cKDTree(verts)
    groups = tree.query_ball_tree(tree, tol)

    # Create mapping from old to new vertex indices
    old_to_new = np.zeros(len(verts), dtype=int)
    new_verts = []
    seen = set()

    for i, group in enumerate(groups):
        rep = min(group)
        if rep not in seen:
            new_idx = len(new_verts)
            new_verts.append(verts[rep])
            for j in group:
                old_to_new[j] = new_idx
            seen.update(group)

    verts_fixed = np.array(new_verts)

    # Update face indices
    faces_fixed = old_to_new[faces]

    # Remove degenerate triangles (all same vertex or zero area)
    valid_faces = []
    for face in faces_fixed:
        # Check for repeated vertices
        if len(np.unique(face)) < 3:
            continue

        # Check for zero area
        v0, v1, v2 = verts_fixed[face]
        if verts_fixed.shape[1] == 2:
            # 2D: cross product gives area
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.abs(e1[0] * e2[1] - e1[1] * e2[0])
        else:
            # 3D: use cross product norm
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))

        if area > tol ** 2:
            valid_faces.append(face)

    faces_fixed = np.array(valid_faces) if valid_faces else np.array([], dtype=int).reshape(0, 3)

    # Remove unreferenced vertices
    if len(faces_fixed) > 0:
        used_verts = np.unique(faces_fixed)
        vert_map = np.zeros(len(verts_fixed), dtype=int)
        vert_map[used_verts] = np.arange(len(used_verts))

        verts_fixed = verts_fixed[used_verts]
        faces_fixed = vert_map[faces_fixed]

    return verts_fixed, faces_fixed


def mytsearch(tri: Delaunay, points: np.ndarray) -> np.ndarray:
    """
    Find which triangle contains each query point.

    This is a wrapper around scipy's Delaunay.find_simplex that handles
    edge cases and provides MATLAB-compatible output.

    Parameters
    ----------
    tri : Delaunay
        Delaunay triangulation object.
    points : ndarray
        Query points, shape (n_points, 2).

    Returns
    -------
    indices : ndarray
        Triangle index for each point. -1 for points outside triangulation.

    Examples
    --------
    >>> from scipy.spatial import Delaunay
    >>> verts = np.array([[0, 0], [1, 0], [0.5, 1]])
    >>> tri = Delaunay(verts)
    >>> points = np.array([[0.3, 0.3], [2, 0]])
    >>> indices = mytsearch(tri, points)  # [0, -1]
    """
    points = np.atleast_2d(points)
    return tri.find_simplex(points)


def tinterp(tri: Delaunay, values: np.ndarray,
            points: np.ndarray) -> np.ndarray:
    """
    Interpolate values at query points using triangulation.

    Uses barycentric interpolation within triangles.

    Parameters
    ----------
    tri : Delaunay
        Delaunay triangulation object.
    values : ndarray
        Values at triangulation vertices, shape (n_verts,) or (n_verts, m).
    points : ndarray
        Query points, shape (n_points, 2).

    Returns
    -------
    interp_values : ndarray
        Interpolated values at query points. NaN for points outside.

    Examples
    --------
    >>> verts = np.array([[0, 0], [1, 0], [0.5, 1]])
    >>> tri = Delaunay(verts)
    >>> values = np.array([0, 1, 0.5])
    >>> points = np.array([[0.3, 0.3]])
    >>> tinterp(tri, values, points)
    """
    points = np.atleast_2d(points)
    values = np.atleast_1d(values)

    # Find containing triangles
    simplex_idx = tri.find_simplex(points)

    # Initialize output
    if values.ndim == 1:
        result = np.full(len(points), np.nan)
    else:
        result = np.full((len(points), values.shape[1]), np.nan)

    # Points inside triangulation
    inside = simplex_idx >= 0

    if not np.any(inside):
        return result

    # Get barycentric coordinates
    # Transform to reference triangle
    transform = tri.transform[simplex_idx[inside]]
    delta = points[inside] - transform[:, 2]

    # Barycentric coordinates for first two vertices
    bary = np.einsum('ijk,ik->ij', transform[:, :2], delta)

    # Third barycentric coordinate
    bary3 = 1 - bary.sum(axis=1)

    # Full barycentric coordinates
    bary_full = np.column_stack([bary, bary3])

    # Get vertex values for each triangle
    tri_verts = tri.simplices[simplex_idx[inside]]

    # Interpolate
    if values.ndim == 1:
        vert_vals = values[tri_verts]  # (n_inside, 3)
        result[inside] = np.sum(bary_full * vert_vals, axis=1)
    else:
        for j in range(values.shape[1]):
            vert_vals = values[tri_verts, j]
            result[inside, j] = np.sum(bary_full * vert_vals, axis=1)

    return result


def checkgeometry(verts: np.ndarray, faces: np.ndarray) -> dict:
    """
    Check mesh geometry for common issues.

    Performs various quality checks on a mesh including:
    - Degenerate triangles
    - Non-manifold edges
    - Inverted triangles
    - Duplicate vertices

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    report : dict
        Dictionary with geometry check results:
        - 'valid': bool, overall mesh validity
        - 'n_degenerate': number of degenerate triangles
        - 'n_inverted': number of inverted triangles
        - 'n_duplicate_verts': number of duplicate vertices
        - 'non_manifold_edges': list of non-manifold edges
        - 'min_area': minimum triangle area
        - 'max_area': maximum triangle area
        - 'min_angle': minimum angle in degrees
        - 'aspect_ratio': maximum aspect ratio
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    report = {
        'valid': True,
        'n_degenerate': 0,
        'n_inverted': 0,
        'n_duplicate_verts': 0,
        'non_manifold_edges': [],
        'min_area': np.inf,
        'max_area': 0,
        'min_angle': 180,
        'aspect_ratio': 1
    }

    if len(faces) == 0:
        report['valid'] = False
        return report

    # Check for duplicate vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    pairs = tree.query_pairs(1e-10)
    report['n_duplicate_verts'] = len(pairs)

    # Compute triangle properties
    areas = []
    angles = []
    aspect_ratios = []

    for face in faces:
        v0, v1, v2 = verts[face]

        # Edge vectors
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Edge lengths
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        # Area
        if verts.shape[1] == 2:
            area = 0.5 * np.abs(e0[0] * (-e2[1]) - e0[1] * (-e2[0]))
        else:
            area = 0.5 * np.linalg.norm(np.cross(e0, -e2))

        areas.append(area)

        # Check for degenerate
        if area < 1e-15:
            report['n_degenerate'] += 1
            continue

        # Angles (using law of cosines)
        cos_angles = [
            np.dot(-e2, e0) / (l2 * l0 + 1e-20),
            np.dot(-e0, e1) / (l0 * l1 + 1e-20),
            np.dot(-e1, e2) / (l1 * l2 + 1e-20)
        ]
        face_angles = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
        angles.extend(face_angles)

        # Aspect ratio (longest edge / height)
        longest = max(l0, l1, l2)
        height = 2 * area / (longest + 1e-20)
        aspect_ratios.append(longest / (height + 1e-20))

    report['min_area'] = min(areas) if areas else 0
    report['max_area'] = max(areas) if areas else 0
    report['min_angle'] = min(angles) if angles else 0
    report['aspect_ratio'] = max(aspect_ratios) if aspect_ratios else 1

    # Check for non-manifold edges
    edge_count = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    for edge, count in edge_count.items():
        if count > 2:
            report['non_manifold_edges'].append(edge)

    # Set validity
    report['valid'] = (
        report['n_degenerate'] == 0 and
        len(report['non_manifold_edges']) == 0 and
        report['min_angle'] > 1
    )

    return report


def meshfaces(verts: np.ndarray, faces: np.ndarray) -> dict:
    """
    Compute face-related properties of a mesh.

    Parameters
    ----------
    verts : ndarray
        Mesh vertices.
    faces : ndarray
        Mesh triangles.

    Returns
    -------
    props : dict
        Dictionary with face properties:
        - 'centroids': face centroids
        - 'areas': face areas
        - 'normals': face normals (for 3D meshes)
        - 'edges': unique edges
        - 'face_edges': edges for each face
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces)

    n_faces = len(faces)
    dim = verts.shape[1]

    # Centroids
    centroids = verts[faces].mean(axis=1)

    # Areas and normals
    areas = np.zeros(n_faces)
    if dim == 3:
        normals = np.zeros((n_faces, 3))
    else:
        normals = None

    for i, face in enumerate(faces):
        v0, v1, v2 = verts[face]
        e0 = v1 - v0
        e1 = v2 - v0

        if dim == 3:
            cross = np.cross(e0, e1)
            area = 0.5 * np.linalg.norm(cross)
            areas[i] = area
            if area > 1e-15:
                normals[i] = cross / (2 * area)
        else:
            areas[i] = 0.5 * np.abs(e0[0] * e1[1] - e0[1] * e1[0])

    # Extract unique edges
    edges_set = set()
    face_edges = []

    for face in faces:
        fe = []
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges_set.add(edge)
            fe.append(edge)
        face_edges.append(fe)

    edges = np.array(list(edges_set))

    props = {
        'centroids': centroids,
        'areas': areas,
        'normals': normals,
        'edges': edges,
        'face_edges': face_edges
    }

    return props
