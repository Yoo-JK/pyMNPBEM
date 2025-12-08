"""
Geometric utility functions for MNPBEM.

Includes distance calculations, point queries, and geometric operations.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from scipy.spatial import cKDTree


def distmin3(points1: np.ndarray, points2: np.ndarray) -> Tuple[float, int, int]:
    """
    Find minimum distance between two point sets.

    Parameters
    ----------
    points1 : ndarray
        First point set (n1, 3)
    points2 : ndarray
        Second point set (n2, 3)

    Returns
    -------
    dist : float
        Minimum distance
    idx1 : int
        Index in points1 of closest point
    idx2 : int
        Index in points2 of closest point
    """
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)

    tree = cKDTree(points2)
    distances, indices = tree.query(points1)

    min_idx = np.argmin(distances)
    return distances[min_idx], min_idx, indices[min_idx]


def distmin_particle(particle1, particle2) -> Tuple[float, int, int]:
    """
    Find minimum distance between two particles.

    Parameters
    ----------
    particle1 : Particle
        First particle
    particle2 : Particle
        Second particle

    Returns
    -------
    dist : float
        Minimum distance between surfaces
    face1 : int
        Face index on particle1
    face2 : int
        Face index on particle2
    """
    return distmin3(particle1.pos, particle2.pos)


def point_in_particle(points: np.ndarray, particle) -> np.ndarray:
    """
    Test if points are inside a closed particle.

    Uses ray casting algorithm.

    Parameters
    ----------
    points : ndarray
        Test points (n, 3)
    particle : Particle
        Closed particle surface

    Returns
    -------
    inside : ndarray
        Boolean array indicating if each point is inside
    """
    points = np.atleast_2d(points)
    n_pts = len(points)

    inside = np.zeros(n_pts, dtype=bool)

    # Use ray casting along z direction
    for i, pt in enumerate(points):
        # Count intersections with faces
        n_intersect = 0

        for j in range(particle.n_faces):
            face = particle.faces[j]
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)

            if len(indices) < 3:
                continue

            vertices = particle.verts[indices]

            # Check if ray from pt in +z direction intersects triangle
            if _ray_triangle_intersect(pt, vertices):
                n_intersect += 1

        # Inside if odd number of intersections
        inside[i] = (n_intersect % 2) == 1

    return inside


def _ray_triangle_intersect(point: np.ndarray, vertices: np.ndarray) -> bool:
    """
    Test if ray from point in +z direction intersects triangle.

    Parameters
    ----------
    point : ndarray
        Ray origin (3,)
    vertices : ndarray
        Triangle vertices (3, 3) or more for polygon

    Returns
    -------
    bool
        True if ray intersects
    """
    # For polygon, split into triangles
    if len(vertices) > 3:
        for i in range(1, len(vertices) - 1):
            tri = np.array([vertices[0], vertices[i], vertices[i+1]])
            if _ray_triangle_intersect(point, tri):
                return True
        return False

    # Möller–Trumbore algorithm
    v0, v1, v2 = vertices
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Ray direction is [0, 0, 1]
    ray_dir = np.array([0, 0, 1])

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if abs(a) < 1e-10:
        return False  # Ray parallel to triangle

    f = 1.0 / a
    s = point - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)

    # Intersection at positive t (ray going in +z direction)
    return t > 1e-10


def nearest_face(point: np.ndarray, particle) -> Tuple[int, float]:
    """
    Find nearest face to a point.

    Parameters
    ----------
    point : ndarray
        Query point (3,)
    particle : Particle
        Particle

    Returns
    -------
    face_idx : int
        Index of nearest face
    dist : float
        Distance to nearest face centroid
    """
    point = np.asarray(point)
    distances = np.linalg.norm(particle.pos - point, axis=1)
    idx = np.argmin(distances)
    return idx, distances[idx]


def project_to_surface(point: np.ndarray, particle) -> Tuple[np.ndarray, int]:
    """
    Project point onto particle surface.

    Parameters
    ----------
    point : ndarray
        Point to project (3,)
    particle : Particle
        Particle surface

    Returns
    -------
    proj_point : ndarray
        Projected point (3,)
    face_idx : int
        Face index of projection
    """
    point = np.asarray(point)

    # Find nearest face
    face_idx, _ = nearest_face(point, particle)

    # Project onto face plane
    face = particle.faces[face_idx]
    valid = ~np.isnan(face)
    indices = face[valid].astype(int)
    vertices = particle.verts[indices]

    # Face normal and centroid
    nvec = particle.nvec[face_idx]
    centroid = particle.pos[face_idx]

    # Project point onto plane
    d = np.dot(point - centroid, nvec)
    proj_point = point - d * nvec

    return proj_point, face_idx


def surface_distance(particle1, particle2) -> float:
    """
    Compute surface-to-surface distance between particles.

    Parameters
    ----------
    particle1 : Particle
        First particle
    particle2 : Particle
        Second particle

    Returns
    -------
    float
        Minimum surface distance
    """
    dist, _, _ = distmin_particle(particle1, particle2)
    return dist


def gap_distance(particle, substrate_z: float = 0) -> float:
    """
    Compute gap distance between particle and substrate.

    Parameters
    ----------
    particle : Particle
        Particle
    substrate_z : float
        Z-coordinate of substrate plane

    Returns
    -------
    float
        Gap distance
    """
    min_z = np.min(particle.verts[:, 2])
    return min_z - substrate_z


def compute_solid_angle(point: np.ndarray, particle) -> float:
    """
    Compute solid angle subtended by particle at a point.

    Parameters
    ----------
    point : ndarray
        Observation point (3,)
    particle : Particle
        Particle surface

    Returns
    -------
    float
        Solid angle in steradians
    """
    point = np.asarray(point)
    solid_angle = 0.0

    for i in range(particle.n_faces):
        face = particle.faces[i]
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)

        if len(indices) < 3:
            continue

        vertices = particle.verts[indices]

        # For triangle, use spherical excess formula
        # Simplified: use projected area / r^2
        r_vec = particle.pos[i] - point
        r = np.linalg.norm(r_vec)

        if r > 1e-10:
            # cos(theta) = (n . r) / |r|
            cos_theta = np.dot(particle.nvec[i], r_vec / r)
            solid_angle += particle.area[i] * np.abs(cos_theta) / r**2

    return solid_angle


def mesh_quality(particle) -> dict:
    """
    Compute mesh quality metrics for a particle.

    Parameters
    ----------
    particle : Particle
        Particle to analyze

    Returns
    -------
    dict
        Quality metrics including min/max/mean aspect ratio,
        skewness, and area statistics
    """
    aspect_ratios = []
    skewness = []

    for i in range(particle.n_faces):
        face = particle.faces[i]
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)

        if len(indices) < 3:
            continue

        vertices = particle.verts[indices]

        # Compute edge lengths
        edges = []
        for j in range(len(vertices)):
            edge = vertices[(j + 1) % len(vertices)] - vertices[j]
            edges.append(np.linalg.norm(edge))

        edges = np.array(edges)

        # Aspect ratio = max_edge / min_edge
        ar = edges.max() / max(edges.min(), 1e-10)
        aspect_ratios.append(ar)

        # Skewness for triangles
        if len(indices) == 3:
            # Ideal equilateral triangle
            area = particle.area[i]
            perimeter = np.sum(edges)
            # Compare to equilateral with same perimeter
            ideal_area = (np.sqrt(3) / 36) * perimeter**2
            skew = 1 - area / max(ideal_area, 1e-10)
            skewness.append(max(0, min(1, skew)))

    aspect_ratios = np.array(aspect_ratios)
    skewness = np.array(skewness) if skewness else np.array([0])

    return {
        'n_faces': particle.n_faces,
        'n_verts': particle.n_verts,
        'total_area': np.sum(particle.area),
        'min_area': np.min(particle.area),
        'max_area': np.max(particle.area),
        'mean_area': np.mean(particle.area),
        'min_aspect_ratio': np.min(aspect_ratios),
        'max_aspect_ratio': np.max(aspect_ratios),
        'mean_aspect_ratio': np.mean(aspect_ratios),
        'mean_skewness': np.mean(skewness),
        'max_skewness': np.max(skewness)
    }
