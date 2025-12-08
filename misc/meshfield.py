"""
Mesh field computation and interpolation utilities.

These functions handle field calculations on particle surfaces
and interpolation between mesh representations.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from scipy import interpolate
from scipy.spatial import cKDTree


class MeshField:
    """
    Field defined on a triangular mesh.

    Supports interpolation, integration, and field operations.

    Parameters
    ----------
    mesh : object
        Mesh object with verts, faces, pos, area attributes.
    values : ndarray, optional
        Field values at face centroids.

    Attributes
    ----------
    mesh : object
        Underlying mesh.
    values : ndarray
        Field values.
    """

    def __init__(self, mesh, values: Optional[np.ndarray] = None):
        """Initialize mesh field."""
        self.mesh = mesh
        self.values = values
        self._tree = None

    @property
    def pos(self) -> np.ndarray:
        """Face centroids."""
        if hasattr(self.mesh, 'pos'):
            return self.mesh.pos
        elif hasattr(self.mesh, 'pc'):
            return self.mesh.pc.pos
        else:
            raise AttributeError("Mesh has no position data")

    @property
    def area(self) -> np.ndarray:
        """Face areas."""
        if hasattr(self.mesh, 'area'):
            return self.mesh.area
        elif hasattr(self.mesh, 'pc'):
            return self.mesh.pc.area
        else:
            raise AttributeError("Mesh has no area data")

    @property
    def nvec(self) -> np.ndarray:
        """Face normal vectors."""
        if hasattr(self.mesh, 'nvec'):
            return self.mesh.nvec
        elif hasattr(self.mesh, 'pc'):
            return self.mesh.pc.nvec
        else:
            raise AttributeError("Mesh has no normal vector data")

    def _build_tree(self):
        """Build spatial search tree."""
        if self._tree is None:
            self._tree = cKDTree(self.pos)

    def integrate(self) -> Union[float, complex, np.ndarray]:
        """
        Integrate field over mesh surface.

        Returns
        -------
        float or complex
            Integrated value.
        """
        if self.values is None:
            raise ValueError("No field values to integrate")

        if self.values.ndim == 1:
            return np.sum(self.values * self.area)
        else:
            return np.sum(self.values * self.area[:, np.newaxis], axis=0)

    def mean(self) -> Union[float, complex, np.ndarray]:
        """
        Compute area-weighted mean of field.

        Returns
        -------
        float or complex
            Mean value.
        """
        return self.integrate() / np.sum(self.area)

    def interpolate(self, points: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        Interpolate field to arbitrary points.

        Parameters
        ----------
        points : ndarray
            Query points (n_points, 3).
        method : str
            Interpolation method: 'nearest', 'linear', 'rbf'.

        Returns
        -------
        ndarray
            Interpolated values.
        """
        points = np.atleast_2d(points)

        if method == 'nearest':
            return self._interpolate_nearest(points)
        elif method == 'linear':
            return self._interpolate_linear(points)
        elif method == 'rbf':
            return self._interpolate_rbf(points)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _interpolate_nearest(self, points: np.ndarray) -> np.ndarray:
        """Nearest-neighbor interpolation."""
        self._build_tree()
        _, indices = self._tree.query(points)
        return self.values[indices]

    def _interpolate_linear(self, points: np.ndarray) -> np.ndarray:
        """Linear interpolation using inverse distance weighting."""
        self._build_tree()
        k = min(4, len(self.pos))
        distances, indices = self._tree.query(points, k=k)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)

        # Inverse distance weights
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)

        if self.values.ndim == 1:
            return np.sum(weights * self.values[indices], axis=1)
        else:
            result = np.zeros((len(points), self.values.shape[1]),
                            dtype=self.values.dtype)
            for i in range(len(points)):
                result[i] = np.sum(weights[i, :, np.newaxis] *
                                  self.values[indices[i]], axis=0)
            return result

    def _interpolate_rbf(self, points: np.ndarray) -> np.ndarray:
        """Radial basis function interpolation."""
        if self.values.ndim == 1:
            rbf = interpolate.RBFInterpolator(self.pos, self.values)
            return rbf(points)
        else:
            result = np.zeros((len(points), self.values.shape[1]),
                            dtype=self.values.dtype)
            for j in range(self.values.shape[1]):
                rbf = interpolate.RBFInterpolator(self.pos, self.values[:, j])
                result[:, j] = rbf(points)
            return result

    def gradient(self) -> 'MeshField':
        """
        Compute gradient of scalar field.

        Returns
        -------
        MeshField
            Gradient field (vector values).
        """
        if self.values is None or self.values.ndim != 1:
            raise ValueError("Gradient requires scalar field")

        # Approximate gradient using neighboring faces
        n_faces = len(self.pos)
        grad = np.zeros((n_faces, 3), dtype=self.values.dtype)

        self._build_tree()

        for i in range(n_faces):
            # Find neighbors
            _, neighbors = self._tree.query(self.pos[i], k=min(7, n_faces))

            # Least squares gradient estimation
            dx = self.pos[neighbors] - self.pos[i]
            dv = self.values[neighbors] - self.values[i]

            # Solve least squares: dx @ grad = dv
            if len(neighbors) >= 3:
                grad[i], _, _, _ = np.linalg.lstsq(dx, dv, rcond=None)

        return MeshField(self.mesh, grad)

    def divergence(self) -> 'MeshField':
        """
        Compute divergence of vector field.

        Returns
        -------
        MeshField
            Divergence field (scalar values).
        """
        if self.values is None or self.values.ndim != 2 or self.values.shape[1] != 3:
            raise ValueError("Divergence requires vector field")

        # Approximate using surface divergence theorem
        n_faces = len(self.pos)
        div = np.zeros(n_faces, dtype=self.values.dtype)

        # div(F) ~ integral(F.n) / area
        for i in range(n_faces):
            div[i] = np.dot(self.values[i], self.nvec[i])

        return MeshField(self.mesh, div)

    def __add__(self, other):
        """Add fields."""
        if isinstance(other, MeshField):
            return MeshField(self.mesh, self.values + other.values)
        return MeshField(self.mesh, self.values + other)

    def __mul__(self, other):
        """Multiply field."""
        if isinstance(other, MeshField):
            return MeshField(self.mesh, self.values * other.values)
        return MeshField(self.mesh, self.values * other)

    def __repr__(self) -> str:
        shape = self.values.shape if self.values is not None else None
        return f"MeshField(n_faces={len(self.pos)}, shape={shape})"


def meshfield(mesh, values: Optional[np.ndarray] = None) -> MeshField:
    """
    Create a mesh field.

    Parameters
    ----------
    mesh : object
        Mesh or particle object.
    values : ndarray, optional
        Field values.

    Returns
    -------
    MeshField
        Mesh field object.
    """
    return MeshField(mesh, values)


def interpolate_field(
    source_pos: np.ndarray,
    source_values: np.ndarray,
    target_pos: np.ndarray,
    method: str = 'linear',
    **kwargs
) -> np.ndarray:
    """
    Interpolate field from source to target positions.

    Parameters
    ----------
    source_pos : ndarray
        Source positions (n_source, 3).
    source_values : ndarray
        Source values (n_source,) or (n_source, ...).
    target_pos : ndarray
        Target positions (n_target, 3).
    method : str
        Interpolation method.

    Returns
    -------
    ndarray
        Interpolated values at target positions.
    """
    source_pos = np.atleast_2d(source_pos)
    target_pos = np.atleast_2d(target_pos)

    tree = cKDTree(source_pos)

    if method == 'nearest':
        _, indices = tree.query(target_pos)
        return source_values[indices]

    elif method == 'linear':
        k = min(kwargs.get('k', 4), len(source_pos))
        distances, indices = tree.query(target_pos, k=k)

        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)

        if source_values.ndim == 1:
            return np.sum(weights * source_values[indices], axis=1)
        else:
            result = np.zeros((len(target_pos), source_values.shape[1]),
                            dtype=source_values.dtype)
            for i in range(len(target_pos)):
                result[i] = np.sum(weights[i, :, np.newaxis] *
                                  source_values[indices[i]], axis=0)
            return result

    elif method == 'rbf':
        if source_values.ndim == 1:
            rbf = interpolate.RBFInterpolator(source_pos, source_values, **kwargs)
            return rbf(target_pos)
        else:
            result = np.zeros((len(target_pos), source_values.shape[1]),
                            dtype=source_values.dtype)
            for j in range(source_values.shape[1]):
                rbf = interpolate.RBFInterpolator(source_pos, source_values[:, j], **kwargs)
                result[:, j] = rbf(target_pos)
            return result

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def field_at_points(
    particle,
    sig,
    points: np.ndarray,
    field_type: str = 'potential'
) -> np.ndarray:
    """
    Compute field at arbitrary points from BEM solution.

    Parameters
    ----------
    particle : ComParticle
        Particle object.
    sig : CompStruct
        BEM solution.
    points : ndarray
        Evaluation points (n_points, 3).
    field_type : str
        Type: 'potential', 'field', 'charge'.

    Returns
    -------
    ndarray
        Field values at points.
    """
    points = np.atleast_2d(points)
    n_points = len(points)

    # Get surface data
    pos_surf = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
    area = particle.area if hasattr(particle, 'area') else particle.pc.area
    charges = sig.get('sig')

    if field_type == 'potential':
        # Coulomb potential
        phi = np.zeros(n_points, dtype=complex)

        for i, pt in enumerate(points):
            r = np.linalg.norm(pt - pos_surf, axis=1)
            r[r < 1e-10] = 1e-10

            if charges.ndim == 1:
                phi[i] = np.sum(charges * area / (4 * np.pi * r))
            else:
                phi[i] = np.sum(charges[:, 0] * area / (4 * np.pi * r))

        return phi

    elif field_type == 'field':
        # Electric field
        E = np.zeros((n_points, 3), dtype=complex)

        for i, pt in enumerate(points):
            r_vec = pt - pos_surf
            r = np.linalg.norm(r_vec, axis=1)
            r[r < 1e-10] = 1e-10
            r_hat = r_vec / r[:, np.newaxis]

            if charges.ndim == 1:
                E[i] = np.sum(charges[:, np.newaxis] * area[:, np.newaxis] *
                             r_hat / (4 * np.pi * r[:, np.newaxis]**2), axis=0)
            else:
                E[i] = np.sum(charges[:, 0, np.newaxis] * area[:, np.newaxis] *
                             r_hat / (4 * np.pi * r[:, np.newaxis]**2), axis=0)

        return E

    elif field_type == 'charge':
        # Just return interpolated charge density
        return interpolate_field(pos_surf, charges, points, method='nearest')

    else:
        raise ValueError(f"Unknown field type: {field_type}")
