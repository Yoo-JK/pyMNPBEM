"""
Mesh field computation and interpolation utilities.

These functions handle field calculations on particle surfaces
and interpolation between mesh representations.

This module provides:
- MeshField: Field defined on a triangular mesh with interpolation
- GridField: Field computation on a 3D grid using Green functions (MATLAB meshfield)
- Utility functions for field interpolation and computation
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any
from scipy import interpolate
from scipy.spatial import cKDTree


class GridField:
    """
    Compute electromagnetic fields on a grid of positions.

    This class is equivalent to MATLAB's meshfield class. It sets up a grid
    of points and computes electromagnetic fields using Green functions.

    Parameters
    ----------
    particle : ComParticle
        Particle object.
    x : ndarray
        X-coordinates for field evaluation.
    y : ndarray
        Y-coordinates for field evaluation.
    z : ndarray
        Z-coordinates for field evaluation.
    green_cls : class, optional
        Green function class to use (GreenStat, GreenRet, etc.).
    nmax : int, optional
        Maximum number of points to process at once (memory optimization).
    **kwargs
        Additional arguments for Green function initialization.

    Examples
    --------
    >>> from mnpbem import ComParticle, trisphere, GridField
    >>> p = ComParticle(trisphere(10, 20))
    >>> x = np.linspace(-20, 20, 50)
    >>> y = np.linspace(-20, 20, 50)
    >>> xx, yy = np.meshgrid(x, y)
    >>> mf = GridField(p, xx, yy, 0)  # z=0 plane
    >>> e, h = mf.field(sig, enei)
    """

    def __init__(
        self,
        particle: Any,
        x: np.ndarray,
        y: np.ndarray,
        z: Union[np.ndarray, float],
        green_cls: Any = None,
        nmax: Optional[int] = None,
        **kwargs
    ):
        """Initialize grid field object."""
        self.particle = particle
        self.nmax = nmax
        self.options = kwargs

        # Expand coordinates to matching shapes
        x, y, z = self._expand_coordinates(x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.original_shape = x.shape

        # Create position array
        self._pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Store Green function class
        self._green_cls = green_cls
        self._green = None

    def _expand_coordinates(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Union[np.ndarray, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand coordinates to matching 3D arrays."""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        # Check if shapes already match
        shapes = [x.shape, y.shape, z.shape]
        if len(set(str(s) for s in shapes)) == 1:
            return x, y, z

        # Expand scalar z
        if z.size == 1:
            z = np.full_like(x, z.item())
            return x, y, z

        # Handle 2D + 1D case: expand to 3D
        if x.shape == y.shape and x.ndim == 2:
            siz = x.shape
            nz = z.size
            x = np.broadcast_to(x[:, :, np.newaxis], (*siz, nz))
            y = np.broadcast_to(y[:, :, np.newaxis], (*siz, nz))
            z = np.broadcast_to(z.reshape(1, 1, nz), (*siz, nz))
            return x.copy(), y.copy(), z.copy()

        # Handle 1D arrays - create meshgrid
        if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
            x, y, z = np.meshgrid(x, y, z, indexing='ij')
            return x, y, z

        return x, y, z

    @property
    def pos(self) -> np.ndarray:
        """Flattened position array (n_points, 3)."""
        return self._pos

    @property
    def n_points(self) -> int:
        """Number of evaluation points."""
        return len(self._pos)

    @property
    def pt(self) -> 'GridFieldPoints':
        """
        Point object for grid positions (MATLAB meshfield.pt equivalent).

        This property provides access to grid positions in a format
        compatible with excitation field computation.

        Returns
        -------
        GridFieldPoints
            Point-like object with pos attribute for field evaluation.

        Examples
        --------
        >>> mf = GridField(p, xx, yy, zz)
        >>> # Compute incident field at grid points
        >>> E_inc, H_inc = exc.fields(mf.pt.pos, wavelength)
        """
        return GridFieldPoints(self._pos, self.original_shape)

    def _init_green(self, enei: float):
        """Initialize Green function if needed."""
        if self._green_cls is None:
            # Try to import and use GreenStat as default
            try:
                from ..greenfun import GreenStat
                self._green_cls = GreenStat
            except ImportError:
                raise ValueError("No Green function class specified")

        # Create a point-like object for the field positions
        from ..particles import ComPoint
        pt = ComPoint(self.particle, self._pos, **self.options)

        self._green = self._green_cls(pt, self.particle, **self.options)
        self._green_enei = enei

    def field(
        self,
        sig: Any,
        enei: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute electromagnetic fields.

        Parameters
        ----------
        sig : CompStruct
            Surface charges and currents from BEM solution.
        enei : float, optional
            Wavelength (for retarded calculations).
        **kwargs
            Additional parameters for Green function.

        Returns
        -------
        e : ndarray
            Electric field with shape matching input grid + (3,) for components.
        h : ndarray or None
            Magnetic field (None for quasistatic simulations).
        """
        # Check if sig already contains field
        if hasattr(sig, 'e') and sig.e is not None:
            return self._reshape_field(sig.e, sig.get('h', None))

        # Use chunked computation if nmax is set
        if self.nmax is not None and self.n_points > self.nmax:
            return self._field_chunked(sig, enei, **kwargs)

        return self._field_single(sig, enei, **kwargs)

    def _field_single(
        self,
        sig: Any,
        enei: Optional[float],
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute field using precomputed Green function."""
        # Initialize Green function if needed
        if self._green is None or (enei is not None and enei != self._green_enei):
            self._init_green(enei)

        # Compute field through Green function
        f = self._green.field(sig, **kwargs)

        e = f.get('e') if hasattr(f, 'get') else getattr(f, 'e', None)
        h = f.get('h') if hasattr(f, 'get') else getattr(f, 'h', None)

        return self._reshape_field(e, h)

    def _field_chunked(
        self,
        sig: Any,
        enei: Optional[float],
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute field in chunks to save memory."""
        n = self.n_points
        nmax = self.nmax

        # Initialize output arrays
        e_out = None
        h_out = None

        # Process in chunks
        for start in range(0, n, nmax):
            end = min(start + nmax, n)
            indices = slice(start, end)

            # Create sub-point object
            from ..particles import ComPoint
            pos_chunk = self._pos[indices]
            pt = ComPoint(self.particle, pos_chunk, **self.options)

            # Create Green function for chunk
            g = self._green_cls(pt, self.particle, **self.options)

            # Compute field
            f = g.field(sig, **kwargs)
            e_chunk = f.get('e') if hasattr(f, 'get') else getattr(f, 'e', None)
            h_chunk = f.get('h') if hasattr(f, 'get') else getattr(f, 'h', None)

            # Allocate output on first chunk
            if e_out is None and e_chunk is not None:
                shape = (n,) + e_chunk.shape[1:]
                e_out = np.zeros(shape, dtype=e_chunk.dtype)
            if h_out is None and h_chunk is not None:
                shape = (n,) + h_chunk.shape[1:]
                h_out = np.zeros(shape, dtype=h_chunk.dtype)

            # Store chunk results
            if e_chunk is not None:
                e_out[indices] = e_chunk
            if h_chunk is not None:
                h_out[indices] = h_chunk

        return self._reshape_field(e_out, h_out)

    def _reshape_field(
        self,
        e: Optional[np.ndarray],
        h: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Reshape field arrays to match original grid shape."""
        if e is not None:
            # Shape: original_shape + field_components
            field_shape = e.shape[1:] if e.ndim > 1 else ()
            new_shape = self.original_shape + field_shape
            e = e.reshape(new_shape)

        if h is not None:
            field_shape = h.shape[1:] if h.ndim > 1 else ()
            new_shape = self.original_shape + field_shape
            h = h.reshape(new_shape)

        return e, h

    def potential(self, sig: Any, enei: Optional[float] = None) -> np.ndarray:
        """
        Compute electrostatic potential on grid.

        Parameters
        ----------
        sig : CompStruct
            Surface charges.
        enei : float, optional
            Wavelength.

        Returns
        -------
        ndarray
            Potential values on grid.
        """
        # Get surface data
        if hasattr(self.particle, 'pos'):
            pos_surf = self.particle.pos
            area = self.particle.area
        else:
            pos_surf = self.particle.pc.pos
            area = self.particle.pc.area

        charges = sig.get('sig') if hasattr(sig, 'get') else sig.sig

        # Compute potential at each point
        phi = np.zeros(self.n_points, dtype=complex)

        for i, pt in enumerate(self._pos):
            r = np.linalg.norm(pt - pos_surf, axis=1)
            r[r < 1e-10] = 1e-10

            if charges.ndim == 1:
                phi[i] = np.sum(charges * area / (4 * np.pi * r))
            else:
                phi[i] = np.sum(charges[:, 0] * area / (4 * np.pi * r))

        return phi.reshape(self.original_shape)

    def total_field(
        self,
        sig: Any,
        exc: Any,
        enei: Optional[float] = None,
        eps_out: float = 1.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute total electromagnetic fields (scattered + incident).

        This method computes the total field by adding:
        1. Scattered field from BEM solution (at grid points)
        2. Incident field from excitation (at grid points)

        This is equivalent to MATLAB's:
            e = emesh(sig) + emesh(exc.field(emesh.pt, enei))

        Parameters
        ----------
        sig : CompStruct
            Surface charges and currents from BEM solution.
        exc : PlaneWave or similar
            Excitation object with fields() method.
        enei : float, optional
            Wavelength (nm). If None, uses sig.enei.
        eps_out : float, optional
            Dielectric function of surrounding medium.

        Returns
        -------
        e_total : ndarray
            Total electric field with shape matching input grid + (3,).
        h_total : ndarray or None
            Total magnetic field (None for quasistatic simulations).

        Examples
        --------
        >>> mf = GridField(p, xx, yy, 0)
        >>> e_total, h_total = mf.total_field(sig, exc, wavelength)
        """
        # Get wavelength from sig if not provided
        if enei is None:
            enei = getattr(sig, 'enei', None)
            if enei is None and hasattr(sig, 'get'):
                enei = sig.get('enei')

        if enei is None:
            raise ValueError("Wavelength (enei) must be provided or available in sig")

        # Compute scattered field at grid points
        e_scat, h_scat = self.field(sig, enei)

        # Compute incident field at grid points
        # The excitation object should have a fields() method
        if hasattr(exc, 'fields'):
            E_inc, H_inc = exc.fields(self._pos, enei, eps_out)
        elif hasattr(exc, 'planewave') and hasattr(exc.planewave, 'fields'):
            # Handle excitation objects that wrap a planewave
            E_inc, H_inc = exc.planewave.fields(self._pos, enei, eps_out)
        else:
            raise ValueError("Excitation object must have a fields() method")

        # Handle different polarization dimensions
        # E_inc shape is typically (n_pos, n_pol, 3)
        # e_scat shape is typically (*original_shape, 3) or (*original_shape, 3, n_pol)
        if E_inc.ndim == 3 and E_inc.shape[1] > 1:
            # Multiple polarizations - reshape incident field to match grid
            n_pol = E_inc.shape[1]
            E_inc_reshaped = E_inc.reshape(self.original_shape + (n_pol, 3))
            # Swap axes to match scattered field shape
            E_inc_reshaped = np.moveaxis(E_inc_reshaped, -2, -1)  # (..., 3, n_pol)
        elif E_inc.ndim == 3 and E_inc.shape[1] == 1:
            # Single polarization
            E_inc_reshaped = E_inc[:, 0, :].reshape(self.original_shape + (3,))
        else:
            E_inc_reshaped = E_inc.reshape(self.original_shape + (3,))

        # Add scattered and incident fields
        # Need to handle shape broadcasting
        if e_scat.shape == E_inc_reshaped.shape:
            e_total = e_scat + E_inc_reshaped
        elif e_scat.ndim > E_inc_reshaped.ndim:
            # e_scat has extra dimension for polarizations
            E_inc_expanded = E_inc_reshaped[..., np.newaxis]
            e_total = e_scat + E_inc_expanded
        else:
            # Try to broadcast
            try:
                e_total = e_scat + E_inc_reshaped
            except ValueError:
                # Reshape E_inc to match e_scat
                E_inc_flat = E_inc.reshape(-1, E_inc.shape[-1]) if E_inc.ndim > 2 else E_inc
                e_scat_flat = e_scat.reshape(-1, 3) if e_scat.ndim > 1 else e_scat
                if E_inc_flat.shape[0] == e_scat_flat.shape[0]:
                    e_total = (e_scat_flat + E_inc_flat[:, :3]).reshape(e_scat.shape)
                else:
                    raise ValueError(
                        f"Cannot combine scattered field (shape {e_scat.shape}) "
                        f"with incident field (shape {E_inc.shape}). "
                        "Ensure both are computed at the same grid points."
                    )

        # Handle magnetic field
        h_total = None
        if h_scat is not None and H_inc is not None:
            if H_inc.ndim == 3 and H_inc.shape[1] > 1:
                n_pol = H_inc.shape[1]
                H_inc_reshaped = H_inc.reshape(self.original_shape + (n_pol, 3))
                H_inc_reshaped = np.moveaxis(H_inc_reshaped, -2, -1)
            elif H_inc.ndim == 3 and H_inc.shape[1] == 1:
                H_inc_reshaped = H_inc[:, 0, :].reshape(self.original_shape + (3,))
            else:
                H_inc_reshaped = H_inc.reshape(self.original_shape + (3,))

            try:
                h_total = h_scat + H_inc_reshaped
            except ValueError:
                h_total = h_scat  # Fall back to just scattered field

        return e_total, h_total

    def __repr__(self) -> str:
        return f"GridField(shape={self.original_shape}, n_points={self.n_points})"


class GridFieldPoints:
    """
    Point object for GridField grid positions.

    This class provides a MATLAB-compatible interface for accessing
    grid positions, similar to meshfield.pt in MATLAB MNPBEM.

    Parameters
    ----------
    pos : ndarray
        Flattened position array (n_points, 3).
    original_shape : tuple
        Original grid shape for reshaping results.

    Attributes
    ----------
    pos : ndarray
        Position array.
    n : int
        Number of points.

    Examples
    --------
    >>> mf = GridField(p, xx, yy, zz)
    >>> E_inc, H_inc = exc.fields(mf.pt.pos, wavelength)
    """

    def __init__(self, pos: np.ndarray, original_shape: tuple):
        """Initialize grid field points."""
        self._pos = pos
        self.original_shape = original_shape

    @property
    def pos(self) -> np.ndarray:
        """Position array (n_points, 3)."""
        return self._pos

    @property
    def n(self) -> int:
        """Number of points."""
        return len(self._pos)

    def __len__(self) -> int:
        """Number of points."""
        return len(self._pos)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """
        Reshape field array to original grid shape.

        This mimics MATLAB's meshfield.pt(field) behavior.

        Parameters
        ----------
        field : ndarray
            Field values at grid points, shape (n_points, ...).

        Returns
        -------
        ndarray
            Reshaped field with original grid shape.
        """
        if field.shape[0] != len(self._pos):
            raise ValueError(
                f"Field size ({field.shape[0]}) doesn't match "
                f"number of grid points ({len(self._pos)})"
            )

        # Reshape to original grid shape
        extra_dims = field.shape[1:] if field.ndim > 1 else ()
        new_shape = self.original_shape + extra_dims
        return field.reshape(new_shape)

    def __repr__(self) -> str:
        return f"GridFieldPoints(n={self.n}, shape={self.original_shape})"


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


def gridfield(
    particle,
    x: np.ndarray,
    y: np.ndarray,
    z: Union[np.ndarray, float],
    green_cls=None,
    nmax: Optional[int] = None,
    **kwargs
) -> GridField:
    """
    Create a grid field object for electromagnetic field computation.

    This is the Python equivalent of MATLAB's meshfield function.

    Parameters
    ----------
    particle : ComParticle
        Particle object.
    x : ndarray
        X-coordinates for field evaluation.
    y : ndarray
        Y-coordinates for field evaluation.
    z : ndarray or float
        Z-coordinates for field evaluation.
    green_cls : class, optional
        Green function class (GreenStat, GreenRet, etc.).
    nmax : int, optional
        Maximum points per chunk (memory optimization).
    **kwargs
        Additional Green function arguments.

    Returns
    -------
    GridField
        Grid field object.

    Examples
    --------
    >>> # Create evaluation grid
    >>> x = np.linspace(-20, 20, 50)
    >>> y = np.linspace(-20, 20, 50)
    >>> xx, yy = np.meshgrid(x, y)
    >>> mf = gridfield(particle, xx, yy, 0)
    >>> e, h = mf.field(sig)
    """
    return GridField(particle, x, y, z, green_cls=green_cls, nmax=nmax, **kwargs)
