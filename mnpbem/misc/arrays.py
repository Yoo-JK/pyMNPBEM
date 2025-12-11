"""
Array utilities for MNPBEM.

Provides specialized array classes and grid generation functions.
"""

import numpy as np
from typing import Optional, Union, Tuple, List


class ValArray:
    """
    Wavelength/energy indexed array.

    Stores values associated with wavelengths for spectral data.

    Parameters
    ----------
    enei : ndarray
        Wavelengths or energies.
    values : ndarray
        Values for each wavelength.

    Attributes
    ----------
    enei : ndarray
        Wavelengths.
    values : ndarray
        Values array.

    Examples
    --------
    >>> wl = np.linspace(400, 800, 100)
    >>> spectrum = ValArray(wl, absorption)
    >>> print(spectrum(550))  # Interpolate at 550 nm
    """

    def __init__(self, enei: np.ndarray, values: np.ndarray):
        """Initialize wavelength-indexed array."""
        self.enei = np.asarray(enei)
        self.values = np.asarray(values)

        if len(self.enei) != len(self.values):
            raise ValueError("enei and values must have same length")

    def __len__(self) -> int:
        return len(self.enei)

    def __getitem__(self, idx):
        """Index by position."""
        return ValArray(self.enei[idx], self.values[idx])

    def __call__(self, wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate value at wavelength.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength(s) to interpolate.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """
        from scipy import interpolate
        interp = interpolate.interp1d(self.enei, self.values, kind='linear',
                                      fill_value='extrapolate')
        return interp(wavelength)

    @property
    def max(self) -> Tuple[float, float]:
        """(wavelength, value) at maximum."""
        idx = np.argmax(self.values)
        return self.enei[idx], self.values[idx]

    @property
    def min(self) -> Tuple[float, float]:
        """(wavelength, value) at minimum."""
        idx = np.argmin(self.values)
        return self.enei[idx], self.values[idx]

    def peak_wavelength(self) -> float:
        """Wavelength at peak value."""
        return self.enei[np.argmax(self.values)]

    def fwhm(self) -> float:
        """Full width at half maximum."""
        half_max = np.max(self.values) / 2
        above = self.values >= half_max
        indices = np.where(above)[0]
        if len(indices) < 2:
            return 0.0
        return self.enei[indices[-1]] - self.enei[indices[0]]

    def integrate(self) -> float:
        """Integrate spectrum."""
        return np.trapz(self.values, self.enei)

    def normalize(self) -> 'ValArray':
        """Return normalized array (max = 1)."""
        return ValArray(self.enei, self.values / np.max(np.abs(self.values)))

    def smooth(self, window: int = 5) -> 'ValArray':
        """Apply smoothing."""
        kernel = np.ones(window) / window
        smoothed = np.convolve(self.values, kernel, mode='same')
        return ValArray(self.enei, smoothed)

    def derivative(self) -> 'ValArray':
        """Compute derivative."""
        deriv = np.gradient(self.values, self.enei)
        return ValArray(self.enei, deriv)

    def __add__(self, other):
        if isinstance(other, ValArray):
            if not np.allclose(self.enei, other.enei):
                raise ValueError("Wavelengths must match")
            return ValArray(self.enei, self.values + other.values)
        return ValArray(self.enei, self.values + other)

    def __mul__(self, other):
        if isinstance(other, ValArray):
            if not np.allclose(self.enei, other.enei):
                raise ValueError("Wavelengths must match")
            return ValArray(self.enei, self.values * other.values)
        return ValArray(self.enei, self.values * other)

    def __repr__(self) -> str:
        return f"ValArray(n={len(self)}, range=[{self.enei[0]:.1f}, {self.enei[-1]:.1f}])"


class VecArray:
    """
    Vector array for 3D vector fields.

    Stores vector data with convenient access methods.

    Parameters
    ----------
    data : ndarray
        Vector data, shape (n, 3) or (n, m, 3).

    Attributes
    ----------
    data : ndarray
        Raw data array.
    """

    def __init__(self, data: np.ndarray):
        """Initialize vector array."""
        self.data = np.asarray(data)

        if self.data.shape[-1] != 3:
            raise ValueError("Last dimension must be 3 (vectors)")

    @property
    def x(self) -> np.ndarray:
        """X components."""
        return self.data[..., 0]

    @property
    def y(self) -> np.ndarray:
        """Y components."""
        return self.data[..., 1]

    @property
    def z(self) -> np.ndarray:
        """Z components."""
        return self.data[..., 2]

    @property
    def magnitude(self) -> np.ndarray:
        """Vector magnitudes."""
        return np.linalg.norm(self.data, axis=-1)

    @property
    def unit(self) -> 'VecArray':
        """Unit vectors."""
        mag = self.magnitude
        mag = np.where(mag < 1e-10, 1.0, mag)
        return VecArray(self.data / mag[..., np.newaxis])

    def dot(self, other: Union['VecArray', np.ndarray]) -> np.ndarray:
        """Dot product."""
        if isinstance(other, VecArray):
            other = other.data
        return np.sum(self.data * other, axis=-1)

    def cross(self, other: Union['VecArray', np.ndarray]) -> 'VecArray':
        """Cross product."""
        if isinstance(other, VecArray):
            other = other.data
        return VecArray(np.cross(self.data, other))

    def component(self, direction: np.ndarray) -> np.ndarray:
        """Component along direction."""
        direction = np.asarray(direction)
        direction = direction / np.linalg.norm(direction)
        return self.dot(direction)

    def rotate(self, angle: float, axis: np.ndarray) -> 'VecArray':
        """
        Rotate vectors around axis.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : ndarray
            Rotation axis (will be normalized).

        Returns
        -------
        VecArray
            Rotated vectors.
        """
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)

        c, s = np.cos(angle), np.sin(angle)
        ux, uy, uz = axis

        # Rotation matrix (Rodrigues formula)
        R = np.array([
            [c + ux**2*(1-c),    ux*uy*(1-c) - uz*s,  ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s, c + uy**2*(1-c),     uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s,  c + uz**2*(1-c)]
        ])

        return VecArray(self.data @ R.T)

    def __getitem__(self, idx):
        return VecArray(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, VecArray):
            return VecArray(self.data + other.data)
        return VecArray(self.data + other)

    def __mul__(self, other):
        if isinstance(other, VecArray):
            return VecArray(self.data * other.data)
        return VecArray(self.data * np.asarray(other)[..., np.newaxis])

    def __neg__(self):
        return VecArray(-self.data)

    def __repr__(self) -> str:
        return f"VecArray(shape={self.data.shape})"


def valarray(enei: np.ndarray, values: np.ndarray) -> ValArray:
    """Create a wavelength-indexed array."""
    return ValArray(enei, values)


def vecarray(data: np.ndarray) -> VecArray:
    """Create a vector array."""
    return VecArray(data)


def igrid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z: float = 0.0,
    n_points: Union[int, Tuple[int, int]] = 50,
    plane: str = 'xy'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create interpolation grid for field calculations.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max) range.
    y_range : tuple
        (y_min, y_max) range.
    z : float
        Fixed coordinate value for third dimension.
    n_points : int or tuple
        Number of grid points.
    plane : str
        Grid plane: 'xy', 'xz', or 'yz'.

    Returns
    -------
    X : ndarray
        X coordinate grid.
    Y : ndarray
        Y coordinate grid.
    pts : ndarray
        Grid points as (n*n, 3) array.

    Examples
    --------
    >>> X, Y, pts = igrid((-20, 20), (-20, 20), z=0, n_points=50)
    >>> # pts has shape (2500, 3) for field evaluation
    """
    if isinstance(n_points, int):
        n_x, n_y = n_points, n_points
    else:
        n_x, n_y = n_points

    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    X, Y = np.meshgrid(x, y)

    if plane == 'xy':
        Z = np.full_like(X, z)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    elif plane == 'xz':
        # X-Z plane: y is fixed
        pts = np.column_stack([X.ravel(), np.full(X.size, z), Y.ravel()])
    elif plane == 'yz':
        # Y-Z plane: x is fixed
        pts = np.column_stack([np.full(X.size, z), X.ravel(), Y.ravel()])
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return X, Y, pts


def meshgrid3d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    n_points: Union[int, Tuple[int, int, int]] = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3D grid for volume field calculations.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max).
    y_range : tuple
        (y_min, y_max).
    z_range : tuple
        (z_min, z_max).
    n_points : int or tuple
        Number of points per dimension.

    Returns
    -------
    X, Y, Z : ndarray
        3D coordinate grids.
    pts : ndarray
        Flattened points array (n^3, 3).
    """
    if isinstance(n_points, int):
        n_x = n_y = n_z = n_points
    else:
        n_x, n_y, n_z = n_points

    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    z = np.linspace(z_range[0], z_range[1], n_z)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return X, Y, Z, pts


def linspace_grid(start: np.ndarray, end: np.ndarray, n_points: int) -> np.ndarray:
    """
    Create line of points from start to end.

    Parameters
    ----------
    start : ndarray
        Start point (3,).
    end : ndarray
        End point (3,).
    n_points : int
        Number of points.

    Returns
    -------
    ndarray
        Points along line (n_points, 3).
    """
    start = np.asarray(start)
    end = np.asarray(end)
    t = np.linspace(0, 1, n_points)
    return start + np.outer(t, end - start)


def sphere_grid(
    center: np.ndarray = (0, 0, 0),
    radius: float = 1.0,
    n_theta: int = 20,
    n_phi: int = 40
) -> np.ndarray:
    """
    Create spherical grid of points.

    Parameters
    ----------
    center : ndarray
        Sphere center.
    radius : float
        Sphere radius.
    n_theta : int
        Number of polar angles.
    n_phi : int
        Number of azimuthal angles.

    Returns
    -------
    ndarray
        Grid points (n_theta * n_phi, 3).
    """
    center = np.asarray(center)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    THETA, PHI = np.meshgrid(theta, phi)

    x = center[0] + radius * np.sin(THETA) * np.cos(PHI)
    y = center[1] + radius * np.sin(THETA) * np.sin(PHI)
    z = center[2] + radius * np.cos(THETA)

    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def cylinder_grid(
    center: np.ndarray = (0, 0, 0),
    radius: float = 1.0,
    height: float = 1.0,
    n_phi: int = 40,
    n_z: int = 20
) -> np.ndarray:
    """
    Create cylindrical grid of points.

    Parameters
    ----------
    center : ndarray
        Cylinder center.
    radius : float
        Cylinder radius.
    height : float
        Cylinder height.
    n_phi : int
        Number of azimuthal angles.
    n_z : int
        Number of z-points.

    Returns
    -------
    ndarray
        Grid points.
    """
    center = np.asarray(center)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    z = np.linspace(-height/2, height/2, n_z)

    PHI, Z = np.meshgrid(phi, z)

    x = center[0] + radius * np.cos(PHI)
    y = center[1] + radius * np.sin(PHI)
    z_pts = center[2] + Z

    return np.column_stack([x.ravel(), y.ravel(), z_pts.ravel()])
