"""
Tabulated Green functions for layer structures.

Pre-computes Green functions on a grid for fast interpolation.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import interpolate


class GreenTableLayer:
    """
    Tabulated Green function for layer structures.

    Pre-computes the reflected Green function on an (r, z) grid
    for fast interpolation during BEM calculations.

    Parameters
    ----------
    layer : LayerStructure
        Layer structure.
    r_range : tuple
        (r_min, r_max) for radial coordinate.
    z_range : tuple
        (z_min, z_max) for vertical coordinate.
    n_r : int
        Number of radial grid points.
    n_z : int
        Number of vertical grid points.

    Examples
    --------
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.material import EpsConst
    >>> from pymnpbem.greenfun import GreenTableLayer
    >>>
    >>> eps = [EpsConst(1), EpsConst(2.25)]
    >>> layer = LayerStructure(eps)
    >>> gtab = GreenTableLayer(layer, (0.1, 100), (1, 200), n_r=100, n_z=100)
    """

    def __init__(
        self,
        layer,
        r_range: Tuple[float, float],
        z_range: Tuple[float, float],
        n_r: int = 100,
        n_z: int = 100
    ):
        """Initialize tabulated Green function."""
        self.layer = layer
        self.r_range = r_range
        self.z_range = z_range
        self.n_r = n_r
        self.n_z = n_z

        # Create grid (use log spacing for r)
        self.r_tab = np.logspace(
            np.log10(max(r_range[0], 0.1)),
            np.log10(r_range[1]),
            n_r
        )
        self.z_tab = np.linspace(z_range[0], z_range[1], n_z)

        # Table storage
        self._tables = {}
        self._interpolators = {}

    def compute_table(self, enei: float, component: str = 'scalar') -> np.ndarray:
        """
        Compute Green function table for given wavelength.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        component : str
            'scalar' or 'zz' component.

        Returns
        -------
        ndarray
            Green function table (n_r, n_z, n_z).
        """
        key = (enei, component)
        if key in self._tables:
            return self._tables[key]

        n_r, n_z = self.n_r, self.n_z

        # For each (r, z1, z2), compute reflected Green function
        table = np.zeros((n_r, n_z, n_z), dtype=complex)

        for i, r in enumerate(self.r_tab):
            for j, z1 in enumerate(self.z_tab):
                for k, z2 in enumerate(self.z_tab):
                    pos1 = np.array([0, 0, z1])
                    pos2 = np.array([r, 0, z2])

                    if component == 'scalar':
                        G = self.layer._sommerfeld_integral(
                            enei, pos1, pos2,
                            self.layer.indlayer(np.array([z1]))[0],
                            self.layer.indlayer(np.array([z2]))[0]
                        )
                    else:
                        G = self._compute_component(enei, pos1, pos2, component)

                    table[i, j, k] = G

        self._tables[key] = table
        self._create_interpolator(key)

        return table

    def _compute_component(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        component: str
    ) -> complex:
        """Compute specific dyadic component."""
        layer1 = self.layer.indlayer(np.array([pos1[2]]))[0]
        layer2 = self.layer.indlayer(np.array([pos2[2]]))[0]

        if component == 'zz':
            G_dyad = self.layer._green_dyadic(enei, pos1, pos2, layer1, layer2)
            return G_dyad[2, 2]
        elif component == 'xx':
            G_dyad = self.layer._green_dyadic(enei, pos1, pos2, layer1, layer2)
            return G_dyad[0, 0]
        else:
            return self.layer._green_scalar(enei, pos1, pos2, layer1, layer2)

    def _create_interpolator(self, key: Tuple) -> None:
        """Create interpolator for cached table."""
        table = self._tables[key]

        # Use log(r) for better interpolation
        log_r = np.log10(self.r_tab)

        # Create interpolator (real and imaginary parts separately)
        self._interpolators[key] = {
            'real': interpolate.RegularGridInterpolator(
                (log_r, self.z_tab, self.z_tab),
                table.real,
                method='linear',
                bounds_error=False,
                fill_value=0
            ),
            'imag': interpolate.RegularGridInterpolator(
                (log_r, self.z_tab, self.z_tab),
                table.imag,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
        }

    def interp(
        self,
        enei: float,
        r: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
        component: str = 'scalar'
    ) -> np.ndarray:
        """
        Interpolate Green function values.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        r : ndarray
            Radial distances.
        z1 : ndarray
            Source z-coordinates.
        z2 : ndarray
            Field z-coordinates.
        component : str
            Component to interpolate.

        Returns
        -------
        ndarray
            Interpolated Green function values.
        """
        key = (enei, component)
        if key not in self._interpolators:
            self.compute_table(enei, component)

        interp = self._interpolators[key]

        # Prepare points
        r = np.atleast_1d(r)
        z1 = np.atleast_1d(z1)
        z2 = np.atleast_1d(z2)

        # Handle broadcasting
        if r.shape != z1.shape:
            if len(r) == 1:
                r = np.full_like(z1, r[0])
            elif len(z1) == 1:
                z1 = np.full_like(r, z1[0])
                z2 = np.full_like(r, z2[0])

        # Clip r to valid range
        r_clipped = np.clip(r, self.r_range[0], self.r_range[1])
        log_r = np.log10(r_clipped)

        points = np.column_stack([log_r, z1, z2])

        # Interpolate
        real_part = interp['real'](points)
        imag_part = interp['imag'](points)

        return real_part + 1j * imag_part

    def inside(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Check if points are inside tabulation range.

        Parameters
        ----------
        r : ndarray
            Radial distances.
        z : ndarray
            Z-coordinates.

        Returns
        -------
        ndarray
            Boolean array.
        """
        r = np.atleast_1d(r)
        z = np.atleast_1d(z)

        in_r = (r >= self.r_range[0]) & (r <= self.r_range[1])
        in_z = (z >= self.z_range[0]) & (z <= self.z_range[1])

        return in_r & in_z

    def ismember(self, enei: float) -> bool:
        """Check if wavelength is in cache."""
        return any(key[0] == enei for key in self._tables)

    def clear(self) -> None:
        """Clear cached tables."""
        self._tables.clear()
        self._interpolators.clear()

    def __repr__(self) -> str:
        return (f"GreenTableLayer(r_range={self.r_range}, "
                f"z_range={self.z_range}, n_r={self.n_r}, n_z={self.n_z})")


class CompGreenTableLayer:
    """
    Composite tabulated Green function for layer structures.

    Wraps GreenTableLayer for use with composite particles.

    Parameters
    ----------
    p1 : ComParticle
        First particle (positions).
    p2 : ComParticle
        Second particle (positions).
    layer : LayerStructure
        Layer structure.
    **kwargs : dict
        Options for GreenTableLayer.
    """

    def __init__(self, p1, p2, layer, **kwargs):
        """Initialize composite tabulated Green function."""
        self.p1 = p1
        self.p2 = p2
        self.layer = layer

        # Determine grid ranges from particle positions
        pos1 = p1.pos if hasattr(p1, 'pos') else p1.pc.pos
        pos2 = p2.pos if hasattr(p2, 'pos') else p2.pc.pos

        all_pos = np.vstack([pos1, pos2])

        # Radial range
        r_max = np.max(np.sqrt(all_pos[:, 0]**2 + all_pos[:, 1]**2))
        r_range = kwargs.pop('r_range', (0.1, 2 * r_max + 100))

        # Z range
        z_min = np.min(all_pos[:, 2])
        z_max = np.max(all_pos[:, 2])
        z_margin = 0.2 * (z_max - z_min)
        z_range = kwargs.pop('z_range', (z_min - z_margin, z_max + z_margin))

        # Create table
        self.table = GreenTableLayer(layer, r_range, z_range, **kwargs)

        self._enei = None
        self._G = None

    def eval(self, enei: float) -> np.ndarray:
        """
        Evaluate Green function matrix.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Green function matrix (n2, n1).
        """
        if self._enei == enei and self._G is not None:
            return self._G

        pos1 = self.p1.pos if hasattr(self.p1, 'pos') else self.p1.pc.pos
        pos2 = self.p2.pos if hasattr(self.p2, 'pos') else self.p2.pc.pos

        n1, n2 = len(pos1), len(pos2)

        # Compute distances
        r = np.zeros((n2, n1))
        z1_grid = np.zeros((n2, n1))
        z2_grid = np.zeros((n2, n1))

        for i in range(n1):
            for j in range(n2):
                dx = pos2[j, 0] - pos1[i, 0]
                dy = pos2[j, 1] - pos1[i, 1]
                r[j, i] = np.sqrt(dx**2 + dy**2)
                z1_grid[j, i] = pos1[i, 2]
                z2_grid[j, i] = pos2[j, 2]

        # Interpolate reflected Green function
        G_refl = self.table.interp(
            enei,
            r.flatten(),
            z1_grid.flatten(),
            z2_grid.flatten()
        ).reshape((n2, n1))

        # Add direct Green function
        k0 = 2 * np.pi / enei
        eps0 = self.layer.eps[0](enei)[0]
        k = np.sqrt(eps0) * k0

        G_direct = np.zeros((n2, n1), dtype=complex)
        for i in range(n1):
            for j in range(n2):
                dist = np.linalg.norm(pos2[j] - pos1[i])
                if dist < 1e-10:
                    dist = 1e-10
                G_direct[j, i] = np.exp(1j * k * dist) / (4 * np.pi * dist)

        self._G = G_direct + G_refl
        self._enei = enei

        return self._G

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        if self._G is None:
            raise ValueError("Must call eval() first")
        return self._G

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """Compute potential from charges."""
        if self._G is None:
            raise ValueError("Must call eval() first")
        return self._G @ sig

    def field(self, sig: np.ndarray) -> np.ndarray:
        """Compute field from charges (approximate)."""
        # Use numerical gradient
        phi = self.potential(sig)
        return -np.gradient(phi)

    def __repr__(self) -> str:
        return f"CompGreenTableLayer(table={self.table})"
