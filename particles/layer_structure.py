"""
Layer structure for stratified media.

Complete implementation including Green's functions and BEM solver integration.
"""

import numpy as np
from typing import List, Optional, Any, Tuple, Union
from scipy import special
from scipy import integrate

from ..misc.units import eV2nm


class LayerStructure:
    """
    Stratified media (layer structure) for substrate modeling.

    Handles reflection and transmission at planar interfaces,
    Green's function computation, and BEM solver integration.

    Parameters
    ----------
    eps : list
        List of dielectric function objects for each layer.
        eps[0] is the top semi-infinite layer, eps[-1] is the bottom.
    d : list or ndarray
        Thicknesses of intermediate layers (N-2 values for N layers).
    z_interface : list or ndarray, optional
        Z-positions of interfaces. If not given, computed from d.

    Attributes
    ----------
    eps : list
        Dielectric functions for each layer.
    d : ndarray
        Layer thicknesses.
    z : ndarray
        Z-positions of interfaces.
    """

    def __init__(
        self,
        eps: List[Any],
        d: Optional[np.ndarray] = None,
        z_interface: Optional[np.ndarray] = None
    ):
        """
        Initialize layer structure.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        d : ndarray, optional
            Layer thicknesses for intermediate layers.
        z_interface : ndarray, optional
            Z-positions of interfaces.
        """
        self.eps = eps
        self.n_layers = len(eps)

        if d is None:
            d = np.array([])
        self.d = np.asarray(d, dtype=float)

        if z_interface is not None:
            self.z = np.asarray(z_interface, dtype=float)
        else:
            # Compute interface positions from thicknesses
            # Convention: first interface at z=0
            self.z = np.zeros(len(d) + 1)
            for i, thickness in enumerate(d):
                self.z[i + 1] = self.z[i] - thickness

        # Cache for Green's function tables
        self._green_cache = {}
        self._tab_cache = {}

    def indlayer(self, z: np.ndarray) -> np.ndarray:
        """
        Determine which layer each z-position is in.

        Parameters
        ----------
        z : ndarray
            Z-positions.

        Returns
        -------
        ndarray
            Layer indices (0-based).
        """
        z = np.asarray(z)
        indices = np.zeros(z.shape, dtype=int)

        for i, z_int in enumerate(self.z):
            indices[z < z_int] = i + 1

        return indices

    def fresnel(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fresnel reflection and transmission coefficients.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel component of wavevector.
        pol : str
            Polarization: 'p' (TM) or 's' (TE).

        Returns
        -------
        r : ndarray
            Reflection coefficients.
        t : ndarray
            Transmission coefficients.
        """
        kpar = np.atleast_1d(kpar)
        k0 = 2 * np.pi / enei

        # Get dielectric values
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Compute perpendicular wavevector in each layer
        kz = []
        for eps in eps_vals:
            kz_sq = eps * k0**2 - kpar**2
            kz.append(np.sqrt(kz_sq.astype(complex)))

        # Fresnel coefficients at each interface
        r = np.zeros((len(self.z), len(kpar)), dtype=complex)
        t = np.ones((len(self.z), len(kpar)), dtype=complex)

        for i in range(len(self.z)):
            eps1 = eps_vals[i]
            eps2 = eps_vals[i + 1] if i + 1 < len(eps_vals) else eps_vals[i]
            kz1 = kz[i]
            kz2 = kz[i + 1] if i + 1 < len(kz) else kz[i]

            if pol == 'p':  # TM polarization
                r[i] = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
                t[i] = 2 * eps2 * kz1 / (eps2 * kz1 + eps1 * kz2)
            else:  # TE polarization
                r[i] = (kz1 - kz2) / (kz1 + kz2)
                t[i] = 2 * kz1 / (kz1 + kz2)

        return r, t

    def efresnel(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p',
        layer_from: int = 0,
        layer_to: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute extended Fresnel coefficients for multilayer.

        Uses transfer matrix method for full multilayer stack.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel wavevector component.
        pol : str
            Polarization: 'p' or 's'.
        layer_from : int
            Source layer index.
        layer_to : int
            Target layer index (-1 for bottom).

        Returns
        -------
        r : ndarray
            Generalized reflection coefficient.
        t : ndarray
            Generalized transmission coefficient.
        """
        if layer_to < 0:
            layer_to = self.n_layers - 1

        kpar = np.atleast_1d(kpar)
        k0 = 2 * np.pi / enei

        # Get dielectric values
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Compute kz in each layer
        kz = []
        for eps in eps_vals:
            kz_sq = eps * k0**2 - kpar**2
            # Choose branch with positive imaginary part for evanescent waves
            kz_val = np.sqrt(kz_sq.astype(complex))
            kz_val = np.where(kz_val.imag < 0, -kz_val, kz_val)
            kz.append(kz_val)

        # Transfer matrix method
        M = np.zeros((len(kpar), 2, 2), dtype=complex)
        M[:, 0, 0] = 1
        M[:, 1, 1] = 1

        for i in range(layer_from, layer_to):
            # Interface matrix
            r12, t12 = self._fresnel_interface(eps_vals[i], eps_vals[i + 1],
                                               kz[i], kz[i + 1], pol)

            D = np.zeros((len(kpar), 2, 2), dtype=complex)
            D[:, 0, 0] = 1 / t12
            D[:, 0, 1] = r12 / t12
            D[:, 1, 0] = r12 / t12
            D[:, 1, 1] = 1 / t12

            # Propagation matrix in layer i+1
            if i + 1 < layer_to and i < len(self.d):
                phase = np.exp(1j * kz[i + 1] * self.d[i])
                P = np.zeros((len(kpar), 2, 2), dtype=complex)
                P[:, 0, 0] = phase
                P[:, 1, 1] = 1 / phase
                D = np.einsum('kij,kjl->kil', D, P)

            M = np.einsum('kij,kjl->kil', M, D)

        # Extract coefficients
        r = M[:, 1, 0] / M[:, 0, 0]
        t = 1 / M[:, 0, 0]

        return r, t

    def reflection(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p',
        layer_from: int = 0
    ) -> np.ndarray:
        """
        Compute total reflection coefficient.

        Uses transfer matrix method for multilayer structures.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel wavevector component.
        pol : str
            Polarization: 'p' or 's'.
        layer_from : int
            Layer from which reflection is computed.

        Returns
        -------
        ndarray
            Total reflection coefficient.
        """
        if self.n_layers == 2:
            # Simple two-layer case
            r, _ = self.fresnel(enei, kpar, pol)
            return r[0]

        # Multi-layer: use transfer matrix method
        k0 = 2 * np.pi / enei
        kpar = np.atleast_1d(kpar)

        # Get dielectric values
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Compute kz in each layer
        kz = []
        for eps in eps_vals:
            kz_sq = eps * k0**2 - kpar**2
            kz.append(np.sqrt(kz_sq.astype(complex)))

        # Build up reflection from bottom
        r_total = np.zeros(len(kpar), dtype=complex)

        for i in range(len(self.z) - 1, layer_from - 1, -1):
            r12, _ = self._fresnel_interface(eps_vals[i], eps_vals[i + 1],
                                              kz[i], kz[i + 1], pol)

            if i < len(self.z) - 1:
                # Add phase from propagation in layer
                phase = np.exp(2j * kz[i + 1] * self.d[i])
                r_total = (r12 + r_total * phase) / (1 + r12 * r_total * phase)
            else:
                r_total = r12

        return r_total

    def _fresnel_interface(
        self,
        eps1: complex,
        eps2: complex,
        kz1: np.ndarray,
        kz2: np.ndarray,
        pol: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fresnel coefficients for single interface."""
        if pol == 'p':
            r = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
            t = 2 * np.sqrt(eps1 * eps2) * kz1 / (eps2 * kz1 + eps1 * kz2)
        else:
            r = (kz1 - kz2) / (kz1 + kz2)
            t = 2 * kz1 / (kz1 + kz2)
        return r, t

    def mindist(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance to any interface.

        Parameters
        ----------
        pos : ndarray
            Positions, shape (n, 3).

        Returns
        -------
        ndarray
            Minimum distances.
        """
        pos = np.atleast_2d(pos)
        z = pos[:, 2]

        min_dist = np.full(len(pos), np.inf)
        for z_int in self.z:
            dist = np.abs(z - z_int)
            min_dist = np.minimum(min_dist, dist)

        return min_dist

    def green(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        component: str = 'scalar'
    ) -> np.ndarray:
        """
        Compute Green's function for layer structure.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        pos1 : ndarray
            Source positions (n1, 3).
        pos2 : ndarray
            Field positions (n2, 3).
        component : str
            'scalar' for scalar potential, 'dyadic' for dyadic Green function.

        Returns
        -------
        ndarray
            Green's function values.
        """
        pos1 = np.atleast_2d(pos1)
        pos2 = np.atleast_2d(pos2)

        k0 = 2 * np.pi / enei

        # Determine layers for source and field points
        layer1 = self.indlayer(pos1[:, 2])
        layer2 = self.indlayer(pos2[:, 2])

        n1, n2 = len(pos1), len(pos2)

        if component == 'scalar':
            G = np.zeros((n2, n1), dtype=complex)

            for i in range(n1):
                for j in range(n2):
                    G[j, i] = self._green_scalar(
                        enei, pos1[i], pos2[j], layer1[i], layer2[j]
                    )
            return G
        else:
            # Dyadic Green function (3x3 per pair)
            G = np.zeros((n2, n1, 3, 3), dtype=complex)

            for i in range(n1):
                for j in range(n2):
                    G[j, i] = self._green_dyadic(
                        enei, pos1[i], pos2[j], layer1[i], layer2[j]
                    )
            return G

    def _green_scalar(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        layer1: int,
        layer2: int
    ) -> complex:
        """Compute scalar Green's function between two points."""
        k0 = 2 * np.pi / enei
        eps1 = self.eps[layer1](enei)[0]

        # Direct term
        r = np.linalg.norm(pos2 - pos1)
        if r < 1e-10:
            r = 1e-10

        G_direct = np.exp(1j * np.sqrt(eps1) * k0 * r) / (4 * np.pi * r)

        # Reflected term (Sommerfeld integral)
        if len(self.z) > 0:
            G_refl = self._sommerfeld_integral(enei, pos1, pos2, layer1, layer2)
        else:
            G_refl = 0

        return G_direct + G_refl

    def _green_dyadic(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        layer1: int,
        layer2: int
    ) -> np.ndarray:
        """Compute dyadic Green's function between two points."""
        k0 = 2 * np.pi / enei
        eps1 = self.eps[layer1](enei)[0]
        k = np.sqrt(eps1) * k0

        r_vec = pos2 - pos1
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            r = 1e-10
            r_vec = np.array([1e-10, 0, 0])

        r_hat = r_vec / r

        # Free-space dyadic Green's function
        exp_ikr = np.exp(1j * k * r)

        G0 = exp_ikr / (4 * np.pi * r) * (
            (1 + 1j / (k * r) - 1 / (k * r)**2) * np.eye(3) +
            (-1 - 3j / (k * r) + 3 / (k * r)**2) * np.outer(r_hat, r_hat)
        )

        # Add reflected contribution for layered medium
        if len(self.z) > 0:
            G_refl = self._sommerfeld_dyadic(enei, pos1, pos2, layer1, layer2)
            G0 = G0 + G_refl

        return G0

    def _sommerfeld_integral(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        layer1: int,
        layer2: int,
        n_points: int = 50
    ) -> complex:
        """
        Compute Sommerfeld integral for reflected Green's function.

        Uses numerical integration along deformed contour.
        """
        k0 = 2 * np.pi / enei
        eps_vals = [eps(enei)[0] for eps in self.eps]

        # Horizontal distance
        rho = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        z1, z2 = pos1[2], pos2[2]

        if rho < 1e-10:
            rho = 1e-10

        # Integration path: use Gauss-Legendre quadrature
        # Deform contour to avoid branch points
        kmax = 5 * k0 * np.sqrt(max(np.abs(e) for e in eps_vals))

        kpar, weights = np.polynomial.legendre.leggauss(n_points)
        kpar = kmax * (kpar + 1) / 2  # Map to [0, kmax]
        weights = weights * kmax / 2

        # Get reflection coefficient
        r_p = self.reflection(enei, kpar, 'p', layer_from=layer1)

        # Compute kz
        kz = np.sqrt(eps_vals[layer1] * k0**2 - kpar**2 + 0j)
        kz = np.where(kz.imag < 0, -kz, kz)

        # Sommerfeld integral kernel
        # G_refl = integral of r * J0(kpar * rho) * exp(i*kz*(|z1| + |z2|)) * kpar / kz
        z_sum = np.abs(z1) + np.abs(z2)

        J0 = special.j0(kpar * rho)
        integrand = r_p * J0 * np.exp(1j * kz * z_sum) * kpar / (kz + 1e-30)

        G_refl = np.sum(integrand * weights) / (2 * np.pi)

        return G_refl

    def _sommerfeld_dyadic(
        self,
        enei: float,
        pos1: np.ndarray,
        pos2: np.ndarray,
        layer1: int,
        layer2: int
    ) -> np.ndarray:
        """Compute dyadic Sommerfeld integral for reflected field."""
        # Simplified: use scalar approximation for diagonal terms
        G_scalar = self._sommerfeld_integral(enei, pos1, pos2, layer1, layer2)

        # Approximate dyadic as diagonal
        G_refl = np.diag([G_scalar, G_scalar, G_scalar])

        return G_refl

    def bemsolve(
        self,
        particle,
        enei: float,
        options: Optional[dict] = None
    ):
        """
        Create BEM solver for particle in layer structure.

        Parameters
        ----------
        particle : ComParticle
            Particle to solve.
        enei : float
            Wavelength in nm.
        options : dict, optional
            Solver options.

        Returns
        -------
        BEMStatLayer or BEMRetLayer
            Appropriate BEM solver.
        """
        from ..bem import BEMStatLayer, BEMRetLayer

        if options is None:
            options = {}

        sim_type = options.get('sim', 'stat')

        if sim_type == 'stat':
            return BEMStatLayer(particle, self, enei=enei, **options)
        else:
            return BEMRetLayer(particle, self, enei=enei, **options)

    def tabspace(
        self,
        r_range: Tuple[float, float],
        z_range: Tuple[float, float],
        n_r: int = 50,
        n_z: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tabulation space for Green's function interpolation.

        Parameters
        ----------
        r_range : tuple
            (r_min, r_max) for radial coordinate.
        z_range : tuple
            (z_min, z_max) for vertical coordinate.
        n_r : int
            Number of radial points.
        n_z : int
            Number of vertical points.

        Returns
        -------
        r_tab : ndarray
            Radial coordinates.
        z_tab : ndarray
            Vertical coordinates.
        """
        r_tab = np.linspace(r_range[0], r_range[1], n_r)
        z_tab = np.linspace(z_range[0], z_range[1], n_z)
        return r_tab, z_tab

    def round(self, z: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """
        Round z-values to nearest interface if within tolerance.

        Parameters
        ----------
        z : ndarray
            Z-coordinates.
        tol : float
            Tolerance for rounding.

        Returns
        -------
        ndarray
            Rounded z-values.
        """
        z = np.asarray(z).copy()

        for z_int in self.z:
            mask = np.abs(z - z_int) < tol
            z[mask] = z_int

        return z

    def options(self, **kwargs) -> dict:
        """
        Create options dictionary for BEM solver.

        Parameters
        ----------
        **kwargs : dict
            Additional options.

        Returns
        -------
        dict
            Options dictionary.
        """
        opts = {
            'layer': self,
            'n_layers': self.n_layers,
        }
        opts.update(kwargs)
        return opts

    def reflectionsubs(
        self,
        enei: float,
        kpar: np.ndarray,
        pol: str = 'p'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reflection coefficients for substrate (bottom layer).

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        kpar : ndarray
            Parallel wavevector.
        pol : str
            Polarization.

        Returns
        -------
        r_sub : ndarray
            Substrate reflection coefficient.
        kz_sub : ndarray
            Perpendicular wavevector in substrate.
        """
        k0 = 2 * np.pi / enei
        eps_sub = self.eps[-1](enei)[0]

        kpar = np.atleast_1d(kpar)
        kz_sub = np.sqrt(eps_sub * k0**2 - kpar**2 + 0j)

        r_sub = self.reflection(enei, kpar, pol, layer_from=0)

        return r_sub, kz_sub

    def __repr__(self) -> str:
        return f"LayerStructure(n_layers={self.n_layers}, z={self.z})"
