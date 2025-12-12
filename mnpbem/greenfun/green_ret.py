"""
Retarded Green function for full electromagnetic simulations.

This module implements the retarded (full Maxwell) Green function
that accounts for electromagnetic wave propagation effects.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import sparse

from ..particles import Particle, Point
from ..misc.units import SPEED_OF_LIGHT


class GreenRet:
    """
    Green function for retarded (full electromagnetic) simulations.

    The retarded Green function includes wave propagation effects:
        G(r, r', k) = exp(i*k*|r-r'|) / (4*pi*|r-r'|)

    Parameters
    ----------
    p1 : Particle or Point
        Source points/particle.
    p2 : Particle
        Target particle (boundary elements).
    k : complex
        Wavenumber in the medium.
    rel_cutoff : float
        Cutoff parameter for face integration refinement.

    Attributes
    ----------
    p1 : Particle or Point
        Source points.
    p2 : Particle
        Target particle.
    k : complex
        Wavenumber.
    """

    def __init__(
        self,
        p1: Union[Particle, Point],
        p2: Particle,
        k: complex = None,
        rel_cutoff: float = 3.0,
        **kwargs
    ):
        """
        Initialize retarded Green function.

        Parameters
        ----------
        p1 : Particle or Point
            Source points/particle.
        p2 : Particle
            Target particle.
        k : complex
            Wavenumber.
        rel_cutoff : float
            Cutoff for integration refinement.
        """
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.rel_cutoff = rel_cutoff
        self.options = kwargs

        # Cache for computed matrices
        self._G = None
        self._F = None
        self._Gp = None
        self._L = None
        self._cached_k = None

    @property
    def pos1(self) -> np.ndarray:
        """Source positions."""
        return self.p1.pos

    @property
    def pos2(self) -> np.ndarray:
        """Target positions (face centroids)."""
        return self.p2.pos

    @property
    def n1(self) -> int:
        """Number of source points."""
        return len(self.pos1)

    @property
    def n2(self) -> int:
        """Number of target faces."""
        return self.p2.n_faces

    def _compute_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances and distance vectors between all point pairs.

        Returns
        -------
        d : ndarray
            Distance matrix, shape (n1, n2).
        dr : ndarray
            Distance vectors, shape (n1, n2, 3).
        """
        # dr[i, j] = pos2[j] - pos1[i]
        dr = self.pos2[np.newaxis, :, :] - self.pos1[:, np.newaxis, :]
        d = np.linalg.norm(dr, axis=2)
        return d, dr

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        self._G = None
        self._F = None
        self._Gp = None
        self._L = None
        self._cached_k = None

    def set_k(self, k: complex) -> None:
        """
        Set wavenumber and clear cache if changed.

        Parameters
        ----------
        k : complex
            Wavenumber in the medium.
        """
        if k != self._cached_k:
            self.k = k
            self.clear_cache()
            self._cached_k = k

    def G(self, k: complex = None) -> np.ndarray:
        """
        Compute retarded Green function matrix (scalar potential).

        Following MATLAB greenret/eval1.m:
        G[i, j] = exp(i*k*r) / r * area[j]

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor in the Green function.
        This is because the boundary element method integrates the factor into
        the boundary conditions.

        Parameters
        ----------
        k : complex, optional
            Wavenumber. Uses stored value if not provided.

        Returns
        -------
        ndarray
            Green function matrix, shape (n1, n2), complex.
        """
        if k is not None:
            self.set_k(k)
        k = self.k

        if self._G is not None and self._cached_k == k:
            return self._G

        d, _ = self._compute_distances()

        # Identify self-terms (zero distance)
        self_mask = (d == 0)

        # Use placeholder value for self-terms to avoid NaN from exp(1j*k*inf)
        d_safe = np.where(self_mask, 1.0, d)

        # Retarded Green function: exp(i*k*r) / r  (MATLAB convention, no 4*pi)
        G = np.exp(1j * k * d_safe) / d_safe

        # Set self-terms to zero (they would be singular anyway)
        G[self_mask] = 0

        # Multiply by face areas for integration
        G = G * self.p2.area[np.newaxis, :]

        self._G = G
        return self._G

    def F(self, k: complex = None) -> np.ndarray:
        """
        Compute surface derivative of retarded Green function.

        Following MATLAB greenret/eval1.m:
        F[i, j] = (n_j . r_ij) * (i*k - 1/r) / r^2 * area[j] * exp(i*k*r)

        where r_ij = pos2[j] - pos1[i], r = |r_ij|

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor.

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Surface derivative matrix, shape (n1, n2), complex.
        """
        if k is not None:
            self.set_k(k)
        k = self.k

        if self._F is not None and self._cached_k == k:
            return self._F

        d, dr = self._compute_distances()

        # Identify self-terms (zero distance)
        self_mask = (d == 0)

        # Use placeholder value for self-terms to avoid NaN
        d_safe = np.where(self_mask, 1.0, d)

        # Normal vectors at source (p1) - MATLAB uses p1.nvec
        nvec = self.p1.nvec if hasattr(self.p1, 'nvec') else self.p2.nvec

        # n . dr (inner product of normal with distance vector)
        # dr = pos2 - pos1, so n . dr is negative of what we might expect
        n_dot_dr = np.sum(nvec[:, np.newaxis, :] * dr, axis=2)

        # F = (n . r) * (i*k - 1/r) / r^2 * area * exp(ikr)
        # (MATLAB formula from eval1.m)
        exp_ikr = np.exp(1j * k * d_safe)
        F = n_dot_dr * (1j * k - 1.0 / d_safe) / (d_safe ** 2) * exp_ikr

        # Set self-terms to zero (singular)
        F[self_mask] = 0

        # Multiply by face areas
        F = F * self.p2.area[np.newaxis, :]

        self._F = F
        return self._F

    def H1(self, k: complex = None) -> np.ndarray:
        """
        F + 2*pi (for inside boundary condition).

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F(k) + 2 * np.pi * np.eye(self.n1, self.n2)

    def H2(self, k: complex = None) -> np.ndarray:
        """
        F - 2*pi (for outside boundary condition).

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F(k) - 2 * np.pi * np.eye(self.n1, self.n2)

    def Gp(self, k: complex = None) -> np.ndarray:
        """
        Compute gradient of retarded Green function.

        Following MATLAB greenret/eval1.m:
        Gp[i, j, :] = (i*k - 1/r) / r^2 * dr * area[j] * exp(i*k*r)

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor.

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Green function gradient, shape (n1, n2, 3), complex.
        """
        if k is not None:
            self.set_k(k)
        k = self.k

        if self._Gp is not None and self._cached_k == k:
            return self._Gp

        d, dr = self._compute_distances()

        # Identify self-terms (zero distance)
        self_mask = (d == 0)

        # Use placeholder value for self-terms to avoid NaN
        d_safe = np.where(self_mask, 1.0, d)

        # grad G = (i*k - 1/r) * exp(i*k*r) / r^2 * dr  (no 4*pi)
        exp_ikr = np.exp(1j * k * d_safe)
        factor = (1j * k - 1.0 / d_safe) * exp_ikr / (d_safe ** 2)

        # Gp[i, j, :] = factor[i,j] * dr[i,j,:]
        Gp = factor[:, :, np.newaxis] * dr

        # Set self-terms to zero (they would be singular anyway)
        Gp[self_mask, :] = 0

        # Multiply by face areas
        Gp = Gp * self.p2.area[np.newaxis, :, np.newaxis]

        self._Gp = Gp
        return self._Gp

    def L(self, k: complex = None) -> np.ndarray:
        """
        Compute the L matrix for vector potential (dyadic Green function).

        L is related to the transverse part of the Green function
        used for magnetic field contributions.

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            L matrix, shape (n1, n2, 3, 3), complex.
        """
        if k is not None:
            self.set_k(k)
        k = self.k

        if self._L is not None and self._cached_k == k:
            return self._L

        d, dr = self._compute_distances()

        # Identify self-terms (zero distance)
        self_mask = (d == 0)

        # Use placeholder value for self-terms to avoid NaN from exp(1j*k*inf)
        d_safe = np.where(self_mask, 1.0, d)

        # Unit vectors
        r_hat = dr / d_safe[:, :, np.newaxis]

        # Scalar Green function
        exp_ikr = np.exp(1j * k * d_safe)
        g0 = exp_ikr / (4 * np.pi * d_safe)

        # Dyadic Green function components
        # G_ij = g0 * [(1 + i/(kr) - 1/(kr)^2) * delta_ij
        #              + (-1 - 3i/(kr) + 3/(kr)^2) * r_i * r_j / r^2]
        kr = k * d_safe
        kr_safe = np.where(self_mask, 1.0, kr)  # Avoid division by zero

        coef1 = 1 + 1j / kr_safe - 1 / kr_safe ** 2
        coef2 = -1 - 3j / kr_safe + 3 / kr_safe ** 2

        # Build dyadic
        L = np.zeros((self.n1, self.n2, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                L[:, :, i, j] = g0 * (coef1 * delta_ij + coef2 * r_hat[:, :, i] * r_hat[:, :, j])

        # Set self-terms to zero (they would be singular anyway)
        L[self_mask, :, :] = 0

        # Multiply by face areas
        L = L * self.p2.area[np.newaxis, :, np.newaxis, np.newaxis]

        self._L = L
        return self._L

    def potential(self, sig: np.ndarray, k: complex = None) -> np.ndarray:
        """
        Compute potential from surface charges.

        phi = G @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Potential at source points.
        """
        return self.G(k) @ sig

    def field(self, sig: np.ndarray, k: complex = None) -> np.ndarray:
        """
        Compute electric field from surface charges.

        E = -Gp @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Electric field at source points, shape (n1, 3) or (n1, 3, n_exc).
        """
        Gp = self.Gp(k)

        # Handle sig with shape (n2, 1) - squeeze to 1D for single excitation
        if sig.ndim == 2 and sig.shape[1] == 1:
            sig = sig.squeeze(axis=1)

        if sig.ndim == 1:
            return -np.einsum('ijk,j->ik', Gp, sig)
        else:
            return -np.einsum('ijk,jl->ikl', Gp, sig)

    def __repr__(self) -> str:
        return f"GreenRet(n1={self.n1}, n2={self.n2}, k={self.k})"


class GreenRetLayer(GreenRet):
    """
    Retarded Green function with layer (substrate) effects.

    Includes reflected Green function contributions from
    planar interfaces using Sommerfeld integrals or tabulated values.

    Parameters
    ----------
    p1 : Particle or Point
        Source points/particle.
    p2 : Particle
        Target particle.
    layer : LayerStructure
        Layer structure defining interfaces.
    k : complex
        Wavenumber.
    tab : GreenTableLayer, optional
        Pre-computed Green function table for interpolation.
    deriv : str
        Type of derivative: 'norm' for surface normal, 'cart' for Cartesian.

    Attributes
    ----------
    G_dict : dict
        Reflected Green function components (p, s polarizations).
    F_dict : dict
        Surface derivatives of reflected Green function.
    """

    def __init__(
        self,
        p1: Union[Particle, Point],
        p2: Particle,
        layer,
        k: complex = None,
        tab=None,
        deriv: str = 'norm',
        **kwargs
    ):
        super().__init__(p1, p2, k, **kwargs)
        self.layer = layer
        self.tab = tab
        self.deriv = deriv

        # Reflected Green function storage
        self._G_refl = None
        self._F_refl = None
        self._Gp_refl = None
        self._enei = None
        self._G_dict = None
        self._F_dict = None

        # Indices for diagonal refinement
        self._id = None
        self._ind = None

        # Initialize refinement indices
        self._init_refinement()

    def _init_refinement(self) -> None:
        """Initialize indices for diagonal element refinement."""
        # Only set up diagonal refinement if p1 and p2 are the same object
        # or have the same positions (same shape required for comparison)
        if self.p1 is self.p2:
            n = self.n1
            self._id = np.arange(n) * n + np.arange(n)
        elif (hasattr(self.p1, 'pos') and hasattr(self.p2, 'pos') and
              self.p1.pos.shape == self.p2.pos.shape and
              np.allclose(self.p1.pos, self.p2.pos)):
            n = self.n1
            self._id = np.arange(n) * n + np.arange(n)

    def initrefl(self, enei: float, ind: np.ndarray = None) -> 'GreenRetLayer':
        """
        Initialize reflected part of Green function.

        Computes the reflected Green function using tabulated values
        and Sommerfeld integral interpolation.

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum (nm).
        ind : ndarray, optional
            Index to specific matrix elements to compute.

        Returns
        -------
        GreenRetLayer
            Self with computed reflected Green functions.
        """
        if self._enei is not None and enei == self._enei:
            return self

        self._enei = enei

        if ind is not None:
            return self._initrefl3(enei, ind)
        elif self.deriv == 'norm':
            return self._initrefl1(enei)
        else:
            return self._initrefl2(enei)

    def _initrefl1(self, enei: float) -> 'GreenRetLayer':
        """
        Initialize reflected Green function with surface normal derivatives.
        Computes G and F (surface derivative).
        """
        if self.tab is not None:
            self.tab.compute_table(enei)

        pos1 = self.pos1
        pos2 = self.pos2

        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        r = np.sqrt(x**2 + y**2)
        z1 = np.tile(pos1[:, 2:3], (1, len(pos2)))
        z2 = np.tile(pos2[:, 2].T, (len(pos1), 1))

        nvec = self.p1.nvec if hasattr(self.p1, 'nvec') else np.zeros((self.n1, 3))
        r_safe = np.where(r < 1e-10, 1e-10, r)
        in_xy = (nvec[:, 0:1] * x + nvec[:, 1:2] * y) / r_safe

        G_dict, Fr_dict, Fz_dict = self._compute_sommerfeld(enei, r, z1, z2)

        F_dict = {}
        area = self.p2.area if hasattr(self.p2, 'area') else np.ones(self.n2)

        for name in G_dict.keys():
            G_dict[name] = G_dict[name] * area[np.newaxis, :]
            F_dict[name] = (Fr_dict[name] * in_xy +
                          Fz_dict[name] * nvec[:, 2:3]) * area[np.newaxis, :]

        if self._id is not None and len(self._id) > 0:
            self._refine_diagonal(enei, G_dict, F_dict, Fr_dict, Fz_dict)

        self._G_dict = G_dict
        self._F_dict = F_dict
        return self

    def _initrefl2(self, enei: float) -> 'GreenRetLayer':
        """Initialize with Cartesian derivatives. Computes G and Gp (gradient)."""
        if self.tab is not None:
            self.tab.compute_table(enei)

        pos1 = self.pos1
        pos2 = self.pos2

        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        r = np.sqrt(x**2 + y**2)
        z1 = np.tile(pos1[:, 2:3], (1, len(pos2)))
        z2 = np.tile(pos2[:, 2].T, (len(pos1), 1))

        G_dict, Fr_dict, Fz_dict = self._compute_sommerfeld(enei, r, z1, z2)

        Gp_dict = {}
        area = self.p2.area if hasattr(self.p2, 'area') else np.ones(self.n2)
        r_safe = np.where(r < 1e-10, 1e-10, r)

        for name in G_dict.keys():
            G_dict[name] = G_dict[name] * area[np.newaxis, :]
            Gp = np.zeros((self.n1, 3, self.n2), dtype=complex)
            Gp[:, 0, :] = Fr_dict[name] * (x / r_safe) * area[np.newaxis, :]
            Gp[:, 1, :] = Fr_dict[name] * (y / r_safe) * area[np.newaxis, :]
            Gp[:, 2, :] = Fz_dict[name] * area[np.newaxis, :]
            Gp_dict[name] = Gp

        F_dict = {}
        nvec = self.p1.nvec if hasattr(self.p1, 'nvec') else np.zeros((self.n1, 3))
        for name in G_dict.keys():
            F_dict[name] = np.sum(nvec[:, :, np.newaxis] * Gp_dict[name], axis=1)

        self._G_dict = G_dict
        self._F_dict = F_dict
        self._Gp_refl = Gp_dict
        return self

    def _initrefl3(self, enei: float, ind: np.ndarray) -> 'GreenRetLayer':
        """Initialize for specific indices only."""
        if self.tab is not None:
            self.tab.compute_table(enei)

        row, col = np.unravel_index(ind, (self.n1, self.n2))
        pos1 = self.pos1[row]
        pos2 = self.pos2[col]

        x = pos1[:, 0] - pos2[:, 0]
        y = pos1[:, 1] - pos2[:, 1]
        r = np.sqrt(x**2 + y**2)
        z1 = pos1[:, 2]
        z2 = pos2[:, 2]

        nvec = self.p1.nvec[row] if hasattr(self.p1, 'nvec') else np.zeros((len(row), 3))
        area = self.p2.area[col] if hasattr(self.p2, 'area') else np.ones(len(col))
        r_safe = np.where(r < 1e-10, 1e-10, r)
        in_xy = (nvec[:, 0] * x + nvec[:, 1] * y) / r_safe

        G_vals, Fr_vals, Fz_vals = self._compute_sommerfeld_1d(enei, r, z1, z2)

        G_dict = {}
        F_dict = {}
        for name in G_vals.keys():
            G_dict[name] = G_vals[name] * area
            F_dict[name] = (Fr_vals[name] * in_xy + Fz_vals[name] * nvec[:, 2]) * area

        self._G_dict = G_dict
        self._F_dict = F_dict
        self._refl_ind = ind
        return self

    def _compute_sommerfeld(
        self,
        enei: float,
        r: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray
    ) -> Tuple[dict, dict, dict]:
        """
        Compute reflected Green function using Sommerfeld integrals.
        Uses image charge approximation with Fresnel coefficients.
        """
        k0 = 2 * np.pi / enei
        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0

        eps_above = self.layer.eps[0](enei) if callable(self.layer.eps[0]) else self.layer.eps[0]
        eps_below = self.layer.eps[1](enei) if len(self.layer.eps) > 1 else eps_above
        if hasattr(eps_above, '__len__'):
            eps_above = eps_above[0]
        if hasattr(eps_below, '__len__'):
            eps_below = eps_below[0]

        k1 = k0 * np.sqrt(eps_above)
        z1_mirror = 2 * z_interface - z1
        z_diff = z1_mirror - z2
        d_mirror = np.sqrt(r**2 + z_diff**2)
        d_safe = np.where(d_mirror < 1e-10, 1e-10, d_mirror)

        r_p = (eps_below - eps_above) / (eps_below + eps_above)
        r_s = (np.sqrt(eps_above) - np.sqrt(eps_below)) / (np.sqrt(eps_above) + np.sqrt(eps_below))

        G_p = r_p * np.exp(1j * k1 * d_safe) / (4 * np.pi * d_safe)
        G_s = r_s * np.exp(1j * k1 * d_safe) / (4 * np.pi * d_safe)

        factor = (1j * k1 - 1/d_safe) / d_safe
        Fr_p = G_p * factor * r / d_safe
        Fr_s = G_s * factor * r / d_safe
        Fz_p = G_p * factor * z_diff / d_safe
        Fz_s = G_s * factor * z_diff / d_safe

        return (
            {'p': G_p, 's': G_s},
            {'p': Fr_p, 's': Fr_s},
            {'p': Fz_p, 's': Fz_s}
        )

    def _compute_sommerfeld_1d(
        self,
        enei: float,
        r: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray
    ) -> Tuple[dict, dict, dict]:
        """Compute for 1D arrays."""
        G, Fr, Fz = self._compute_sommerfeld(enei, r[:, np.newaxis], z1[:, np.newaxis], z2[:, np.newaxis])
        return (
            {k: v.flatten() for k, v in G.items()},
            {k: v.flatten() for k, v in Fr.items()},
            {k: v.flatten() for k, v in Fz.items()}
        )

    def _refine_diagonal(
        self,
        enei: float,
        G_dict: dict,
        F_dict: dict,
        Fr_dict: dict,
        Fz_dict: dict
    ) -> None:
        """
        Refine diagonal elements using polar integration.
        Implements Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015).
        """
        if self._id is None or len(self._id) == 0:
            return

        p = self.p1
        n = p.n_faces if hasattr(p, 'n_faces') else len(p.pos)
        diag_idx = np.arange(n)

        n_phi = 12
        n_r = 6
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        r_quad = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
        w_r = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

        for name in G_dict.keys():
            G_refined = np.zeros(n, dtype=complex)
            F_refined = np.zeros(n, dtype=complex)

            for i in range(n):
                pos_c = p.pos[i]
                nvec_i = p.nvec[i] if hasattr(p, 'nvec') else np.array([0, 0, 1])
                face_radius = np.sqrt(p.area[i] / np.pi) if hasattr(p, 'area') else 1.0

                G_sum = 0.0
                F_sum = 0.0

                for j, rj in enumerate(r_quad):
                    r_actual = rj * face_radius
                    for phi_k in phi:
                        dx = r_actual * np.cos(phi_k)
                        dy = r_actual * np.sin(phi_k)
                        r_dist = np.sqrt(dx**2 + dy**2)
                        z1_q = pos_c[2]
                        z2_q = pos_c[2]

                        G_q, Fr_q, Fz_q = self._compute_sommerfeld_1d(
                            enei, np.array([r_dist]), np.array([z1_q]), np.array([z2_q])
                        )
                        weight = w_r[j] * r_actual * (2 * np.pi / n_phi)
                        G_sum += G_q[name][0] * weight

                        r_safe = max(r_dist, 1e-10)
                        in_xy = (nvec_i[0] * dx + nvec_i[1] * dy) / r_safe
                        F_sum += (Fr_q[name][0] * in_xy + Fz_q[name][0] * nvec_i[2]) * weight

                G_refined[i] = G_sum
                F_refined[i] = F_sum

            G_dict[name].flat[self._id] = G_refined
            F_dict[name].flat[self._id] = F_refined

    def shapefunction(self, ind: int) -> np.ndarray:
        """
        Compute shape function for boundary element.

        Parameters
        ----------
        ind : int
            Index to boundary element.

        Returns
        -------
        ndarray
            Shape functions for element vertices.
        """
        p = self.p1
        if not hasattr(p, 'faces'):
            return np.array([1.0])

        face = p.faces[ind]
        if np.isnan(face[-1]) or face[-1] < 0:
            xi = np.array([0, 1, 0])
            eta = np.array([0, 0, 1])
            return np.column_stack([xi, eta, 1 - xi - eta])
        else:
            xi = np.array([-1, 1, 1, -1])
            eta = np.array([-1, -1, 1, 1])
            return 0.25 * np.column_stack([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])

    def G_reflected(self, k: complex = None) -> np.ndarray:
        """Compute total reflected Green function (sum of polarizations)."""
        if self._G_dict is None:
            return self._simple_reflected_G(k)

        G_total = np.zeros((self.n1, self.n2), dtype=complex)
        for name, G in self._G_dict.items():
            G_total += G
        return G_total

    def F_reflected(self, k: complex = None) -> np.ndarray:
        """Compute surface derivative of reflected Green function."""
        if self._F_dict is None:
            return np.zeros((self.n1, self.n2), dtype=complex)

        F_total = np.zeros((self.n1, self.n2), dtype=complex)
        for name, F in self._F_dict.items():
            F_total += F
        return F_total

    def _simple_reflected_G(self, k: complex = None) -> np.ndarray:
        """Simple image charge approximation for backward compatibility."""
        if k is not None:
            self.set_k(k)
        k = self.k

        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0
        pos1 = self.pos1
        pos2 = self.pos2

        pos1_mirror = pos1.copy()
        pos1_mirror[:, 2] = 2 * z_interface - pos1[:, 2]

        dr = pos2[np.newaxis, :, :] - pos1_mirror[:, np.newaxis, :]
        d = np.linalg.norm(dr, axis=2)
        d_safe = np.where(d == 0, np.inf, d)

        if len(self.layer.eps) >= 2:
            eps1, eps2 = self.layer.eps[0], self.layer.eps[1]
            n1 = np.sqrt(eps1) if np.isscalar(eps1) else np.sqrt(eps1(2*np.pi/k)[0])
            n2 = np.sqrt(eps2) if np.isscalar(eps2) else np.sqrt(eps2(2*np.pi/k)[0])
            r_fresnel = (n1 - n2) / (n1 + n2)
        else:
            r_fresnel = 0

        G_refl = r_fresnel * np.exp(1j * k * d_safe) / (4 * np.pi * d_safe)
        G_refl = G_refl * self.p2.area[np.newaxis, :]
        return G_refl

    def G(self, k: complex = None) -> np.ndarray:
        """Total Green function including reflection."""
        return super().G(k) + self.G_reflected(k)

    def F(self, k: complex = None) -> np.ndarray:
        """Total surface derivative including reflection."""
        return super().F(k) + self.F_reflected(k)

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        super().clear_cache()
        self._G_refl = None
        self._F_refl = None
        self._Gp_refl = None
        self._G_dict = None
        self._F_dict = None
        self._enei = None
