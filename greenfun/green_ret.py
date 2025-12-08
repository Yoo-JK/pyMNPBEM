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

        G[i, j] = exp(i*k*r) / (4*pi*r) * area[j]

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

        # Avoid division by zero on diagonal
        d_safe = np.where(d == 0, np.inf, d)

        # Retarded Green function: exp(i*k*r) / (4*pi*r)
        G = np.exp(1j * k * d_safe) / (4 * np.pi * d_safe)

        # Handle self-terms (diagonal)
        if self.n1 == self.n2:
            np.fill_diagonal(G, 0)

        # Multiply by face areas for integration
        G = G * self.p2.area[np.newaxis, :]

        self._G = G
        return self._G

    def F(self, k: complex = None) -> np.ndarray:
        """
        Compute surface derivative of retarded Green function.

        F[i, j] = n_j . grad_j G(r_i, r_j) * area[j]

        For retarded Green function:
        grad G = (i*k - 1/r) * exp(i*k*r) / (4*pi*r) * (r_j - r_i) / r

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

        # Avoid division by zero
        d_safe = np.where(d == 0, np.inf, d)

        # grad G = (i*k - 1/r) * G * r_hat
        # where r_hat = (r_j - r_i) / r
        exp_ikr = np.exp(1j * k * d_safe)
        factor = (1j * k - 1.0 / d_safe) * exp_ikr / (4 * np.pi * d_safe)

        # Normal vectors at target faces
        nvec = self.p2.nvec  # (n2, 3)

        # n . dr / r
        n_dot_dr = np.sum(dr * nvec[np.newaxis, :, :], axis=2)
        n_dot_rhat = n_dot_dr / d_safe

        F = factor * n_dot_rhat

        # Handle self-terms
        if self.n1 == self.n2:
            np.fill_diagonal(F, 0)

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

        Gp[i, j, :] = grad_j G(r_i, r_j) * area[j]

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

        # Avoid division by zero
        d_safe = np.where(d == 0, np.inf, d)

        # grad G = (i*k - 1/r) * exp(i*k*r) / (4*pi*r) * r_hat
        exp_ikr = np.exp(1j * k * d_safe)
        factor = (1j * k - 1.0 / d_safe) * exp_ikr / (4 * np.pi * d_safe ** 2)

        # Gp[i, j, :] = factor[i,j] * dr[i,j,:]
        Gp = factor[:, :, np.newaxis] * dr

        # Handle self-terms
        if self.n1 == self.n2:
            for i in range(min(self.n1, self.n2)):
                Gp[i, i, :] = 0

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
        d_safe = np.where(d == 0, np.inf, d)

        # Unit vectors
        r_hat = dr / d_safe[:, :, np.newaxis]

        # Scalar Green function
        exp_ikr = np.exp(1j * k * d_safe)
        g0 = exp_ikr / (4 * np.pi * d_safe)

        # Dyadic Green function components
        # G_ij = g0 * [(1 + i/(kr) - 1/(kr)^2) * delta_ij
        #              + (-1 - 3i/(kr) + 3/(kr)^2) * r_i * r_j / r^2]
        kr = k * d_safe
        kr_safe = np.where(kr == 0, np.inf, kr)

        coef1 = 1 + 1j / kr_safe - 1 / kr_safe ** 2
        coef2 = -1 - 3j / kr_safe + 3 / kr_safe ** 2

        # Build dyadic
        L = np.zeros((self.n1, self.n2, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                L[:, :, i, j] = g0 * (coef1 * delta_ij + coef2 * r_hat[:, :, i] * r_hat[:, :, j])

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
    planar interfaces using Fresnel coefficients.

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
    """

    def __init__(
        self,
        p1: Union[Particle, Point],
        p2: Particle,
        layer,
        k: complex = None,
        **kwargs
    ):
        super().__init__(p1, p2, k, **kwargs)
        self.layer = layer
        self._G_refl = None

    def G_reflected(self, k: complex = None) -> np.ndarray:
        """
        Compute reflected Green function from layer interfaces.

        Parameters
        ----------
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Reflected Green function contribution.
        """
        if k is not None:
            self.set_k(k)
        k = self.k

        if self._G_refl is not None and self._cached_k == k:
            return self._G_refl

        # Get layer parameters
        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0

        # Source and target positions
        pos1 = self.pos1
        pos2 = self.pos2

        # Mirror image positions (reflection about z=z_interface)
        pos1_mirror = pos1.copy()
        pos1_mirror[:, 2] = 2 * z_interface - pos1[:, 2]

        # Distance from mirror sources to targets
        dr = pos2[np.newaxis, :, :] - pos1_mirror[:, np.newaxis, :]
        d = np.linalg.norm(dr, axis=2)
        d_safe = np.where(d == 0, np.inf, d)

        # Fresnel reflection coefficient (simplified for normal incidence)
        if len(self.layer.eps) >= 2:
            eps1, eps2 = self.layer.eps[0], self.layer.eps[1]
            n1 = np.sqrt(eps1) if np.isscalar(eps1) else np.sqrt(eps1(2*np.pi/k)[0])
            n2 = np.sqrt(eps2) if np.isscalar(eps2) else np.sqrt(eps2(2*np.pi/k)[0])
            r_fresnel = (n1 - n2) / (n1 + n2)
        else:
            r_fresnel = 0

        # Reflected Green function
        G_refl = r_fresnel * np.exp(1j * k * d_safe) / (4 * np.pi * d_safe)
        G_refl = G_refl * self.p2.area[np.newaxis, :]

        self._G_refl = G_refl
        return self._G_refl

    def G(self, k: complex = None) -> np.ndarray:
        """Total Green function including reflection."""
        return super().G(k) + self.G_reflected(k)

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        super().clear_cache()
        self._G_refl = None
