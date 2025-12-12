"""
Quasistatic Green function.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import sparse

from ..particles import Particle, Point


class GreenStat:
    """
    Green function for quasistatic approximation (Coulomb).

    The quasistatic Green function is G(r, r') = 1 / (4*pi*|r - r'|).

    Parameters
    ----------
    p1 : Particle or Point
        Source points/particle.
    p2 : Particle
        Target particle (boundary elements).
    rel_cutoff : float
        Cutoff parameter for face integration refinement.

    Attributes
    ----------
    p1 : Particle or Point
        Source points.
    p2 : Particle
        Target particle.
    """

    def __init__(
        self,
        p1: Union[Particle, Point],
        p2: Particle,
        rel_cutoff: float = 3.0,
        deriv: str = 'cart',
        **kwargs
    ):
        """
        Initialize quasistatic Green function.

        Parameters
        ----------
        p1 : Particle or Point
            Source points/particle.
        p2 : Particle
            Target particle.
        rel_cutoff : float
            Cutoff for integration refinement.
        deriv : str
            Derivative mode: 'cart' or 'norm'.
        """
        self.p1 = p1
        self.p2 = p2
        self.rel_cutoff = rel_cutoff
        self.deriv = deriv

        # Cache for computed matrices
        self._G = None
        self._F = None
        self._Gp = None

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

    def G(self) -> np.ndarray:
        """
        Compute Green function matrix.

        Following MATLAB greenstat/eval1.m:
        G[i, j] = 1 / |r_j - r_i| * area[j]

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor.

        Returns
        -------
        ndarray
            Green function matrix, shape (n1, n2).
        """
        if self._G is not None:
            return self._G

        d, _ = self._compute_distances()

        # Avoid division by zero on diagonal
        d = np.where(d == 0, np.inf, d)

        # Green function: 1 / r  (MATLAB convention, no 4*pi)
        G = 1.0 / d

        # Multiply by face areas for integration
        G = G * self.p2.area[np.newaxis, :]

        self._G = G
        return self._G

    def F(self) -> np.ndarray:
        """
        Compute surface derivative of Green function.

        Following MATLAB greenstat/eval1.m:
        F[i, j] = -n . (r1 - r2) / |r_j - r_i|^3 * area[j]

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor.
        Also note the sign: MATLAB uses -(n.(x,y,z)) where (x,y,z) = pos1 - pos2

        Returns
        -------
        ndarray
            Surface derivative matrix, shape (n1, n2).
        """
        if self._F is not None:
            return self._F

        d, dr = self._compute_distances()

        # Avoid division by zero
        d = np.where(d == 0, np.inf, d)
        d3 = d ** 3

        # Normal vectors at p1 (source) - MATLAB uses p1.nvec
        nvec = self.p1.nvec if hasattr(self.p1, 'nvec') else self.p2.nvec

        # MATLAB: F = -(in(x,1) + in(y,2) + in(z,3)) ./ d.^3 * area
        # where x = pos1 - pos2 = -dr
        # n . (-dr) / r^3 = -n . dr / r^3
        n_dot_dr = np.sum(nvec[:, np.newaxis, :] * dr, axis=2)
        F = -n_dot_dr / d3

        # Multiply by face areas
        F = F * self.p2.area[np.newaxis, :]

        self._F = F
        return self._F

    def H1(self) -> np.ndarray:
        """
        F + 2*pi (for inside boundary condition).

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F() + 2 * np.pi * np.eye(self.n1, self.n2)

    def H2(self) -> np.ndarray:
        """
        F - 2*pi (for outside boundary condition).

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F() - 2 * np.pi * np.eye(self.n1, self.n2)

    def Gp(self) -> np.ndarray:
        """
        Compute gradient of Green function.

        Following MATLAB greenstat/eval1.m:
        Gp[i, j, :] = -(r1 - r2) / |r_j - r_i|^3 * area[j]

        Note: MATLAB MNPBEM does NOT use the 1/(4*pi) factor.

        Returns
        -------
        ndarray
            Green function gradient, shape (n1, n2, 3).
        """
        if self._Gp is not None:
            return self._Gp

        d, dr = self._compute_distances()

        # Avoid division by zero
        d = np.where(d == 0, np.inf, d)
        d3 = d ** 3

        # MATLAB: Gp = -[x./d.^3, y./d.^3, z./d.^3] * area
        # where x = pos1 - pos2 = -dr
        # So Gp = -(-dr) / d^3 = dr / d^3
        Gp = -(-dr) / d3[:, :, np.newaxis]

        # Multiply by face areas
        Gp = Gp * self.p2.area[np.newaxis, :, np.newaxis]

        self._Gp = Gp
        return self._Gp

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute potential from surface charges.

        phi = G @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).

        Returns
        -------
        ndarray
            Potential at source points.
        """
        return self.G() @ sig

    def field(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute electric field from surface charges.

        E = -Gp @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).

        Returns
        -------
        ndarray
            Electric field at source points, shape (n1, 3) or (n1, 3, n_exc).
        """
        Gp = self.Gp()

        # Handle sig with shape (n2, 1) - squeeze to 1D for single excitation
        if sig.ndim == 2 and sig.shape[1] == 1:
            sig = sig.squeeze(axis=1)

        if sig.ndim == 1:
            # E[i, k] = -sum_j Gp[i, j, k] * sig[j]
            return -np.einsum('ijk,j->ik', Gp, sig)
        else:
            # Multiple excitations
            return -np.einsum('ijk,jl->ikl', Gp, sig)

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        self._G = None
        self._F = None
        self._Gp = None

    def __repr__(self) -> str:
        return f"GreenStat(n1={self.n1}, n2={self.n2})"
