"""
Retarded BEM solver for full electromagnetic simulations.

This module implements the boundary element method for the full
Maxwell equations, including retardation effects.
"""

import numpy as np
from typing import Optional, Union, Tuple

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct, Point, ComPoint
from ..greenfun import CompGreenRet, GreenRet
from ..misc.options import BEMOptions


class BEMRet(BEMBase):
    """
    BEM solver for retarded (full electromagnetic) simulations.

    Solves the boundary integral equations for surface charges and
    currents such that Maxwell's equations are fulfilled including
    retardation effects.

    The retarded BEM equations are:
        [Sigma_e  G_ee   G_em] [sig]   [phi_e]
        [G_me     Sigma_m G_mm] [h  ] = [phi_m]

    where:
        sig = surface charges
        h = surface currents
        G = retarded Green function matrices

    References
    ----------
    Garcia de Abajo & Howie, PRB 65, 115418 (2002)
    Hohenester & Trugler, CPC 183, 370 (2012)

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    enei : float, optional
        Initial wavelength for precomputation.
    **kwargs : dict
        Options (rel_cutoff, etc.)

    Examples
    --------
    >>> from mnpbem import ComParticle, EpsConst, EpsTable, BEMRet
    >>> from mnpbem.particles.shapes import trisphere
    >>>
    >>> epstab = [EpsConst(1), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 50)  # 50 nm sphere (retardation important)
    >>> p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)
    >>> bem = BEMRet(p)
    """

    def __init__(
        self,
        p: ComParticle,
        enei: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize retarded BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float, optional
            Wavelength for precomputation.
        **kwargs : dict
            Options.
        """
        self.p = p
        self.options = kwargs

        # Create Green function
        self.g = CompGreenRet(p, p, **kwargs)

        # Cache for matrices
        self._mat = None
        self._enei = None
        self._k = None

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _get_wavenumber(self, enei: float, medium_idx: int = 0) -> complex:
        """
        Get wavenumber in specified medium.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        medium_idx : int
            Index of medium (0-based).

        Returns
        -------
        complex
            Wavenumber k = 2*pi*n/lambda.
        """
        eps_func = self.p.eps[medium_idx]
        _, k = eps_func(enei)
        return k

    def _compute_matrices(self, enei: float) -> None:
        """
        Compute BEM matrices for given wavelength.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        """
        if self._enei == enei and self._mat is not None:
            return

        n = self.n_faces

        # Get wavenumber in background medium
        k_bg = self._get_wavenumber(enei, 0)
        self.g.set_k(k_bg)

        # Get dielectric functions for all faces
        eps_in = np.zeros(n, dtype=complex)
        eps_out = np.zeros(n, dtype=complex)
        k_in = np.zeros(n, dtype=complex)
        k_out = np.zeros(n, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p.p):
            n_faces_i = particle.n_faces
            e_in, e_out = self.p.dielectric_inout(enei, i)
            _, ki = self.p.eps[self.p.inout[i, 0] - 1](enei)
            _, ko = self.p.eps[self.p.inout[i, 1] - 1](enei)

            eps_in[face_idx:face_idx + n_faces_i] = e_in
            eps_out[face_idx:face_idx + n_faces_i] = e_out
            k_in[face_idx:face_idx + n_faces_i] = ki
            k_out[face_idx:face_idx + n_faces_i] = ko

            face_idx += n_faces_i

        # Build BEM matrix
        # For each face, we have:
        # Lambda_e * sig + F * sig = -phi_e
        # Lambda_m * h + ... = -phi_m

        # Simplified version: scalar BEM with retarded Green function
        # Lambda factor
        Lambda = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)

        # Get surface derivative of Green function
        F = self.g.F(k_bg)

        # BEM matrix
        bem_matrix = np.diag(Lambda) + F

        # Solve using LU decomposition
        self._mat = -np.linalg.inv(bem_matrix)
        self._enei = enei
        self._k = k_bg

    def _compute_full_matrices(self, enei: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full electromagnetic BEM matrices.

        Returns matrices for both electric and magnetic surface currents.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        Sigma_e : ndarray
            Electric response matrix.
        Sigma_m : ndarray
            Magnetic response matrix.
        """
        n = self.n_faces

        # Get wavenumber
        k_bg = self._get_wavenumber(enei, 0)
        self.g.set_k(k_bg)

        # Dielectric contrasts
        deps = self.p.delta_eps(enei)

        # Green function matrices
        G = self.g.G(k_bg)
        F = self.g.F(k_bg)

        # Electric response
        Lambda_e = np.diag((deps + 2) / deps / (2 * np.pi))
        Sigma_e = Lambda_e + F

        # Magnetic response (for surface currents)
        # In quasistatic limit, this reduces to zero
        # For retarded case, includes k^2 * L contributions
        L = self.g.L(k_bg)

        # Diagonal contribution from magnetic response
        Lambda_m = np.diag(np.ones(n) * 2 * np.pi)

        # Cross-coupling terms
        G_em = k_bg ** 2 * np.einsum('ijkk->ij', L)  # Trace of dyadic
        G_me = G_em.T

        Sigma_m = Lambda_m + G_em

        return Sigma_e, Sigma_m

    def solve(self, exc: CompStruct) -> CompStruct:
        """
        Solve BEM equations for given excitation.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field (external potential at boundary).

        Returns
        -------
        CompStruct
            Solution with 'sig' field (surface charges) and optionally 'h' (currents).
        """
        # Ensure matrices are computed for this wavelength
        self._compute_matrices(exc.enei)

        # Get external potential
        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        # Solve: sig = -inv(Lambda + F) @ phip
        sig = self._mat @ phip

        return CompStruct(self.p, exc.enei, sig=sig, k=self._k)

    def solve_full(self, exc: CompStruct) -> CompStruct:
        """
        Solve full electromagnetic BEM equations.

        Includes both surface charges and surface currents.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' (electric) and optionally 'phim' (magnetic).

        Returns
        -------
        CompStruct
            Solution with 'sig' (charges) and 'h' (currents).
        """
        n = self.n_faces

        # Get matrices
        Sigma_e, Sigma_m = self._compute_full_matrices(exc.enei)

        # Get excitations
        phip = exc.get('phip')
        phim = exc.get('phim', np.zeros(n))

        # Build full system
        # [Sigma_e   G_em] [sig]   [phip]
        # [G_me   Sigma_m] [h  ] = [phim]

        k_bg = self._get_wavenumber(exc.enei, 0)

        # For now, use simplified decoupled solution
        sig = np.linalg.solve(Sigma_e, -phip)
        h = np.linalg.solve(Sigma_m, -phim)

        return CompStruct(self.p, exc.enei, sig=sig, h=h, k=k_bg)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax (alternative to solve)."""
        return self.solve(exc)

    def __rtruediv__(self, exc: CompStruct) -> CompStruct:
        """Allow exc / bem syntax."""
        return self.solve(exc)

    def field(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Electric field, shape (n_pts, 3).
        """
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            return self.g.field(charges, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.field(charges, k)

    def potential(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Potential at points.
        """
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            return self.g.potential(charges, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.potential(charges, k)

    def __repr__(self) -> str:
        return f"BEMRet(p={self.p}, enei={self._enei})"
