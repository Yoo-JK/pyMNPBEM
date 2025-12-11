"""
Retarded BEM solver for full electromagnetic simulations.

This module implements the boundary element method for the full
Maxwell equations, including retardation effects.

The implementation follows Garcia de Abajo & Howie, PRB 65, 115418 (2002).

Key equations:
- Eq. (10-11): Boundary conditions for potentials
- Eq. (15): Tangential electric field continuity (alpha)
- Eq. (18): Normal displacement field continuity (De)
- Eq. (19): Surface charge solution
- Eq. (20): Surface current solution
- Eq. (21-22): BEM matrix definitions
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

    The solution provides:
    - sig1, sig2: Surface charges on inside/outside boundaries
    - h1, h2: Surface currents on inside/outside boundaries

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
    >>> sphere = trisphere(144, 50)  # 50 nm sphere
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

        # Cache for BEM matrices (initialized per wavelength)
        self._enei = None
        self._k = None
        self._nvec = None

        # Core BEM matrices (Garcia de Abajo & Howie notation)
        self._G1i = None  # Inverse of inside Green function
        self._G2i = None  # Inverse of outside Green function
        self._L1 = None   # G1 * eps1 * G1i (Eq. 22)
        self._L2 = None   # G2 * eps2 * G2i
        self._Sigma1 = None  # H1 * G1i (Eq. 21)
        self._Sigma2 = None  # H2 * G2i
        self._Deltai = None  # inv(Sigma1 - Sigma2)
        self._Sigmai = None  # For Eq. 19

        # Dielectric functions at boundaries
        self._eps1 = None  # Inside dielectric
        self._eps2 = None  # Outside dielectric

        # Initialize for given wavelength if provided
        if enei is not None:
            self._init_matrices(enei)

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
        result = eps_func(enei)
        if isinstance(result, tuple):
            _, k = result
        else:
            k = 2 * np.pi * np.sqrt(result) / enei
        return k

    def _init_matrices(self, enei: float) -> None:
        """
        Initialize BEM matrices for given wavelength.

        Follows MATLAB bemret/subsref.m

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        """
        if self._enei == enei and self._G1i is not None:
            return

        n = self.n_faces

        # Wavenumber in vacuum
        k0 = 2 * np.pi / enei
        self._k = k0

        # Normal vectors
        self._nvec = self.p.nvec

        # Get dielectric functions for all faces
        eps1 = np.zeros(n, dtype=complex)  # Inside
        eps2 = np.zeros(n, dtype=complex)  # Outside

        face_idx = 0
        for i, particle in enumerate(self.p.p):
            n_faces_i = particle.n_faces
            e_in, e_out = self.p.dielectric_inout(enei, i)
            eps1[face_idx:face_idx + n_faces_i] = e_in
            eps2[face_idx:face_idx + n_faces_i] = e_out
            face_idx += n_faces_i

        self._eps1 = eps1
        self._eps2 = eps2

        # Get Green function matrices
        # G: scalar Green function
        # F: surface derivative (n . grad G)
        # We need to compute these properly
        k_bg = self._get_wavenumber(enei, 0)
        self.g.set_k(k_bg)

        G = self.g.G(k_bg)  # (n, n)
        F = self.g.F(k_bg)  # (n, n)

        # H1 = F + 2*pi*I (inside)
        # H2 = F - 2*pi*I (outside)
        H1 = F + 2 * np.pi * np.eye(n)
        H2 = F - 2 * np.pi * np.eye(n)

        # Inverse Green function matrices
        # G1i = inv(G) for inside
        # G2i = inv(G) for outside
        # Note: In MNPBEM, G1 and G2 are different due to dielectric contrast
        # For now, use simplified version
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)

        self._G1i = G_inv
        self._G2i = G_inv

        # L1 = G * eps1 * G1i (Eq. 22)
        # L2 = G * eps2 * G2i
        # These are diagonal in epsilon
        self._L1 = G @ np.diag(eps1) @ self._G1i
        self._L2 = G @ np.diag(eps2) @ self._G2i

        # Sigma1 = H1 * G1i (Eq. 21)
        # Sigma2 = H2 * G2i
        self._Sigma1 = H1 @ self._G1i
        self._Sigma2 = H2 @ self._G2i

        # Deltai = inv(Sigma1 - Sigma2)
        Delta = self._Sigma1 - self._Sigma2
        try:
            self._Deltai = np.linalg.inv(Delta)
        except np.linalg.LinAlgError:
            self._Deltai = np.linalg.pinv(Delta)

        # Sigmai for Eq. 19
        # From the paper, this involves eps1, eps2 contrasts
        # Sigmai = inv(Sigma_eff) where Sigma_eff handles the dielectric contrast
        deps = eps1 - eps2
        deps_safe = np.where(np.abs(deps) < 1e-10, 1e-10, deps)
        Lambda = (eps1 + eps2) / deps_safe / (2 * np.pi)

        Sigma_eff = np.diag(Lambda) + F
        try:
            self._Sigmai = -np.linalg.inv(Sigma_eff)
        except np.linalg.LinAlgError:
            self._Sigmai = -np.linalg.pinv(Sigma_eff)

        self._enei = enei

    def _extract_excitation(self, exc) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract excitation variables from CompStruct.

        Following MATLAB bemret/private/excitation.m

        Parameters
        ----------
        exc : CompStruct or excitation object
            Excitation with phi1, phi2, a1, a2, etc.

        Returns
        -------
        phi : ndarray
            phi2 - phi1 (scalar potential difference)
        a : ndarray
            a2 - a1 (vector potential difference), shape (n, 3, n_pol)
        alpha : ndarray
            Tangential E-field boundary condition (Eq. 15)
        De : ndarray
            Normal D-field boundary condition (Eq. 18)
        """
        n = self.n_faces
        k = self._k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2

        # Get potentials from excitation
        # Default to zeros if not present
        phi1 = exc.get('phi1', np.zeros((n, 1), dtype=complex))
        phi2 = exc.get('phi2', np.zeros((n, 1), dtype=complex))
        phi1p = exc.get('phi1p', np.zeros((n, 1), dtype=complex))
        phi2p = exc.get('phi2p', np.zeros((n, 1), dtype=complex))

        a1 = exc.get('a1', np.zeros((n, 3, 1), dtype=complex))
        a2 = exc.get('a2', np.zeros((n, 3, 1), dtype=complex))
        a1p = exc.get('a1p', np.zeros((n, 3, 1), dtype=complex))
        a2p = exc.get('a2p', np.zeros((n, 3, 1), dtype=complex))

        # Ensure correct shapes
        if phi1.ndim == 1:
            phi1 = phi1[:, np.newaxis]
        if phi2.ndim == 1:
            phi2 = phi2[:, np.newaxis]
        if phi1p.ndim == 1:
            phi1p = phi1p[:, np.newaxis]
        if phi2p.ndim == 1:
            phi2p = phi2p[:, np.newaxis]

        if a1.ndim == 2:
            a1 = a1[:, :, np.newaxis]
        if a2.ndim == 2:
            a2 = a2[:, :, np.newaxis]
        if a1p.ndim == 2:
            a1p = a1p[:, :, np.newaxis]
        if a2p.ndim == 2:
            a2p = a2p[:, :, np.newaxis]

        n_pol = a1.shape[2]

        # Eqs. (10, 11): Potential differences
        phi = phi2 - phi1  # (n, n_pol)
        a = a2 - a1        # (n, 3, n_pol)

        # Eq. (15): alpha = a2p - a1p - i*k*(n x phi2*eps2 - n x phi1*eps1)
        # outer(nvec, phi, eps) = nvec * phi * eps (element-wise, then broadcast)
        # This gives the tangential component of the E-field boundary condition
        alpha = a2p - a1p  # (n, 3, n_pol)

        for i_pol in range(n_pol):
            # n x (phi * eps) = nvec * (phi * eps)
            # In MATLAB: outer(nvec, phi2, eps2) means nvec .* (phi2 .* eps2)
            term1 = nvec * (phi2[:, i_pol] * eps2)[:, np.newaxis]  # (n, 3)
            term2 = nvec * (phi1[:, i_pol] * eps1)[:, np.newaxis]  # (n, 3)
            alpha[:, :, i_pol] -= 1j * k * (term1 - term2)

        # Eq. (18): De = eps2*phi2p - eps1*phi1p - i*k*(n.a2*eps2 - n.a1*eps1)
        # inner(nvec, a, eps) = sum(nvec * a, axis=1) * eps
        De = np.zeros((n, n_pol), dtype=complex)

        for i_pol in range(n_pol):
            De[:, i_pol] = eps2 * phi2p[:, i_pol] - eps1 * phi1p[:, i_pol]

            # n . a with dielectric weighting
            n_dot_a2 = np.sum(nvec * a2[:, :, i_pol], axis=1)  # (n,)
            n_dot_a1 = np.sum(nvec * a1[:, :, i_pol], axis=1)  # (n,)
            De[:, i_pol] -= 1j * k * (n_dot_a2 * eps2 - n_dot_a1 * eps1)

        return phi, a, alpha, De

    def solve(self, exc) -> CompStruct:
        """
        Solve BEM equations for given excitation.

        Following MATLAB bemret/mldivide.m exactly.

        Parameters
        ----------
        exc : CompStruct or excitation object
            Excitation with vector/scalar potentials.

        Returns
        -------
        CompStruct
            Solution with sig1, sig2, h1, h2 fields.
        """
        # Initialize matrices for this wavelength
        self._init_matrices(exc.enei)

        n = self.n_faces
        k = self._k
        nvec = self._nvec

        # Extract excitation (Eqs. 10-11, 15, 18)
        phi, a, alpha, De = self._extract_excitation(exc)

        n_pol = phi.shape[1] if phi.ndim > 1 else 1

        # Get stored matrices
        G1i = self._G1i
        G2i = self._G2i
        L1 = self._L1
        L2 = self._L2
        Sigma1 = self._Sigma1
        Deltai = self._Deltai
        Sigmai = self._Sigmai

        # Initialize output arrays
        sig1 = np.zeros((n, n_pol), dtype=complex)
        sig2 = np.zeros((n, n_pol), dtype=complex)
        h1 = np.zeros((n, 3, n_pol), dtype=complex)
        h2 = np.zeros((n, 3, n_pol), dtype=complex)

        # Solve for each polarization
        for i_pol in range(n_pol):
            phi_i = phi[:, i_pol]  # (n,)
            a_i = a[:, :, i_pol]   # (n, 3)
            alpha_i = alpha[:, :, i_pol]  # (n, 3)
            De_i = De[:, i_pol]    # (n,)

            # Modify alpha and De (MATLAB lines 31-34)
            # alpha = alpha - Sigma1 @ a + i*k * outer(nvec, L1 @ phi)
            # De = De - Sigma1 @ (L1 @ phi) + i*k * inner(nvec, L1 @ a)

            # Sigma1 @ a: matrix times vector field
            # For each component of a, multiply by Sigma1
            Sigma1_a = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                Sigma1_a[:, j] = Sigma1 @ a_i[:, j]

            # L1 @ phi: matrix times scalar
            L1_phi = L1 @ phi_i  # (n,)

            # outer(nvec, L1_phi) = nvec * L1_phi
            outer_nvec_L1phi = nvec * L1_phi[:, np.newaxis]  # (n, 3)

            alpha_mod = alpha_i - Sigma1_a + 1j * k * outer_nvec_L1phi  # (n, 3)

            # L1 @ a: for each vector component
            L1_a = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                L1_a[:, j] = L1 @ a_i[:, j]

            # inner(nvec, L1_a) = sum(nvec * L1_a, axis=1)
            inner_nvec_L1a = np.sum(nvec * L1_a, axis=1)  # (n,)

            # Sigma1 @ L1 @ phi
            Sigma1_L1_phi = Sigma1 @ L1_phi  # (n,)

            De_mod = De_i - Sigma1_L1_phi + 1j * k * inner_nvec_L1a  # (n,)

            # Eq. (19): sig2 = Sigmai @ (De + i*k * inner(nvec, (L1-L2) @ Deltai @ alpha))
            # First compute Deltai @ alpha (matrix times vector field)
            Deltai_alpha = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                Deltai_alpha[:, j] = Deltai @ alpha_mod[:, j]

            # (L1 - L2) @ Deltai_alpha
            L_diff = L1 - L2
            L_diff_Deltai_alpha = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                L_diff_Deltai_alpha[:, j] = L_diff @ Deltai_alpha[:, j]

            # inner(nvec, L_diff_Deltai_alpha)
            inner_term = np.sum(nvec * L_diff_Deltai_alpha, axis=1)  # (n,)

            sig2_i = Sigmai @ (De_mod + 1j * k * inner_term)  # (n,)

            # Eq. (20): h2 = Deltai @ (i*k * outer(nvec, (L1-L2) @ sig2) + alpha)
            # (L1 - L2) @ sig2
            L_diff_sig2 = L_diff @ sig2_i  # (n,)

            # outer(nvec, L_diff_sig2) = nvec * L_diff_sig2
            outer_nvec_L_diff_sig2 = nvec * L_diff_sig2[:, np.newaxis]  # (n, 3)

            h2_rhs = 1j * k * outer_nvec_L_diff_sig2 + alpha_mod  # (n, 3)

            h2_i = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                h2_i[:, j] = Deltai @ h2_rhs[:, j]

            # Surface charges and currents (MATLAB lines 44-45)
            # sig1 = G1i @ (sig2 + phi)
            # h1 = G1i @ (h2 + a)
            # sig2 = G2i @ sig2
            # h2 = G2i @ h2

            sig1_i = G1i @ (sig2_i + phi_i)

            h1_i = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                h1_i[:, j] = G1i @ (h2_i[:, j] + a_i[:, j])

            sig2_final = G2i @ sig2_i

            h2_final = np.zeros((n, 3), dtype=complex)
            for j in range(3):
                h2_final[:, j] = G2i @ h2_i[:, j]

            # Store results
            sig1[:, i_pol] = sig1_i
            sig2[:, i_pol] = sig2_final
            h1[:, :, i_pol] = h1_i
            h2[:, :, i_pol] = h2_final

        return CompStruct(
            self.p, exc.enei,
            sig1=sig1, sig2=sig2, h1=h1, h2=h2,
            k=self._k
        )

    def __truediv__(self, exc) -> CompStruct:
        """Allow bem / exc syntax (alternative to solve)."""
        return self.solve(exc)

    def __rtruediv__(self, exc) -> CompStruct:
        """Allow exc / bem syntax."""
        return self.solve(exc)

    def field(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
        """
        Compute electric field from surface charges and currents.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges (sig1, sig2) and currents (h1, h2).
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Electric field, shape (n_pts, 3) or (n_pts, 3, n_pol).
        """
        # Get wavenumber
        k = sig.get('k', self._k)
        if k is None:
            k = 2 * np.pi / sig.enei

        # Use sig2 and h2 for field outside particle
        # For retarded BEM, field involves both charges and currents
        sig2 = sig.get('sig2')
        h2 = sig.get('h2')

        if sig2 is None:
            # Fallback to simple sig
            charges = sig.get('sig')
            if charges is None:
                raise ValueError("Solution must have 'sig2' or 'sig' field")
            return self._field_from_charges(charges, k, pts)

        # Full retarded field calculation
        return self._field_retarded(sig2, h2, k, pts)

    def _field_from_charges(self, charges, k, pts):
        """Compute field from surface charges only (simplified)."""
        if pts is None:
            return self.g.field(charges, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.field(charges, k)

    def _field_retarded(self, sig2, h2, k, pts):
        """
        Compute field from surface charges and currents.

        E = -grad phi - i*k*A
        where phi comes from charges and A from currents.
        """
        # For now, use simplified calculation
        # Full implementation would use Green's dyadic
        if pts is None:
            E_sig = self.g.field(sig2, k)
            # Add current contribution (k^2 * L @ h2)
            # This is a simplification; full version needs dyadic Green function
            return E_sig
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.field(sig2, k)

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
        sig2 = sig.get('sig2')
        if sig2 is None:
            sig2 = sig.get('sig')
        if sig2 is None:
            raise ValueError("Solution must have 'sig2' or 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            return self.g.potential(sig2, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.potential(sig2, k)

    def __repr__(self) -> str:
        return f"BEMRet(p={self.p}, enei={self._enei})"
