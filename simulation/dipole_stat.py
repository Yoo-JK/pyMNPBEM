"""
Dipole excitation for quasistatic BEM.
"""

import numpy as np
from typing import Optional, Union

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions


class DipoleStat:
    """
    Dipole excitation within quasistatic approximation.

    A point dipole creates a potential:
        phi(r) = (1/4*pi*eps) * p . (r - r0) / |r - r0|^3

    Parameters
    ----------
    pt : ndarray
        Dipole positions, shape (n_dip, 3) or (3,).
    dip : ndarray
        Dipole moments, shape (n_dip, 3) or (3,).
    medium : int
        Index of medium containing the dipole.

    Examples
    --------
    >>> # Single dipole at origin, pointing in z
    >>> exc = DipoleStat([0, 0, 0], [0, 0, 1])
    """

    def __init__(
        self,
        pt: np.ndarray,
        dip: np.ndarray,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize dipole excitation.

        Parameters
        ----------
        pt : ndarray
            Dipole positions.
        dip : ndarray
            Dipole moments.
        medium : int
            Index of excitation medium (1-based).
        """
        self.pt = np.atleast_2d(pt)
        self.dip = np.atleast_2d(dip)
        self.medium = medium
        self.options = kwargs

        if len(self.pt) != len(self.dip):
            raise ValueError("Number of positions and dipoles must match")

    @property
    def n_dip(self) -> int:
        """Number of dipoles."""
        return len(self.pt)

    def __call__(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute external potential for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        return self.potential(p, enei)

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential derivative at particle boundary.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        pos = p.pos  # Face centroids (n_faces, 3)
        nvec = p.nvec  # Normal vectors (n_faces, 3)

        phip = np.zeros((p.n_faces, self.n_dip))

        for i, (r0, dip) in enumerate(zip(self.pt, self.dip)):
            # Distance vector: r - r0
            dr = pos - r0  # (n_faces, 3)
            r = np.linalg.norm(dr, axis=1)  # (n_faces,)

            # Avoid division by zero
            r = np.where(r == 0, np.inf, r)
            r3 = r ** 3
            r5 = r ** 5

            # Dipole potential gradient dotted with normal
            # phi = (1/4pi) * p . (r-r0) / r^3
            # grad phi = (1/4pi) * [p/r^3 - 3*(p.(r-r0))*(r-r0)/r^5]
            # phip = grad phi . n

            p_dot_dr = np.sum(dip * dr, axis=1)  # (n_faces,)
            term1 = np.sum(dip * nvec, axis=1) / r3
            term2 = 3 * p_dot_dr * np.sum(dr * nvec, axis=1) / r5

            phip[:, i] = (1 / (4 * np.pi)) * (term1 - term2)

        return CompStruct(p, enei, phip=phip)

    def field(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute electric field from dipole excitation.

        Based on Jackson Eq. (4.13):
        E = (3*r_hat*(r_hat.p) - p) / (4*pi*eps*r^3)

        Parameters
        ----------
        p : ComParticle
            Particle or points where field is computed.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Object containing electric field 'e' with shape
            (n_pos, 3, n_dip, n_orient) for multiple dipole orientations.
        """
        pos = p.pos if hasattr(p, 'pos') else np.atleast_2d(p)
        n_pos = len(pos)

        # Get dielectric function of embedding medium
        eps_medium = 1.0
        if hasattr(p, 'eps') and self.medium <= len(p.eps):
            eps_val = p.eps[self.medium - 1]
            if callable(eps_val):
                eps_medium = eps_val(enei)
            else:
                eps_medium = eps_val

        # Handle dipole orientations - can be (n_dip, 3) or (n_dip, 3, n_orient)
        dip = self.dip
        if dip.ndim == 2:
            dip = dip[:, :, np.newaxis]  # (n_dip, 3, 1)
        n_orient = dip.shape[2]

        # Allocate output: (n_pos, 3, n_dip, n_orient)
        e = np.zeros((n_pos, 3, self.n_dip, n_orient))

        for i in range(self.n_dip):
            # Distance vector from dipole to field point
            dr = pos - self.pt[i]  # (n_pos, 3)
            r = np.linalg.norm(dr, axis=1)  # (n_pos,)

            # Avoid division by zero
            r = np.where(r < 1e-10, 1e-10, r)
            r3 = r ** 3

            # Normalized distance vector
            r_hat = dr / r[:, np.newaxis]  # (n_pos, 3)

            for j in range(n_orient):
                d = dip[i, :, j]  # dipole moment (3,)

                # Inner product r_hat . d
                r_dot_d = np.sum(r_hat * d, axis=1)  # (n_pos,)

                # Electric field: E = (3*r_hat*(r_hat.p) - p) / (4*pi*eps*r^3)
                # Note: screening by dielectric function of embedding medium
                e[:, 0, i, j] = (3 * r_hat[:, 0] * r_dot_d - d[0]) / (4 * np.pi * eps_medium * r3)
                e[:, 1, i, j] = (3 * r_hat[:, 1] * r_dot_d - d[1]) / (4 * np.pi * eps_medium * r3)
                e[:, 2, i, j] = (3 * r_hat[:, 2] * r_dot_d - d[2]) / (4 * np.pi * eps_medium * r3)

        # Squeeze if only one orientation
        if n_orient == 1:
            e = e.squeeze(axis=-1)

        return CompStruct(p, enei, e=e)

    def farfield(self, spec, enei: float) -> CompStruct:
        """
        Compute far-field electromagnetic fields from dipoles.

        Parameters
        ----------
        spec : object
            Spectrum object with pinfty (sphere at infinity) and medium index.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Object containing far-field 'e' and 'h' fields.
        """
        # Get directions at infinity
        if hasattr(spec, 'pinfty'):
            pinfty = spec.pinfty
            directions = pinfty.nvec if hasattr(pinfty, 'nvec') else pinfty
        else:
            directions = np.atleast_2d(spec)

        n_dir = len(directions)

        # Normalize directions
        dir_norm = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(dir_norm < 1e-10, 1, dir_norm)

        # Get medium properties
        medium = getattr(spec, 'medium', self.medium)
        eps_medium = 1.0
        if hasattr(spec, 'pinfty') and hasattr(spec.pinfty, 'eps'):
            eps_tab = spec.pinfty.eps
            if medium <= len(eps_tab):
                eps_val = eps_tab[medium - 1]
                if callable(eps_val):
                    eps_medium = eps_val(enei)
                else:
                    eps_medium = eps_val

        # Wave number and refractive index
        k = 2 * np.pi * np.sqrt(eps_medium) / enei
        nb = np.sqrt(eps_medium)

        # Handle dipole orientations
        dip = self.dip
        if dip.ndim == 2:
            dip = dip[:, :, np.newaxis]
        n_orient = dip.shape[2]

        # Screen dipoles by dielectric environment
        eps_dip = np.ones(self.n_dip)
        # Approximate screening: dip_screened = eps_medium / eps_local * dip

        # Allocate far-fields
        e = np.zeros((n_dir, 3, self.n_dip, n_orient), dtype=complex)
        h = np.zeros((n_dir, 3, self.n_dip, n_orient), dtype=complex)

        for i in range(self.n_dip):
            # Green function for k*r -> infinity: exp(-i*k*r_hat.r0)
            # where r0 is dipole position
            phase = np.exp(-1j * k * np.dot(directions, self.pt[i]))  # (n_dir,)

            for j in range(n_orient):
                d = dip[i, :, j]  # (3,)

                # Far-field amplitude
                # E ~ k^2 * (n x p) x n * G = k^2 * [p - n*(n.p)] * G
                # H ~ k^2 * (n x p) * G

                for l, ndir in enumerate(directions):
                    n_cross_p = np.cross(ndir, d)
                    e_ff = np.cross(n_cross_p, ndir)

                    e[l, :, i, j] = k**2 * e_ff * phase[l] / eps_medium
                    h[l, :, i, j] = k**2 * n_cross_p * phase[l] / nb

        # Squeeze if only one orientation
        if n_orient == 1:
            e = e.squeeze(axis=-1)
            h = h.squeeze(axis=-1)

        # Create result
        if hasattr(spec, 'pinfty'):
            result = CompStruct(spec.pinfty, enei, e=e, h=h)
        else:
            result = CompStruct(None, enei, e=e, h=h)

        return result

    def decay_rate(self, sig: CompStruct) -> np.ndarray:
        """
        Compute radiative decay rate enhancement.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Decay rate enhancement for each dipole.
        """
        # Compute electric field at dipole positions from surface charges
        area = sig.p.area
        pos = sig.p.pos
        charges = sig.get('sig')

        gamma = np.zeros(self.n_dip)

        for i, (r0, dip) in enumerate(zip(self.pt, self.dip)):
            # Distance from surface elements to dipole
            dr = r0 - pos  # (n_faces, 3)
            r = np.linalg.norm(dr, axis=1)
            r = np.where(r == 0, np.inf, r)
            r3 = r ** 3

            # Electric field from surface charges
            # E = sum_j sigma_j * (r0 - pos_j) / (4*pi*r^3) * area_j
            sig_vals = charges[:, i] if charges.ndim > 1 else charges
            E = np.sum(sig_vals[:, np.newaxis] * dr / (4 * np.pi * r3[:, np.newaxis]) * area[:, np.newaxis], axis=0)

            # Decay rate proportional to Im(dip . E)
            gamma[i] = np.imag(np.sum(dip * E))

        return gamma

    def __repr__(self) -> str:
        return f"DipoleStat(n_dip={self.n_dip}, medium={self.medium})"


def dipole(
    pt: np.ndarray,
    dip: np.ndarray,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> DipoleStat:
    """
    Factory function for dipole excitation.

    Parameters
    ----------
    pt : ndarray
        Dipole positions.
    dip : ndarray
        Dipole moments.
    options : BEMOptions or dict, optional
        Simulation options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    DipoleStat
        Dipole excitation object.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}
    medium = all_options.pop('medium', 1)

    return DipoleStat(pt, dip, medium=medium, **all_options)
