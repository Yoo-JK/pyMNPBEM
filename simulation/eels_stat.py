"""
Electron Energy Loss Spectroscopy (EELS) simulation.

Simulates energy loss of swift electrons passing near or through
metallic nanoparticles.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct, Point
from ..misc.options import BEMOptions
from ..misc.units import SPEED_OF_LIGHT, eV2nm


class EELSStat:
    """
    EELS simulation in quasistatic approximation.

    Computes the energy loss probability for an electron beam
    passing near or through a nanoparticle.

    The electron creates a time-dependent electric field that
    excites the particle's plasmon modes.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions (x, y) in nm.
    velocity : float
        Electron velocity (relative to speed of light, v/c).
        Typical TEM: 0.5-0.7 (100-300 keV).
    width : float, optional
        Beam width for extended beam simulation.

    Examples
    --------
    >>> from mnpbem import EELSStat
    >>> # Single beam position at x=15 nm, y=0
    >>> eels = EELSStat([15, 0], velocity=0.5)
    """

    def __init__(
        self,
        impact: np.ndarray,
        velocity: float = 0.5,
        width: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize EELS simulation.

        Parameters
        ----------
        impact : ndarray
            Impact parameter (x, y) positions in nm.
        velocity : float
            Electron velocity v/c (typically 0.5-0.7).
        width : float, optional
            Beam width.
        """
        self.impact = np.atleast_2d(impact)
        if self.impact.shape[1] == 2:
            # Add z=0 if only (x,y) given
            self.impact = np.column_stack([self.impact, np.zeros(len(self.impact))])

        self.velocity = velocity  # v/c
        self.width = width
        self.options = kwargs

        # Electron properties
        self.v = velocity * SPEED_OF_LIGHT  # nm/fs

    @property
    def n_beams(self) -> int:
        """Number of beam positions."""
        return len(self.impact)

    @property
    def gamma_lorentz(self) -> float:
        """Lorentz factor gamma = 1/sqrt(1 - v^2/c^2)."""
        return 1.0 / np.sqrt(1 - self.velocity ** 2)

    def __call__(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute external potential for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm (or energy if in eV).

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        return self.potential(p, enei)

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential from electron beam.

        The electron creates a time-dependent potential that,
        after Fourier transform, gives the excitation at frequency omega.

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
        nvec = p.nvec

        # Angular frequency
        omega = 2 * np.pi * SPEED_OF_LIGHT / enei

        phip = np.zeros((p.n_faces, self.n_beams), dtype=complex)

        for i, r_beam in enumerate(self.impact):
            # Distance from beam to face centroids
            # Beam travels along z-axis through (x0, y0)
            dx = pos[:, 0] - r_beam[0]
            dy = pos[:, 1] - r_beam[1]
            rho = np.sqrt(dx ** 2 + dy ** 2)  # Perpendicular distance
            z = pos[:, 2]  # Distance along beam

            # Avoid division by zero
            rho = np.where(rho < 1e-10, 1e-10, rho)

            # Electric field from electron (in frequency domain)
            # Using modified Bessel functions for retardation
            # In quasistatic limit, field is approximately:
            # E_rho ~ (2*omega/v^2) * K_1(omega*rho/v) * exp(i*omega*z/v)
            # where K_1 is modified Bessel function

            from scipy.special import kv

            arg = omega * rho / self.v / self.gamma_lorentz
            phase = np.exp(1j * omega * z / self.v)

            # Bessel function K_0 for potential
            K0 = kv(0, arg)
            K1 = kv(1, arg)

            # Potential at surfaces
            prefactor = 2 / (self.v * self.gamma_lorentz)
            phi = prefactor * K0 * phase

            # Normal derivative for BEM
            # d(phi)/dn = grad(phi) . n
            rho_hat = np.stack([dx / rho, dy / rho, np.zeros_like(rho)], axis=1)

            grad_phi_rho = -prefactor * omega / (self.v * self.gamma_lorentz) * K1 * phase
            grad_phi_z = 1j * omega / self.v * phi

            grad_phi = grad_phi_rho[:, np.newaxis] * rho_hat
            grad_phi[:, 2] += grad_phi_z

            phip[:, i] = np.sum(grad_phi * nvec, axis=1)

        return CompStruct(p, enei, phip=phip)

    def loss(self, sig: CompStruct) -> np.ndarray:
        """
        Compute energy loss probability.

        The loss probability is proportional to the work done by
        the induced field on the electron.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Energy loss probability for each beam position.
        """
        # Get induced potential at beam positions
        omega = 2 * np.pi * SPEED_OF_LIGHT / sig.enei

        charges = sig.get('sig')
        pos_surf = sig.p.pos
        area = sig.p.area

        loss = np.zeros(self.n_beams)

        for i, r_beam in enumerate(self.impact):
            # Integrate induced field along beam trajectory
            # Loss ~ Im[integral E_z(z) * exp(-i*omega*z/v) dz]

            sig_vals = charges[:, i] if charges.ndim > 1 else charges

            # Sample points along z
            z_min = pos_surf[:, 2].min() - 50
            z_max = pos_surf[:, 2].max() + 50
            z_points = np.linspace(z_min, z_max, 100)

            Ez_integral = 0j
            for z in z_points:
                # Distance from surface elements to point on beam
                dx = r_beam[0] - pos_surf[:, 0]
                dy = r_beam[1] - pos_surf[:, 1]
                dz = z - pos_surf[:, 2]
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                r = np.where(r < 1e-10, 1e-10, r)

                # Z-component of field from surface charges
                Ez = np.sum(sig_vals * dz / (4 * np.pi * r ** 3) * area)

                # Fourier integral weight
                phase = np.exp(-1j * omega * z / self.v)
                Ez_integral += Ez * phase

            dz = (z_max - z_min) / 100
            loss[i] = np.imag(Ez_integral * dz)

        return loss

    def loss_map(
        self,
        p: ComParticle,
        bem,
        enei: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute EELS loss probability map.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        bem : BEM solver
            BEM solver (BEMStat or BEMRet).
        enei : float
            Wavelength in nm.
        x_range : tuple
            (x_min, x_max) in nm.
        y_range : tuple
            (y_min, y_max) in nm.
        n_points : int
            Number of points per dimension.

        Returns
        -------
        x : ndarray
            X coordinates.
        y : ndarray
            Y coordinates.
        loss_map : ndarray
            Loss probability map, shape (n_points, n_points).
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        loss_map = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                eels = EELSStat([X[i, j], Y[i, j]], self.velocity)
                exc = eels(p, enei)
                sig = bem.solve(exc)
                loss_map[i, j] = eels.loss(sig)[0]

        return x, y, loss_map

    def __repr__(self) -> str:
        return f"EELSStat(n_beams={self.n_beams}, velocity={self.velocity})"


def eels(
    impact: np.ndarray,
    velocity: float = 0.5,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> EELSStat:
    """
    Factory function for EELS excitation.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions.
    velocity : float
        Electron velocity v/c.
    options : BEMOptions or dict, optional
        Options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    EELSStat
        EELS simulation object.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}

    return EELSStat(impact, velocity, **all_options)
