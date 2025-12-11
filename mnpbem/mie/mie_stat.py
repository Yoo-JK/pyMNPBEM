"""
Quasistatic Mie theory for spherical particles.
"""

import numpy as np
from typing import Any, Union, Tuple

from .spherical_harmonics import (
    sphtable, adipole, dipole_from_coeffs, field_from_coeffs
)


class MieStat:
    """
    Mie theory for spherical particles in quasistatic approximation.

    The quasistatic (dipole) approximation is valid when the particle
    is much smaller than the wavelength (diameter << lambda).

    In this limit, the polarizability is:
        alpha = (eps_in - eps_out) / (eps_in + 2*eps_out) * a^3

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside the sphere.
    epsout : dielectric function
        Dielectric function outside the sphere.
    diameter : float
        Diameter of the sphere in nm.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, MieStat
    >>> eps_out = EpsConst(1.0)  # vacuum
    >>> eps_in = EpsTable('gold.dat')
    >>> mie = MieStat(eps_in, eps_out, 10)  # 10 nm gold sphere
    >>> sca = mie.sca(np.linspace(400, 800, 100))
    """

    def __init__(
        self,
        epsin: Any,
        epsout: Any,
        diameter: float,
        **kwargs
    ):
        """
        Initialize quasistatic Mie solver.

        Parameters
        ----------
        epsin : dielectric function
            Dielectric function inside sphere.
        epsout : dielectric function
            Dielectric function outside sphere.
        diameter : float
            Sphere diameter in nm.
        """
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        self.options = kwargs

    @property
    def radius(self) -> float:
        """Sphere radius."""
        return self.diameter / 2

    def polarizability(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute dipole polarizability.

        alpha = (eps_in - eps_out) / (eps_in + 2*eps_out) * a^3

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Complex polarizability.
        """
        enei = np.atleast_1d(enei)

        # Get dielectric functions
        eps_in, _ = self.epsin(enei)
        eps_out, _ = self.epsout(enei)

        # Polarizability (Clausius-Mossotti)
        a = self.radius
        alpha = (eps_in - eps_out) / (eps_in + 2 * eps_out) * a ** 3

        return alpha

    def scattering(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute scattering cross section.

        sca = (8*pi/3) * k^4 * |alpha|^2

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Scattering cross section in nm^2.
        """
        return self.sca(enei)

    def sca(self, enei: np.ndarray) -> np.ndarray:
        """Alias for scattering."""
        enei = np.atleast_1d(enei)

        # Get background dielectric and wavenumber
        eps_out, k = self.epsout(enei)

        # Polarizability
        alpha = self.polarizability(enei)

        # Scattering cross section
        sca = (8 * np.pi / 3) * np.real(k) ** 4 * np.abs(alpha) ** 2

        return np.real(sca)

    def absorption(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute absorption cross section.

        abs = 4*pi*k * Im(alpha)

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Absorption cross section in nm^2.
        """
        return self.abs(enei)

    def abs(self, enei: np.ndarray) -> np.ndarray:
        """Alias for absorption."""
        enei = np.atleast_1d(enei)

        # Get wavenumber
        _, k = self.epsout(enei)

        # Polarizability
        alpha = self.polarizability(enei)

        # Absorption cross section
        abs_cs = 4 * np.pi * np.real(k) * np.imag(alpha)

        return np.real(abs_cs)

    def extinction(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute extinction cross section.

        ext = sca + abs

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Extinction cross section in nm^2.
        """
        return self.ext(enei)

    def ext(self, enei: np.ndarray) -> np.ndarray:
        """Alias for extinction."""
        return self.sca(enei) + self.abs(enei)

    def decay_rate(self, enei: float, z: np.ndarray, l_max: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Total and radiative decay rate for oscillating dipole near sphere.

        Scattering rates are given in units of the free-space decay rate.

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum (single value).
        z : ndarray
            Dipole positions along z axis in nm.
        l_max : int
            Maximum spherical harmonic degree.

        Returns
        -------
        tot : ndarray
            Total scattering rate for dipole orientations x and z, shape (len(z), 2).
        rad : ndarray
            Radiative scattering rate for dipole orientations x and z, shape (len(z), 2).
        """
        enei = float(enei)
        z = np.atleast_1d(z)

        # Background dielectric function
        epsb, _ = self.epsout(enei)
        nb = np.sqrt(epsb)

        # Free space radiative lifetime (Wigner-Weisskopf)
        gamma_ww = 4 / 3 * np.real(nb) * (2 * np.pi / enei) ** 3

        # Ratio of dielectric functions
        eps_in, _ = self.epsin(enei)
        epsz = eps_in / epsb

        # Generate spherical harmonic tables
        ltab, mtab = sphtable(l_max, 'full')

        # Total and radiative scattering rate
        tot = np.zeros((len(z), 2))
        rad = np.zeros((len(z), 2))

        # Dipole orientations: x and z
        dip_orientations = [
            np.array([1.0, 0.0, 0.0]),  # x
            np.array([0.0, 0.0, 1.0])   # z
        ]

        # Loop over dipole positions and orientations
        for iz, z_pos in enumerate(z):
            for idip, dip in enumerate(dip_orientations):
                # Position of dipole (normalized by diameter)
                pos = np.array([0.0, 0.0, z_pos]) / self.diameter

                # Spherical harmonics coefficients for dipole
                adip = adipole(pos, dip, ltab, mtab)

                # Induced dipole moment of sphere
                indip = dipole_from_coeffs(ltab, mtab, adip, 1.0, epsz)

                # Radiative decay rate (in units of free decay rate)
                rad[iz, idip] = np.linalg.norm(dip + indip) ** 2

                # Create vertices on sphere surface for field computation
                # Use a single point in the direction of the dipole
                r_surf = self.diameter / 2
                verts = pos * self.diameter
                if np.linalg.norm(verts) < 1e-10:
                    verts = np.array([[0, 0, r_surf]])
                else:
                    verts = verts.reshape(1, 3)

                # Induced electric field
                efield, _ = field_from_coeffs(ltab, mtab, verts, adip, 1.0, epsz)
                efield = efield[0] / (np.real(epsb) * self.diameter ** 3)

                # Total decay rate
                tot[iz, idip] = rad[iz, idip] + np.imag(np.dot(efield, dip)) / (gamma_ww / 2)

        return tot, rad

    def loss(self, enei: np.ndarray) -> np.ndarray:
        """
        Energy loss probability (for EELS) in quasistatic limit.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Energy loss probability (proportional to extinction).
        """
        # In quasistatic limit, loss is proportional to absorption
        return self.abs(enei)

    def __repr__(self) -> str:
        return f"MieStat(diameter={self.diameter})"
