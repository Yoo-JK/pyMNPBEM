"""
Retarded Mie theory for spherical particles.

Full Mie solution including all multipole orders for arbitrary
particle sizes.
"""

import numpy as np
from scipy import special
from typing import Any, Union, Tuple, Optional

from .spherical_harmonics import (
    mie_coefficients, mie_efficiencies, sphtable, aeels,
    riccati_bessel, BOHR, HARTREE, FINE_STRUCTURE
)


class MieRet:
    """
    Full Mie theory for spherical particles.

    Valid for any particle size, includes all multipole contributions.

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside the sphere.
    epsout : dielectric function
        Dielectric function outside the sphere.
    diameter : float
        Diameter of the sphere in nm.
    n_max : int, optional
        Maximum multipole order. If None, determined automatically.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, MieRet
    >>> eps_out = EpsConst(1.0)  # vacuum
    >>> eps_in = EpsTable('gold.dat')
    >>> mie = MieRet(eps_in, eps_out, 100)  # 100 nm gold sphere
    >>> sca = mie.sca(np.linspace(400, 800, 100))
    """

    def __init__(
        self,
        epsin: Any,
        epsout: Any,
        diameter: float,
        n_max: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Mie solver.

        Parameters
        ----------
        epsin : dielectric function
            Dielectric function inside sphere.
        epsout : dielectric function
            Dielectric function outside sphere.
        diameter : float
            Sphere diameter in nm.
        n_max : int, optional
            Maximum multipole order.
        """
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        self.n_max = n_max
        self.options = kwargs

    @property
    def radius(self) -> float:
        """Sphere radius in nm."""
        return self.diameter / 2

    def size_parameter(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute size parameter x = k * a = 2*pi*n*a/lambda.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Size parameter.
        """
        enei = np.atleast_1d(enei)
        _, k = self.epsout(enei)
        return np.real(k) * self.radius

    def relative_index(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute relative refractive index m = n_in / n_out.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Complex relative refractive index.
        """
        enei = np.atleast_1d(enei)

        eps_in, _ = self.epsin(enei)
        eps_out, _ = self.epsout(enei)

        n_in = np.sqrt(eps_in)
        n_out = np.sqrt(eps_out)

        return n_in / n_out

    def coefficients(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mie coefficients a_n and b_n.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        a_n : ndarray
            Electric multipole coefficients, shape (n_wavelengths, n_max).
        b_n : ndarray
            Magnetic multipole coefficients, shape (n_wavelengths, n_max).
        """
        enei = np.atleast_1d(enei)
        n_wl = len(enei)

        x = self.size_parameter(enei)
        m = self.relative_index(enei)

        # Determine n_max
        if self.n_max is None:
            n_max = int(np.max(x) + 4 * np.max(x) ** (1 / 3) + 2)
            n_max = max(n_max, 10)
        else:
            n_max = self.n_max

        a_n_all = np.zeros((n_wl, n_max), dtype=complex)
        b_n_all = np.zeros((n_wl, n_max), dtype=complex)

        for i in range(n_wl):
            a_n, b_n = mie_coefficients(m[i], x[i], n_max)
            a_n_all[i] = a_n
            b_n_all[i] = b_n

        return a_n_all, b_n_all

    def efficiencies(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Mie efficiency factors.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        Q_ext : ndarray
            Extinction efficiency.
        Q_sca : ndarray
            Scattering efficiency.
        Q_abs : ndarray
            Absorption efficiency.
        """
        enei = np.atleast_1d(enei)
        n_wl = len(enei)

        x = self.size_parameter(enei)
        m = self.relative_index(enei)

        Q_ext = np.zeros(n_wl)
        Q_sca = np.zeros(n_wl)
        Q_abs = np.zeros(n_wl)

        for i in range(n_wl):
            Q_ext[i], Q_sca[i], Q_abs[i] = mie_efficiencies(m[i], x[i], self.n_max)

        return Q_ext, Q_sca, Q_abs

    def scattering(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute scattering cross section.

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
        """Scattering cross section."""
        enei = np.atleast_1d(enei)
        _, Q_sca, _ = self.efficiencies(enei)

        # Geometric cross section
        geo_cs = np.pi * self.radius ** 2

        return Q_sca * geo_cs

    def absorption(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute absorption cross section.

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
        """Absorption cross section."""
        enei = np.atleast_1d(enei)
        _, _, Q_abs = self.efficiencies(enei)

        geo_cs = np.pi * self.radius ** 2

        return Q_abs * geo_cs

    def extinction(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute extinction cross section.

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
        """Extinction cross section."""
        enei = np.atleast_1d(enei)
        Q_ext, _, _ = self.efficiencies(enei)

        geo_cs = np.pi * self.radius ** 2

        return Q_ext * geo_cs

    def decay_rate(self, enei: float, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Total and radiative decay rate for oscillating dipole near sphere.

        Scattering rates are given in units of the free-space decay rate.
        See Kim et al., Surf. Science 195, 1 (1988).

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum (single value).
        z : ndarray
            Dipole positions along z axis in nm.

        Returns
        -------
        tot : ndarray
            Total scattering rate for dipole orientations x and z, shape (len(z), 2).
        rad : ndarray
            Radiative scattering rate for dipole orientations x and z, shape (len(z), 2).
        """
        enei = float(enei)
        z = np.atleast_1d(z)

        # Dielectric functions
        epsb, k = self.epsout(np.array([enei]))
        epsb = epsb[0]
        k = k[0]
        eps_in, k_in = self.epsin(np.array([enei]))
        eps_in = eps_in[0]
        k_in = k_in[0]

        # Total and radiative scattering rate
        tot = np.zeros((len(z), 2))
        rad = np.zeros((len(z), 2))

        # Use unique spherical harmonic degrees
        if self.n_max is None:
            x = np.real(k) * self.radius
            l_max = int(x + 4 * x ** (1 / 3) + 2)
            l_max = max(l_max, 10)
        else:
            l_max = self.n_max

        l_values = np.arange(1, l_max + 1)

        # Compute Riccati-Bessel functions for Mie coefficients
        j1, h1, zjp1, zhp1 = riccati_bessel(0.5 * k * self.diameter, l_values)
        j2, _, zjp2, _ = riccati_bessel(0.5 * k_in * self.diameter, l_values)

        # Modified Mie coefficients [Eq. (11)]
        A = (j1 * zjp2 - j2 * zjp1) / (j2 * zhp1 - h1 * zjp2)
        B = (epsb * j1 * zjp2 - eps_in * j2 * zjp1) / (eps_in * j2 * zhp1 - epsb * h1 * zjp2)

        # Loop over dipole positions
        for iz, z_pos in enumerate(z):
            # Background wavenumber * dipole position
            y = k * z_pos

            # Get spherical Bessel and Hankel functions at position of dipole
            j, h, zjp, zhp = riccati_bessel(y, l_values)

            # Normalized nonradiative decay rates [Eq. (17, 19)]
            tot[iz, 0] = 1 + 1.5 * np.real(np.sum((l_values + 0.5) *
                (B * (zhp / y) ** 2 + A * h ** 2)))
            tot[iz, 1] = 1 + 1.5 * np.real(np.sum(
                (2 * l_values + 1) * l_values * (l_values + 1) * B * (h / y) ** 2))

            # Normalized radiative decay rates [Eq. (18, 20)]
            rad[iz, 0] = 0.75 * np.sum((2 * l_values + 1) *
                (np.abs(j + A * h) ** 2 + np.abs((zjp + B * zhp) / y) ** 2))
            rad[iz, 1] = 1.5 * np.sum((2 * l_values + 1) * l_values * (l_values + 1) *
                np.abs((j + B * h) / y) ** 2)

        return np.real(tot), np.real(rad)

    def loss(self, b: np.ndarray, enei: np.ndarray, beta: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Energy loss probability for fast electron in vicinity of dielectric sphere.

        See F. J. Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).

        Parameters
        ----------
        b : ndarray
            Impact parameter (distance from sphere surface) in nm.
        enei : ndarray
            Wavelength of light in vacuum in nm.
        beta : float
            Electron velocity in units of speed of light (default 0.7).

        Returns
        -------
        prob : ndarray
            EELS probability, see Eq. (29), shape (len(b), len(enei)).
        prad : ndarray
            Photon emission probability, Eq. (37), shape (len(b), len(enei)).
        """
        b = np.atleast_1d(b)
        enei = np.atleast_1d(enei)

        # Make sure electron trajectory does not penetrate sphere
        assert np.all(b > 0), "Impact parameter must be positive"

        # Add sphere radius to impact parameter
        b_total = 0.5 * self.diameter + b

        # Gamma factor
        gamma = 1 / np.sqrt(1 - beta ** 2)

        # EELS and photon loss probability
        prob = np.zeros((len(b), len(enei)))
        prad = np.zeros((len(b), len(enei)))

        # Determine l_max
        if self.n_max is None:
            x_max = np.max(2 * np.pi / enei) * self.radius
            l_max = int(x_max + 4 * x_max ** (1 / 3) + 10)
            l_max = max(l_max, 20)
        else:
            l_max = max(self.n_max, 20)

        # Spherical harmonics tables
        ltab, mtab = sphtable(l_max, 'full')

        # Spherical harmonics coefficients for EELS
        ce, cm = aeels(ltab, mtab, beta)

        for ien, wavelength in enumerate(enei):
            # Wavenumber of light in medium
            epsb, k = self.epsout(np.array([wavelength]))
            epsb = epsb[0]
            k = np.real(k[0])

            # Mie expressions only implemented for epsb = 1
            # For other media, we use effective values
            eps_in, _ = self.epsin(np.array([wavelength]))
            epsz = eps_in[0] / epsb

            # Get unique l values and compute Mie coefficients
            l_unique = np.unique(ltab)

            # Build te, tm arrays for all (l, m) combinations
            te = np.zeros(len(ltab), dtype=complex)
            tm = np.zeros(len(ltab), dtype=complex)

            for l in l_unique:
                l = int(l)
                # Mie coefficients for this l
                a_n, b_n = mie_coefficients(np.sqrt(epsz), k * self.radius, l)

                # a, b correspond to 1i * [tE, tM] of Garcia [Eqs. (20, 21)]
                te_l = 1j * a_n[l - 1] if l <= len(a_n) else 0
                tm_l = 1j * b_n[l - 1] if l <= len(b_n) else 0

                # Assign to all m values for this l
                idx_l = np.where(ltab == l)[0]
                te[idx_l] = te_l
                tm[idx_l] = tm_l

            for ib, b_val in enumerate(b_total):
                # Modified Bessel function K_m
                K = special.kv(np.abs(mtab), k * b_val / (beta * gamma))

                # Energy loss probability, Eq. (29)
                prob[ib, ien] = np.sum(K ** 2 * (cm * np.imag(tm) + ce * np.imag(te))) / k

                # Photon loss probability, Eq. (37)
                prad[ib, ien] = np.sum(K ** 2 * (cm * np.abs(tm) ** 2 + ce * np.abs(te) ** 2)) / k

        # Convert to units of (1/eV)
        prob = FINE_STRUCTURE ** 2 / (BOHR * HARTREE) * prob
        prad = FINE_STRUCTURE ** 2 / (BOHR * HARTREE) * prad

        return prob, prad

    def loss_simple(self, enei: np.ndarray) -> np.ndarray:
        """
        Simplified energy loss probability (proportional to extinction).

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Energy loss probability (proportional to extinction).
        """
        return self.ext(enei)

    def __repr__(self) -> str:
        return f"MieRet(diameter={self.diameter}, n_max={self.n_max})"
