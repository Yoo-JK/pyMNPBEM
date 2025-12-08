"""
Retarded Mie theory for spherical particles.

Full Mie solution including all multipole orders for arbitrary
particle sizes.
"""

import numpy as np
from typing import Any, Union, Tuple, Optional

from .spherical_harmonics import mie_coefficients, mie_efficiencies


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

    def decay_rate(self, enei: np.ndarray, r: float, orientation: str = 'radial') -> np.ndarray:
        """
        Compute decay rate enhancement for a dipole near the sphere.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        r : float
            Distance from sphere center in nm.
        orientation : str
            Dipole orientation: 'radial' or 'tangential'.

        Returns
        -------
        ndarray
            Decay rate enhancement factor.
        """
        enei = np.atleast_1d(enei)
        a_n, b_n = self.coefficients(enei)

        _, k = self.epsout(enei)
        kr = np.real(k) * r

        # This is a simplified version
        # Full implementation requires spherical Bessel functions
        gamma = np.ones(len(enei))

        n_max = a_n.shape[1]
        for n in range(1, n_max + 1):
            factor = (2 * n + 1) * n * (n + 1)

            if orientation == 'radial':
                gamma += factor * np.abs(a_n[:, n - 1]) ** 2 / kr ** (2 * n + 4)
            else:
                gamma += factor * (np.abs(a_n[:, n - 1]) ** 2 + np.abs(b_n[:, n - 1]) ** 2) / kr ** (2 * n + 4) / 4

        return np.real(gamma)

    def loss(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute energy loss probability (for EELS).

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Energy loss probability.
        """
        # Simplified model based on extinction
        return self.ext(enei)

    def __repr__(self) -> str:
        return f"MieRet(diameter={self.diameter}, n_max={self.n_max})"
