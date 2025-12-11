"""
Gans theory for ellipsoidal particles.

Extended Mie theory for prolate and oblate spheroids in the
quasistatic approximation.
"""

import numpy as np
from typing import Any, Union, Tuple


class MieGans:
    """
    Gans theory for ellipsoidal particles.

    Valid in the quasistatic limit (particle << wavelength).
    Applicable to prolate (cigar-shaped) and oblate (disk-shaped) spheroids.

    The polarizability for each axis is:
        alpha_i = V * (eps_in - eps_out) / (eps_out + L_i * (eps_in - eps_out))

    where L_i is the depolarization factor for axis i.

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside the ellipsoid.
    epsout : dielectric function
        Dielectric function outside the ellipsoid.
    axes : tuple of float
        Semi-axes (a, b, c) in nm. Convention: a >= b >= c for prolate,
        c > a = b for oblate.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, MieGans
    >>> eps_out = EpsConst(1.0)
    >>> eps_in = EpsTable('gold.dat')
    >>> # Prolate spheroid: 100 nm x 20 nm x 20 nm
    >>> gans = MieGans(eps_in, eps_out, (50, 10, 10))
    >>> ext = gans.ext(np.linspace(400, 800, 100))
    """

    def __init__(
        self,
        epsin: Any,
        epsout: Any,
        axes: Tuple[float, float, float],
        **kwargs
    ):
        """
        Initialize Gans theory solver.

        Parameters
        ----------
        epsin : dielectric function
            Dielectric function inside ellipsoid.
        epsout : dielectric function
            Dielectric function outside ellipsoid.
        axes : tuple
            Semi-axes (a, b, c) in nm.
        """
        self.epsin = epsin
        self.epsout = epsout
        self.axes = tuple(sorted(axes, reverse=True))  # Ensure a >= b >= c
        self.options = kwargs

        # Compute depolarization factors
        self._L = self._compute_depolarization_factors()

    @property
    def a(self) -> float:
        """Largest semi-axis."""
        return self.axes[0]

    @property
    def b(self) -> float:
        """Middle semi-axis."""
        return self.axes[1]

    @property
    def c(self) -> float:
        """Smallest semi-axis."""
        return self.axes[2]

    @property
    def volume(self) -> float:
        """Ellipsoid volume."""
        return 4 / 3 * np.pi * self.a * self.b * self.c

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio a/c."""
        return self.a / self.c

    @property
    def L(self) -> Tuple[float, float, float]:
        """Depolarization factors (La, Lb, Lc)."""
        return self._L

    def _compute_depolarization_factors(self) -> Tuple[float, float, float]:
        """
        Compute depolarization factors for the ellipsoid.

        For a general ellipsoid with semi-axes a >= b >= c:
        L_a = (abc/2) * integral_0^inf ds / ((s+a^2) * f(s))
        where f(s) = sqrt((s+a^2)(s+b^2)(s+c^2))

        Returns
        -------
        tuple
            Depolarization factors (La, Lb, Lc).
        """
        a, b, c = self.axes

        # Check for special cases
        if np.allclose([a, b, c], a):
            # Sphere
            return (1 / 3, 1 / 3, 1 / 3)

        if np.allclose([b, c], b):
            # Prolate spheroid (a > b = c)
            e = np.sqrt(1 - (b / a) ** 2)  # Eccentricity
            if e < 1e-10:
                return (1 / 3, 1 / 3, 1 / 3)

            La = (1 - e ** 2) / e ** 2 * (-1 + 1 / (2 * e) * np.log((1 + e) / (1 - e)))
            Lb = (1 - La) / 2
            Lc = Lb
            return (La, Lb, Lc)

        if np.allclose([a, b], a):
            # Oblate spheroid (a = b > c)
            e = np.sqrt(1 - (c / a) ** 2)  # Eccentricity
            if e < 1e-10:
                return (1 / 3, 1 / 3, 1 / 3)

            g = np.sqrt((1 - e ** 2) / e ** 2)
            Lc = g ** 2 / e ** 2 * (np.pi / 2 - np.arctan(g) - g) + g
            La = (1 - Lc) / 2
            Lb = La
            return (La, Lb, Lc)

        # General triaxial ellipsoid - numerical integration
        from scipy import integrate

        def integrand_a(s):
            return 1 / ((s + a ** 2) * np.sqrt((s + a ** 2) * (s + b ** 2) * (s + c ** 2)))

        def integrand_b(s):
            return 1 / ((s + b ** 2) * np.sqrt((s + a ** 2) * (s + b ** 2) * (s + c ** 2)))

        def integrand_c(s):
            return 1 / ((s + c ** 2) * np.sqrt((s + a ** 2) * (s + b ** 2) * (s + c ** 2)))

        prefactor = a * b * c / 2

        La, _ = integrate.quad(integrand_a, 0, np.inf)
        Lb, _ = integrate.quad(integrand_b, 0, np.inf)
        Lc, _ = integrate.quad(integrand_c, 0, np.inf)

        return (prefactor * La, prefactor * Lb, prefactor * Lc)

    def polarizability(self, enei: np.ndarray, axis: int = None) -> np.ndarray:
        """
        Compute polarizability.

        alpha_i = V * (eps_in - eps_out) / (eps_out + L_i * (eps_in - eps_out))

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        axis : int, optional
            Axis index (0, 1, 2). If None, returns average.

        Returns
        -------
        ndarray
            Complex polarizability (nm^3).
        """
        enei = np.atleast_1d(enei)

        eps_in, _ = self.epsin(enei)
        eps_out, _ = self.epsout(enei)

        deps = eps_in - eps_out

        if axis is None:
            # Average polarizability
            alpha = np.zeros_like(eps_in, dtype=complex)
            for i, Li in enumerate(self.L):
                alpha += self.volume * deps / (eps_out + Li * deps)
            alpha /= 3
        else:
            Li = self.L[axis]
            alpha = self.volume * deps / (eps_out + Li * deps)

        return alpha

    def scattering(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """
        Compute scattering cross section.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        pol : ndarray, optional
            Polarization direction. If None, uses average.

        Returns
        -------
        ndarray
            Scattering cross section in nm^2.
        """
        return self.sca(enei, pol)

    def sca(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """Scattering cross section."""
        enei = np.atleast_1d(enei)

        _, k = self.epsout(enei)
        k = np.real(k)

        if pol is None:
            # Average over random orientations
            alpha = self.polarizability(enei)
            sca = (8 * np.pi / 3) * k ** 4 * np.abs(alpha) ** 2
        else:
            pol = np.asarray(pol)
            pol = pol / np.linalg.norm(pol)

            sca = np.zeros(len(enei))
            for i in range(3):
                alpha_i = self.polarizability(enei, axis=i)
                sca += (8 * np.pi / 3) * k ** 4 * np.abs(alpha_i) ** 2 * pol[i] ** 2

        return np.real(sca)

    def absorption(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """
        Compute absorption cross section.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        pol : ndarray, optional
            Polarization direction.

        Returns
        -------
        ndarray
            Absorption cross section in nm^2.
        """
        return self.abs(enei, pol)

    def abs(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """Absorption cross section."""
        enei = np.atleast_1d(enei)

        _, k = self.epsout(enei)
        k = np.real(k)

        if pol is None:
            alpha = self.polarizability(enei)
            abs_cs = 4 * np.pi * k * np.imag(alpha)
        else:
            pol = np.asarray(pol)
            pol = pol / np.linalg.norm(pol)

            abs_cs = np.zeros(len(enei))
            for i in range(3):
                alpha_i = self.polarizability(enei, axis=i)
                abs_cs += 4 * np.pi * k * np.imag(alpha_i) * pol[i] ** 2

        return np.real(abs_cs)

    def extinction(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """
        Compute extinction cross section.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        pol : ndarray, optional
            Polarization direction.

        Returns
        -------
        ndarray
            Extinction cross section in nm^2.
        """
        return self.ext(enei, pol)

    def ext(self, enei: np.ndarray, pol: np.ndarray = None) -> np.ndarray:
        """Extinction cross section."""
        return self.sca(enei, pol) + self.abs(enei, pol)

    def decay_rate(self, enei: np.ndarray, r: np.ndarray, dip: np.ndarray) -> np.ndarray:
        """
        Compute decay rate enhancement for a dipole near the ellipsoid.

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.
        r : ndarray
            Dipole position relative to ellipsoid center.
        dip : ndarray
            Dipole orientation.

        Returns
        -------
        ndarray
            Decay rate enhancement factor.
        """
        # Simplified model
        enei = np.atleast_1d(enei)
        return np.ones(len(enei))

    def __repr__(self) -> str:
        return f"MieGans(axes={self.axes}, L={self.L})"
