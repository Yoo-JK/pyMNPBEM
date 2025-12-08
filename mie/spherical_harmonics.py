"""
Spherical harmonics and related functions for Mie theory.

This module provides:
- Spherical Bessel functions
- Associated Legendre polynomials
- Vector spherical harmonics
- Spherical harmonic tables
"""

import numpy as np
from scipy import special
from typing import Tuple, Optional


def spherical_jn(n: int, z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Spherical Bessel function of the first kind.

    j_n(z) = sqrt(pi/(2z)) * J_{n+1/2}(z)

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    z : ndarray
        Argument.
    derivative : bool
        If True, return the derivative.

    Returns
    -------
    ndarray
        Spherical Bessel function or its derivative.
    """
    return special.spherical_jn(n, z, derivative=derivative)


def spherical_yn(n: int, z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Spherical Bessel function of the second kind.

    y_n(z) = sqrt(pi/(2z)) * Y_{n+1/2}(z)

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    z : ndarray
        Argument.
    derivative : bool
        If True, return the derivative.

    Returns
    -------
    ndarray
        Spherical Bessel function or its derivative.
    """
    return special.spherical_yn(n, z, derivative=derivative)


def spherical_hn1(n: int, z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Spherical Hankel function of the first kind.

    h1_n(z) = j_n(z) + i*y_n(z)

    Parameters
    ----------
    n : int
        Order.
    z : ndarray
        Argument.
    derivative : bool
        If True, return the derivative.

    Returns
    -------
    ndarray
        Spherical Hankel function.
    """
    jn = spherical_jn(n, z, derivative=derivative)
    yn = spherical_yn(n, z, derivative=derivative)
    return jn + 1j * yn


def spherical_hn2(n: int, z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Spherical Hankel function of the second kind.

    h2_n(z) = j_n(z) - i*y_n(z)

    Parameters
    ----------
    n : int
        Order.
    z : ndarray
        Argument.
    derivative : bool
        If True, return the derivative.

    Returns
    -------
    ndarray
        Spherical Hankel function.
    """
    jn = spherical_jn(n, z, derivative=derivative)
    yn = spherical_yn(n, z, derivative=derivative)
    return jn - 1j * yn


def riccati_bessel_psi(n: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Riccati-Bessel function psi_n(z) = z * j_n(z) and its derivative.

    Parameters
    ----------
    n : int
        Order.
    z : ndarray
        Argument.

    Returns
    -------
    psi : ndarray
        Riccati-Bessel function.
    psi_prime : ndarray
        Derivative.
    """
    z = np.asarray(z)
    jn = spherical_jn(n, z)
    jn_prime = spherical_jn(n, z, derivative=True)

    psi = z * jn
    psi_prime = jn + z * jn_prime

    return psi, psi_prime


def riccati_bessel_xi(n: int, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Riccati-Bessel function xi_n(z) = z * h1_n(z) and its derivative.

    Parameters
    ----------
    n : int
        Order.
    z : ndarray
        Argument.

    Returns
    -------
    xi : ndarray
        Riccati-Bessel function.
    xi_prime : ndarray
        Derivative.
    """
    z = np.asarray(z)
    hn = spherical_hn1(n, z)
    hn_prime = spherical_hn1(n, z, derivative=True)

    xi = z * hn
    xi_prime = hn + z * hn_prime

    return xi, xi_prime


def legendre_p(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Associated Legendre polynomial P_l^m(x).

    Parameters
    ----------
    l : int
        Degree.
    m : int
        Order.
    x : ndarray
        Argument (typically cos(theta)).

    Returns
    -------
    ndarray
        Associated Legendre polynomial.
    """
    return special.lpmv(m, l, x)


def legendre_p_derivative(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Derivative of associated Legendre polynomial dP_l^m/dx.

    Parameters
    ----------
    l : int
        Degree.
    m : int
        Order.
    x : ndarray
        Argument.

    Returns
    -------
    ndarray
        Derivative of associated Legendre polynomial.
    """
    x = np.asarray(x)

    # Use recurrence relation
    # (1-x^2) dP_l^m/dx = l*x*P_l^m - (l+m)*P_{l-1}^m
    plm = legendre_p(l, m, x)

    if l > 0:
        plm1 = legendre_p(l - 1, m, x)
        factor = 1 - x ** 2
        factor = np.where(np.abs(factor) < 1e-14, 1e-14, factor)
        dplm = (l * x * plm - (l + m) * plm1) / factor
    else:
        dplm = np.zeros_like(x)

    return dplm


def spharm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Spherical harmonic Y_l^m(theta, phi).

    Parameters
    ----------
    l : int
        Degree.
    m : int
        Order.
    theta : ndarray
        Polar angle (0 to pi).
    phi : ndarray
        Azimuthal angle (0 to 2*pi).

    Returns
    -------
    ndarray
        Complex spherical harmonic.
    """
    return special.sph_harm(m, l, phi, theta)


def vecspharm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vector spherical harmonics (VSH).

    Returns the three VSH components: Y, Psi, Phi

    Y_lm = Y_lm(theta, phi) * r_hat
    Psi_lm = r * grad(Y_lm)
    Phi_lm = r x grad(Y_lm)

    Parameters
    ----------
    l : int
        Degree.
    m : int
        Order.
    theta : ndarray
        Polar angle.
    phi : ndarray
        Azimuthal angle.

    Returns
    -------
    Y : ndarray
        Radial VSH component.
    Psi : ndarray
        Gradient VSH component.
    Phi : ndarray
        Curl VSH component.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_theta = np.where(np.abs(sin_theta) < 1e-14, 1e-14, sin_theta)

    # Spherical harmonic
    ylm = spharm(l, m, theta, phi)

    # Associated Legendre polynomial and derivative
    plm = legendre_p(l, m, cos_theta)
    dplm = legendre_p_derivative(l, m, cos_theta)

    # Normalization factor
    if m == 0:
        norm = np.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        norm = np.sqrt((2 * l + 1) / (4 * np.pi) *
                       special.factorial(l - abs(m)) / special.factorial(l + abs(m)))

    exp_imphi = np.exp(1j * m * phi)

    # Y component (radial)
    Y = ylm

    # Psi components (theta and phi directions)
    # Psi_theta = dY/dtheta
    # Psi_phi = (1/sin(theta)) * dY/dphi
    Psi_theta = norm * (-sin_theta) * dplm * exp_imphi
    Psi_phi = norm * plm * (1j * m / sin_theta) * exp_imphi

    # Combine into vector
    Psi = np.stack([Psi_theta, Psi_phi], axis=-1)

    # Phi components (cross product with r_hat)
    Phi_theta = -Psi_phi
    Phi_phi = Psi_theta
    Phi = np.stack([Phi_theta, Phi_phi], axis=-1)

    return Y, Psi, Phi


class SphTable:
    """
    Table of precomputed spherical harmonic values.

    Useful for efficient repeated evaluations at the same angles.

    Parameters
    ----------
    l_max : int
        Maximum degree.
    theta : ndarray
        Polar angles.
    phi : ndarray
        Azimuthal angles.
    """

    def __init__(self, l_max: int, theta: np.ndarray, phi: np.ndarray):
        """
        Initialize spherical harmonic table.

        Parameters
        ----------
        l_max : int
            Maximum degree.
        theta : ndarray
            Polar angles.
        phi : ndarray
            Azimuthal angles.
        """
        self.l_max = l_max
        self.theta = np.asarray(theta)
        self.phi = np.asarray(phi)

        # Precompute spherical harmonics
        self._ylm = {}
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                self._ylm[(l, m)] = spharm(l, m, self.theta, self.phi)

    def __call__(self, l: int, m: int) -> np.ndarray:
        """Get precomputed Y_l^m."""
        return self._ylm.get((l, m), np.zeros_like(self.theta, dtype=complex))

    def vecspharm(self, l: int, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get vector spherical harmonics."""
        return vecspharm(l, m, self.theta, self.phi)


def mie_coefficients(
    m_rel: complex,
    x: float,
    n_max: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Mie scattering coefficients a_n and b_n.

    Parameters
    ----------
    m_rel : complex
        Relative refractive index (n_particle / n_medium).
    x : float
        Size parameter (2*pi*n_medium*radius/wavelength).
    n_max : int, optional
        Maximum order. If None, uses Wiscombe criterion.

    Returns
    -------
    a_n : ndarray
        Electric multipole coefficients.
    b_n : ndarray
        Magnetic multipole coefficients.
    """
    # Wiscombe criterion for n_max
    if n_max is None:
        n_max = int(x + 4 * x ** (1 / 3) + 2)
        n_max = max(n_max, 3)

    mx = m_rel * x

    a_n = np.zeros(n_max, dtype=complex)
    b_n = np.zeros(n_max, dtype=complex)

    for n in range(1, n_max + 1):
        # Riccati-Bessel functions
        psi_x, psi_x_prime = riccati_bessel_psi(n, x)
        psi_mx, psi_mx_prime = riccati_bessel_psi(n, mx)
        xi_x, xi_x_prime = riccati_bessel_xi(n, x)

        # Mie coefficients
        # a_n = (m*psi_n(mx)*psi_n'(x) - psi_n(x)*psi_n'(mx)) /
        #       (m*psi_n(mx)*xi_n'(x) - xi_n(x)*psi_n'(mx))
        # b_n = (psi_n(mx)*psi_n'(x) - m*psi_n(x)*psi_n'(mx)) /
        #       (psi_n(mx)*xi_n'(x) - m*xi_n(x)*psi_n'(mx))

        num_a = m_rel * psi_mx * psi_x_prime - psi_x * psi_mx_prime
        den_a = m_rel * psi_mx * xi_x_prime - xi_x * psi_mx_prime

        num_b = psi_mx * psi_x_prime - m_rel * psi_x * psi_mx_prime
        den_b = psi_mx * xi_x_prime - m_rel * xi_x * psi_mx_prime

        a_n[n - 1] = num_a / den_a
        b_n[n - 1] = num_b / den_b

    return a_n, b_n


def mie_efficiencies(
    m_rel: complex,
    x: float,
    n_max: int = None
) -> Tuple[float, float, float]:
    """
    Compute Mie efficiency factors.

    Parameters
    ----------
    m_rel : complex
        Relative refractive index.
    x : float
        Size parameter.
    n_max : int, optional
        Maximum order.

    Returns
    -------
    Q_ext : float
        Extinction efficiency.
    Q_sca : float
        Scattering efficiency.
    Q_abs : float
        Absorption efficiency.
    """
    a_n, b_n = mie_coefficients(m_rel, x, n_max)

    n = np.arange(1, len(a_n) + 1)
    factor = 2 * n + 1

    # Efficiencies
    Q_ext = 2 / x ** 2 * np.sum(factor * np.real(a_n + b_n))
    Q_sca = 2 / x ** 2 * np.sum(factor * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))
    Q_abs = Q_ext - Q_sca

    return Q_ext, Q_sca, Q_abs
