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


def sphtable(l_max: int, mode: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tables of spherical harmonic degrees and orders.

    Parameters
    ----------
    l_max : int
        Maximum degree.
    mode : str
        'full' for all m from -l to l, 'pos' for m >= 0 only.

    Returns
    -------
    ltab : ndarray
        Array of degrees.
    mtab : ndarray
        Array of orders.
    """
    ltab = []
    mtab = []

    for l in range(1, l_max + 1):
        if mode == 'full':
            for m in range(-l, l + 1):
                ltab.append(l)
                mtab.append(m)
        else:
            for m in range(0, l + 1):
                ltab.append(l)
                mtab.append(m)

    return np.array(ltab), np.array(mtab)


def lglnodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss-Lobatto nodes and weights.

    Adapted from Greg von Winkel's MATLAB implementation.

    Parameters
    ----------
    n : int
        Number of nodes - 1 (polynomial degree).

    Returns
    -------
    x : ndarray
        Nodes on [-1, 1].
    w : ndarray
        Quadrature weights.
    """
    n1 = n + 1

    # Initial guess using Chebyshev-Gauss-Lobatto nodes
    x = np.cos(np.pi * np.arange(n + 1) / n)

    # Legendre Vandermonde matrix
    p = np.zeros((n1, n1))

    # Newton-Raphson iteration
    x_old = 2 * np.ones_like(x)

    while np.max(np.abs(x - x_old)) > np.finfo(float).eps:
        x_old = x.copy()
        p[:, 0] = 1
        p[:, 1] = x

        for k in range(2, n1):
            p[:, k] = ((2 * k - 1) * x * p[:, k - 1] - (k - 1) * p[:, k - 2]) / k

        x = x_old - (x * p[:, n] - p[:, n - 1]) / (n1 * p[:, n])

    # Compute weights
    w = 2 / (n * n1 * p[:, n] ** 2)

    return x, w


def fac2(n: int) -> int:
    """
    Compute double factorial n!!.

    Parameters
    ----------
    n : int
        Argument.

    Returns
    -------
    int
        Double factorial.
    """
    if n <= 0:
        return 1
    if n % 2 == 0:
        return int(np.prod(np.arange(2, n + 1, 2)))
    else:
        return int(np.prod(np.arange(1, n + 1, 2)))


def riccati_bessel(z: complex, l_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Riccati-Bessel functions for Mie coefficients.

    Parameters
    ----------
    z : complex
        Argument (k * r).
    l_values : ndarray
        Array of l values.

    Returns
    -------
    j : ndarray
        z * j_l(z) (Riccati-Bessel psi).
    h : ndarray
        z * h_l^(1)(z) (Riccati-Bessel xi).
    zjp : ndarray
        Derivative of z * j_l(z).
    zhp : ndarray
        Derivative of z * h_l^(1)(z).
    """
    l_values = np.atleast_1d(l_values)
    n_l = len(l_values)

    j = np.zeros(n_l, dtype=complex)
    h = np.zeros(n_l, dtype=complex)
    zjp = np.zeros(n_l, dtype=complex)
    zhp = np.zeros(n_l, dtype=complex)

    for i, l in enumerate(l_values):
        psi, psi_prime = riccati_bessel_psi(int(l), z)
        xi, xi_prime = riccati_bessel_xi(int(l), z)
        j[i] = psi
        h[i] = xi
        zjp[i] = psi_prime
        zhp[i] = xi_prime

    return j, h, zjp, zhp


def adipole(pos: np.ndarray, dip: np.ndarray, ltab: np.ndarray, mtab: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics coefficients for dipole in the quasistatic limit.

    The dipole potential for a given degree l and order m is of the form:
        Phi = 4*pi / (2*l + 1) * a(l, m) * Y_l^m(theta, phi) / r^(l+1)

    Parameters
    ----------
    pos : ndarray
        Position of dipole (x, y, z), normalized by sphere radius.
    dip : ndarray
        Dipole vector (dx, dy, dz).
    ltab : ndarray
        Table of spherical harmonic degrees.
    mtab : ndarray
        Table of spherical harmonic orders.

    Returns
    -------
    a : ndarray
        Expansion coefficients.
    """
    pos = np.asarray(pos).flatten()
    dip = np.asarray(dip).flatten()
    ltab = np.asarray(ltab).flatten()
    mtab = np.asarray(mtab).flatten()

    # Convert position to spherical coordinates
    r = np.linalg.norm(pos)
    if r < 1e-14:
        return np.zeros(len(ltab), dtype=complex)

    theta = np.arccos(pos[2] / r)  # polar angle
    phi = np.arctan2(pos[1], pos[0])  # azimuthal angle

    # Unit vector to dipole position
    e = pos / r

    # Spherical harmonics and vector spherical harmonics
    n_coeffs = len(ltab)
    a = np.zeros(n_coeffs, dtype=complex)

    for i in range(n_coeffs):
        l = int(ltab[i])
        m = int(mtab[i])

        # Scalar spherical harmonic (complex conjugate)
        y = np.conj(spharm(l, m, theta, phi))

        # Vector spherical harmonic contribution
        # This involves the derivative of Y_l^m
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta) if np.abs(np.sin(theta)) > 1e-14 else 1e-14

        # dY/dtheta component
        plm = legendre_p(l, abs(m), cos_theta)
        dplm = legendre_p_derivative(l, abs(m), cos_theta)

        # Normalization
        norm = np.sqrt((2 * l + 1) / (4 * np.pi) *
                      special.factorial(l - abs(m)) / special.factorial(l + abs(m)))

        exp_imphi = np.exp(1j * m * phi)

        # Vector spherical harmonic components
        x_theta = norm * (-sin_theta) * dplm * exp_imphi
        x_phi = norm * plm * (1j * m / sin_theta) * exp_imphi

        # Unit vectors in spherical coordinates
        e_theta = np.array([np.cos(theta) * np.cos(phi),
                          np.cos(theta) * np.sin(phi),
                          -np.sin(theta)])
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0])

        # x vector (3D)
        x_vec = x_theta * e_theta + x_phi * e_phi

        # Expansion coefficient
        e_dot_dip = np.dot(e, dip)
        cross_e_dip = np.cross(e, dip)

        a[i] = -((l + 1) * e_dot_dip * y +
                 1j * np.sqrt(l * (l + 1)) * np.dot(np.conj(x_vec), cross_e_dip)) / r ** (l + 2)

    return a


def dipole_from_coeffs(ltab: np.ndarray, mtab: np.ndarray, a: np.ndarray,
                       diameter: float, epsz: complex) -> np.ndarray:
    """
    Dipole moment of sphere within quasistatic Mie theory.

    Parameters
    ----------
    ltab : ndarray
        Table of spherical harmonic degrees.
    mtab : ndarray
        Table of spherical harmonic orders.
    a : ndarray
        Expansion coefficients.
    diameter : float
        Diameter of sphere.
    epsz : complex
        Ratio of dielectric functions (eps_in / eps_out).

    Returns
    -------
    dip : ndarray
        Dipole moment (3D vector).
    """
    ltab = np.asarray(ltab).flatten()
    mtab = np.asarray(mtab).flatten()
    a = np.asarray(a).flatten()

    # Static Mie coefficients [Jackson eq. (4.5)]
    c = (1 - epsz) * ltab / ((1 + epsz) * ltab + 1) * (diameter / 2) ** (2 * ltab + 1) * a

    # Find indices for l=1, m=-1,0,1
    idx_z = np.where((ltab == 1) & (mtab == 0))[0]
    idx_p = np.where((ltab == 1) & (mtab == 1))[0]
    idx_m = np.where((ltab == 1) & (mtab == -1))[0]

    qz = np.sqrt(4 * np.pi / 3) * c[idx_z[0]] if len(idx_z) > 0 else 0
    qp = -np.sqrt(4 * np.pi / 3) * c[idx_p[0]] if len(idx_p) > 0 else 0
    qm = np.sqrt(4 * np.pi / 3) * c[idx_m[0]] if len(idx_m) > 0 else 0

    # Compute dipole moment
    dip = qz * np.array([0, 0, 1]) + (qp * np.array([1, 1j, 0]) + qm * np.array([1, -1j, 0])) / np.sqrt(2)

    return np.real(dip)


def field_from_coeffs(ltab: np.ndarray, mtab: np.ndarray, verts: np.ndarray,
                      a: np.ndarray, diameter: float, epsz: complex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Electric field and potential from Mie theory within quasistatic approximation.

    Parameters
    ----------
    ltab : ndarray
        Table of spherical harmonic degrees.
    mtab : ndarray
        Table of spherical harmonic orders.
    verts : ndarray
        Vertices where field is computed (should be on sphere surface).
    a : ndarray
        Expansion coefficients.
    diameter : float
        Diameter of sphere.
    epsz : complex
        Ratio of dielectric functions (eps_in / eps_out).

    Returns
    -------
    e : ndarray
        Electric field at vertices.
    phi : ndarray
        Scalar potential at vertices.
    """
    ltab = np.asarray(ltab).flatten()
    mtab = np.asarray(mtab).flatten()
    a = np.asarray(a).flatten()
    verts = np.atleast_2d(verts)

    # Static Mie coefficients
    c = (1 - epsz) * ltab / ((1 + epsz) * ltab + 1) * (diameter / 2) ** (2 * ltab + 1) * a

    nverts = verts.shape[0]

    # Convert to spherical coordinates
    r = np.linalg.norm(verts, axis=1)
    r_mean = np.mean(r)
    theta = np.arccos(np.clip(verts[:, 2] / r, -1, 1))
    phi_coord = np.arctan2(verts[:, 1], verts[:, 0])

    # Unit vectors
    unit = verts / r[:, np.newaxis]

    # Compute potential and field
    phi_result = np.zeros(nverts, dtype=complex)
    e_result = np.zeros((nverts, 3), dtype=complex)

    for i in range(len(ltab)):
        l = int(ltab[i])
        m = int(mtab[i])

        # Spherical harmonics at all vertices
        y = spharm(l, m, theta, phi_coord)

        # Potential contribution
        fac_phi = 4 * np.pi / (2 * l + 1) * c[i] / r_mean ** (l + 1)
        phi_result += fac_phi * y

        # Electric field contribution (radial part)
        fac_e = 4 * np.pi / (2 * l + 1) * c[i] * (l + 1) / r_mean ** (l + 2)
        e_result += fac_e * y[:, np.newaxis] * unit

    return e_result, phi_result


def aeels(ltab: np.ndarray, mtab: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical harmonics coefficients for EELS.

    See F. J. Garcia de Abajo, Phys. Rev. B 59, 3095 (1999).

    Parameters
    ----------
    ltab : ndarray
        Table of spherical harmonic degrees.
    mtab : ndarray
        Table of spherical harmonic orders.
    beta : float
        Ratio of electron velocity to speed of light.

    Returns
    -------
    ce : ndarray
        Electric expansion coefficient [eq. (31)].
    cm : ndarray
        Magnetic expansion coefficient [eq. (30)].
    """
    ltab = np.asarray(ltab).flatten()
    mtab = np.asarray(mtab).flatten()

    # Legendre-Gauss-Lobatto nodes and weights
    x, w = lglnodes(100)

    # Table of factorials
    max_lm = int(np.max(ltab + np.abs(mtab))) + 2
    fac = np.array([special.factorial(i, exact=True) for i in range(max_lm + 1)])

    # Gamma value
    gamma = 1 / np.sqrt(1 - beta ** 2)

    # Coupling coefficients
    a = np.zeros(len(ltab), dtype=complex)

    # Loop over unique spherical harmonic degrees
    for l in np.unique(ltab):
        l = int(l)
        # Legendre polynomial
        p = special.lpmv(np.arange(l + 1)[:, np.newaxis], l, x)  # shape: (l+1, len(x))

        # Loop over spherical harmonics with m >= 0
        for m in range(l + 1):
            # Coefficient of Eq. (A9)
            aa = 0.0 + 0.0j

            # Alpha factor
            alpha = np.sqrt((2 * l + 1) / (4 * np.pi) * fac[l - m] / fac[l + m])

            for j in range(m, l + 1):
                # Restrict sum to even j + m integers
                if (j + m) % 2 == 0:
                    # Integral (A7)
                    I = ((-1) ** m * np.sum(w * p[m, :] *
                         (1 - x ** 2) ** (j / 2) * x ** (l - j)))

                    # C factor (A9)
                    C = ((1j ** (l - j) * alpha * fac2(2 * l + 1) /
                         (2 ** j * fac[l - j] *
                          fac[(j - m) // 2] * fac[(j + m) // 2])) * I)

                    # Add contribution
                    aa += C / (beta ** (l + 1) * gamma ** j)

            # Assign values to coefficients
            idx_pos = np.where((ltab == l) & (mtab == m))[0]
            idx_neg = np.where((ltab == l) & (mtab == -m))[0]

            if len(idx_pos) > 0:
                a[idx_pos] = aa
            if len(idx_neg) > 0:
                a[idx_neg] = ((-1) ** m) * aa

    # Expansion coefficient of Eq. (15)
    b = np.zeros(len(ltab), dtype=complex)

    # Build index arrays
    for i in range(len(ltab)):
        l = int(ltab[i])
        m = int(mtab[i])

        if m != l:  # ip condition
            # Find index with same l, m+1
            idx_next = np.where((ltab == l) & (mtab == m + 1))[0]
            if len(idx_next) > 0:
                b[i] += a[idx_next[0]] * np.sqrt((l + m + 1) * (l - m))

        if m != -l:  # im condition
            # Find index with same l, m-1
            idx_prev = np.where((ltab == l) & (mtab == m - 1))[0]
            if len(idx_prev) > 0:
                b[i] -= a[idx_prev[0]] * np.sqrt((l - m + 1) * (l + m))

    # Magnetic and electric expansion coefficients of Eqs. (30, 31)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = 1 / (ltab * (ltab + 1)) * np.abs(2 * beta * mtab * a) ** 2
        ce = 1 / (ltab * (ltab + 1)) * np.abs(b / gamma) ** 2

    # Handle l=0 case
    cm = np.where(ltab == 0, 0, cm)
    ce = np.where(ltab == 0, 0, ce)

    return ce, cm


# Atomic units for EELS calculations
BOHR = 0.0529177  # nm
HARTREE = 27.2116  # eV
FINE_STRUCTURE = 1 / 137.036  # fine structure constant
