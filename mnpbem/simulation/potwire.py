"""
Wire potential calculation for electron beam excitation.

This module provides the PotWire function for computing the potential
and derivatives of a charged wire (electron beam) segment, following
the MATLAB MNPBEM implementation.

The potential integral is:
    Integrate[ Exp[ I q zz ] / Sqrt[ r ^ 2 + ( zz - z ) ^ 2 ], { zz, z0, z1 } ]

Reference:
    F. J. Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010)
"""

import numpy as np
from typing import Tuple, Optional


def lglnodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss-Lobatto nodes and weights.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    x : ndarray
        Quadrature nodes in [-1, 1].
    w : ndarray
        Quadrature weights.
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Initial guess using Chebyshev nodes
    x = np.cos(np.pi * np.arange(n) / (n - 1))

    # Legendre-Gauss-Lobatto points are zeros of derivative of Legendre polynomial
    # Newton-Raphson iteration
    x_old = np.ones_like(x) * 2

    while np.max(np.abs(x - x_old)) > 1e-15:
        x_old = x.copy()

        # Compute Legendre polynomial and derivative
        P = np.zeros((n, n))
        P[:, 0] = 1
        P[:, 1] = x

        for k in range(2, n):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        # Newton-Raphson update for interior points
        x = x_old - (x * P[:, n - 1] - P[:, n - 2]) / (n * P[:, n - 1])

    # Ensure endpoints are exactly -1 and 1
    x[0] = -1.0
    x[-1] = 1.0

    # Compute weights
    w = 2 / (n * (n - 1) * P[:, n - 1] ** 2)

    return x, w


def potwire(
    r: np.ndarray,
    z: np.ndarray,
    q: float,
    k: float,
    z0: np.ndarray,
    z1: np.ndarray,
    n_quad: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Potential for charged wire segment (electron beam).

    Computes the integral:
        Integrate[ Exp[ I q zz ] / Sqrt[ r ^ 2 + ( zz - z ) ^ 2 ], { zz, z0, z1 } ]

    and its derivatives with respect to r and z.

    Parameters
    ----------
    r : ndarray
        Distance normal to electron beam, shape (n_points,).
    z : ndarray
        Distance along electron beam, shape (n_points,).
    q : float
        Wavenumber of electron beam.
    k : float
        Wavenumber of light.
    z0 : ndarray
        Beginning of wire segments, shape (n_wires,).
    z1 : ndarray
        End of wire segments, shape (n_wires,).
    n_quad : int
        Number of quadrature points (default: 10).

    Returns
    -------
    phi : ndarray
        Potential, shape (n_points, n_wires).
    phir : ndarray
        Derivative of potential wrt r, shape (n_points, n_wires).
    phiz : ndarray
        Derivative of potential wrt z, shape (n_points, n_wires).

    Notes
    -----
    The solution uses the transformation:
        u = zz - z
        v = log( u + sqrt( r^2 + u^2 ) )

    Reference:
        F. J. Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010)
    """
    r = np.atleast_1d(r).reshape(-1, 1)  # (n_points, 1)
    z = np.atleast_1d(z).reshape(-1, 1)  # (n_points, 1)
    z0 = np.atleast_1d(z0).reshape(1, -1)  # (1, n_wires)
    z1 = np.atleast_1d(z1).reshape(1, -1)  # (1, n_wires)

    # Adapt integration limits
    # z0_shifted and z1_shifted have shape (n_points, n_wires)
    z0_shifted = z0 - z
    z1_shifted = z1 - z

    # Transform integration limits
    # v = log( u + sqrt( r^2 + u^2 ) )
    v0 = np.log(z0_shifted + np.sqrt(r ** 2 + z0_shifted ** 2))
    v1 = np.log(z1_shifted + np.sqrt(r ** 2 + z1_shifted ** 2))

    # Initialize integrals
    phi = np.zeros_like(v0, dtype=complex)
    phir = np.zeros_like(v0, dtype=complex)
    phiz = np.zeros_like(v0, dtype=complex)

    # Get quadrature nodes and weights
    x_quad, w_quad = lglnodes(n_quad)

    # Loop over integration points
    for i in range(len(x_quad)):
        x_i = x_quad[i]
        w_i = w_quad[i]

        # Transform integration variable to u
        v = 0.5 * ((1 - x_i) * v0 + (1 + x_i) * v1)
        u = 0.5 * (np.exp(v) - r ** 2 * np.exp(-v))

        # Distance
        rr = np.sqrt(r ** 2 + u ** 2)

        # Exponential factor
        fac = np.exp(1j * (q * u + k * rr))

        # Integral and derivatives
        phi = phi + w_i * fac
        phir = phir + w_i * r * (1j * k / rr - 1 / rr ** 2) * fac
        phiz = phiz - w_i * u * (1j * k / rr - 1 / rr ** 2) * fac

    # Multiply with integration limits and phase factor
    dv = 0.5 * (v1 - v0)
    phase = np.exp(1j * q * z)

    phi = dv * phase * phi
    phir = dv * phase * phir
    phiz = dv * phase * phiz

    return phi, phir, phiz


def potwire_single(
    r: float,
    z: float,
    q: float,
    k: float,
    z0: float,
    z1: float,
    n_quad: int = 10
) -> Tuple[complex, complex, complex]:
    """
    Potential for a single wire segment at a single point.

    Convenience wrapper around potwire for single point evaluation.

    Parameters
    ----------
    r : float
        Distance normal to electron beam.
    z : float
        Distance along electron beam.
    q : float
        Wavenumber of electron beam.
    k : float
        Wavenumber of light.
    z0 : float
        Beginning of wire.
    z1 : float
        End of wire.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    phi : complex
        Potential.
    phir : complex
        Derivative of potential wrt r.
    phiz : complex
        Derivative of potential wrt z.
    """
    phi, phir, phiz = potwire(
        np.array([r]),
        np.array([z]),
        q, k,
        np.array([z0]),
        np.array([z1]),
        n_quad
    )
    return phi[0, 0], phir[0, 0], phiz[0, 0]


def electron_beam_potential(
    pos: np.ndarray,
    beam_axis: np.ndarray,
    beam_start: np.ndarray,
    beam_end: np.ndarray,
    q: float,
    k: float,
    n_quad: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute electron beam potential at arbitrary positions.

    Parameters
    ----------
    pos : ndarray
        Evaluation positions, shape (n_points, 3).
    beam_axis : ndarray
        Unit vector along beam direction, shape (3,).
    beam_start : ndarray
        Starting point of beam, shape (3,).
    beam_end : ndarray
        End point of beam, shape (3,).
    q : float
        Electron beam wavenumber (omega / v).
    k : float
        Light wavenumber.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    phi : ndarray
        Scalar potential at positions, shape (n_points,).
    grad_phi : ndarray
        Gradient of potential, shape (n_points, 3).
    """
    pos = np.atleast_2d(pos)
    beam_axis = np.array(beam_axis) / np.linalg.norm(beam_axis)

    # Project positions onto beam coordinate system
    # z: along beam, r: perpendicular distance
    pos_rel = pos - beam_start

    # z coordinate (along beam)
    z = np.dot(pos_rel, beam_axis)

    # r coordinate (perpendicular distance)
    pos_perp = pos_rel - np.outer(z, beam_axis)
    r = np.linalg.norm(pos_perp, axis=1)

    # Unit vector perpendicular to beam (towards each point)
    r_hat = np.zeros_like(pos)
    nonzero = r > 1e-10
    r_hat[nonzero] = pos_perp[nonzero] / r[nonzero, np.newaxis]

    # Wire endpoints in beam coordinates
    z0 = 0.0  # beam_start
    z1 = np.linalg.norm(beam_end - beam_start)  # beam_end

    # Compute potential
    phi, phir, phiz = potwire(
        r, z, q, k,
        np.array([z0]), np.array([z1]),
        n_quad
    )

    # Flatten results
    phi = phi[:, 0]
    phir = phir[:, 0]
    phiz = phiz[:, 0]

    # Convert derivatives to gradient in Cartesian coordinates
    grad_phi = (phir[:, np.newaxis] * r_hat +
                phiz[:, np.newaxis] * beam_axis)

    return phi, grad_phi


class WirePotential:
    """
    Wire potential calculator for electron beam excitation.

    This class encapsulates the wire potential calculation for use
    in EELS simulations.

    Parameters
    ----------
    v : float
        Electron velocity (fraction of c).
    enei : float
        Wavelength in nm (for light wavenumber).
    n_quad : int
        Number of quadrature points.

    Examples
    --------
    >>> wp = WirePotential(v=0.7, enei=500)
    >>> phi, grad = wp.evaluate(pos, beam_start, beam_end)
    """

    def __init__(self, v: float = 0.7, enei: float = 500.0, n_quad: int = 10):
        """Initialize wire potential calculator."""
        self.v = v  # Electron velocity (fraction of c)
        self.enei = enei
        self.n_quad = n_quad

    @property
    def speed_of_light(self) -> float:
        """Speed of light in nm/s (approximately)."""
        return 299792458.0 * 1e9  # nm/s

    @property
    def k(self) -> float:
        """Light wavenumber."""
        return 2 * np.pi / self.enei

    def q(self, omega: Optional[float] = None) -> float:
        """
        Electron beam wavenumber.

        Parameters
        ----------
        omega : float, optional
            Angular frequency. If None, computed from wavelength.

        Returns
        -------
        float
            Electron wavenumber omega/v.
        """
        if omega is None:
            omega = self.speed_of_light * self.k
        return omega / (self.v * self.speed_of_light)

    def evaluate(
        self,
        pos: np.ndarray,
        beam_start: np.ndarray,
        beam_end: np.ndarray,
        beam_axis: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate wire potential at positions.

        Parameters
        ----------
        pos : ndarray
            Evaluation positions, shape (n_points, 3).
        beam_start : ndarray
            Starting point of beam.
        beam_end : ndarray
            End point of beam.
        beam_axis : ndarray, optional
            Beam direction (computed from start/end if not given).

        Returns
        -------
        phi : ndarray
            Scalar potential.
        grad_phi : ndarray
            Gradient of potential.
        """
        if beam_axis is None:
            beam_axis = beam_end - beam_start
            beam_axis = beam_axis / np.linalg.norm(beam_axis)

        return electron_beam_potential(
            pos, beam_axis, beam_start, beam_end,
            self.q(), self.k, self.n_quad
        )

    def __repr__(self) -> str:
        return f"WirePotential(v={self.v}, enei={self.enei})"
