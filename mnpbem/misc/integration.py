"""
Numerical integration utilities for BEM calculations.

This module provides quadrature rules for boundary element integration:
- Legendre-Gauss-Lobatto (LGL) nodes and weights
- Legendre-Gauss (LG) nodes and weights
- Triangle and quadrilateral face integration (QuadFace)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


def lglnodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss-Lobatto nodes and weights.

    LGL quadrature includes the endpoints [-1, 1] and is useful for
    spectral element methods and certain BEM integrations.

    Parameters
    ----------
    n : int
        Number of quadrature points (including endpoints).

    Returns
    -------
    x : ndarray
        Quadrature nodes in [-1, 1], shape (n,).
    w : ndarray
        Quadrature weights, shape (n,).

    Notes
    -----
    Based on algorithm by Greg von Winckel.
    Uses Newton-Raphson iteration to find roots of Legendre polynomial derivative.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for LGL quadrature")

    n1 = n

    # Initial guess: Chebyshev-Gauss-Lobatto nodes
    x = np.cos(np.pi * np.arange(n) / (n - 1))

    # Legendre Vandermonde matrix
    P = np.zeros((n1, n1))

    # Newton-Raphson iteration
    x_old = np.ones_like(x) * 2

    while np.max(np.abs(x - x_old)) > 1e-15:
        x_old = x.copy()

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(2, n):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        # Newton-Raphson update
        x = x_old - (x * P[:, n - 1] - P[:, n - 2]) / (n * P[:, n - 1])

    # Ensure endpoints are exact
    x[0] = -1.0
    x[-1] = 1.0

    # Compute weights
    w = 2.0 / (n * (n - 1) * P[:, n - 1] ** 2)

    return x, w


def lgwt(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss quadrature nodes and weights.

    LG quadrature does not include endpoints and achieves higher accuracy
    for smooth functions.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a : float
        Lower bound of interval (default: -1).
    b : float
        Upper bound of interval (default: 1).

    Returns
    -------
    x : ndarray
        Quadrature nodes in [a, b], shape (n,).
    w : ndarray
        Quadrature weights, shape (n,).

    Notes
    -----
    Based on algorithm by Greg von Winckel.
    Computes n-point Gauss-Legendre quadrature rule on [a, b].
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    N = n - 1
    N1 = N + 1
    N2 = N + 2

    # Initial guess
    xu = np.linspace(-1, 1, N1)
    y = np.cos((2 * np.arange(N1) + 1) * np.pi / (2 * N + 2)) + \
        (0.27 / N1) * np.sin(np.pi * xu * N / N2)

    # Legendre-Gauss Vandermonde matrix and derivative
    L = np.zeros((N1, N2))

    # Newton-Raphson iteration
    y0 = np.ones_like(y) * 2

    while np.max(np.abs(y - y0)) > 1e-15:
        L[:, 0] = 1
        L[:, 1] = y

        for k in range(2, N2):
            L[:, k] = ((2 * k - 1) * y * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k

        # Derivative of Legendre polynomial
        Lp = N2 * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y ** 2)

        y0 = y
        y = y0 - L[:, N2 - 1] / Lp

    # Linear map from [-1, 1] to [a, b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Compute weights
    Lp = N2 * (L[:, N1 - 1] - y * L[:, N2 - 1]) / (1 - y ** 2)
    w = (b - a) / ((1 - y ** 2) * Lp ** 2) * (N2 / N1) ** 2

    return x, w


def triangle_quadrature(order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get quadrature points and weights for unit triangle.

    The unit triangle has vertices at (0,0), (1,0), (0,1).

    Parameters
    ----------
    order : int
        Quadrature order (1-7 supported, higher orders use degree 7).

    Returns
    -------
    xi : ndarray
        First barycentric coordinate, shape (n_points,).
    eta : ndarray
        Second barycentric coordinate, shape (n_points,).
    w : ndarray
        Quadrature weights (sum to 0.5 for unit triangle).
    """
    if order <= 1:
        # 1-point rule (centroid)
        xi = np.array([1/3])
        eta = np.array([1/3])
        w = np.array([0.5])

    elif order <= 2:
        # 3-point rule
        xi = np.array([1/6, 2/3, 1/6])
        eta = np.array([1/6, 1/6, 2/3])
        w = np.array([1/6, 1/6, 1/6])

    elif order <= 3:
        # 4-point rule
        xi = np.array([1/3, 0.6, 0.2, 0.2])
        eta = np.array([1/3, 0.2, 0.6, 0.2])
        w = np.array([-27/96, 25/96, 25/96, 25/96])

    elif order <= 4:
        # 6-point rule
        a1 = 0.816847572980459
        a2 = 0.091576213509771
        b1 = 0.108103018168070
        b2 = 0.445948490915965
        w1 = 0.109951743655322 / 2
        w2 = 0.223381589678011 / 2

        xi = np.array([a1, a2, a2, b1, b2, b2])
        eta = np.array([a2, a1, a2, b2, b1, b2])
        w = np.array([w1, w1, w1, w2, w2, w2])

    else:
        # 7-point rule (order 5+)
        a = 0.797426985353087
        b = 0.101286507323456
        c = 0.470142064105115
        d = 0.059715871789770

        w1 = 0.225 / 2
        w2 = 0.125939180544827 / 2
        w3 = 0.132394152788506 / 2

        xi = np.array([1/3, a, b, b, c, d, c])
        eta = np.array([1/3, b, a, b, d, c, c])
        w = np.array([w1, w2, w2, w2, w3, w3, w3])

    return xi, eta, w


def quad_quadrature(order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get quadrature points and weights for unit square [-1,1] x [-1,1].

    Parameters
    ----------
    order : int
        Quadrature order (uses tensor product of 1D Gauss rules).

    Returns
    -------
    xi : ndarray
        X coordinates, shape (n_points,).
    eta : ndarray
        Y coordinates, shape (n_points,).
    w : ndarray
        Quadrature weights.
    """
    n = max(1, (order + 1) // 2)
    x1d, w1d = lgwt(n, -1, 1)

    # Tensor product
    xi, eta = np.meshgrid(x1d, x1d)
    xi = xi.ravel()
    eta = eta.ravel()

    wx, wy = np.meshgrid(w1d, w1d)
    w = (wx * wy).ravel()

    return xi, eta, w


@dataclass
class QuadFace:
    """
    Integration over triangular or quadrilateral boundary elements.

    This class provides quadrature rules for integrating functions
    over triangular and quadrilateral faces in BEM calculations.

    Parameters
    ----------
    order : int
        Integration order (default: 3).
    npol : int
        Number of points for polar integration (default: 10).
    refine : int
        Number of refinement levels (default: 0).

    Attributes
    ----------
    x : ndarray
        X coordinates of integration points for triangles.
    y : ndarray
        Y coordinates of integration points for triangles.
    w : ndarray
        Weights for triangle integration.
    x4, y4, w4 : ndarray
        Coordinates and weights for quadrilateral integration.
    """
    order: int = 3
    npol: int = 10
    refine: int = 0

    # Triangle quadrature
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    w: np.ndarray = field(default_factory=lambda: np.array([]))

    # Quad quadrature
    x4: np.ndarray = field(default_factory=lambda: np.array([]))
    y4: np.ndarray = field(default_factory=lambda: np.array([]))
    w4: np.ndarray = field(default_factory=lambda: np.array([]))

    # Polar integration (for singular elements)
    x3_pol: np.ndarray = field(default_factory=lambda: np.array([]))
    y3_pol: np.ndarray = field(default_factory=lambda: np.array([]))
    w3_pol: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Initialize quadrature rules."""
        self._init_triangle()
        self._init_quad()
        if self.npol > 0:
            self._init_polar()

    def _init_triangle(self):
        """Initialize triangle quadrature."""
        xi, eta, w = triangle_quadrature(self.order)

        # Apply refinement if requested
        if self.refine > 0:
            xi, eta, w = self._refine_triangle(xi, eta, w, self.refine)

        self.x = xi
        self.y = eta
        self.w = w

    def _init_quad(self):
        """Initialize quadrilateral quadrature."""
        self.x4, self.y4, self.w4 = quad_quadrature(self.order)

    def _init_polar(self):
        """Initialize polar quadrature for singular integration."""
        n_theta = self.npol
        n_r = self.npol

        # Angular and radial nodes
        theta, w_theta = lgwt(n_theta, 0, 2 * np.pi)
        r, w_r = lgwt(n_r, 0, 1)

        # Create tensor product grid
        theta_grid, r_grid = np.meshgrid(theta, r)
        theta_grid = theta_grid.ravel()
        r_grid = r_grid.ravel()

        w_theta_grid, w_r_grid = np.meshgrid(w_theta, w_r)
        w_grid = (w_theta_grid * w_r_grid * r_grid).ravel()  # r for Jacobian

        # Convert to Cartesian
        self.x3_pol = r_grid * np.cos(theta_grid)
        self.y3_pol = r_grid * np.sin(theta_grid)
        self.w3_pol = w_grid

    def _refine_triangle(
        self,
        xi: np.ndarray,
        eta: np.ndarray,
        w: np.ndarray,
        levels: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine triangle quadrature by subdivision."""
        for _ in range(levels):
            # Each triangle subdivides into 4
            n_pts = len(xi)
            new_xi = []
            new_eta = []
            new_w = []

            # Four sub-triangles with vertices:
            # (0,0), (0.5,0), (0,0.5)
            # (0.5,0), (1,0), (0.5,0.5)
            # (0,0.5), (0.5,0.5), (0,1)
            # (0.5,0), (0.5,0.5), (0,0.5)

            transforms = [
                (0.0, 0.0, 0.5),  # scale by 0.5, offset (0,0)
                (0.5, 0.0, 0.5),  # scale by 0.5, offset (0.5,0)
                (0.0, 0.5, 0.5),  # scale by 0.5, offset (0,0.5)
                (0.25, 0.25, 0.5),  # center sub-triangle
            ]

            for dx, dy, scale in transforms:
                new_xi.extend(xi * scale + dx)
                new_eta.extend(eta * scale + dy)
                new_w.extend(w * scale ** 2)

            xi = np.array(new_xi)
            eta = np.array(new_eta)
            w = np.array(new_w)

        return xi, eta, w

    @property
    def n_triangle(self) -> int:
        """Number of triangle quadrature points."""
        return len(self.x)

    @property
    def n_quad(self) -> int:
        """Number of quadrilateral quadrature points."""
        return len(self.x4)

    def integrate_triangle(
        self,
        func: callable,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> complex:
        """
        Integrate function over a triangle.

        Parameters
        ----------
        func : callable
            Function f(x, y, z) to integrate.
        v0, v1, v2 : ndarray
            Triangle vertices, each shape (3,).

        Returns
        -------
        complex
            Integral value.
        """
        # Map quadrature points to triangle
        # P = (1-xi-eta)*v0 + xi*v1 + eta*v2
        result = 0.0
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        for i in range(len(self.x)):
            xi, eta = self.x[i], self.y[i]
            p = (1 - xi - eta) * v0 + xi * v1 + eta * v2
            result += self.w[i] * func(p[0], p[1], p[2])

        return result * 2 * area  # Factor of 2 because weights sum to 0.5

    def integrate_quad(
        self,
        func: callable,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray
    ) -> complex:
        """
        Integrate function over a quadrilateral.

        Parameters
        ----------
        func : callable
            Function f(x, y, z) to integrate.
        v0, v1, v2, v3 : ndarray
            Quadrilateral vertices (counter-clockwise), each shape (3,).

        Returns
        -------
        complex
            Integral value.
        """
        result = 0.0

        for i in range(len(self.x4)):
            xi, eta = self.x4[i], self.y4[i]
            # Bilinear mapping
            p = (1-xi)*(1-eta)/4 * v0 + (1+xi)*(1-eta)/4 * v1 + \
                (1+xi)*(1+eta)/4 * v2 + (1-xi)*(1+eta)/4 * v3

            # Jacobian
            dpdxi = (-(1-eta)*v0 + (1-eta)*v1 + (1+eta)*v2 - (1+eta)*v3) / 4
            dpdeta = (-(1-xi)*v0 - (1+xi)*v1 + (1+xi)*v2 + (1-xi)*v3) / 4
            jac = np.linalg.norm(np.cross(dpdxi, dpdeta))

            result += self.w4[i] * func(p[0], p[1], p[2]) * jac

        return result

    def __repr__(self) -> str:
        return f"QuadFace(order={self.order}, n_tri={self.n_triangle}, n_quad={self.n_quad})"


def quadface(order: int = 3, npol: int = 10, refine: int = 0) -> QuadFace:
    """
    Create quadrature object for face integration.

    Parameters
    ----------
    order : int
        Integration order.
    npol : int
        Number of points for polar integration.
    refine : int
        Refinement levels.

    Returns
    -------
    QuadFace
        Quadrature object.
    """
    return QuadFace(order=order, npol=npol, refine=refine)
