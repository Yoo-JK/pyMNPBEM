"""
Mie theory for spherical and ellipsoidal particles.

This module provides analytical Mie theory solutions:
- MieStat: Quasistatic Mie theory for small spheres
- MieRet: Full retarded Mie theory for arbitrary size spheres
- MieGans: Gans theory for ellipsoidal particles
- miesolver: Factory function for Mie solvers
- Spherical harmonics and Bessel functions
"""

from .mie_stat import MieStat
from .mie_ret import MieRet
from .mie_gans import MieGans
from .factory import miesolver
from .spherical_harmonics import (
    spharm,
    vecspharm,
    SphTable,
    spherical_jn,
    spherical_yn,
    spherical_hn1,
    spherical_hn2,
    riccati_bessel_psi,
    riccati_bessel_xi,
    legendre_p,
    mie_coefficients,
    mie_efficiencies,
)

__all__ = [
    "MieStat",
    "MieRet",
    "MieGans",
    "miesolver",
    "spharm",
    "vecspharm",
    "SphTable",
    "spherical_jn",
    "spherical_yn",
    "spherical_hn1",
    "spherical_hn2",
    "riccati_bessel_psi",
    "riccati_bessel_xi",
    "legendre_p",
    "mie_coefficients",
    "mie_efficiencies",
]
