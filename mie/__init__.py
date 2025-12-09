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
    riccati_bessel,
    legendre_p,
    legendre_p_derivative,
    mie_coefficients,
    mie_efficiencies,
    sphtable,
    lglnodes,
    fac2,
    adipole,
    dipole_from_coeffs,
    field_from_coeffs,
    aeels,
    BOHR,
    HARTREE,
    FINE_STRUCTURE,
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
    "riccati_bessel",
    "legendre_p",
    "legendre_p_derivative",
    "mie_coefficients",
    "mie_efficiencies",
    "sphtable",
    "lglnodes",
    "fac2",
    "adipole",
    "dipole_from_coeffs",
    "field_from_coeffs",
    "aeels",
    "BOHR",
    "HARTREE",
    "FINE_STRUCTURE",
]
