"""
Factory function for Mie solvers.
"""

from typing import Optional, Union, Any, Tuple

from .mie_stat import MieStat
from .mie_ret import MieRet
from .mie_gans import MieGans
from ..misc.options import BEMOptions


def miesolver(
    epsin: Any,
    epsout: Any,
    diameter: Union[float, Tuple[float, float, float]],
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> Union[MieStat, MieRet, MieGans]:
    """
    Factory function to create Mie solver.

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside sphere/ellipsoid.
    epsout : dielectric function
        Dielectric function outside sphere/ellipsoid.
    diameter : float or tuple
        Sphere diameter in nm, or tuple of (2*a, 2*b, 2*c) for ellipsoid.
    options : BEMOptions or dict, optional
        Simulation options. Key option is 'sim':
        - 'stat': Quasistatic Mie/Gans theory
        - 'ret': Full retarded Mie theory
    **kwargs : dict
        Additional options.

    Returns
    -------
    MieStat, MieRet, or MieGans
        Mie solver instance.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, miesolver, bemoptions
    >>> # Quasistatic sphere
    >>> op = bemoptions(sim='stat')
    >>> eps_gold = EpsTable('gold.dat')
    >>> eps_vac = EpsConst(1)
    >>> mie = miesolver(eps_gold, eps_vac, 10, op)
    >>>
    >>> # Retarded sphere (large particle)
    >>> op_ret = bemoptions(sim='ret')
    >>> mie_ret = miesolver(eps_gold, eps_vac, 100, op_ret)
    >>>
    >>> # Ellipsoid (automatically uses Gans theory)
    >>> mie_ell = miesolver(eps_gold, eps_vac, (100, 20, 20), op)
    """
    # Handle options
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = {
            'sim': options.sim,
            **options.extra
        }

    all_options = {**options, **kwargs}
    sim = all_options.pop('sim', 'stat')

    # Check if ellipsoid
    if isinstance(diameter, (tuple, list)) and len(diameter) == 3:
        # Ellipsoid - use Gans theory
        axes = tuple(d / 2 for d in diameter)  # Convert diameter to semi-axes
        return MieGans(epsin, epsout, axes, **all_options)

    # Sphere
    if sim == 'stat':
        return MieStat(epsin, epsout, diameter, **all_options)
    elif sim == 'ret':
        return MieRet(epsin, epsout, diameter, **all_options)
    else:
        raise ValueError(f"Unknown simulation type: {sim}. Use 'stat' or 'ret'.")
