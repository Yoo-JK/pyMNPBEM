"""
Factory functions for BEM solvers.
"""

from typing import Optional, Union

from .bem_stat import BEMStat
from .bem_ret import BEMRet
from ..particles import ComParticle
from ..misc.options import BEMOptions


def bemsolver(
    p: ComParticle,
    options: Optional[Union[BEMOptions, dict]] = None,
    enei: Optional[float] = None,
    **kwargs
) -> Union[BEMStat, BEMRet]:
    """
    Factory function to create appropriate BEM solver.

    Creates a BEM solver based on the simulation type specified
    in the options.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    options : BEMOptions or dict, optional
        Simulation options. Key option is 'sim':
        - 'stat': Quasistatic solver (BEMStat) - for small particles
        - 'ret': Retarded solver (BEMRet) - for larger particles
    enei : float, optional
        Wavelength for precomputation.
    **kwargs : dict
        Additional options.

    Returns
    -------
    BEMStat or BEMRet
        BEM solver instance.

    Examples
    --------
    >>> from mnpbem import bemoptions, bemsolver, ComParticle
    >>> op = bemoptions(sim='stat')
    >>> bem = bemsolver(p, op)
    >>>
    >>> # For larger particles, use retarded solver
    >>> op_ret = bemoptions(sim='ret')
    >>> bem_ret = bemsolver(p, op_ret)
    """
    # Handle options
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = {
            'sim': options.sim,
            'rel_cutoff': options.rel_cutoff,
            **options.extra
        }

    # Merge with kwargs
    all_options = {**options, **kwargs}

    # Get simulation type
    sim = all_options.pop('sim', 'stat')

    # Create appropriate solver
    if sim == 'stat':
        return BEMStat(p, enei=enei, **all_options)
    elif sim == 'ret':
        return BEMRet(p, enei=enei, **all_options)
    else:
        raise ValueError(f"Unknown simulation type: {sim}. Use 'stat' or 'ret'.")
