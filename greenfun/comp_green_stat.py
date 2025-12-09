"""
Composite Green function for compound particles.
"""

import numpy as np
from typing import Optional, Union

from .green_stat import GreenStat
from ..particles import ComParticle, ComPoint


class CompGreenStat:
    """
    Green function for composite particles in quasistatic approximation.

    Wraps GreenStat for use with compound particles that have
    multiple dielectric regions.

    Parameters
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    **kwargs : dict
        Options passed to GreenStat.

    Attributes
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    g : GreenStat
        Underlying Green function object.
    """

    def __init__(
        self,
        p1: Union[ComParticle, ComPoint],
        p2: ComParticle,
        **kwargs
    ):
        """
        Initialize composite Green function.

        Parameters
        ----------
        p1 : ComParticle or ComPoint
            Source compound.
        p2 : ComParticle
            Target compound particle.
        """
        self.p1 = p1
        self.p2 = p2

        # Create underlying Green function using combined particles
        self.g = GreenStat(p1.pc, p2.pc, **kwargs)

    @property
    def G(self) -> np.ndarray:
        """Green function matrix."""
        return self.g.G()

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F()

    @property
    def H1(self) -> np.ndarray:
        """F + 2*pi."""
        return self.g.H1()

    @property
    def H2(self) -> np.ndarray:
        """F - 2*pi."""
        return self.g.H2()

    @property
    def Gp(self) -> np.ndarray:
        """Gradient of Green function."""
        return self.g.Gp()

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.

        Returns
        -------
        ndarray
            Potential at p1 positions.
        """
        return self.g.potential(sig)

    def field(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.

        Returns
        -------
        ndarray
            Electric field at p1 positions.
        """
        return self.g.field(sig)

    def eval(self, enei: float = None, name: str = None) -> np.ndarray:
        """
        Evaluate Green function matrices.

        This method provides a MATLAB-compatible interface for evaluating
        Green function matrices by name.

        Parameters
        ----------
        enei : float, optional
            Wavelength in nm (not used in quasistatic limit but kept for
            interface compatibility).
        name : str, optional
            Name of matrix to return:
            - 'G': Green function
            - 'F': Surface derivative of Green function
            - 'H1': F + 2*pi (for inside-to-outside)
            - 'H2': F - 2*pi (for outside-to-inside)
            - 'Gp': Gradient of Green function
            If None, returns G.

        Returns
        -------
        ndarray
            The requested Green function matrix.

        Examples
        --------
        >>> g = CompGreenStat(p1, p2)
        >>> G = g.eval(name='G')
        >>> F = g.eval(name='F')
        """
        if name is None or name == 'G':
            return self.G
        elif name == 'F':
            return self.F
        elif name == 'H1':
            return self.H1
        elif name == 'H2':
            return self.H2
        elif name == 'Gp':
            return self.Gp
        else:
            raise ValueError(f"Unknown matrix name: {name}. "
                           f"Valid names: 'G', 'F', 'H1', 'H2', 'Gp'")

    def diag(self, name: str = 'G') -> np.ndarray:
        """
        Get diagonal elements of Green function matrix.

        Parameters
        ----------
        name : str
            Name of matrix ('G', 'F', 'H1', 'H2').

        Returns
        -------
        ndarray
            Diagonal elements.
        """
        mat = self.eval(name=name)
        return np.diag(mat)

    def __repr__(self) -> str:
        return f"CompGreenStat(p1={self.p1}, p2={self.p2})"
