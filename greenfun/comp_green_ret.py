"""
Composite retarded Green function for compound particles.
"""

import numpy as np
from typing import Optional, Union

from .green_ret import GreenRet, GreenRetLayer
from ..particles import ComParticle, ComPoint


class CompGreenRet:
    """
    Retarded Green function for composite particles.

    Wraps GreenRet for use with compound particles that have
    multiple dielectric regions.

    Parameters
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    k : complex, optional
        Wavenumber.
    **kwargs : dict
        Options passed to GreenRet.

    Attributes
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    g : GreenRet
        Underlying Green function object.
    """

    def __init__(
        self,
        p1: Union[ComParticle, ComPoint],
        p2: ComParticle,
        k: complex = None,
        **kwargs
    ):
        """
        Initialize composite retarded Green function.

        Parameters
        ----------
        p1 : ComParticle or ComPoint
            Source compound.
        p2 : ComParticle
            Target compound particle.
        k : complex, optional
            Wavenumber.
        """
        self.p1 = p1
        self.p2 = p2

        # Create underlying Green function using combined particles
        self.g = GreenRet(p1.pc, p2.pc, k=k, **kwargs)

    def set_k(self, k: complex) -> None:
        """Set wavenumber."""
        self.g.set_k(k)

    @property
    def k(self) -> complex:
        """Current wavenumber."""
        return self.g.k

    def G(self, k: complex = None) -> np.ndarray:
        """Green function matrix."""
        return self.g.G(k)

    def F(self, k: complex = None) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F(k)

    def H1(self, k: complex = None) -> np.ndarray:
        """F + 2*pi."""
        return self.g.H1(k)

    def H2(self, k: complex = None) -> np.ndarray:
        """F - 2*pi."""
        return self.g.H2(k)

    def Gp(self, k: complex = None) -> np.ndarray:
        """Gradient of Green function."""
        return self.g.Gp(k)

    def L(self, k: complex = None) -> np.ndarray:
        """Dyadic Green function for vector potential."""
        return self.g.L(k)

    def potential(self, sig: np.ndarray, k: complex = None) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Potential at p1 positions.
        """
        return self.g.potential(sig, k)

    def field(self, sig: np.ndarray, k: complex = None) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.
        k : complex, optional
            Wavenumber.

        Returns
        -------
        ndarray
            Electric field at p1 positions.
        """
        return self.g.field(sig, k)

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        self.g.clear_cache()

    def eval(self, enei: float = None, name: str = None) -> np.ndarray:
        """
        Evaluate Green function matrices.

        This method provides a MATLAB-compatible interface for evaluating
        Green function matrices by name.

        Parameters
        ----------
        enei : float, optional
            Wavelength in nm. If provided, sets the wavenumber k = 2*pi/enei.
        name : str, optional
            Name of matrix to return:
            - 'G': Green function
            - 'F': Surface derivative of Green function
            - 'H1': F + 2*pi (for inside-to-outside)
            - 'H2': F - 2*pi (for outside-to-inside)
            - 'Gp': Gradient of Green function
            - 'L': Dyadic Green function for vector potential
            If None, returns G.

        Returns
        -------
        ndarray
            The requested Green function matrix.

        Examples
        --------
        >>> g = CompGreenRet(p1, p2)
        >>> G = g.eval(enei=500, name='G')
        >>> F = g.eval(enei=500, name='F')
        """
        # Set wavenumber from wavelength
        k = None
        if enei is not None:
            k = 2 * np.pi / enei

        if name is None or name == 'G':
            return self.G(k)
        elif name == 'F':
            return self.F(k)
        elif name == 'H1':
            return self.H1(k)
        elif name == 'H2':
            return self.H2(k)
        elif name == 'Gp':
            return self.Gp(k)
        elif name == 'L':
            return self.L(k)
        else:
            raise ValueError(f"Unknown matrix name: {name}. "
                           f"Valid names: 'G', 'F', 'H1', 'H2', 'Gp', 'L'")

    def diag(self, enei: float = None, name: str = 'G') -> np.ndarray:
        """
        Get diagonal elements of Green function matrix.

        Parameters
        ----------
        enei : float, optional
            Wavelength in nm.
        name : str
            Name of matrix ('G', 'F', 'H1', 'H2', 'L').

        Returns
        -------
        ndarray
            Diagonal elements.
        """
        mat = self.eval(enei=enei, name=name)
        return np.diag(mat)

    def __repr__(self) -> str:
        return f"CompGreenRet(p1={self.p1}, p2={self.p2}, k={self.k})"


class CompGreenRetLayer(CompGreenRet):
    """
    Composite retarded Green function with layer effects.

    Parameters
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    layer : LayerStructure
        Layer structure.
    k : complex, optional
        Wavenumber.
    """

    def __init__(
        self,
        p1: Union[ComParticle, ComPoint],
        p2: ComParticle,
        layer,
        k: complex = None,
        **kwargs
    ):
        self.p1 = p1
        self.p2 = p2
        self.layer = layer

        # Create layer Green function
        self.g = GreenRetLayer(p1.pc, p2.pc, layer, k=k, **kwargs)

    def G_reflected(self, k: complex = None) -> np.ndarray:
        """Reflected Green function from layers."""
        return self.g.G_reflected(k)
