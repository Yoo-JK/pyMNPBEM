"""
Quasistatic BEM solver for layer structures (substrates).
"""

import numpy as np
from typing import Optional, Union

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct, Point, ComPoint
from ..greenfun import CompGreenStatLayer


class BEMStatLayer(BEMBase):
    """
    BEM solver for quasistatic approximation with layer structure.

    Handles particles on or near substrates using image charge method.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    layer : LayerStructure
        Layer structure (substrate).
    enei : float, optional
        Initial wavelength for precomputation.
    **kwargs : dict
        Options.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.bem import BEMStatLayer
    >>>
    >>> # Substrate at z=0
    >>> eps_air = EpsConst(1)
    >>> eps_glass = EpsConst(2.25)
    >>> layer = LayerStructure([eps_air, eps_glass])
    >>>
    >>> # Gold sphere above substrate
    >>> eps_gold = EpsTable('gold.dat')
    >>> sphere = trisphere(144, 10).shift([0, 0, 15])  # 5 nm above substrate
    >>> p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    >>> bem = BEMStatLayer(p, layer)
    """

    def __init__(
        self,
        p: ComParticle,
        layer,
        enei: Optional[float] = None,
        **kwargs
    ):
        """Initialize layer BEM solver."""
        self.p = p
        self.layer = layer
        self.options = kwargs

        # Create Green function with layer
        self.g = CompGreenStatLayer(p, p, layer, **kwargs)

        # Cache for resolvent matrix
        self._mat = None
        self._enei = None

        # Precompute if wavelength given
        if enei is not None:
            self._compute_resolvent(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _compute_resolvent(self, enei: float) -> None:
        """Compute BEM resolvent matrix for given wavelength."""
        if self._enei == enei and self._mat is not None:
            return

        # Compute Lambda factor for each face
        Lambda = self.p.lambda_factor(enei)

        # Get F matrix with layer corrections
        F = self.g.eval(enei, 'F')

        # BEM matrix: Lambda + F
        bem_matrix = np.diag(Lambda) + F
        self._mat = -np.linalg.inv(bem_matrix)
        self._enei = enei

    def solve(self, exc: CompStruct) -> CompStruct:
        """
        Solve BEM equations for given excitation.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field.

        Returns
        -------
        CompStruct
            Solution with 'sig' field.
        """
        self._compute_resolvent(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        sig = self._mat @ phip
        return CompStruct(self.p, exc.enei, sig=sig)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def field(self, sig: CompStruct, pts=None) -> np.ndarray:
        """Compute electric field from surface charges."""
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        return self.g.field(charges, sig.enei)

    def potential(self, sig: CompStruct, pts=None) -> np.ndarray:
        """Compute potential from surface charges."""
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        return self.g.potential(charges, sig.enei)

    def __repr__(self) -> str:
        return f"BEMStatLayer(p={self.p}, enei={self._enei})"
