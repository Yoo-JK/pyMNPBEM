"""
Quasistatic BEM solver with mirror symmetry.
"""

import numpy as np
from typing import Optional, Union, List

from .bem_base import BEMBase
from ..particles import ComParticleMirror, CompStruct, CompStructMirror


class BEMStatMirror(BEMBase):
    """
    BEM solver for quasistatic approximation with mirror symmetry.

    Exploits mirror symmetry to reduce computational cost.

    Parameters
    ----------
    p : ComParticleMirror
        Compound particle with mirror symmetry.
    enei : float, optional
        Initial wavelength.
    **kwargs : dict
        Options.

    Examples
    --------
    >>> from pymnpbem import EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import ComParticleMirror
    >>> from pymnpbem.bem import BEMStatMirror
    >>>
    >>> eps = [EpsConst(1), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10)
    >>> p = ComParticleMirror(eps, [sphere], [[2, 1]], sym='x')
    >>> bem = BEMStatMirror(p)
    """

    def __init__(
        self,
        p: ComParticleMirror,
        enei: Optional[float] = None,
        **kwargs
    ):
        """Initialize mirror BEM solver."""
        from ..greenfun import CompGreenStatMirror

        self.p = p
        self.options = kwargs

        # Green function for mirror symmetry
        self.g = CompGreenStatMirror(p, **kwargs)

        # Surface derivative of Green function
        self._F = self.g.F

        # Cache for resolvent matrices (one per symmetry configuration)
        self._mat = {}
        self._enei = None

        if enei is not None:
            self._compute_resolvent(enei)

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function matrix."""
        return self._F

    @property
    def n_faces(self) -> int:
        """Number of boundary elements (reduced)."""
        return self.p.n_faces

    def _compute_resolvent(self, enei: float) -> None:
        """Compute BEM resolvent matrices for each symmetry configuration."""
        if self._enei == enei and self._mat:
            return

        self._mat = {}

        # Compute Lambda factor
        Lambda = self.p.lambda_factor(enei)

        # For each symmetry configuration
        for i, symrow in enumerate(self.p.symtable):
            # Modify F matrix based on symmetry
            F_sym = self._apply_symmetry_to_F(symrow)

            # BEM matrix
            bem_matrix = np.diag(Lambda) + F_sym
            self._mat[i] = -np.linalg.inv(bem_matrix)

        self._enei = enei

    def _apply_symmetry_to_F(self, symrow: np.ndarray) -> np.ndarray:
        """Apply symmetry factors to F matrix."""
        # Get base F connecting reduced to full particle
        F_full = self._F

        # Extract the relevant block for this symmetry
        n = self.n_faces
        n_full = F_full.shape[1]
        n_copies = n_full // n

        # Sum contributions with symmetry factors
        F_sym = np.zeros((n, n), dtype=F_full.dtype)
        for k in range(n_copies):
            factor = symrow[k] if k < len(symrow) else 1
            F_sym += factor * F_full[:, k*n:(k+1)*n]

        return F_sym

    def solve(self, exc: CompStruct, symkey: str = '+') -> CompStruct:
        """
        Solve BEM equations for given excitation and symmetry.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field.
        symkey : str
            Symmetry key: '+', '-' for x/y, or '++', '+-', etc. for xy.

        Returns
        -------
        CompStruct
            Solution with 'sig' field.
        """
        self._compute_resolvent(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        # Get symmetry index
        symval = self.p.symvalue(symkey)
        sym_idx = self.p.symindex(symval)

        if sym_idx < 0:
            raise ValueError(f"Unknown symmetry key: {symkey}")

        # Solve
        sig = self._mat[sym_idx] @ phip

        # Create result with symmetry info
        result = CompStruct(self.p, exc.enei, sig=sig)
        result.symval = symval

        return result

    def solve_all(self, exc: CompStruct) -> List[CompStruct]:
        """
        Solve for all symmetry configurations.

        Parameters
        ----------
        exc : CompStruct
            Excitation.

        Returns
        -------
        list of CompStruct
            Solutions for each symmetry.
        """
        self._compute_resolvent(exc.enei)

        results = []
        symkeys = ['+', '-'] if self.p.sym in ('x', 'y') else ['++', '+-', '-+', '--']

        for key in symkeys:
            results.append(self.solve(exc, key))

        return results

    def expand_solution(self, sig: CompStruct) -> CompStruct:
        """
        Expand solution to full particle.

        Parameters
        ----------
        sig : CompStruct
            Solution for reduced particle.

        Returns
        -------
        CompStruct
            Solution expanded to full particle.
        """
        if not hasattr(sig, 'symval'):
            raise ValueError("Solution must have symmetry information")

        charges = sig.get('sig')
        symval = sig.symval

        # Expand using symmetry
        expanded = self.p.expand_scalar(charges, symval)

        return CompStruct(self.p.full, sig.enei, sig=expanded)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax (solves for '+' symmetry)."""
        return self.solve(exc, '+')

    def __repr__(self) -> str:
        return f"BEMStatMirror(p={self.p}, sym='{self.p.sym}')"
