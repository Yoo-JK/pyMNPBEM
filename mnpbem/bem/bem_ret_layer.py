"""
Retarded BEM solver for layer structures (substrates).
"""

import numpy as np
from typing import Optional, Tuple

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct


class BEMRetLayer(BEMBase):
    """
    BEM solver for retarded simulations with layer structure.

    Handles particles on or near substrates with full electromagnetic
    (retarded) treatment.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    layer : LayerStructure
        Layer structure (substrate).
    enei : float, optional
        Initial wavelength.
    **kwargs : dict
        Options.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.bem import BEMRetLayer
    >>>
    >>> eps_air = EpsConst(1)
    >>> eps_glass = EpsConst(2.25)
    >>> layer = LayerStructure([eps_air, eps_glass])
    >>>
    >>> eps_gold = EpsTable('gold.dat')
    >>> sphere = trisphere(144, 50).shift([0, 0, 60])
    >>> p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    >>> bem = BEMRetLayer(p, layer)
    """

    def __init__(
        self,
        p: ComParticle,
        layer,
        enei: Optional[float] = None,
        **kwargs
    ):
        """Initialize retarded layer BEM solver."""
        from ..greenfun import CompGreenRetLayer

        self.p = p
        self.layer = layer
        self.options = kwargs

        # Create retarded Green function with layer
        self.g = CompGreenRetLayer(p, p, layer=layer, **kwargs)

        # Cache
        self._mat = None
        self._enei = None
        self._k = None

        if enei is not None:
            self._compute_matrices(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _get_wavenumber(self, enei: float, medium_idx: int = 0) -> complex:
        """Get wavenumber in specified medium."""
        eps_func = self.p.eps[medium_idx]
        _, k = eps_func(enei)
        return k

    def _compute_matrices(self, enei: float) -> None:
        """Compute BEM matrices for given wavelength."""
        if self._enei == enei and self._mat is not None:
            return

        n = self.n_faces

        # Get wavenumber in background medium
        k_bg = self._get_wavenumber(enei, 0)
        self.g.set_k(k_bg)

        # Get dielectric functions
        eps_in = np.zeros(n, dtype=complex)
        eps_out = np.zeros(n, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p.p):
            n_faces_i = particle.n_faces
            e_in, e_out = self.p.dielectric_inout(enei, i)
            eps_in[face_idx:face_idx + n_faces_i] = e_in
            eps_out[face_idx:face_idx + n_faces_i] = e_out
            face_idx += n_faces_i

        # Lambda factor
        Lambda = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)

        # Get F matrix with layer corrections
        F = self.g.F(k_bg)

        # BEM matrix
        bem_matrix = np.diag(Lambda) + F
        self._mat = -np.linalg.inv(bem_matrix)
        self._enei = enei
        self._k = k_bg

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
        self._compute_matrices(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        sig = self._mat @ phip

        return CompStruct(self.p, exc.enei, sig=sig, k=self._k)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def field(self, sig: CompStruct, pts=None) -> np.ndarray:
        """Compute electric field from surface charges."""
        charges = sig.get('sig')
        k = sig.get('k', self._k)
        return self.g.field(charges, k)

    def potential(self, sig: CompStruct, pts=None) -> np.ndarray:
        """Compute potential from surface charges."""
        charges = sig.get('sig')
        k = sig.get('k', self._k)
        return self.g.potential(charges, k)

    def __repr__(self) -> str:
        return f"BEMRetLayer(p={self.p}, enei={self._enei})"


class BEMRetMirror(BEMBase):
    """
    Retarded BEM solver with mirror symmetry.

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry.
    enei : float, optional
        Initial wavelength.
    **kwargs : dict
        Options.
    """

    def __init__(self, p, enei: Optional[float] = None, **kwargs):
        """Initialize retarded mirror BEM solver."""
        from ..greenfun import CompGreenRetMirror

        self.p = p
        self.options = kwargs

        self.g = CompGreenRetMirror(p, **kwargs)

        self._mat = {}
        self._enei = None
        self._k = None

        if enei is not None:
            self._compute_matrices(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements (reduced)."""
        return self.p.n_faces

    def _get_wavenumber(self, enei: float, medium_idx: int = 0) -> complex:
        """Get wavenumber."""
        eps_func = self.p.eps[medium_idx]
        _, k = eps_func(enei)
        return k

    def _compute_matrices(self, enei: float) -> None:
        """Compute BEM matrices for each symmetry configuration."""
        if self._enei == enei and self._mat:
            return

        self._mat = {}
        n = self.n_faces

        k_bg = self._get_wavenumber(enei, 0)
        self.g.set_k(k_bg)

        # Get dielectric functions
        eps_in = np.zeros(n, dtype=complex)
        eps_out = np.zeros(n, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p.p):
            n_faces_i = particle.n_faces
            e_in, e_out = self.p.dielectric_inout(enei, i)
            eps_in[face_idx:face_idx + n_faces_i] = e_in
            eps_out[face_idx:face_idx + n_faces_i] = e_out
            face_idx += n_faces_i

        Lambda = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)

        # For each symmetry configuration
        F_base = self.g.F(k_bg)

        for i, symrow in enumerate(self.p.symtable):
            F_sym = self._apply_symmetry_to_F(F_base, symrow)
            bem_matrix = np.diag(Lambda) + F_sym
            self._mat[i] = -np.linalg.inv(bem_matrix)

        self._enei = enei
        self._k = k_bg

    def _apply_symmetry_to_F(self, F_full: np.ndarray, symrow: np.ndarray) -> np.ndarray:
        """Apply symmetry factors to F matrix."""
        n = self.n_faces
        n_full = F_full.shape[1]
        n_copies = n_full // n

        F_sym = np.zeros((n, n), dtype=F_full.dtype)
        for k in range(n_copies):
            factor = symrow[k] if k < len(symrow) else 1
            F_sym += factor * F_full[:, k*n:(k+1)*n]

        return F_sym

    def solve(self, exc: CompStruct, symkey: str = '+') -> CompStruct:
        """Solve BEM equations for given excitation and symmetry."""
        self._compute_matrices(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        symval = self.p.symvalue(symkey)
        sym_idx = self.p.symindex(symval)

        if sym_idx < 0:
            raise ValueError(f"Unknown symmetry key: {symkey}")

        sig = self._mat[sym_idx] @ phip

        result = CompStruct(self.p, exc.enei, sig=sig, k=self._k)
        result.symval = symval

        return result

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc, '+')

    def __repr__(self) -> str:
        return f"BEMRetMirror(p={self.p}, sym='{self.p.sym}')"
