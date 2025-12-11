"""
Iterative retarded BEM solver for layer structures.

Provides efficient solving for large particles on substrates
using H-matrix/ACA compression.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from scipy.sparse.linalg import gmres, bicgstab, LinearOperator

from .bem_base import BEMBase
from .bem_iter import BEMIter
from ..particles import ComParticle, CompStruct


class BEMRetLayerIter(BEMIter, BEMBase):
    """
    Iterative retarded BEM solver with layer structure.

    Uses H-matrix compression for efficient large-scale simulations
    of particles on or near substrates.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    layer : LayerStructure
        Layer structure (substrate).
    enei : float, optional
        Initial wavelength.
    use_hmatrix : bool
        Whether to use H-matrix compression.
    use_aca : bool
        Whether to use ACA for low-rank blocks.
    **kwargs : dict
        Options including tol, maxiter, method.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.bem import BEMRetLayerIter
    >>>
    >>> eps_air = EpsConst(1)
    >>> eps_glass = EpsConst(2.25)
    >>> layer = LayerStructure([eps_air, eps_glass])
    >>>
    >>> eps_gold = EpsTable('gold.dat')
    >>> sphere = trisphere(1000, 50).shift([0, 0, 60])  # Large particle
    >>> p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    >>> bem = BEMRetLayerIter(p, layer, use_hmatrix=True)
    """

    def __init__(
        self,
        p: ComParticle,
        layer,
        enei: Optional[float] = None,
        use_hmatrix: bool = True,
        use_aca: bool = True,
        **kwargs
    ):
        """Initialize iterative retarded layer solver."""
        # Extract iterative solver options
        tol = kwargs.pop('tol', 1e-6)
        maxiter = kwargs.pop('maxiter', 100)
        method = kwargs.pop('method', 'gmres')

        BEMIter.__init__(self, tol, maxiter, method)

        self.p = p
        self.layer = layer
        self.use_hmatrix = use_hmatrix
        self.use_aca = use_aca
        self.options = kwargs

        # Initialize Green function with compression and layer
        if use_hmatrix:
            from ..greenfun import HMatrixGreen
            self.g = HMatrixGreen(p, p, layer=layer, retarded=True, **kwargs)
        else:
            from ..greenfun import CompGreenRetLayer
            self.g = CompGreenRetLayer(p, p, layer=layer, **kwargs)

        # Cache
        self._Lambda = None
        self._enei = None
        self._k = None
        self._precond = None

        if enei is not None:
            self._setup(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _get_wavenumber(self, enei: float, medium_idx: int = 0) -> complex:
        """Get wavenumber in specified medium."""
        eps_func = self.p.eps[medium_idx]
        _, k = eps_func(enei)
        return k

    def _setup(self, enei: float) -> None:
        """Setup for given wavelength."""
        if self._enei == enei:
            return

        n = self.n_faces
        k_bg = self._get_wavenumber(enei, 0)

        # Set wavenumber in Green function
        if hasattr(self.g, 'set_k'):
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
        self._Lambda = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)
        self._enei = enei
        self._k = k_bg

        # Setup preconditioner
        if self.precond:
            self._setup_preconditioner()

    def _setup_preconditioner(self) -> None:
        """Setup preconditioner for iterative solver."""
        # Diagonal (Jacobi) preconditioner
        diag = self._Lambda.copy()
        diag[np.abs(diag) < 1e-10] = 1.0

        def precond_matvec(x):
            return x / diag

        n = self.n_faces
        self._precond = LinearOperator((n, n), matvec=precond_matvec, dtype=complex)

    def _F_matvec(self, x: np.ndarray) -> np.ndarray:
        """Matrix-vector product with F matrix (including layer effects)."""
        if self.use_hmatrix:
            return self.g.matvec(x, self._k)
        else:
            return self.g.F(self._k) @ x

    def solve(self, exc: CompStruct) -> CompStruct:
        """
        Solve BEM equations iteratively.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field.

        Returns
        -------
        CompStruct
            Solution with 'sig' field.
        """
        self._setup(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        # Create matrix-vector product
        matvec = self._create_matvec(self._Lambda, self._F_matvec)

        # Solve iteratively
        sig, info = self._solve_iterative(matvec, -phip, self.n_faces, self._precond)

        if info != 0:
            import warnings
            warnings.warn(f"Iterative solver did not converge: info={info}")

        return CompStruct(self.p, exc.enei, sig=sig, k=self._k)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def field(self, sig: CompStruct, pts=None) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Electric field, shape (n_pts, 3).
        """
        from ..greenfun import GreenRetLayer
        from ..particles import ComPoint, Point

        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            # Field at particle surface
            return self.g.field(charges, k)
        else:
            # Field at external points
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRetLayer(pts_obj, self.p.pc, self.layer, k=k, **self.options)
            return g_pts.field(charges, k)

    def potential(self, sig: CompStruct, pts=None) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Potential at points.
        """
        from ..greenfun import GreenRetLayer
        from ..particles import ComPoint, Point

        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            return self.g.potential(charges, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRetLayer(pts_obj, self.p.pc, self.layer, k=k, **self.options)
            return g_pts.potential(charges, k)

    def get_info(self) -> dict:
        """
        Get solver information and statistics.

        Returns
        -------
        dict
            Solver statistics.
        """
        return {
            'n_faces': self.n_faces,
            'method': self.method,
            'tol': self.tol,
            'maxiter': self.maxiter,
            'iterations': self.iterations,
            'residuals': self.residuals,
            'use_hmatrix': self.use_hmatrix,
            'use_aca': self.use_aca
        }

    def __repr__(self) -> str:
        return f"BEMRetLayerIter(n={self.n_faces}, method='{self.method}')"


class BEMStatLayerIter(BEMIter, BEMBase):
    """
    Iterative quasistatic BEM solver with layer structure.

    Uses H-matrix compression for efficient large-scale simulations
    of particles on substrates.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    layer : LayerStructure
        Layer structure (substrate).
    enei : float, optional
        Initial wavelength.
    use_hmatrix : bool
        Whether to use H-matrix compression.
    **kwargs : dict
        Options.
    """

    def __init__(
        self,
        p: ComParticle,
        layer,
        enei: Optional[float] = None,
        use_hmatrix: bool = True,
        **kwargs
    ):
        """Initialize iterative quasistatic layer solver."""
        tol = kwargs.pop('tol', 1e-6)
        maxiter = kwargs.pop('maxiter', 100)
        method = kwargs.pop('method', 'gmres')

        BEMIter.__init__(self, tol, maxiter, method)

        self.p = p
        self.layer = layer
        self.use_hmatrix = use_hmatrix
        self.options = kwargs

        # Initialize Green function with layer
        if use_hmatrix:
            from ..greenfun import HMatrixGreen
            self.g = HMatrixGreen(p, p, layer=layer, **kwargs)
        else:
            from ..greenfun import CompGreenStatLayer
            self.g = CompGreenStatLayer(p, p, layer=layer, **kwargs)

        self._Lambda = None
        self._enei = None
        self._precond = None

        if enei is not None:
            self._setup(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _setup(self, enei: float) -> None:
        """Setup for given wavelength."""
        if self._enei == enei:
            return

        self._Lambda = self.p.lambda_factor(enei)
        self._enei = enei

        if self.precond:
            self._setup_preconditioner()

    def _setup_preconditioner(self) -> None:
        """Setup preconditioner."""
        diag = self._Lambda.copy()
        diag[np.abs(diag) < 1e-10] = 1.0

        def precond_matvec(x):
            return x / diag

        n = self.n_faces
        self._precond = LinearOperator((n, n), matvec=precond_matvec, dtype=complex)

    def _F_matvec(self, x: np.ndarray) -> np.ndarray:
        """Matrix-vector product with F matrix."""
        if self.use_hmatrix:
            return self.g.matvec(x)
        else:
            return self.g.F @ x

    def solve(self, exc: CompStruct) -> CompStruct:
        """Solve BEM equations iteratively."""
        self._setup(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        matvec = self._create_matvec(self._Lambda, self._F_matvec)
        sig, info = self._solve_iterative(matvec, -phip, self.n_faces, self._precond)

        if info != 0:
            import warnings
            warnings.warn(f"Iterative solver did not converge: info={info}")

        return CompStruct(self.p, exc.enei, sig=sig)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def __repr__(self) -> str:
        return f"BEMStatLayerIter(n={self.n_faces}, method='{self.method}')"
