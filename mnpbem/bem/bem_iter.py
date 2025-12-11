"""
Iterative BEM solvers using H-matrix/ACA acceleration.

These solvers are efficient for large particles where direct
matrix inversion becomes expensive.
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union, Any
from scipy.sparse.linalg import gmres, bicgstab, LinearOperator

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct, ComPoint, Point


class BEMIter:
    """
    Base class for iterative BEM solvers.

    Provides common functionality for iterative solvers that use
    Krylov subspace methods (GMRES, BiCGSTAB) combined with
    H-matrix or ACA compression.

    Parameters
    ----------
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    method : str
        Iterative method: 'gmres' or 'bicgstab'.
    precond : str or callable
        Preconditioner type or custom preconditioner.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        maxiter: int = 100,
        method: str = 'gmres',
        precond: Optional[str] = None
    ):
        """Initialize iterative solver base."""
        self.tol = tol
        self.maxiter = maxiter
        self.method = method
        self.precond = precond

        # Statistics
        self.iterations = 0
        self.residuals = []

    def _create_matvec(self, Lambda: np.ndarray, F_func: Callable) -> Callable:
        """
        Create matrix-vector product function.

        Parameters
        ----------
        Lambda : ndarray
            Diagonal Lambda factor.
        F_func : callable
            Function to compute F @ x.

        Returns
        -------
        callable
            Matrix-vector product function.
        """
        def matvec(x):
            return Lambda * x + F_func(x)
        return matvec

    def _solve_iterative(
        self,
        matvec: Callable,
        rhs: np.ndarray,
        n: int,
        M: Optional[LinearOperator] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Solve system using iterative method.

        Parameters
        ----------
        matvec : callable
            Matrix-vector product.
        rhs : ndarray
            Right-hand side.
        n : int
            System size.
        M : LinearOperator, optional
            Preconditioner.

        Returns
        -------
        x : ndarray
            Solution.
        info : int
            Convergence info (0 = success).
        """
        A_op = LinearOperator((n, n), matvec=matvec, dtype=complex)

        self.residuals = []

        def callback(residual):
            if hasattr(residual, '__len__'):
                self.residuals.append(np.linalg.norm(residual))
            else:
                self.residuals.append(residual)

        if self.method == 'gmres':
            x, info = gmres(A_op, rhs, tol=self.tol, maxiter=self.maxiter,
                          M=M, callback=callback, callback_type='pr_norm')
        else:  # bicgstab
            x, info = bicgstab(A_op, rhs, tol=self.tol, maxiter=self.maxiter,
                              M=M, callback=callback)

        self.iterations = len(self.residuals)
        return x, info


class BEMStatIter(BEMIter, BEMBase):
    """
    Iterative quasistatic BEM solver.

    Uses H-matrix compression for efficient large-scale simulations.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
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
    >>> from pymnpbem.bem import BEMStatIter
    >>>
    >>> eps = [EpsConst(1), EpsTable('gold.dat')]
    >>> sphere = trisphere(1000, 50)  # Large particle
    >>> p = ComParticle(eps, [sphere], [[2, 1]])
    >>> bem = BEMStatIter(p, use_hmatrix=True)
    """

    def __init__(
        self,
        p: ComParticle,
        enei: Optional[float] = None,
        use_hmatrix: bool = True,
        use_aca: bool = True,
        **kwargs
    ):
        """Initialize iterative quasistatic solver."""
        # Extract iterative solver options
        tol = kwargs.pop('tol', 1e-6)
        maxiter = kwargs.pop('maxiter', 100)
        method = kwargs.pop('method', 'gmres')

        BEMIter.__init__(self, tol, maxiter, method)

        self.p = p
        self.use_hmatrix = use_hmatrix
        self.use_aca = use_aca
        self.options = kwargs

        # Initialize Green function with compression
        if use_hmatrix:
            from ..greenfun import HMatrixGreen
            self.g = HMatrixGreen(p, p, **kwargs)
        else:
            from ..greenfun import CompGreenStat
            self.g = CompGreenStat(p, p, **kwargs)

        # Cache
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

        # Compute Lambda factor
        self._Lambda = self.p.lambda_factor(enei)
        self._enei = enei

        # Setup preconditioner if needed
        if self.precond:
            self._setup_preconditioner()

    def _setup_preconditioner(self) -> None:
        """Setup preconditioner."""
        # Simple diagonal preconditioner
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

        return CompStruct(self.p, exc.enei, sig=sig)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def field(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
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
        from ..greenfun import GreenStat

        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        if pts is None:
            # Field at particle surface
            return self.g.field(charges)
        else:
            # Field at external points
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenStat(pts_obj, self.p.pc, **self.options)
            return g_pts.field(charges)

    def potential(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
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
        from ..greenfun import GreenStat

        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        if pts is None:
            return self.g.potential(charges)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenStat(pts_obj, self.p.pc, **self.options)
            return g_pts.potential(charges)

    def __repr__(self) -> str:
        return f"BEMStatIter(n={self.n_faces}, method='{self.method}')"


class BEMRetIter(BEMIter, BEMBase):
    """
    Iterative retarded BEM solver.

    Uses H-matrix compression for efficient large-scale simulations.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
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
        enei: Optional[float] = None,
        use_hmatrix: bool = True,
        **kwargs
    ):
        """Initialize iterative retarded solver."""
        tol = kwargs.pop('tol', 1e-6)
        maxiter = kwargs.pop('maxiter', 100)
        method = kwargs.pop('method', 'gmres')

        BEMIter.__init__(self, tol, maxiter, method)

        self.p = p
        self.use_hmatrix = use_hmatrix
        self.options = kwargs

        # Initialize Green function
        if use_hmatrix:
            from ..greenfun import HMatrixGreen
            self.g = HMatrixGreen(p, p, retarded=True, **kwargs)
        else:
            from ..greenfun import CompGreenRet
            self.g = CompGreenRet(p, p, **kwargs)

        self._Lambda = None
        self._enei = None
        self._k = None

        if enei is not None:
            self._setup(enei)

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _get_wavenumber(self, enei: float, medium_idx: int = 0) -> complex:
        """Get wavenumber."""
        eps_func = self.p.eps[medium_idx]
        _, k = eps_func(enei)
        return k

    def _setup(self, enei: float) -> None:
        """Setup for given wavelength."""
        if self._enei == enei:
            return

        n = self.n_faces
        k_bg = self._get_wavenumber(enei, 0)

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

        self._Lambda = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)
        self._enei = enei
        self._k = k_bg

    def _F_matvec(self, x: np.ndarray) -> np.ndarray:
        """Matrix-vector product with F matrix."""
        if self.use_hmatrix:
            return self.g.matvec(x, self._k)
        else:
            return self.g.F(self._k) @ x

    def solve(self, exc: CompStruct) -> CompStruct:
        """Solve BEM equations iteratively."""
        self._setup(exc.enei)

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        matvec = self._create_matvec(self._Lambda, self._F_matvec)
        sig, info = self._solve_iterative(matvec, -phip, self.n_faces)

        if info != 0:
            import warnings
            warnings.warn(f"Iterative solver did not converge: info={info}")

        return CompStruct(self.p, exc.enei, sig=sig, k=self._k)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def field(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
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
        from ..greenfun import GreenRet

        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        k = sig.get('k', self._k)

        if pts is None:
            return self.g.field(charges, k)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.field(charges, k)

    def potential(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
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
        from ..greenfun import GreenRet

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

            g_pts = GreenRet(pts_obj, self.p.pc, k=k, **self.options)
            return g_pts.potential(charges, k)

    def __repr__(self) -> str:
        return f"BEMRetIter(n={self.n_faces}, method='{self.method}')"
