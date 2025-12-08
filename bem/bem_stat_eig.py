"""
Eigenvalue BEM solver for plasmon mode analysis.

This solver computes plasmonic eigenmodes without requiring
a specific material, using the eigenvalue decomposition of
the BEM matrix.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, List, Any

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct


class BEMStatEig(BEMBase):
    """
    Eigenvalue BEM solver in quasistatic approximation.

    Computes the eigenvalues and eigenmodes of the BEM matrix,
    which correspond to plasmonic resonances.

    The eigenvalue problem is:
        Lambda * sig = F * sig

    where Lambda = (eps_in + eps_out) / (eps_in - eps_out)
    determines the resonance condition for a given material.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    n_modes : int, optional
        Number of modes to compute.
    **kwargs : dict
        Options.

    Attributes
    ----------
    eigenvalues : ndarray
        Eigenvalues Lambda.
    eigenmodes : ndarray
        Eigenmodes (columns are surface charge distributions).
    resonance_eps : ndarray
        Dielectric function at resonance for each mode.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, trisphere
    >>> from pymnpbem.bem import BEMStatEig
    >>>
    >>> eps = [EpsConst(1), EpsConst(-10)]  # Dummy dielectric
    >>> sphere = trisphere(144, 10)
    >>> p = ComParticle(eps, [sphere], [[2, 1]])
    >>> bem = BEMStatEig(p)
    >>> bem.compute()
    >>> print(bem.eigenvalues[:5])  # First 5 eigenvalues
    """

    def __init__(
        self,
        p: ComParticle,
        n_modes: Optional[int] = None,
        **kwargs
    ):
        """Initialize eigenvalue BEM solver."""
        from ..greenfun import CompGreenStat

        self.p = p
        self.n_modes = n_modes
        self.options = kwargs

        # Green function
        self.g = CompGreenStat(p, p, **kwargs)

        # Results
        self.eigenvalues = None
        self.eigenmodes = None
        self.resonance_eps = None
        self._computed = False

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F

    def compute(self, n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute plasmon eigenmodes.

        Parameters
        ----------
        n_modes : int, optional
            Number of modes to compute.

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues Lambda.
        eigenmodes : ndarray
            Eigenmodes (surface charges).
        """
        if n_modes is not None:
            self.n_modes = n_modes

        F = self.F
        n = self.n_faces

        if self.n_modes is None or self.n_modes >= n:
            # Full eigenvalue decomposition
            eigenvalues, eigenmodes = linalg.eig(F)
        else:
            # Sparse eigenvalue decomposition
            from scipy.sparse.linalg import eigs
            eigenvalues, eigenmodes = eigs(F, k=self.n_modes, which='LM')

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenmodes = eigenmodes[:, idx]

        # Compute resonance dielectric functions
        # Lambda = (eps_in + eps_out) / (eps_in - eps_out)
        # For eps_out = 1: eps_in = (Lambda + 1) / (Lambda - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.resonance_eps = (self.eigenvalues + 1) / (self.eigenvalues - 1)

        self._computed = True
        return self.eigenvalues, self.eigenmodes

    def solve(self, exc: CompStruct, mode_indices: Optional[List[int]] = None) -> CompStruct:
        """
        Solve for response by projecting onto eigenmodes.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field.
        mode_indices : list of int, optional
            Which modes to include. Default is all modes.

        Returns
        -------
        CompStruct
            Solution with 'sig' field.
        """
        if not self._computed:
            self.compute()

        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        # Get Lambda factor for given wavelength/material
        Lambda_target = self.p.lambda_factor(exc.enei)

        # Average Lambda (assume uniform material)
        Lambda_avg = np.mean(Lambda_target)

        # Select modes
        if mode_indices is None:
            mode_indices = list(range(len(self.eigenvalues)))

        # Project excitation onto eigenmodes and solve
        sig = np.zeros(self.n_faces, dtype=complex)

        for i in mode_indices:
            if i >= len(self.eigenvalues):
                continue

            mode = self.eigenmodes[:, i]
            eigenval = self.eigenvalues[i]

            # Projection coefficient
            coeff = np.dot(mode.conj(), phip)

            # Response: -1 / (Lambda_avg - eigenval)
            denom = Lambda_avg - eigenval
            if np.abs(denom) > 1e-10:
                sig += -coeff / denom * mode

        return CompStruct(self.p, exc.enei, sig=sig)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax."""
        return self.solve(exc)

    def mode_dipole(self, mode_index: int) -> np.ndarray:
        """
        Compute dipole moment of a plasmon mode.

        Parameters
        ----------
        mode_index : int
            Mode index.

        Returns
        -------
        ndarray
            Dipole moment vector.
        """
        if not self._computed:
            self.compute()

        sigma = self.eigenmodes[:, mode_index]
        pos = self.p.pc.pos
        area = self.p.pc.area

        return np.sum(sigma[:, np.newaxis] * area[:, np.newaxis] * pos, axis=0)

    def is_dipole_active(self, mode_index: int, threshold: float = 0.1) -> bool:
        """
        Check if mode is dipole-active (bright).

        Parameters
        ----------
        mode_index : int
            Mode index.
        threshold : float
            Relative threshold for dipole moment.

        Returns
        -------
        bool
            True if mode is bright.
        """
        dipole = self.mode_dipole(mode_index)
        dipole_mag = np.linalg.norm(dipole)

        sigma = self.eigenmodes[:, mode_index]
        area = self.p.pc.area
        total_charge = np.sum(np.abs(sigma) * area)

        # Characteristic length
        size = np.max(self.p.pc.pos, axis=0) - np.min(self.p.pc.pos, axis=0)
        char_length = np.mean(size)

        return dipole_mag / (total_charge * char_length + 1e-10) > threshold

    def find_resonance(
        self,
        eps_func: Any,
        wavelength_range: Tuple[float, float] = (300, 1000),
        n_points: int = 100
    ) -> List[dict]:
        """
        Find resonance wavelengths for a given material.

        Parameters
        ----------
        eps_func : callable
            Material dielectric function eps(wavelength).
        wavelength_range : tuple
            (min, max) wavelength in nm.
        n_points : int
            Number of wavelength points.

        Returns
        -------
        list of dict
            Resonance information for each mode.
        """
        if not self._computed:
            self.compute()

        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

        resonances = []

        for i, (eigenval, eps_res) in enumerate(zip(self.eigenvalues, self.resonance_eps)):
            if np.abs(eigenval) < 1e-6 or np.abs(eigenval) > 1e6:
                continue

            # Find wavelength where Re(eps) matches resonance condition
            eps_values = np.array([eps_func(wl)[0] for wl in wavelengths])
            diff = np.real(eps_values) - np.real(eps_res)

            # Find zero crossings
            crossings = np.where(np.diff(np.sign(diff)))[0]

            for cross in crossings:
                # Linear interpolation
                wl_res = wavelengths[cross] - diff[cross] * \
                        (wavelengths[cross+1] - wavelengths[cross]) / \
                        (diff[cross+1] - diff[cross])

                resonances.append({
                    'mode': i,
                    'eigenvalue': eigenval,
                    'resonance_eps': eps_res,
                    'wavelength': wl_res,
                    'dipole_active': self.is_dipole_active(i)
                })

        return resonances

    def __repr__(self) -> str:
        status = "computed" if self._computed else "not computed"
        return f"BEMStatEig(n={self.n_faces}, {status})"


class BEMStatEigMirror(BEMStatEig):
    """
    Eigenvalue BEM solver with mirror symmetry.

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry.
    **kwargs : dict
        Options.
    """

    def __init__(self, p, n_modes: Optional[int] = None, **kwargs):
        """Initialize eigenvalue mirror BEM solver."""
        from ..greenfun import CompGreenStatMirror

        self.p = p
        self.n_modes = n_modes
        self.options = kwargs

        self.g = CompGreenStatMirror(p, **kwargs)

        # Store results for each symmetry
        self.eigenvalues_by_sym = {}
        self.eigenmodes_by_sym = {}
        self.resonance_eps_by_sym = {}
        self._computed = False

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F

    def compute(self, symkey: str = '+', n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenmodes for given symmetry configuration.

        Parameters
        ----------
        symkey : str
            Symmetry key.
        n_modes : int, optional
            Number of modes.

        Returns
        -------
        eigenvalues, eigenmodes
        """
        if n_modes is not None:
            self.n_modes = n_modes

        # Get F matrix with symmetry
        F_base = self.F
        symval = self.p.symvalue(symkey)

        # Apply symmetry
        n = self.p.n_faces
        n_full = F_base.shape[1]
        n_copies = n_full // n

        F_sym = np.zeros((n, n), dtype=F_base.dtype)
        for k in range(n_copies):
            factor = symval[k] if k < len(symval) else 1
            F_sym += factor * F_base[:, k*n:(k+1)*n]

        # Compute eigenvalues
        if self.n_modes is None or self.n_modes >= n:
            eigenvalues, eigenmodes = linalg.eig(F_sym)
        else:
            from scipy.sparse.linalg import eigs
            eigenvalues, eigenmodes = eigs(F_sym, k=self.n_modes, which='LM')

        # Sort
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenmodes = eigenmodes[:, idx]

        # Store results
        self.eigenvalues_by_sym[symkey] = eigenvalues
        self.eigenmodes_by_sym[symkey] = eigenmodes

        with np.errstate(divide='ignore', invalid='ignore'):
            self.resonance_eps_by_sym[symkey] = (eigenvalues + 1) / (eigenvalues - 1)

        # Set default properties
        self.eigenvalues = eigenvalues
        self.eigenmodes = eigenmodes
        self.resonance_eps = self.resonance_eps_by_sym[symkey]

        self._computed = True
        return eigenvalues, eigenmodes

    def compute_all(self, n_modes: Optional[int] = None) -> dict:
        """
        Compute eigenmodes for all symmetry configurations.

        Returns
        -------
        dict
            Results for each symmetry key.
        """
        symkeys = ['+', '-'] if self.p.sym in ('x', 'y') else ['++', '+-', '-+', '--']

        results = {}
        for key in symkeys:
            eigenvalues, eigenmodes = self.compute(key, n_modes)
            results[key] = {'eigenvalues': eigenvalues, 'eigenmodes': eigenmodes}

        return results

    def __repr__(self) -> str:
        status = "computed" if self._computed else "not computed"
        return f"BEMStatEigMirror(n={self.p.n_faces}, sym='{self.p.sym}', {status})"
