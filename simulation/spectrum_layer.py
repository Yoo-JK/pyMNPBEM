"""
Spectral response calculations for layer structures.

Provides spectrum calculations for particles on or near substrates.
"""

import numpy as np
from typing import Optional, List, Union, Callable
from tqdm import tqdm

from ..particles import ComParticle, CompStruct


class SpectrumStatLayer:
    """
    Quasistatic spectrum calculator for layer structures.

    Computes optical spectra for particles on substrates using
    the quasistatic approximation.

    Parameters
    ----------
    bem : BEMStatLayer
        BEM solver with layer structure.
    exc : PlaneWaveStatLayer or DipoleStatLayer
        Excitation source.
    wavelengths : ndarray
        Wavelengths in nm.
    show_progress : bool
        Show progress bar.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.bem import BEMStatLayer
    >>> from pymnpbem.simulation import PlaneWaveStatLayer, SpectrumStatLayer
    >>>
    >>> eps_air = EpsConst(1)
    >>> eps_glass = EpsConst(2.25)
    >>> layer = LayerStructure([eps_air, eps_glass])
    >>>
    >>> eps_gold = EpsTable('gold.dat')
    >>> sphere = trisphere(144, 50).shift([0, 0, 60])
    >>> p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    >>> bem = BEMStatLayer(p, layer)
    >>> exc = PlaneWaveStatLayer([1, 0, 0], layer)
    >>> wavelengths = np.linspace(400, 800, 50)
    >>> spec = SpectrumStatLayer(bem, exc, wavelengths)
    """

    def __init__(
        self,
        bem,
        exc,
        wavelengths: np.ndarray,
        show_progress: bool = True
    ):
        """Initialize spectrum calculator."""
        self.bem = bem
        self.exc = exc
        self.wavelengths = np.asarray(wavelengths)
        self.show_progress = show_progress

        # Results storage
        self._sca = None
        self._ext = None
        self._abs = None

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return len(self.wavelengths)

    def compute(self) -> tuple:
        """
        Compute optical spectra.

        Returns
        -------
        sca : ndarray
            Scattering cross section.
        ext : ndarray
            Extinction cross section.
        """
        n_exc = getattr(self.exc, 'n_exc', 1)

        sca = np.zeros((self.n_wavelengths, n_exc))
        ext = np.zeros((self.n_wavelengths, n_exc))
        abs_cs = np.zeros((self.n_wavelengths, n_exc))

        iterator = enumerate(self.wavelengths)
        if self.show_progress:
            iterator = tqdm(list(iterator), desc="Computing spectrum (layer)")

        for i, enei in iterator:
            # Get excitation
            exc_struct = self.exc(self.bem.p, enei)

            # Solve BEM equations
            sig = self.bem.solve(exc_struct)

            # Compute cross sections
            sca[i] = self.exc.sca(sig) if hasattr(self.exc, 'sca') else 0
            ext[i] = self.exc.ext(sig) if hasattr(self.exc, 'ext') else 0
            abs_cs[i] = self.exc.abs(sig) if hasattr(self.exc, 'abs') else ext[i] - sca[i]

        self._sca = sca
        self._ext = ext
        self._abs = abs_cs

        return sca, ext

    @property
    def sca(self) -> np.ndarray:
        """Scattering cross section."""
        if self._sca is None:
            self.compute()
        return self._sca

    @property
    def ext(self) -> np.ndarray:
        """Extinction cross section."""
        if self._ext is None:
            self.compute()
        return self._ext

    @property
    def abs(self) -> np.ndarray:
        """Absorption cross section."""
        if self._abs is None:
            self.compute()
        return self._abs

    def farfield(self, wavelength: float, directions: np.ndarray) -> np.ndarray:
        """
        Compute far-field at specific wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        directions : ndarray
            Scattering directions (n_dir, 3).

        Returns
        -------
        E_ff : ndarray
            Far-field amplitude.
        """
        exc_struct = self.exc(self.bem.p, wavelength)
        sig = self.bem.solve(exc_struct)

        if hasattr(self.exc, 'farfield'):
            return self.exc.farfield(sig, directions)
        else:
            raise NotImplementedError("Excitation does not support farfield")

    def efarfield(
        self,
        wavelength: float,
        directions: np.ndarray,
        include_reflected: bool = True
    ) -> np.ndarray:
        """
        Compute extended far-field including substrate reflection.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        directions : ndarray
            Scattering directions (n_dir, 3).
        include_reflected : bool
            Include reflection from substrate.

        Returns
        -------
        E_ff : ndarray
            Extended far-field amplitude.
        """
        E_ff = self.farfield(wavelength, directions)

        if include_reflected and hasattr(self.bem, 'layer'):
            # Add reflected component
            layer = self.bem.layer
            for i, d in enumerate(directions):
                if d[2] < 0:  # Downward direction
                    # Compute reflection contribution
                    theta = np.arccos(-d[2])
                    k0 = 2 * np.pi / wavelength
                    kpar = k0 * np.sin(theta)
                    r_p = layer.reflection(wavelength, np.array([kpar]), 'p')[0]
                    E_ff[i] *= (1 + r_p)

        return E_ff

    def __repr__(self) -> str:
        return f"SpectrumStatLayer(n_wavelengths={self.n_wavelengths})"


class SpectrumRetLayer:
    """
    Retarded spectrum calculator for layer structures.

    Computes optical spectra for particles on substrates with
    full electromagnetic retardation.

    Parameters
    ----------
    bem : BEMRetLayer
        Retarded BEM solver with layer structure.
    exc : PlaneWaveRetLayer or DipoleRetLayer
        Excitation source.
    wavelengths : ndarray
        Wavelengths in nm.
    show_progress : bool
        Show progress bar.

    Examples
    --------
    >>> from pymnpbem import ComParticle, EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import LayerStructure
    >>> from pymnpbem.bem import BEMRetLayer
    >>> from pymnpbem.simulation import PlaneWaveRetLayer, SpectrumRetLayer
    >>>
    >>> eps_air = EpsConst(1)
    >>> eps_glass = EpsConst(2.25)
    >>> layer = LayerStructure([eps_air, eps_glass])
    >>>
    >>> eps_gold = EpsTable('gold.dat')
    >>> sphere = trisphere(144, 50).shift([0, 0, 60])
    >>> p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    >>> bem = BEMRetLayer(p, layer)
    >>> exc = PlaneWaveRetLayer([1, 0, 0], layer)
    >>> wavelengths = np.linspace(400, 800, 50)
    >>> spec = SpectrumRetLayer(bem, exc, wavelengths)
    """

    def __init__(
        self,
        bem,
        exc,
        wavelengths: np.ndarray,
        show_progress: bool = True
    ):
        """Initialize spectrum calculator."""
        self.bem = bem
        self.exc = exc
        self.wavelengths = np.asarray(wavelengths)
        self.show_progress = show_progress

        # Results storage
        self._sca = None
        self._ext = None
        self._abs = None
        self._solutions = {}

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return len(self.wavelengths)

    def compute(self, quantities: Optional[List[str]] = None) -> dict:
        """
        Compute optical spectra.

        Parameters
        ----------
        quantities : list, optional
            Quantities to compute: 'sca', 'ext', 'abs'.

        Returns
        -------
        dict
            Dictionary with computed spectra.
        """
        if quantities is None:
            quantities = ['sca', 'ext', 'abs']

        n_exc = getattr(self.exc, 'n_exc', 1)
        n_pol = getattr(self.exc, 'n_pol', n_exc)

        sca = np.zeros((self.n_wavelengths, n_pol))
        ext = np.zeros((self.n_wavelengths, n_pol))
        abs_cs = np.zeros((self.n_wavelengths, n_pol))

        iterator = enumerate(self.wavelengths)
        if self.show_progress:
            iterator = tqdm(list(iterator), desc="Computing spectrum (ret layer)")

        for i, enei in iterator:
            # Get excitation
            exc_struct = self.exc(self.bem.p, enei)

            # Solve BEM equations
            sig = self.bem.solve(exc_struct)

            # Store solution
            self._solutions[enei] = sig

            # Compute cross sections
            if 'sca' in quantities:
                sca[i] = self.exc.sca(sig) if hasattr(self.exc, 'sca') else 0
            if 'ext' in quantities:
                ext[i] = self.exc.ext(sig) if hasattr(self.exc, 'ext') else 0
            if 'abs' in quantities:
                if hasattr(self.exc, 'abs'):
                    abs_cs[i] = self.exc.abs(sig)
                else:
                    abs_cs[i] = ext[i] - sca[i]

        self._sca = sca
        self._ext = ext
        self._abs = abs_cs

        result = {}
        if 'sca' in quantities:
            result['sca'] = sca
        if 'ext' in quantities:
            result['ext'] = ext
        if 'abs' in quantities:
            result['abs'] = abs_cs

        return result

    def scattering(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute or return scattering spectrum.

        Parameters
        ----------
        wavelengths : ndarray, optional
            Wavelengths to compute.

        Returns
        -------
        sca : ndarray
            Scattering cross section.
        """
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths)
            self.compute(['sca'])
        elif self._sca is None:
            self.compute(['sca'])
        return self._sca

    def extinction(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute or return extinction spectrum.

        Parameters
        ----------
        wavelengths : ndarray, optional
            Wavelengths to compute.

        Returns
        -------
        ext : ndarray
            Extinction cross section.
        """
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths)
            self.compute(['ext'])
        elif self._ext is None:
            self.compute(['ext'])
        return self._ext

    def absorption(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute or return absorption spectrum.

        Parameters
        ----------
        wavelengths : ndarray, optional
            Wavelengths to compute.

        Returns
        -------
        abs : ndarray
            Absorption cross section.
        """
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths)
            self.compute(['abs'])
        elif self._abs is None:
            self.compute(['abs'])
        return self._abs

    @property
    def sca(self) -> np.ndarray:
        """Scattering cross section."""
        if self._sca is None:
            self.compute()
        return self._sca

    @property
    def ext(self) -> np.ndarray:
        """Extinction cross section."""
        if self._ext is None:
            self.compute()
        return self._ext

    @property
    def abs(self) -> np.ndarray:
        """Absorption cross section."""
        if self._abs is None:
            self.compute()
        return self._abs

    def farfield(self, wavelength: float, directions: np.ndarray) -> np.ndarray:
        """
        Compute far-field scattering at specific wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        directions : ndarray
            Scattering directions (n_dir, 3).

        Returns
        -------
        E_ff : ndarray
            Far-field scattering amplitude.
        """
        if wavelength in self._solutions:
            sig = self._solutions[wavelength]
        else:
            exc_struct = self.exc(self.bem.p, wavelength)
            sig = self.bem.solve(exc_struct)
            self._solutions[wavelength] = sig

        if hasattr(self.exc, 'farfield'):
            return self.exc.farfield(sig, directions)
        else:
            raise NotImplementedError("Excitation does not support farfield")

    def efarfield(
        self,
        wavelength: float,
        directions: np.ndarray,
        include_reflected: bool = True
    ) -> np.ndarray:
        """
        Compute extended far-field including substrate effects.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        directions : ndarray
            Scattering directions (n_dir, 3).
        include_reflected : bool
            Include reflection from substrate.

        Returns
        -------
        E_ff : ndarray
            Extended far-field amplitude.
        """
        E_ff = self.farfield(wavelength, directions)

        if include_reflected and hasattr(self.bem, 'layer'):
            layer = self.bem.layer
            k0 = 2 * np.pi / wavelength

            for i, d in enumerate(directions):
                d_norm = d / np.linalg.norm(d)
                if d_norm[2] < 0:  # Downward direction
                    # Compute reflection for this direction
                    sin_theta = np.sqrt(d_norm[0]**2 + d_norm[1]**2)
                    kpar = k0 * sin_theta

                    # Get reflection coefficients
                    r_p = layer.reflection(wavelength, np.array([kpar]), 'p')[0]
                    r_s = layer.reflection(wavelength, np.array([kpar]), 's')[0]

                    # Apply polarization-dependent reflection
                    # Simplified: use average
                    r_avg = (r_p + r_s) / 2
                    E_ff[i] *= (1 + r_avg)

        return E_ff

    def get_solution(self, wavelength: float) -> CompStruct:
        """
        Get stored BEM solution for wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        sig : CompStruct
            BEM solution.
        """
        if wavelength not in self._solutions:
            exc_struct = self.exc(self.bem.p, wavelength)
            sig = self.bem.solve(exc_struct)
            self._solutions[wavelength] = sig

        return self._solutions[wavelength]

    def __repr__(self) -> str:
        return f"SpectrumRetLayer(n_wavelengths={self.n_wavelengths})"


def spectrum_stat_layer(bem, exc, wavelengths, **kwargs):
    """Factory function for quasistatic layer spectrum."""
    return SpectrumStatLayer(bem, exc, wavelengths, **kwargs)


def spectrum_ret_layer(bem, exc, wavelengths, **kwargs):
    """Factory function for retarded layer spectrum."""
    return SpectrumRetLayer(bem, exc, wavelengths, **kwargs)
