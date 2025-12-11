"""
Retarded spectrum calculations for BEM simulations.

This module provides spectral response calculations with
full electromagnetic retardation effects.
"""

import numpy as np
from typing import Optional, List, Union, Callable


class SpectrumRet:
    """
    Retarded spectrum calculator.

    Computes optical spectra (scattering, extinction, absorption)
    over a range of wavelengths using the retarded BEM solver.

    Parameters
    ----------
    excitation : PlaneWaveRet or DipoleRet
        Excitation source
    particle : ComParticle
        Composite particle
    bem : BEMRet
        Retarded BEM solver
    options : BEMOptions, optional
        Simulation options

    Examples
    --------
    >>> spec = SpectrumRet(exc, particle, bem)
    >>> wavelengths = np.linspace(400, 800, 100)
    >>> sca, ext, abs = spec.compute(wavelengths)
    """

    def __init__(self, excitation, particle, bem, options=None):
        """Initialize spectrum calculator."""
        self.excitation = excitation
        self.particle = particle
        self.bem = bem
        self.options = options

        # Storage for computed spectra
        self._wavelengths = None
        self._solutions = {}
        self._sca = None
        self._ext = None
        self._abs = None

    def compute(self, wavelengths, quantities=None):
        """
        Compute optical spectra over wavelength range.

        Parameters
        ----------
        wavelengths : array_like
            Wavelengths in nm
        quantities : list, optional
            List of quantities to compute: 'sca', 'ext', 'abs', 'all'
            Default is ['sca', 'ext', 'abs']

        Returns
        -------
        spectra : dict
            Dictionary with computed spectra
        """
        wavelengths = np.atleast_1d(wavelengths)
        self._wavelengths = wavelengths

        if quantities is None:
            quantities = ['sca', 'ext', 'abs']
        if 'all' in quantities:
            quantities = ['sca', 'ext', 'abs']

        n_wl = len(wavelengths)
        n_pol = self.excitation.n_pol if hasattr(self.excitation, 'n_pol') else 1

        # Initialize arrays
        self._sca = np.zeros((n_wl, n_pol))
        self._ext = np.zeros((n_wl, n_pol))
        self._abs = np.zeros((n_wl, n_pol))

        # Loop over wavelengths
        for i, wl in enumerate(wavelengths):
            # Get excitation
            exc = self.excitation(self.particle, wl)

            # Solve BEM
            sig = self.bem.solve(exc)

            # Store solution
            self._solutions[wl] = sig

            # Compute cross sections
            if 'sca' in quantities or 'abs' in quantities:
                self._sca[i, :] = self.excitation.sca(sig)
            if 'ext' in quantities or 'abs' in quantities:
                self._ext[i, :] = self.excitation.ext(sig)
            if 'abs' in quantities:
                self._abs[i, :] = self._ext[i, :] - self._sca[i, :]

        # Return results
        result = {}
        if 'sca' in quantities:
            result['sca'] = self._sca
        if 'ext' in quantities:
            result['ext'] = self._ext
        if 'abs' in quantities:
            result['abs'] = self._abs

        return result

    def scattering(self, wavelengths=None):
        """
        Compute or return scattering spectrum.

        Parameters
        ----------
        wavelengths : array_like, optional
            Wavelengths to compute. If None, uses previous.

        Returns
        -------
        sca : ndarray
            Scattering cross section (n_wl, n_pol)
        """
        if wavelengths is not None:
            self.compute(wavelengths, ['sca'])
        return self._sca

    def extinction(self, wavelengths=None):
        """
        Compute or return extinction spectrum.

        Parameters
        ----------
        wavelengths : array_like, optional
            Wavelengths to compute

        Returns
        -------
        ext : ndarray
            Extinction cross section (n_wl, n_pol)
        """
        if wavelengths is not None:
            self.compute(wavelengths, ['ext'])
        return self._ext

    def absorption(self, wavelengths=None):
        """
        Compute or return absorption spectrum.

        Parameters
        ----------
        wavelengths : array_like, optional
            Wavelengths to compute

        Returns
        -------
        abs : ndarray
            Absorption cross section (n_wl, n_pol)
        """
        if wavelengths is not None:
            self.compute(wavelengths, ['abs'])
        return self._abs

    def efficiency(self, wavelengths=None, cross_section='ext'):
        """
        Compute efficiency factors Q = C / (pi * a^2).

        Parameters
        ----------
        wavelengths : array_like, optional
            Wavelengths to compute
        cross_section : str
            'sca', 'ext', or 'abs'

        Returns
        -------
        Q : ndarray
            Efficiency factors
        """
        if wavelengths is not None:
            self.compute(wavelengths, [cross_section])

        # Get cross section
        if cross_section == 'sca':
            C = self._sca
        elif cross_section == 'ext':
            C = self._ext
        elif cross_section == 'abs':
            C = self._abs
        else:
            raise ValueError(f"Unknown cross section: {cross_section}")

        # Estimate geometric cross section (area of bounding sphere)
        pos = self.particle.pos if hasattr(self.particle, 'pos') else self.particle.pc.pos
        r_max = np.max(np.linalg.norm(pos, axis=1))
        A_geom = np.pi * r_max**2

        return C / A_geom

    def farfield(self, wavelength, directions):
        """
        Compute far-field scattering at specific wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        directions : ndarray
            Scattering directions (n_dir, 3)

        Returns
        -------
        E_sca : ndarray
            Far-field scattering amplitude
        """
        if wavelength in self._solutions:
            sig = self._solutions[wavelength]
        else:
            exc = self.excitation(self.particle, wavelength)
            sig = self.bem.solve(exc)

        return self.excitation.farfield(sig, directions)

    def differential_scattering(self, wavelength, theta, phi=0):
        """
        Compute differential scattering cross section.

        dC/dOmega as a function of scattering angle.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        theta : array_like
            Polar angles (radians)
        phi : float or array_like
            Azimuthal angles (radians)

        Returns
        -------
        dC_dOmega : ndarray
            Differential cross section
        """
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        if len(phi) == 1:
            phi = np.full_like(theta, phi[0])

        # Convert to Cartesian directions
        directions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Get far-field
        E_ff = self.farfield(wavelength, directions)

        # Differential cross section ~ |E_ff|^2
        k = 2 * np.pi / wavelength

        dC_dOmega = np.sum(np.abs(E_ff)**2, axis=-1) / k**2

        return dC_dOmega

    @property
    def wavelengths(self):
        """Return computed wavelengths."""
        return self._wavelengths

    def get_solution(self, wavelength):
        """
        Get stored BEM solution for wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm

        Returns
        -------
        sig : Solution
            BEM solution object
        """
        if wavelength not in self._solutions:
            exc = self.excitation(self.particle, wavelength)
            sig = self.bem.solve(exc)
            self._solutions[wavelength] = sig

        return self._solutions[wavelength]


class DecayRateSpectrum:
    """
    Compute decay rate spectra for dipole near nanoparticle.

    Parameters
    ----------
    dipole : DipoleRet
        Dipole excitation
    particle : ComParticle
        Composite particle
    bem : BEMRet
        Retarded BEM solver
    """

    def __init__(self, dipole, particle, bem, options=None):
        """Initialize decay rate spectrum calculator."""
        self.dipole = dipole
        self.particle = particle
        self.bem = bem
        self.options = options

        self._wavelengths = None
        self._gamma = None

    def compute(self, wavelengths):
        """
        Compute decay rate enhancement spectrum.

        Parameters
        ----------
        wavelengths : array_like
            Wavelengths in nm

        Returns
        -------
        gamma : ndarray
            Decay rate enhancement (n_wl, n_dip)
        """
        wavelengths = np.atleast_1d(wavelengths)
        self._wavelengths = wavelengths

        n_wl = len(wavelengths)
        n_dip = self.dipole.n_dip

        self._gamma = np.zeros((n_wl, n_dip))

        for i, wl in enumerate(wavelengths):
            exc = self.dipole(self.particle, wl)
            sig = self.bem.solve(exc)
            self._gamma[i, :] = self.dipole.decayrate(sig)

        return self._gamma

    def purcell_factor(self, wavelengths=None):
        """
        Compute Purcell factor spectrum.

        Same as decay rate enhancement.
        """
        if wavelengths is not None:
            return self.compute(wavelengths)
        return self._gamma

    def quantum_efficiency(self, wavelengths, eta_0=1.0):
        """
        Compute quantum efficiency near particle.

        Parameters
        ----------
        wavelengths : array_like
            Wavelengths in nm
        eta_0 : float
            Intrinsic quantum efficiency

        Returns
        -------
        eta : ndarray
            Modified quantum efficiency
        """
        gamma = self.compute(wavelengths)

        # eta = eta_0 * gamma_rad / (eta_0 * gamma_rad + (1-eta_0) * gamma_nrad)
        # Simplified: eta = gamma_rad / gamma_total
        # For now, assume gamma is total decay rate
        eta = eta_0 * gamma / (eta_0 * gamma + (1 - eta_0))

        return eta


def spectrum_ret(excitation, particle, bem, options=None):
    """
    Factory function for retarded spectrum calculator.

    Parameters
    ----------
    excitation : PlaneWaveRet or DipoleRet
        Excitation source
    particle : ComParticle
        Composite particle
    bem : BEMRet
        Retarded BEM solver
    options : BEMOptions, optional
        Options

    Returns
    -------
    SpectrumRet
        Spectrum calculator
    """
    return SpectrumRet(excitation, particle, bem, options)
