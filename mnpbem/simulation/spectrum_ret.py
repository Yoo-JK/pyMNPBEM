"""
Retarded spectrum calculations for BEM simulations.

This module provides spectral response calculations with
full electromagnetic retardation effects.

Implements far-field calculations and optical cross sections
following Garcia de Abajo, RMP 82, 209 (2010).
"""

import numpy as np
from typing import Optional, List, Union, Callable, Tuple

from ..particles import CompStruct


class SpectrumRet:
    """
    Retarded spectrum calculator.

    Computes far-field scattering amplitudes and optical cross sections
    (scattering, extinction, absorption) for retarded BEM solutions.

    The far-field is computed from surface charges (sig1, sig2) and
    surface currents (h1, h2) using:
        E_ff = i*k0 * phase @ h - i*k * dir x (phase @ sig)

    References
    ----------
    Garcia de Abajo, RMP 82, 209 (2010), Eq. (50)
    Garcia de Abajo & Howie, PRB 65, 115418 (2002)

    Parameters
    ----------
    pinfty : Particle, optional
        Discretized unit sphere for scattering integration.
        If None, a default sphere is created.
    medium : int, optional
        Index of embedding medium (1-based, default 1).

    Examples
    --------
    >>> spec = SpectrumRet()
    >>> exc = PlaneWaveRet(pol=[1,0,0], dir=[0,0,1])
    >>> exc.set_spectrum(spec)
    >>> sig = bem.solve(exc(particle, wavelength))
    >>> field, k = spec.farfield(sig, directions)
    """

    def __init__(self, pinfty=None, medium=1, **kwargs):
        """
        Initialize spectrum calculator.

        Parameters
        ----------
        pinfty : Particle, optional
            Unit sphere for angular integration.
        medium : int, optional
            Embedding medium index (1-based).
        """
        self.medium = medium
        self.options = kwargs

        if pinfty is None:
            # Create default unit sphere
            self.pinfty = self._create_unit_sphere()
        else:
            self.pinfty = pinfty

    def _create_unit_sphere(self, n_theta=20, n_phi=40):
        """
        Create discretized unit sphere for angular integration.

        Parameters
        ----------
        n_theta : int
            Number of polar angle divisions.
        n_phi : int
            Number of azimuthal angle divisions.

        Returns
        -------
        UnitSphere
            Object with pos, nvec, area attributes.
        """
        class UnitSphere:
            pass

        sphere = UnitSphere()

        # Create angular grid
        theta = np.linspace(0, np.pi, n_theta + 1)[1:-1]  # Exclude poles
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        # Cartesian positions on unit sphere
        x = np.sin(theta_flat) * np.cos(phi_flat)
        y = np.sin(theta_flat) * np.sin(phi_flat)
        z = np.cos(theta_flat)

        sphere.pos = np.column_stack([x, y, z])
        sphere.nvec = sphere.pos.copy()  # Outward normals = positions for unit sphere

        # Angular element dΩ = sin(θ) dθ dφ
        dtheta = np.pi / n_theta
        dphi = 2 * np.pi / n_phi
        sphere.area = np.sin(theta_flat) * dtheta * dphi

        sphere.n = len(theta_flat)

        return sphere

    def farfield(self, sig: CompStruct, directions=None) -> Tuple[CompStruct, complex]:
        """
        Compute electromagnetic far-field from surface charge distribution.

        Following MATLAB spectrumret/farfield.m and
        Garcia de Abajo, RMP 82, 209 (2010), Eq. (50).

        Parameters
        ----------
        sig : CompStruct
            BEM solution with sig1, sig2 (charges) and h1, h2 (currents).
        directions : ndarray, optional
            Scattering directions (n_dir, 3). If None, uses pinfty.nvec.

        Returns
        -------
        field : CompStruct
            Far-field with 'e' (electric) and 'h' (magnetic) fields.
        k : complex
            Wavenumber in medium.
        """
        if directions is None:
            directions = self.pinfty.nvec

        directions = np.atleast_2d(directions)
        n_dir = len(directions)

        # Normalize directions
        dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(dir_norms > 1e-10, dir_norms, 1.0)

        # Get particle data
        p = sig.p
        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        area = p.area if hasattr(p, 'area') else p.pc.area
        inout = p.inout if hasattr(p, 'inout') else np.array([[2, 1]] * len(p.p))

        # Get wavenumber
        eps_med = p.eps[self.medium - 1](sig.enei)
        if isinstance(eps_med, tuple):
            eps_med, k = eps_med
        else:
            k = 2 * np.pi * np.sqrt(eps_med) / sig.enei

        k0 = 2 * np.pi / sig.enei

        # Get surface charges and currents
        sig1 = sig.get('sig1')
        sig2 = sig.get('sig2')
        h1 = sig.get('h1')
        h2 = sig.get('h2')

        # Determine number of polarizations
        if sig1 is not None:
            n_pol = sig1.shape[1] if sig1.ndim > 1 else 1
        elif sig2 is not None:
            n_pol = sig2.shape[1] if sig2.ndim > 1 else 1
        else:
            # Fallback to simple sig
            sig_simple = sig.get('sig')
            if sig_simple is None:
                raise ValueError("Solution must have sig1/sig2 or sig field")
            n_pol = sig_simple.shape[1] if sig_simple.ndim > 1 else 1
            sig2 = sig_simple
            h2 = np.zeros((len(sig_simple), 3, n_pol), dtype=complex)

        # Initialize far-field arrays
        E_ff = np.zeros((n_dir, 3, n_pol), dtype=complex)
        H_ff = np.zeros((n_dir, 3, n_pol), dtype=complex)

        # Phase factor matrix: exp(-i*k*dir.pos) * area
        # shape: (n_dir, n_faces)
        phase = np.exp(-1j * k * directions @ pos.T) * area[np.newaxis, :]

        # Get face indices for inner and outer surfaces that belong to medium
        face_start = 0
        inner_faces = []
        outer_faces = []

        for p_idx, particle in enumerate(p.p):
            n_faces_p = particle.n_faces
            face_end = face_start + n_faces_p

            if inout[p_idx, 0] == self.medium:  # Inner surface in medium
                inner_faces.extend(range(face_start, face_end))
            if inout[p_idx, 1] == self.medium:  # Outer surface in medium
                outer_faces.extend(range(face_start, face_end))

            face_start = face_end

        inner_faces = np.array(inner_faces, dtype=int)
        outer_faces = np.array(outer_faces, dtype=int)

        # Compute far-field for each polarization
        for i_pol in range(n_pol):
            e = np.zeros((n_dir, 3), dtype=complex)
            h = np.zeros((n_dir, 3), dtype=complex)

            # Contribution from inner surface
            if len(inner_faces) > 0 and sig1 is not None and h1 is not None:
                sig1_i = sig1[:, i_pol] if sig1.ndim > 1 else sig1
                h1_i = h1[:, :, i_pol] if h1.ndim > 2 else h1

                # phase @ h1 (n_dir, 3)
                phase_h1 = phase[:, inner_faces] @ h1_i[inner_faces, :]

                # phase @ sig1 (n_dir,)
                phase_sig1 = phase[:, inner_faces] @ sig1_i[inner_faces]

                # E = i*k0 * phase @ h - i*k * outer(dir, phase @ sig)
                e += 1j * k0 * phase_h1
                # outer(dir, phase_sig) = dir * phase_sig (broadcasting)
                e -= 1j * k * directions * phase_sig1[:, np.newaxis]

                # H = i*k * cross(dir, phase @ h)
                for j_dir in range(n_dir):
                    h[j_dir, :] += 1j * k * np.cross(directions[j_dir], phase_h1[j_dir])

            # Contribution from outer surface
            if len(outer_faces) > 0 and sig2 is not None and h2 is not None:
                sig2_i = sig2[:, i_pol] if sig2.ndim > 1 else sig2
                h2_i = h2[:, :, i_pol] if h2.ndim > 2 else h2

                # phase @ h2 (n_dir, 3)
                phase_h2 = phase[:, outer_faces] @ h2_i[outer_faces, :]

                # phase @ sig2 (n_dir,)
                phase_sig2 = phase[:, outer_faces] @ sig2_i[outer_faces]

                # E contribution
                e += 1j * k0 * phase_h2
                e -= 1j * k * directions * phase_sig2[:, np.newaxis]

                # H contribution
                for j_dir in range(n_dir):
                    h[j_dir, :] += 1j * k * np.cross(directions[j_dir], phase_h2[j_dir])

            E_ff[:, :, i_pol] = e
            H_ff[:, :, i_pol] = h

        # Create output CompStruct
        field = CompStruct(p, sig.enei, e=E_ff, h=H_ff)

        return field, k

    def scattering(self, sig: CompStruct) -> Tuple[np.ndarray, dict]:
        """
        Compute scattering cross section from far-field.

        C_sca = integral of |E_ff|^2 dΩ / (incoming intensity)

        Parameters
        ----------
        sig : CompStruct
            BEM solution.

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization.
        dsca : dict
            Differential scattering cross section {'dsca': array}.
        """
        # Get far-field on unit sphere
        field, k = self.farfield(sig)

        E_ff = field.e  # (n_dir, 3, n_pol)
        area = self.pinfty.area

        if E_ff.ndim == 2:
            E_ff = E_ff[:, :, np.newaxis]

        n_pol = E_ff.shape[2]

        # Differential scattering cross section: dσ/dΩ = |E_ff|^2
        dsca = np.sum(np.abs(E_ff)**2, axis=1)  # (n_dir, n_pol)

        # Total scattering: integrate over sphere
        sca = np.zeros(n_pol)
        for i_pol in range(n_pol):
            sca[i_pol] = np.sum(dsca[:, i_pol] * area)

        return sca, {'dsca': dsca}


class SpectrumRetCalculator:
    """
    Full spectrum calculator for retarded BEM.

    Computes optical spectra over wavelength range.

    Parameters
    ----------
    excitation : PlaneWaveRet
        Excitation source.
    particle : ComParticle
        Composite particle.
    bem : BEMRet
        Retarded BEM solver.
    options : dict, optional
        Simulation options.
    """

    def __init__(self, excitation, particle, bem, options=None):
        """Initialize spectrum calculator."""
        self.excitation = excitation
        self.particle = particle
        self.bem = bem
        self.options = options if options is not None else {}

        # Create spectrum object and link to excitation
        self.spec = SpectrumRet(medium=getattr(excitation, 'medium', 1))
        if hasattr(excitation, 'set_spectrum'):
            excitation.set_spectrum(self.spec)

        # Storage
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
            Wavelengths in nm.
        quantities : list, optional
            List of quantities: 'sca', 'ext', 'abs', 'all'.

        Returns
        -------
        spectra : dict
            Dictionary with computed spectra.
        """
        wavelengths = np.atleast_1d(wavelengths)
        self._wavelengths = wavelengths

        if quantities is None:
            quantities = ['sca', 'ext', 'abs']
        if 'all' in quantities:
            quantities = ['sca', 'ext', 'abs']

        n_wl = len(wavelengths)
        n_pol = self.excitation.n_pol

        self._sca = np.zeros((n_wl, n_pol))
        self._ext = np.zeros((n_wl, n_pol))
        self._abs = np.zeros((n_wl, n_pol))

        for i, wl in enumerate(wavelengths):
            # Get excitation
            exc = self.excitation(self.particle, wl)

            # Solve BEM
            sig = self.bem.solve(exc)
            self._solutions[wl] = sig

            # Compute cross sections
            if 'ext' in quantities or 'abs' in quantities:
                self._ext[i, :] = self.excitation.extinction(sig)

            if 'sca' in quantities or 'abs' in quantities:
                sca, _ = self.spec.scattering(sig)
                # Normalize by incoming intensity
                eps_bg = self.particle.eps[0](wl)
                if isinstance(eps_bg, tuple):
                    eps_bg = eps_bg[0]
                nb = np.sqrt(eps_bg)
                self._sca[i, :] = sca / (0.5 * nb)

            if 'abs' in quantities:
                self._abs[i, :] = self._ext[i, :] - self._sca[i, :]

        result = {}
        if 'sca' in quantities:
            result['sca'] = self._sca
        if 'ext' in quantities:
            result['ext'] = self._ext
        if 'abs' in quantities:
            result['abs'] = self._abs

        return result

    def scattering(self, wavelengths=None):
        """Return or compute scattering spectrum."""
        if wavelengths is not None:
            self.compute(wavelengths, ['sca'])
        return self._sca

    def extinction(self, wavelengths=None):
        """Return or compute extinction spectrum."""
        if wavelengths is not None:
            self.compute(wavelengths, ['ext'])
        return self._ext

    def absorption(self, wavelengths=None):
        """Return or compute absorption spectrum."""
        if wavelengths is not None:
            self.compute(wavelengths, ['abs'])
        return self._abs

    @property
    def wavelengths(self):
        """Return computed wavelengths."""
        return self._wavelengths

    def get_solution(self, wavelength):
        """Get stored BEM solution for wavelength."""
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
        n_dip = self.dipole.n_dip if hasattr(self.dipole, 'n_dip') else 1

        self._gamma = np.zeros((n_wl, n_dip))

        for i, wl in enumerate(wavelengths):
            exc = self.dipole(self.particle, wl)
            sig = self.bem.solve(exc)
            if hasattr(self.dipole, 'decayrate'):
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


def spectrum_ret(pinfty=None, medium=1, **kwargs):
    """
    Factory function for retarded spectrum calculator.

    Parameters
    ----------
    pinfty : Particle, optional
        Unit sphere for angular integration.
    medium : int, optional
        Embedding medium index.
    **kwargs : dict
        Additional options.

    Returns
    -------
    SpectrumRet
        Spectrum calculator.
    """
    return SpectrumRet(pinfty, medium, **kwargs)
