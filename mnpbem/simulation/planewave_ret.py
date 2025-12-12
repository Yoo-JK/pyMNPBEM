"""
Retarded plane wave excitation for BEM simulations.

This module provides full electromagnetic plane wave excitation
that includes retardation effects for larger particles.

The excitation computes vector potentials A and their surface derivatives
for use in the retarded BEM solver (Garcia de Abajo & Howie, PRB 65, 115418).

For plane waves:
- Scalar potential phi = 0 (transverse wave)
- Vector potential A = pol * exp(i*k*r) / (i*k0)
- Surface derivative Ap = (i*k * n.dir) * A
"""

import numpy as np
from typing import Optional, List, Union, Tuple


class PlaneWaveRet:
    """
    Retarded plane wave excitation.

    For particles comparable to or larger than the wavelength,
    retardation effects must be included. This class provides
    the full electromagnetic plane wave fields and potentials.

    Parameters
    ----------
    pol : array_like
        Polarization directions (n_pol, 3)
    dir : array_like
        Propagation directions (n_pol, 3)
    medium : int, optional
        Index of medium through which excitation enters (1-based, default 1).
    options : dict, optional
        BEM simulation options

    Attributes
    ----------
    pol : ndarray
        Normalized polarization vectors
    dir : ndarray
        Normalized propagation directions
    medium : int
        Exciting medium index (1-based, MATLAB convention)
    n_pol : int
        Number of polarizations

    References
    ----------
    Garcia de Abajo & Howie, PRB 65, 115418 (2002)
    """

    def __init__(self, pol, dir, medium=1, options=None):
        """Initialize retarded plane wave excitation."""
        self.pol = np.atleast_2d(pol).astype(float)
        self.dir = np.atleast_2d(dir).astype(float)
        self.medium = medium  # 1-based index (MATLAB convention)
        self.options = options if options is not None else {}

        # Normalize directions
        dir_norms = np.linalg.norm(self.dir, axis=1, keepdims=True)
        self.dir = self.dir / np.where(dir_norms > 1e-10, dir_norms, 1.0)

        # Ensure polarization is perpendicular to direction
        for i in range(len(self.pol)):
            # Remove component parallel to direction
            dot = np.dot(self.pol[i], self.dir[i])
            self.pol[i] = self.pol[i] - dot * self.dir[i]
            # Normalize
            norm = np.linalg.norm(self.pol[i])
            if norm > 1e-10:
                self.pol[i] = self.pol[i] / norm

        self.n_pol = len(self.pol)
        self._wavelength = None
        self._k = None
        self._spec = None  # For spectrum calculations

    def __call__(self, particle, wavelength):
        """
        Compute plane wave excitation for particle.

        Parameters
        ----------
        particle : ComParticle
            Composite particle
        wavelength : float
            Wavelength in nm

        Returns
        -------
        exc : PlaneWaveRetExcitation
            Excitation object with fields and potentials for BEM solver
        """
        self._wavelength = wavelength
        self._k = 2 * np.pi / wavelength  # Wave number in vacuum

        return PlaneWaveRetExcitation(self, particle, wavelength)

    def fields(self, pos, wavelength, eps_out=1.0):
        """
        Compute electric and magnetic fields at positions.

        Parameters
        ----------
        pos : ndarray
            Positions (n_pos, 3)
        wavelength : float
            Wavelength in nm
        eps_out : complex
            Dielectric function of surrounding medium

        Returns
        -------
        E : ndarray
            Electric field (n_pos, n_pol, 3)
        H : ndarray
            Magnetic field (n_pos, n_pol, 3)
        """
        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        # Wave number in vacuum and medium
        k0 = 2 * np.pi / wavelength
        n_med = np.sqrt(eps_out)
        k = k0 * n_med

        E = np.zeros((n_pos, self.n_pol, 3), dtype=complex)
        H = np.zeros((n_pos, self.n_pol, 3), dtype=complex)

        for i in range(self.n_pol):
            # Wave vector in medium
            kvec = k * self.dir[i]

            # Phase factor exp(i k.r)
            phase = np.exp(1j * pos @ kvec)

            # Electric field: E = E0 * pol * exp(i k.r)
            # Note: E0 amplitude is normalized to 1
            E[:, i, :] = phase[:, np.newaxis] * self.pol[i]

            # Magnetic field: H = n * (k_hat x E)
            k_cross_pol = np.cross(self.dir[i], self.pol[i])
            H[:, i, :] = n_med * phase[:, np.newaxis] * k_cross_pol

        return E, H

    def set_spectrum(self, spec):
        """
        Set spectrum object for cross-section calculations.

        Parameters
        ----------
        spec : SpectrumRet
            Spectrum calculator with pinfty (unit sphere)
        """
        self._spec = spec

    def extinction(self, sig):
        """
        Compute extinction cross section using optical theorem.

        C_ext = 4*pi*k * Im(pol* . p)

        where p is the induced dipole moment from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges and currents

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization
        """
        # Try full far-field calculation from sig2 and h2
        try:
            return self._farfield_extinction(sig)
        except Exception:
            pass

        # If SpectrumRet is available, use full far-field calculation
        if self._spec is not None:
            try:
                field, k = self._spec.farfield(sig, self.dir)
                ext = np.zeros(self.n_pol)
                for i in range(self.n_pol):
                    E_forward = field.e[i, :, i] if field.e.ndim == 3 else field.e[i, :]
                    ext[i] = 4 * np.pi / k * np.imag(np.vdot(self.pol[i], E_forward))
                return ext
            except Exception:
                pass  # Fall back to simple method

        # Simple dipole-based extinction (like quasistatic)
        return self._simple_extinction(sig)

    def _farfield_extinction(self, sig):
        """
        Compute extinction using far-field from charges and currents.

        Based on MATLAB spectrumret/farfield.m and Garcia de Abajo, RMP 82, 209 (2010), Eq. (50).

        E_ff = i*k0 * (phase @ h2) - i*k * outer(dir, phase @ sig2)

        Parameters
        ----------
        sig : CompStruct
            BEM solution with sig2 and h2

        Returns
        -------
        ext : ndarray
            Extinction cross section
        """
        # Get particle geometry
        p = sig.p
        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        area = p.area if hasattr(p, 'area') else p.pc.area
        n_faces = len(pos)

        # Get surface charges and currents
        sig2 = sig.get('sig2')
        h2 = sig.get('h2')

        if sig2 is None or h2 is None:
            raise ValueError("Need sig2 and h2 for far-field extinction")

        # Ensure correct shapes
        if sig2.ndim == 1:
            sig2 = sig2[:, np.newaxis]
        if h2.ndim == 2:
            h2 = h2[:, :, np.newaxis]

        n_pol = sig2.shape[1]

        # Get wavenumbers
        eps_result = p.eps[self.medium - 1](sig.enei)
        if isinstance(eps_result, tuple):
            eps_med, k = eps_result
        else:
            eps_med = eps_result
            k = 2 * np.pi * np.sqrt(eps_med) / sig.enei

        k0 = 2 * np.pi / sig.enei

        # Compute extinction for each polarization
        ext = np.zeros(n_pol)

        for i_pol in range(min(self.n_pol, n_pol)):
            dir_i = self.dir[i_pol]  # Propagation direction
            pol_i = self.pol[i_pol]  # Polarization

            # Phase factor: exp(-i*k*dir.pos) * area
            # For forward scattering (observation in dir direction)
            phase = np.exp(-1j * k * (pos @ dir_i)) * area  # (n_faces,)

            # Far-field electric field contributions:
            # E_ff = i*k0 * sum(phase * h2) - i*k * dir * sum(phase * sig2)

            # Current contribution: i*k0 * phase @ h2
            E_h = 1j * k0 * np.sum(phase[:, np.newaxis] * h2[:, :, i_pol], axis=0)  # (3,)

            # Charge contribution: -i*k * dir * (phase @ sig2)
            phase_sig = np.sum(phase * sig2[:, i_pol])  # scalar
            E_sig = -1j * k * dir_i * phase_sig  # (3,)

            # Total forward-scattered field
            E_forward = E_h + E_sig

            # Optical theorem: ext = 4*pi/k * Im(pol* . E_forward)
            pol_dot_E = np.vdot(pol_i, E_forward)
            ext[i_pol] = 4 * np.pi / k * np.imag(pol_dot_E)

        return np.real(ext)

    def _simple_extinction(self, sig):
        """
        Simple extinction using dipole approximation.

        For particles smaller than wavelength, this gives good results.
        C_ext = 4*pi*k * Im(pol* . p)

        Parameters
        ----------
        sig : CompStruct
            BEM solution

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization
        """
        # Get particle geometry
        p = sig.p
        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        area = p.area if hasattr(p, 'area') else p.pc.area

        # Get surface charges (prefer sig2 = outer surface)
        charges = sig.get('sig2')
        if charges is None:
            charges = sig.get('sig')
        if charges is None:
            raise ValueError("No surface charges found in solution")

        # Ensure charges is 2D: (n_faces, n_pol)
        if charges.ndim == 1:
            charges = charges[:, np.newaxis]

        n_pol = charges.shape[1]

        # Induced dipole moment: p = integral(r * sigma * dA)
        # dip[k, j] = sum_i pos[i, k] * area[i] * sig[i, j]
        weighted_pos = pos * area[:, np.newaxis]  # (n_faces, 3)
        dip = weighted_pos.T @ charges  # (3, n_pol)

        # Get wavenumber in medium
        eps_result = p.eps[self.medium - 1](sig.enei)
        if isinstance(eps_result, tuple):
            eps_med, k = eps_result
        else:
            eps_med = eps_result
            k = 2 * np.pi * np.sqrt(eps_med) / sig.enei

        # Extinction: C_ext = 4*pi*k * Im(pol* . p)
        ext = np.zeros(n_pol)
        for i in range(min(self.n_pol, n_pol)):
            pol_dot_dip = np.vdot(self.pol[i], dip[:, i])
            ext[i] = 4 * np.pi * k * np.imag(pol_dot_dip)

        return np.real(ext)

    def scattering(self, sig):
        """
        Compute scattering cross section.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges and currents

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        dsca : dict or None
            Differential scattering cross section (None for simple method)
        """
        # If SpectrumRet is available, use full far-field integration
        if self._spec is not None:
            try:
                sca, dsca = self._spec.scattering(sig)
                # Get refractive index of embedding medium
                eps_bg = sig.p.eps[0](sig.enei)
                if isinstance(eps_bg, tuple):
                    eps_bg = eps_bg[0]
                nb = np.sqrt(eps_bg)
                # Normalize by incoming power
                sca = sca / (0.5 * nb)
                if dsca is not None:
                    dsca['dsca'] = dsca['dsca'] / (0.5 * nb)
                return sca, dsca
            except Exception:
                pass  # Fall back to simple method

        # Simple dipole-based scattering
        return self._simple_scattering(sig), None

    def _simple_scattering(self, sig):
        """
        Simple scattering using dipole approximation.

        C_sca = (8*pi/3) * k^4 * |p|^2

        Parameters
        ----------
        sig : CompStruct
            BEM solution

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        """
        # Get particle geometry
        p = sig.p
        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        area = p.area if hasattr(p, 'area') else p.pc.area

        # Get surface charges
        charges = sig.get('sig2')
        if charges is None:
            charges = sig.get('sig')
        if charges is None:
            raise ValueError("No surface charges found in solution")

        if charges.ndim == 1:
            charges = charges[:, np.newaxis]

        n_pol = charges.shape[1]

        # Induced dipole moment
        weighted_pos = pos * area[:, np.newaxis]
        dip = weighted_pos.T @ charges  # (3, n_pol)

        # Get wavenumber
        eps_result = p.eps[self.medium - 1](sig.enei)
        if isinstance(eps_result, tuple):
            eps_med, k = eps_result
        else:
            eps_med = eps_result
            k = 2 * np.pi * np.sqrt(eps_med) / sig.enei

        # Scattering: C_sca = (8*pi/3) * k^4 * |p|^2
        sca = np.zeros(n_pol)
        for i in range(n_pol):
            sca[i] = (8 * np.pi / 3) * np.real(k)**4 * np.sum(np.abs(dip[:, i])**2)

        return np.real(sca)

    def absorption(self, sig):
        """
        Compute absorption cross section.

        C_abs = C_ext - C_sca

        Parameters
        ----------
        sig : CompStruct
            BEM solution

        Returns
        -------
        abs : ndarray
            Absorption cross section for each polarization
        """
        ext = self.extinction(sig)
        sca_result = self.scattering(sig)
        sca = sca_result[0] if isinstance(sca_result, tuple) else sca_result
        return ext - sca

    # Legacy methods for backward compatibility
    def sca(self, sig):
        """Compute scattering cross section (legacy interface)."""
        sca_result = self.scattering(sig)
        return sca_result[0] if isinstance(sca_result, tuple) else sca_result

    def ext(self, sig):
        """Compute extinction cross section (legacy interface)."""
        return self.extinction(sig)

    def abs(self, sig):
        """Compute absorption cross section (legacy interface)."""
        return self.absorption(sig)


class PlaneWaveRetExcitation:
    """
    Excitation object for retarded plane wave.

    Stores precomputed fields and potentials for BEM solution.
    Computes vector potentials and their surface derivatives as required
    by the retarded BEM solver.

    For plane wave excitation (MATLAB planewaveret/potential.m):
    - phi1, phi2 = 0 (scalar potential is zero for transverse wave)
    - a1, a2 = vector potential inside/outside
    - a1p, a2p = surface derivative of vector potential

    The vector potential is computed as:
        A = pol * exp(i*k*r) / (i*k0)

    And its surface derivative:
        Ap = (i*k * n.dir) * A

    Attributes
    ----------
    planewave : PlaneWaveRet
        Parent plane wave object
    particle : ComParticle
        Composite particle
    wavelength : float
        Wavelength in nm
    enei : float
        Alias for wavelength (BEM solver compatibility)
    a1, a2 : ndarray
        Vector potential inside/outside, shape (n_faces, 3, n_pol)
    a1p, a2p : ndarray
        Surface derivative of vector potential, shape (n_faces, 3, n_pol)
    phi1, phi2 : ndarray
        Scalar potential (zero for plane wave)
    phi1p, phi2p : ndarray
        Surface derivative of scalar potential (zero)
    E_inc, H_inc : ndarray
        Incident electric/magnetic fields
    """

    def __init__(self, planewave, particle, wavelength):
        """
        Initialize excitation.

        Parameters
        ----------
        planewave : PlaneWaveRet
            Parent plane wave object
        particle : ComParticle
            Composite particle
        wavelength : float
            Wavelength in nm
        """
        self.planewave = planewave
        self.particle = particle
        self.wavelength = wavelength
        self.enei = wavelength  # BEM solver expects this attribute

        # Get particle properties
        if hasattr(particle, 'pos'):
            self.pos = particle.pos
            self.nvec = particle.nvec
            self.n_faces = particle.n_faces
        else:
            self.pos = particle.pc.pos
            self.nvec = particle.pc.nvec
            self.n_faces = particle.pc.n_faces

        # Get inout array (which medium each face belongs to)
        if hasattr(particle, 'inout'):
            self.inout = particle.inout
        else:
            # Default: all faces are [2, 1] (material 2 inside, medium 1 outside)
            self.inout = np.array([[2, 1]] * len(particle.p))

        # Compute potentials and fields
        self._compute_potentials()
        self._compute_fields()

    def _compute_potentials(self):
        """
        Compute vector potentials for BEM solver.

        Following MATLAB planewaveret/potential.m exactly.
        """
        pol = self.planewave.pol
        dir = self.planewave.dir
        medium = self.planewave.medium
        n_pol = self.planewave.n_pol

        # Wavenumber in vacuum
        k0 = 2 * np.pi / self.wavelength

        # Get refractive index of exciting medium
        eps_med = self.particle.eps[medium - 1](self.wavelength)  # 0-based
        if isinstance(eps_med, tuple):
            eps_med = eps_med[0]
        nb = np.sqrt(eps_med)

        # Wavenumber in medium
        k = k0 * nb

        # Initialize arrays
        # Shape: (n_faces, 3, n_pol) for vector quantities
        self.a1 = np.zeros((self.n_faces, 3, n_pol), dtype=complex)
        self.a1p = np.zeros((self.n_faces, 3, n_pol), dtype=complex)
        self.a2 = np.zeros((self.n_faces, 3, n_pol), dtype=complex)
        self.a2p = np.zeros((self.n_faces, 3, n_pol), dtype=complex)

        # Scalar potentials are zero for plane wave (transverse)
        self.phi1 = np.zeros((self.n_faces, n_pol), dtype=complex)
        self.phi1p = np.zeros((self.n_faces, n_pol), dtype=complex)
        self.phi2 = np.zeros((self.n_faces, n_pol), dtype=complex)
        self.phi2p = np.zeros((self.n_faces, n_pol), dtype=complex)

        # Loop over inside (inout=0) and outside (inout=1) of particle surfaces
        # Note: Python uses 0-based indexing, MATLAB uses 1-based
        for inout_idx in range(2):  # 0=inside, 1=outside
            # Find faces where this inout side belongs to the exciting medium
            # self.inout has shape (n_particles, 2) where [:, 0]=inside, [:, 1]=outside
            # Need to map particle index to face indices

            face_start = 0
            ind_list = []

            for p_idx, p in enumerate(self.particle.p):
                n_faces_p = p.n_faces
                face_end = face_start + n_faces_p

                # Check if this particle's inout matches exciting medium
                if self.particle.inout[p_idx, inout_idx] == medium:
                    ind_list.extend(range(face_start, face_end))

                face_start = face_end

            ind = np.array(ind_list, dtype=int)

            if len(ind) == 0:
                continue

            # Compute vector potential for each polarization
            for i in range(n_pol):
                # Phase factor: exp(i*k*r) / (i*k0)
                # Vector potential in Lorenz gauge: A = E/(iω) = E/(ik0*c)
                # In natural units (c=1): A = E/(ik0)
                phase = np.exp(1j * k * (self.pos[ind] @ dir[i])) / (1j * k0)

                # Vector potential: A = phase * pol
                a = np.outer(phase, pol[i])  # (len(ind), 3)

                # Surface derivative: Ap = (i*k * n.dir) * A
                # n.dir is the dot product of normal vector with propagation direction
                n_dot_dir = self.nvec[ind] @ dir[i]  # (len(ind),)
                ap = (1j * k * n_dot_dir)[:, np.newaxis] * a  # (len(ind), 3)

                # Store in appropriate array
                if inout_idx == 0:  # inside
                    self.a1[ind, :, i] = a
                    self.a1p[ind, :, i] = ap
                else:  # outside
                    self.a2[ind, :, i] = a
                    self.a2p[ind, :, i] = ap

    def _compute_fields(self):
        """Compute incident electric and magnetic fields at surface positions."""
        # Get medium dielectric function
        eps_out = 1.0
        if hasattr(self.particle, 'eps'):
            eps_result = self.particle.eps[0](self.wavelength)
            if isinstance(eps_result, tuple):
                eps_out = eps_result[0]
            else:
                eps_out = eps_result

        self.E_inc, self.H_inc = self.planewave.fields(
            self.pos, self.wavelength, eps_out
        )

    @property
    def e(self):
        """Incident electric field."""
        return self.E_inc

    @property
    def h(self):
        """Incident magnetic field (not surface current)."""
        return self.H_inc

    def get(self, key: str, default=None):
        """
        Get data by key for BEM solver compatibility.

        Supports all fields required by retarded BEM solver:
        - Vector potentials: a1, a1p, a2, a2p
        - Scalar potentials: phi1, phi1p, phi2, phi2p (zero for plane wave)
        - Fields: e, h

        Parameters
        ----------
        key : str
            Data key to retrieve
        default : any, optional
            Default value if key not found

        Returns
        -------
        any
            Requested data or default
        """
        # Direct attribute access for known keys
        key_map = {
            # Vector potentials (required for retarded BEM)
            'a1': 'a1',
            'a1p': 'a1p',
            'a2': 'a2',
            'a2p': 'a2p',
            # Scalar potentials (zero for plane wave)
            'phi1': 'phi1',
            'phi1p': 'phi1p',
            'phi2': 'phi2',
            'phi2p': 'phi2p',
            # Legacy keys
            'phip': 'phi1p',  # For quasistatic compatibility
            'phi': 'phi1',
            # Fields
            'e': 'E_inc',
            'h': 'H_inc',
            'a': 'a1',  # Legacy: vector potential
        }

        attr_name = key_map.get(key, key)
        return getattr(self, attr_name, default)


def planewave_ret(pol, dir, medium=1, options=None):
    """
    Factory function for retarded plane wave excitation.

    Parameters
    ----------
    pol : array_like
        Polarization directions
    dir : array_like
        Propagation directions
    medium : int, optional
        Index of exciting medium (1-based)
    options : dict, optional
        Simulation options

    Returns
    -------
    PlaneWaveRet
        Plane wave excitation object
    """
    return PlaneWaveRet(pol, dir, medium, options)
