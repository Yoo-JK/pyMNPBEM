"""
Retarded plane wave excitation for BEM simulations.

This module provides full electromagnetic plane wave excitation
that includes retardation effects for larger particles.
"""

import numpy as np
from typing import Optional, List, Union, Tuple


class PlaneWaveRet:
    """
    Retarded plane wave excitation.

    For particles comparable to or larger than the wavelength,
    retardation effects must be included. This class provides
    the full electromagnetic plane wave fields.

    Parameters
    ----------
    pol : array_like
        Polarization directions (n_pol, 3)
    dir : array_like
        Propagation directions (n_pol, 3)
    options : BEMOptions, optional
        BEM simulation options

    Attributes
    ----------
    pol : ndarray
        Normalized polarization vectors
    dir : ndarray
        Normalized propagation directions
    kvec : ndarray
        Wave vectors (computed for each wavelength)
    """

    def __init__(self, pol, dir, options=None):
        """Initialize retarded plane wave excitation."""
        self.pol = np.atleast_2d(pol).astype(float)
        self.dir = np.atleast_2d(dir).astype(float)
        self.options = options

        # Normalize directions
        self.dir = self.dir / np.linalg.norm(self.dir, axis=1, keepdims=True)

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
            Excitation object with fields and potentials
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

        # Wave number in medium
        n_med = np.sqrt(eps_out)
        k = 2 * np.pi * n_med / wavelength

        # Speed of light factor for H field
        # H = sqrt(eps/mu) * (k x E) / |k|
        # In Gaussian units: H = n * (k_hat x E)

        E = np.zeros((n_pos, self.n_pol, 3), dtype=complex)
        H = np.zeros((n_pos, self.n_pol, 3), dtype=complex)

        for i in range(self.n_pol):
            # Wave vector
            kvec = k * self.dir[i]

            # Phase factor exp(i k.r)
            phase = np.exp(1j * pos @ kvec)

            # Electric field: E = E0 * pol * exp(i k.r)
            E[:, i, :] = phase[:, np.newaxis] * self.pol[i]

            # Magnetic field: H = n * (k_hat x E)
            k_cross_E = np.cross(self.dir[i], self.pol[i])
            H[:, i, :] = n_med * phase[:, np.newaxis] * k_cross_E

        return E, H

    def potentials(self, pos, wavelength, eps_out=1.0):
        """
        Compute scalar and vector potentials at positions.

        In Lorenz gauge, for plane wave:
        phi = 0 (transverse wave)
        A = E / (i * omega) = E * wavelength / (2 * pi * i * c)

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
        phi : ndarray
            Scalar potential (n_pos, n_pol) - zero for plane wave
        A : ndarray
            Vector potential (n_pos, n_pol, 3)
        """
        E, H = self.fields(pos, wavelength, eps_out)

        n_pos = len(pos)

        # Scalar potential is zero for transverse plane wave
        phi = np.zeros((n_pos, self.n_pol), dtype=complex)

        # Vector potential A = E / (i * omega)
        # omega = 2 * pi * c / wavelength
        # A = E * wavelength / (2 * pi * i * c)
        # In our units (nm, with c absorbed): A proportional to E / k
        k = 2 * np.pi / wavelength
        A = E / (1j * k)

        return phi, A

    def sca(self, sig):
        """
        Compute scattering cross section.

        Parameters
        ----------
        sig : Solution
            BEM solution with surface charges and currents

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        """
        return self._cross_section(sig, 'sca')

    def ext(self, sig):
        """
        Compute extinction cross section.

        Parameters
        ----------
        sig : Solution
            BEM solution with surface charges and currents

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization
        """
        return self._cross_section(sig, 'ext')

    def abs(self, sig):
        """
        Compute absorption cross section.

        Parameters
        ----------
        sig : Solution
            BEM solution with surface charges and currents

        Returns
        -------
        abs : ndarray
            Absorption cross section for each polarization
        """
        ext = self.ext(sig)
        sca = self.sca(sig)
        return ext - sca

    def _cross_section(self, sig, cstype):
        """Compute cross sections from BEM solution."""
        particle = sig.particle
        wavelength = sig.wavelength

        k = 2 * np.pi / wavelength

        # Get particle surface data
        pos = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        nvec = particle.nvec if hasattr(particle, 'nvec') else particle.pc.nvec
        area = particle.area if hasattr(particle, 'area') else particle.pc.area

        # Surface charges and currents
        sigma = sig.sig  # Surface charge (n_faces, n_pol)

        if cstype == 'ext':
            # Extinction from optical theorem
            # C_ext = 4*pi/k * Im(f(0)) = 4*pi/k * Im(E0* . p)
            # where p is the induced dipole moment

            ext = np.zeros(self.n_pol)

            for i in range(self.n_pol):
                # Induced dipole moment
                if sigma.ndim == 1:
                    charge = sigma
                else:
                    charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

                # Dipole moment from charges
                dipole = np.sum(charge[:, np.newaxis] * area[:, np.newaxis] * pos, axis=0)

                # Extinction
                ext[i] = 4 * np.pi * k * np.imag(np.dot(np.conj(self.pol[i]), dipole))

            return ext

        elif cstype == 'sca':
            # Scattering cross section
            # Integrate scattered power over sphere

            sca = np.zeros(self.n_pol)

            for i in range(self.n_pol):
                if sigma.ndim == 1:
                    charge = sigma
                else:
                    charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

                # Dipole approximation for small particles
                dipole = np.sum(charge[:, np.newaxis] * area[:, np.newaxis] * pos, axis=0)

                # Scattering from dipole: C_sca = k^4 / (6*pi) * |p|^2
                sca[i] = k**4 / (6 * np.pi) * np.abs(np.dot(dipole, dipole))

            return sca

        else:
            raise ValueError(f"Unknown cross section type: {cstype}")

    def farfield(self, sig, directions):
        """
        Compute far-field scattering amplitude.

        Parameters
        ----------
        sig : Solution
            BEM solution
        directions : ndarray
            Scattering directions (n_dir, 3)

        Returns
        -------
        E_sca : ndarray
            Scattered electric field amplitude (n_dir, n_pol, 3)
        """
        directions = np.atleast_2d(directions)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        n_dir = len(directions)
        particle = sig.particle
        wavelength = sig.wavelength

        k = 2 * np.pi / wavelength

        pos = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        area = particle.area if hasattr(particle, 'area') else particle.pc.area
        sigma = sig.sig

        E_sca = np.zeros((n_dir, self.n_pol, 3), dtype=complex)

        for i in range(self.n_pol):
            if sigma.ndim == 1:
                charge = sigma
            else:
                charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

            for j, sdir in enumerate(directions):
                # Phase factors for each surface element
                phase = np.exp(-1j * k * pos @ sdir)

                # Dipole moment contribution with phase
                p_eff = np.sum(charge[:, np.newaxis] * area[:, np.newaxis] *
                              phase[:, np.newaxis] * pos, axis=0)

                # Far-field: E_sca ~ k^2 * (r_hat x (r_hat x p)) * exp(ikr)/r
                # The (r_hat x (r_hat x p)) projects out the radial component
                r_cross_p = np.cross(sdir, p_eff)
                E_sca[j, i, :] = k**2 * np.cross(sdir, r_cross_p)

        return E_sca


class PlaneWaveRetExcitation:
    """
    Excitation object for retarded plane wave.

    Stores precomputed fields and potentials for BEM solution.
    """

    def __init__(self, planewave, particle, wavelength):
        """Initialize excitation."""
        self.planewave = planewave
        self.particle = particle
        self.wavelength = wavelength

        # Get positions
        if hasattr(particle, 'pos'):
            self.pos = particle.pos
        else:
            self.pos = particle.pc.pos

        # Compute incident fields at particle surface
        self._compute_fields()

    def _compute_fields(self):
        """Compute incident fields at surface positions."""
        # Get medium dielectric function
        eps_out = 1.0
        if hasattr(self.particle, 'eps'):
            eps_out = self.particle.eps[0](self.wavelength)

        self.E_inc, self.H_inc = self.planewave.fields(
            self.pos, self.wavelength, eps_out
        )
        self.phi_inc, self.A_inc = self.planewave.potentials(
            self.pos, self.wavelength, eps_out
        )

    @property
    def e(self):
        """Incident electric field."""
        return self.E_inc

    @property
    def h(self):
        """Incident magnetic field."""
        return self.H_inc

    @property
    def phi(self):
        """Incident scalar potential."""
        return self.phi_inc

    @property
    def a(self):
        """Incident vector potential."""
        return self.A_inc


def planewave_ret(pol, dir, options=None):
    """
    Factory function for retarded plane wave excitation.

    Parameters
    ----------
    pol : array_like
        Polarization directions
    dir : array_like
        Propagation directions
    options : BEMOptions, optional
        Simulation options

    Returns
    -------
    PlaneWaveRet
        Plane wave excitation object
    """
    return PlaneWaveRet(pol, dir, options)
