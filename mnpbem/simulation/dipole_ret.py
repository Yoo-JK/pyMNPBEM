"""
Retarded dipole excitation for BEM simulations.

This module provides full electromagnetic dipole excitation
with retardation effects for decay rate and field calculations.
"""

import numpy as np
from typing import Optional, Union, Tuple


class DipoleRet:
    """
    Retarded dipole excitation.

    Computes the electromagnetic fields of an oscillating dipole
    including full retardation effects. Used for calculating
    decay rates and Purcell factors near nanoparticles.

    Parameters
    ----------
    pt : array_like
        Dipole positions (n_dip, 3) or (3,) for single dipole
    dip : array_like
        Dipole moments/orientations (n_dip, 3) or (3,)
    options : BEMOptions, optional
        BEM simulation options

    Attributes
    ----------
    pt : ndarray
        Dipole positions
    dip : ndarray
        Dipole moment directions (normalized)
    """

    def __init__(self, pt, dip, options=None):
        """Initialize retarded dipole excitation."""
        self.pt = np.atleast_2d(pt).astype(float)
        self.dip = np.atleast_2d(dip).astype(float)
        self.options = options

        # Normalize dipole directions
        norms = np.linalg.norm(self.dip, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self.dip = self.dip / norms

        self.n_dip = len(self.pt)
        self._wavelength = None

    def __call__(self, particle, wavelength):
        """
        Compute dipole excitation for particle.

        Parameters
        ----------
        particle : ComParticle
            Composite particle
        wavelength : float
            Wavelength in nm

        Returns
        -------
        exc : DipoleRetExcitation
            Excitation object with fields and potentials
        """
        self._wavelength = wavelength
        return DipoleRetExcitation(self, particle, wavelength)

    def fields(self, pos, wavelength, eps_out=1.0):
        """
        Compute electric and magnetic fields from dipole.

        The retarded dipole fields are:
        E = k^2 (n x p) x n * G + [3n(n.p) - p] * (1/r^3 - ik/r^2) * exp(ikr)
        H = k^2 (n x p) * (1 - 1/(ikr)) * G

        where G = exp(ikr)/(4*pi*r), n = r_hat

        Parameters
        ----------
        pos : ndarray
            Field positions (n_pos, 3)
        wavelength : float
            Wavelength in nm
        eps_out : complex
            Dielectric function of surrounding medium

        Returns
        -------
        E : ndarray
            Electric field (n_pos, n_dip, 3)
        H : ndarray
            Magnetic field (n_pos, n_dip, 3)
        """
        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        # Wave number
        n_med = np.sqrt(eps_out)
        k = 2 * np.pi * n_med / wavelength

        E = np.zeros((n_pos, self.n_dip, 3), dtype=complex)
        H = np.zeros((n_pos, self.n_dip, 3), dtype=complex)

        for i in range(self.n_dip):
            # Vector from dipole to field point
            r_vec = pos - self.pt[i]
            r = np.linalg.norm(r_vec, axis=1)
            r[r < 1e-10] = 1e-10  # Avoid division by zero

            # Unit vector
            n_hat = r_vec / r[:, np.newaxis]

            # Dipole moment
            p = self.dip[i]

            # Useful dot products
            n_dot_p = np.sum(n_hat * p, axis=1)

            # Green function
            kr = k * r
            G = np.exp(1j * kr) / (4 * np.pi * r)

            # Near-field term coefficient: (1/r^3 - ik/r^2)
            near_coef = (1.0 / r**3 - 1j * k / r**2) * np.exp(1j * kr)

            # Far-field term coefficient: k^2 * G
            far_coef = k**2 * G

            # Electric field
            # Far-field: k^2 * G * (n x p) x n = k^2 * G * [p - n(n.p)]
            # Near-field: [3n(n.p) - p] * near_coef
            E_far = far_coef[:, np.newaxis] * (p - n_hat * n_dot_p[:, np.newaxis])
            E_near = near_coef[:, np.newaxis] * (3 * n_hat * n_dot_p[:, np.newaxis] - p)

            E[:, i, :] = E_far + E_near

            # Magnetic field: k^2 * (n x p) * (1 - 1/(ikr)) * G
            n_cross_p = np.cross(n_hat, p)
            H_coef = k**2 * G * (1 - 1.0 / (1j * kr))
            H[:, i, :] = H_coef[:, np.newaxis] * n_cross_p

        return E, H

    def potentials(self, pos, wavelength, eps_out=1.0):
        """
        Compute scalar and vector potentials from dipole.

        phi = (p . r_hat) * (1/r^2 - ik/r) * exp(ikr) / (4*pi)
        A = p * exp(ikr) / (4*pi*r) * (ik)

        Parameters
        ----------
        pos : ndarray
            Positions (n_pos, 3)
        wavelength : float
            Wavelength in nm
        eps_out : complex
            Dielectric function

        Returns
        -------
        phi : ndarray
            Scalar potential (n_pos, n_dip)
        A : ndarray
            Vector potential (n_pos, n_dip, 3)
        """
        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        n_med = np.sqrt(eps_out)
        k = 2 * np.pi * n_med / wavelength

        phi = np.zeros((n_pos, self.n_dip), dtype=complex)
        A = np.zeros((n_pos, self.n_dip, 3), dtype=complex)

        for i in range(self.n_dip):
            r_vec = pos - self.pt[i]
            r = np.linalg.norm(r_vec, axis=1)
            r[r < 1e-10] = 1e-10

            n_hat = r_vec / r[:, np.newaxis]
            p = self.dip[i]

            kr = k * r
            n_dot_p = np.sum(n_hat * p, axis=1)

            # Scalar potential
            phi_coef = (1.0 / r**2 - 1j * k / r) * np.exp(1j * kr) / (4 * np.pi)
            phi[:, i] = n_dot_p * phi_coef

            # Vector potential
            G = np.exp(1j * kr) / (4 * np.pi * r)
            A[:, i, :] = (1j * k * G)[:, np.newaxis] * p

        return phi, A

    def decayrate(self, sig, plane='total'):
        """
        Compute decay rate enhancement (Purcell factor).

        The decay rate enhancement is:
        gamma/gamma_0 = 1 + 6*pi*eps_0*c^3/(omega^3 * |p|^2) * Im(p* . E_sca)

        Parameters
        ----------
        sig : Solution
            BEM solution
        plane : str
            'total', 'parallel', or 'perpendicular' decay rate

        Returns
        -------
        gamma : ndarray
            Decay rate enhancement for each dipole
        """
        # Get scattered field at dipole positions
        E_sca = self._scattered_field_at_dipole(sig)

        gamma = np.zeros(self.n_dip)

        for i in range(self.n_dip):
            p = self.dip[i]
            E = E_sca[i]

            # Decay rate enhancement from Im(p* . E_sca)
            # Normalized by free-space decay rate
            p_dot_E = np.dot(np.conj(p), E)
            gamma[i] = 1 + 6 * np.pi * np.imag(p_dot_E)

        return gamma

    def _scattered_field_at_dipole(self, sig):
        """Compute scattered field at dipole positions."""
        particle = sig.particle
        wavelength = sig.wavelength

        k = 2 * np.pi / wavelength

        # Get particle surface data
        pos_surf = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        area = particle.area if hasattr(particle, 'area') else particle.pc.area
        sigma = sig.sig

        E_sca = np.zeros((self.n_dip, 3), dtype=complex)

        for i in range(self.n_dip):
            pt = self.pt[i]

            # Distance to surface elements
            r_vec = pt - pos_surf
            r = np.linalg.norm(r_vec, axis=1)
            r[r < 1e-10] = 1e-10

            # Green function
            G = np.exp(1j * k * r) / (4 * np.pi * r)

            # Get charges for this dipole
            if sigma.ndim == 1:
                charge = sigma
            else:
                charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

            # Electric field from surface charges (monopole contribution)
            # E = -grad(phi) where phi = sum(sigma * G * area)
            r_hat = r_vec / r[:, np.newaxis]
            grad_G = (1j * k - 1.0 / r)[:, np.newaxis] * G[:, np.newaxis] * r_hat

            E_sca[i] = -np.sum(charge[:, np.newaxis] * area[:, np.newaxis] * grad_G, axis=0)

        return E_sca

    def farfield(self, sig, directions):
        """
        Compute far-field radiation pattern.

        Parameters
        ----------
        sig : Solution
            BEM solution
        directions : ndarray
            Observation directions (n_dir, 3)

        Returns
        -------
        power : ndarray
            Radiated power in each direction (n_dir, n_dip)
        """
        directions = np.atleast_2d(directions)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        n_dir = len(directions)
        wavelength = sig.wavelength
        k = 2 * np.pi / wavelength

        power = np.zeros((n_dir, self.n_dip))

        # Total dipole = original + induced
        p_ind = self._induced_dipole(sig)

        for i in range(self.n_dip):
            p_total = self.dip[i] + p_ind[i]

            for j, n_hat in enumerate(directions):
                # Far-field intensity ~ |n x (n x p)|^2 = |p|^2 - |n.p|^2
                n_cross_p = np.cross(n_hat, p_total)
                n_cross_n_cross_p = np.cross(n_hat, n_cross_p)
                power[j, i] = np.abs(np.dot(n_cross_n_cross_p, np.conj(n_cross_n_cross_p)))

        return power

    def _induced_dipole(self, sig):
        """Compute induced dipole moment from surface charges."""
        particle = sig.particle

        pos = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        area = particle.area if hasattr(particle, 'area') else particle.pc.area
        sigma = sig.sig

        p_ind = np.zeros((self.n_dip, 3), dtype=complex)

        for i in range(self.n_dip):
            if sigma.ndim == 1:
                charge = sigma
            else:
                charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

            p_ind[i] = np.sum(charge[:, np.newaxis] * area[:, np.newaxis] * pos, axis=0)

        return p_ind

    def scattering(self, sig, spec=None):
        """
        Compute scattering cross section for dipole excitation.

        The scattering cross section is computed from the far-field
        radiation pattern using the optical theorem.

        Parameters
        ----------
        sig : Solution
            BEM solution with surface charges and currents.
        spec : Spectrum, optional
            Spectrum object with directions at infinity.
            If None, uses default angular integration.

        Returns
        -------
        sca : ndarray
            Total scattering cross section for each dipole.
        dsca : ndarray or CompStruct, optional
            Differential scattering cross section.
        """
        wavelength = sig.wavelength if hasattr(sig, 'wavelength') else sig.enei

        # Get particle and surface data
        if hasattr(sig, 'particle'):
            particle = sig.particle
        elif hasattr(sig, 'p'):
            particle = sig.p
        else:
            raise ValueError("Cannot find particle in solution")

        pos = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        area = particle.area if hasattr(particle, 'area') else particle.pc.area
        nvec = particle.nvec if hasattr(particle, 'nvec') else particle.pc.nvec
        sigma = sig.sig if hasattr(sig, 'sig') else sig.get('sig')

        # Wave number
        k = 2 * np.pi / wavelength

        # If spectrum provided, compute far-field on that grid
        if spec is not None and hasattr(spec, 'pinfty'):
            # Compute far-field at given directions
            far = self._compute_farfield(sig, spec)

            # Get surface at infinity
            pinfty = spec.pinfty
            directions = pinfty.nvec if hasattr(pinfty, 'nvec') else pinfty
            area_ff = pinfty.area if hasattr(pinfty, 'area') else np.ones(len(directions))

            # Poynting vector: S = 0.5 * Re(E x H*)
            e_ff = far.get('e') if hasattr(far, 'get') else far.e
            h_ff = far.get('h') if hasattr(far, 'get') else far.h

            dsca = np.zeros((len(directions), self.n_dip))
            for i in range(self.n_dip):
                if e_ff.ndim == 3:
                    e_i = e_ff[:, :, i]
                    h_i = h_ff[:, :, i]
                else:
                    e_i = e_ff
                    h_i = h_ff

                # Poynting vector in normal direction
                poynting = 0.5 * np.real(
                    np.sum(directions * np.cross(e_i, np.conj(h_i)), axis=1)
                )
                dsca[:, i] = poynting

            # Total scattering = integral of dsca over solid angle
            sca = np.sum(dsca * area_ff[:, np.newaxis], axis=0)

            return sca, dsca
        else:
            # Default: use induced dipole approximation for small particles
            # Scattering cross section from Rayleigh formula
            p_ind = self._induced_dipole(sig)

            sca = np.zeros(self.n_dip)
            for i in range(self.n_dip):
                p_total = self.dip[i] + p_ind[i]
                # sigma_sca = (k^4 / 6*pi) * |p|^2
                sca[i] = k**4 / (6 * np.pi) * np.abs(np.dot(p_total, np.conj(p_total)))

            return sca

    def extinction(self, sig, exc=None):
        """
        Compute extinction cross section for dipole excitation.

        Extinction = Scattering + Absorption

        Uses the optical theorem: sigma_ext = (4*pi*k) * Im(f(0))
        where f(0) is the forward scattering amplitude.

        Parameters
        ----------
        sig : Solution
            BEM solution.
        exc : Excitation, optional
            Incident field excitation.

        Returns
        -------
        ext : ndarray
            Extinction cross section for each dipole.
        """
        wavelength = sig.wavelength if hasattr(sig, 'wavelength') else sig.enei
        k = 2 * np.pi / wavelength

        # Get scattered field at dipole position
        E_sca = self._scattered_field_at_dipole(sig)

        ext = np.zeros(self.n_dip)

        for i in range(self.n_dip):
            p = self.dip[i]
            E = E_sca[i]

            # Extinction from optical theorem
            # sigma_ext = (4*pi*k) * Im(p* . E_sca) / |E_inc|^2
            # For dipole excitation, normalized by dipole moment
            ext[i] = 4 * np.pi * k * np.imag(np.dot(np.conj(p), E))

        return ext

    def absorption(self, sig, exc=None):
        """
        Compute absorption cross section for dipole excitation.

        Absorption = Extinction - Scattering

        Parameters
        ----------
        sig : Solution
            BEM solution.
        exc : Excitation, optional
            Incident field excitation.

        Returns
        -------
        abs : ndarray
            Absorption cross section for each dipole.
        """
        ext = self.extinction(sig, exc)
        sca_result = self.scattering(sig)

        # Handle both return formats of scattering
        if isinstance(sca_result, tuple):
            sca = sca_result[0]
        else:
            sca = sca_result

        return ext - sca

    def _compute_farfield(self, sig, spec):
        """
        Compute far-field from surface charges/currents.

        Parameters
        ----------
        sig : Solution
            BEM solution.
        spec : Spectrum
            Spectrum with directions at infinity.

        Returns
        -------
        CompStruct
            Far-field E and H.
        """
        from ..particles import CompStruct

        wavelength = sig.wavelength if hasattr(sig, 'wavelength') else sig.enei
        k = 2 * np.pi / wavelength

        # Get particle data
        if hasattr(sig, 'particle'):
            particle = sig.particle
        elif hasattr(sig, 'p'):
            particle = sig.p
        else:
            raise ValueError("Cannot find particle")

        pos = particle.pos if hasattr(particle, 'pos') else particle.pc.pos
        area = particle.area if hasattr(particle, 'area') else particle.pc.area
        sigma = sig.sig if hasattr(sig, 'sig') else sig.get('sig')

        # Get directions
        pinfty = spec.pinfty
        directions = pinfty.nvec if hasattr(pinfty, 'nvec') else np.atleast_2d(pinfty)
        n_dir = len(directions)

        # Allocate far-fields
        e = np.zeros((n_dir, 3, self.n_dip), dtype=complex)
        h = np.zeros((n_dir, 3, self.n_dip), dtype=complex)

        for i in range(self.n_dip):
            if sigma.ndim == 1:
                charge = sigma
            else:
                charge = sigma[:, i] if i < sigma.shape[1] else sigma[:, 0]

            for j, ndir in enumerate(directions):
                # Far-field phase: exp(-i*k*n.r)
                phase = np.exp(-1j * k * np.dot(pos, ndir))

                # Far-field from surface charges (electric dipole radiation)
                # E_ff ~ k^2 * sum(sigma * area * (n x r) x n * phase)
                for m in range(len(pos)):
                    r_perp = pos[m] - np.dot(pos[m], ndir) * ndir
                    e_contrib = k**2 * charge[m] * area[m] * r_perp * phase[m]
                    e[j, :, i] += e_contrib

                # Magnetic field from E: H = n x E / Z0
                h[j, :, i] = np.cross(ndir, e[j, :, i])

        return CompStruct(pinfty, wavelength, e=e, h=h)


class DipoleRetExcitation:
    """
    Excitation object for retarded dipole.
    """

    def __init__(self, dipole, particle, wavelength):
        """Initialize excitation."""
        self.dipole = dipole
        self.particle = particle
        self.wavelength = wavelength
        self.enei = wavelength  # BEM solver expects this attribute

        if hasattr(particle, 'pos'):
            self.pos = particle.pos
        else:
            self.pos = particle.pc.pos

        self._compute_fields()

    def _compute_fields(self):
        """Compute incident fields at surface."""
        eps_out = 1.0
        if hasattr(self.particle, 'eps'):
            eps_result = self.particle.eps[0](self.wavelength)
            # Handle both tuple (eps, k) and scalar returns
            if isinstance(eps_result, tuple):
                eps_out = eps_result[0]
            else:
                eps_out = eps_result

        self.E_inc, self.H_inc = self.dipole.fields(
            self.pos, self.wavelength, eps_out
        )
        self.phi_inc, self.A_inc = self.dipole.potentials(
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

    def get(self, key: str, default=None):
        """
        Get data by key for BEM solver compatibility.

        Maps BEM solver expected keys to internal attributes:
        - 'phip' -> phi_inc (incident scalar potential)
        - 'e' -> E_inc (incident electric field)
        - 'h' -> H_inc (incident magnetic field)
        - 'a' -> A_inc (incident vector potential)

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
        key_map = {
            'phip': 'phi_inc',
            'phi': 'phi_inc',
            'e': 'E_inc',
            'h': 'H_inc',
            'a': 'A_inc',
        }

        attr_name = key_map.get(key, key)
        return getattr(self, attr_name, default)


def dipole_ret(pt, dip, options=None):
    """
    Factory function for retarded dipole excitation.

    Parameters
    ----------
    pt : array_like
        Dipole position(s)
    dip : array_like
        Dipole moment direction(s)
    options : BEMOptions, optional
        Simulation options

    Returns
    -------
    DipoleRet
        Dipole excitation object
    """
    return DipoleRet(pt, dip, options)
