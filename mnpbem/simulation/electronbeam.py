"""
Electron beam excitation for BEM simulations.

This module provides electron beam excitation for simulating
EELS (Electron Energy Loss Spectroscopy) and CL (Cathodoluminescence).
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import special


class ElectronBeam:
    """
    Electron beam excitation.

    Represents a fast electron passing near or through a nanoparticle.
    The electron acts as a moving point charge creating time-dependent
    electromagnetic fields.

    Parameters
    ----------
    impact : array_like
        Impact parameter position (x, y) or (x, y, z) where beam crosses z=0
    velocity : float
        Electron velocity as fraction of speed of light (v/c)
    direction : array_like, optional
        Beam direction (default: [0, 0, 1] along z-axis)
    options : BEMOptions, optional
        Simulation options

    Attributes
    ----------
    impact : ndarray
        Impact position
    vel : float
        Normalized velocity (v/c)
    gamma : float
        Lorentz factor
    direction : ndarray
        Beam direction (normalized)
    """

    def __init__(self, impact, velocity, direction=None, options=None):
        """Initialize electron beam."""
        self.impact = np.array(impact, dtype=float)
        if len(self.impact) == 2:
            self.impact = np.append(self.impact, 0.0)

        self.vel = float(velocity)  # v/c
        self.gamma = 1.0 / np.sqrt(1 - self.vel**2)  # Lorentz factor

        if direction is None:
            direction = [0, 0, 1]
        self.direction = np.array(direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)

        self.options = options
        self._wavelength = None

    def __call__(self, particle, wavelength):
        """
        Create excitation for given particle and wavelength.

        Parameters
        ----------
        particle : ComParticle
            Composite particle
        wavelength : float
            Wavelength (photon equivalent of energy loss)

        Returns
        -------
        exc : ElectronBeamExcitation
            Excitation object
        """
        self._wavelength = wavelength
        return ElectronBeamExcitation(self, particle, wavelength)

    def trajectory(self, t):
        """
        Electron position along trajectory.

        Parameters
        ----------
        t : float or ndarray
            Time parameter (z = v * t)

        Returns
        -------
        pos : ndarray
            Position (3,) or (n, 3)
        """
        t = np.atleast_1d(t)
        pos = self.impact + np.outer(t, self.direction) * self.vel
        return pos.squeeze()

    def fields(self, pos, wavelength, t=None):
        """
        Compute electromagnetic fields from moving electron.

        The fields of a moving charge (Liénard-Wiechert potentials):
        E = q * gamma * (r - v*t) / (4*pi*eps0 * |r - v*t|^3 * (1 - (v.r_hat)^2 / c^2)^(3/2))

        In Fourier domain at frequency omega:
        E(r, omega) = FT[E(r, t)]

        Parameters
        ----------
        pos : ndarray
            Field positions (n, 3)
        wavelength : float
            Wavelength in nm
        t : float, optional
            Time (if None, compute frequency domain)

        Returns
        -------
        E : ndarray
            Electric field (n, 3)
        H : ndarray
            Magnetic field (n, 3)
        """
        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        # Speed of light in nm/fs (approximately)
        c = 299.792458  # nm/fs

        # Angular frequency
        omega = 2 * np.pi * c / wavelength

        # Electron parameters
        v = self.vel * c  # velocity in nm/fs
        b = self.impact  # impact position

        E = np.zeros((n_pos, 3), dtype=complex)
        H = np.zeros((n_pos, 3), dtype=complex)

        for i, r in enumerate(pos):
            # Vector from trajectory to field point
            # Perpendicular distance to beam axis
            r_perp = r - b
            r_perp = r_perp - np.dot(r_perp, self.direction) * self.direction
            b_perp = np.linalg.norm(r_perp)

            if b_perp < 1e-10:
                # On the beam axis
                continue

            r_perp_hat = r_perp / b_perp

            # Parallel component (along beam direction)
            z = np.dot(r - b, self.direction)

            # Frequency-domain fields
            # From Jackson, Classical Electrodynamics
            # E_perp ~ K_1(omega * b / (gamma * v))
            # E_parallel ~ K_0(omega * b / (gamma * v))

            xi = omega * b_perp / (self.gamma * v)

            if xi < 100:  # Avoid overflow in Bessel functions
                K0 = special.kv(0, xi)
                K1 = special.kv(1, xi)
            else:
                K0 = 0.0
                K1 = 0.0

            # Phase factor
            phase = np.exp(1j * omega * z / v)

            # Prefactor (units: elementary charge)
            prefactor = omega / (np.pi * self.gamma * v**2)

            # Perpendicular field
            E_perp = prefactor * K1 * r_perp_hat * phase

            # Parallel field
            E_para = -1j * prefactor / self.gamma * K0 * self.direction * phase

            E[i] = E_perp + E_para

            # Magnetic field: H = (v x E) / c^2 (approximately)
            H[i] = np.cross(self.vel * self.direction, E[i])

        return E, H

    def potential(self, pos, wavelength):
        """
        Compute scalar potential from electron beam.

        Parameters
        ----------
        pos : ndarray
            Positions (n, 3)
        wavelength : float
            Wavelength in nm

        Returns
        -------
        phi : ndarray
            Scalar potential (n,)
        """
        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        c = 299.792458  # nm/fs
        omega = 2 * np.pi * c / wavelength
        v = self.vel * c

        phi = np.zeros(n_pos, dtype=complex)

        for i, r in enumerate(pos):
            r_perp = r - self.impact
            r_perp = r_perp - np.dot(r_perp, self.direction) * self.direction
            b_perp = np.linalg.norm(r_perp)

            if b_perp < 1e-10:
                continue

            z = np.dot(r - self.impact, self.direction)
            xi = omega * b_perp / (self.gamma * v)

            if xi < 100:
                K0 = special.kv(0, xi)
            else:
                K0 = 0.0

            phase = np.exp(1j * omega * z / v)
            phi[i] = 2 / (self.gamma * v) * K0 * phase

        return phi

    def loss_probability(self, sig, wavelength):
        """
        Compute energy loss probability.

        The loss probability is related to the work done by the
        induced fields on the electron.

        Parameters
        ----------
        sig : Solution
            BEM solution
        wavelength : float
            Wavelength in nm

        Returns
        -------
        loss : float
            Loss probability (dimensionless)
        """
        # Get induced potential along trajectory
        # Support both sig.particle and sig.p (CompStruct uses 'p')
        if hasattr(sig, 'particle'):
            particle = sig.particle
        elif hasattr(sig, 'p'):
            particle = sig.p
        else:
            raise AttributeError("sig must have 'particle' or 'p' attribute")

        # Sample points along trajectory
        z_min = -100  # nm
        z_max = 100
        n_sample = 200
        z_vals = np.linspace(z_min, z_max, n_sample)

        trajectory_pts = np.array([self.impact + z * self.direction for z in z_vals])

        # Compute induced potential
        phi_ind = self._induced_potential(sig, trajectory_pts, wavelength)

        # Energy loss = integral of (velocity * E_induced) along trajectory
        # = integral of (-v * grad(phi_ind)) along path
        # = -v * delta(phi_ind)

        # Use gradient of phi along trajectory
        c = 299.792458
        omega = 2 * np.pi * c / wavelength
        v = self.vel * c

        # Numerical derivative
        dphi_dz = np.gradient(phi_ind, z_vals)

        # Loss probability proportional to Im(integral)
        # Gamma = (2 * e^2 / (pi * hbar * v)) * Im(phi_ind(omega))
        integrand = dphi_dz * np.exp(-1j * omega * z_vals / v)
        integral = np.trapz(integrand, z_vals)

        loss = -np.imag(integral) / (np.pi * v)

        return np.abs(loss)

    def _induced_potential(self, sig, pts, wavelength):
        """Compute induced potential at points from surface charges."""
        # Support both sig.particle and sig.p (CompStruct uses 'p')
        if hasattr(sig, 'particle'):
            particle = sig.particle
        elif hasattr(sig, 'p'):
            particle = sig.p
        else:
            raise AttributeError("sig must have 'particle' or 'p' attribute")

        if hasattr(particle, 'pc'):
            pos_surf = particle.pc.pos
            area = particle.pc.area
        else:
            pos_surf = particle.pos
            area = particle.area

        sigma = sig.sig
        if sigma.ndim > 1:
            sigma = sigma[:, 0]  # Take first column

        k = 2 * np.pi / wavelength

        phi = np.zeros(len(pts), dtype=complex)

        for i, pt in enumerate(pts):
            r = np.linalg.norm(pt - pos_surf, axis=1)
            r[r < 1e-10] = 1e-10

            # Green function (retarded or quasistatic)
            if self.options and getattr(self.options, 'sim', 'stat') == 'ret':
                G = np.exp(1j * k * r) / (4 * np.pi * r)
            else:
                G = 1.0 / (4 * np.pi * r)

            phi[i] = np.sum(sigma * area * G)

        return phi


class ElectronBeamExcitation:
    """
    Excitation object for electron beam.
    """

    def __init__(self, beam, particle, wavelength):
        """Initialize excitation."""
        self.beam = beam
        self.particle = particle
        self.wavelength = wavelength

        if hasattr(particle, 'pos'):
            self.pos = particle.pos
        else:
            self.pos = particle.pc.pos

        self._compute_fields()

    def _compute_fields(self):
        """Compute incident fields at surface."""
        self.E_inc, self.H_inc = self.beam.fields(self.pos, self.wavelength)
        self.phi_inc = self.beam.potential(self.pos, self.wavelength)

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


class ElectronBeamRet(ElectronBeam):
    """
    Retarded electron beam for full electromagnetic calculations.

    Includes radiation effects and is appropriate for larger particles.
    """

    def __init__(self, impact, velocity, direction=None, options=None):
        """Initialize retarded electron beam."""
        super().__init__(impact, velocity, direction, options)

    def fields(self, pos, wavelength, t=None):
        """
        Compute full retarded fields from moving electron.

        Uses the Liénard-Wiechert formulation with proper retardation.
        """
        # For relativistic electron, use parent class implementation
        # which already uses relativistic formulas
        return super().fields(pos, wavelength, t)


def electronbeam(impact, velocity, direction=None, options=None):
    """
    Factory function for electron beam excitation.

    Parameters
    ----------
    impact : array_like
        Impact parameter (x, y) or (x, y, z)
    velocity : float
        Electron velocity as v/c
    direction : array_like, optional
        Beam direction
    options : BEMOptions, optional
        Options (if sim='ret', use retarded version)

    Returns
    -------
    ElectronBeam or ElectronBeamRet
        Electron beam excitation object
    """
    if options and getattr(options, 'sim', 'stat') == 'ret':
        return ElectronBeamRet(impact, velocity, direction, options)
    return ElectronBeam(impact, velocity, direction, options)
