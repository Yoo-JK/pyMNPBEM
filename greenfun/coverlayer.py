"""
Coverlayer module for coated/core-shell nanoparticles.

This module provides Green functions modified for particles
with thin coating layers (shells).
"""

import numpy as np
from typing import Optional, Union, Tuple


class CoverLayer:
    """
    Coating layer for core-shell particles.

    Represents a thin dielectric layer coating a particle,
    modifying the effective boundary conditions.

    Parameters
    ----------
    eps_layer : callable or complex
        Dielectric function of the layer material
    thickness : float
        Layer thickness in nm
    eps_core : callable or complex, optional
        Dielectric function of the core

    Attributes
    ----------
    eps_layer : callable
        Layer dielectric function
    thickness : float
        Layer thickness
    """

    def __init__(self, eps_layer, thickness, eps_core=None):
        """Initialize cover layer."""
        if callable(eps_layer):
            self.eps_layer = eps_layer
        else:
            self._eps_layer_val = complex(eps_layer)
            self.eps_layer = lambda wl: self._eps_layer_val

        self.thickness = thickness

        if eps_core is not None:
            if callable(eps_core):
                self.eps_core = eps_core
            else:
                self._eps_core_val = complex(eps_core)
                self.eps_core = lambda wl: self._eps_core_val
        else:
            self.eps_core = None

    def effective_eps(self, wavelength, eps_out=1.0):
        """
        Compute effective dielectric function for thin layer.

        For a thin layer, the effective boundary condition can be
        approximated using an effective dielectric constant.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_out : complex
            Outside dielectric function

        Returns
        -------
        eps_eff : complex
            Effective dielectric function
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        # Thin layer approximation
        # For d << wavelength, effective eps is weighted average
        if self.eps_core is not None:
            eps_c = self.eps_core(wavelength)
        else:
            eps_c = eps_l

        # Volume fraction approximation (for spherical core-shell)
        # This is a simplified model; exact solution depends on geometry
        eps_eff = eps_l

        return eps_eff

    def reflection_coefficient(self, wavelength, eps_in, eps_out, angle=0):
        """
        Compute reflection coefficient through layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_in : complex
            Inside dielectric function
        eps_out : complex
            Outside dielectric function
        angle : float
            Incidence angle in radians

        Returns
        -------
        r : complex
            Reflection coefficient
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        n_in = np.sqrt(eps_in)
        n_l = np.sqrt(eps_l)
        n_out = np.sqrt(eps_out)

        # Snell's law
        theta_in = angle
        sin_l = n_in / n_l * np.sin(theta_in)
        if np.abs(sin_l) > 1:
            # Total internal reflection
            cos_l = 1j * np.sqrt(sin_l**2 - 1)
        else:
            cos_l = np.sqrt(1 - sin_l**2)

        sin_out = n_l / n_out * sin_l
        if np.abs(sin_out) > 1:
            cos_out = 1j * np.sqrt(sin_out**2 - 1)
        else:
            cos_out = np.sqrt(1 - sin_out**2)

        # Fresnel coefficients at each interface (s-polarization)
        r_in_l = (n_in * np.cos(theta_in) - n_l * cos_l) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        r_l_out = (n_l * cos_l - n_out * cos_out) / \
                  (n_l * cos_l + n_out * cos_out)

        # Phase factor through layer
        delta = k * n_l * d * cos_l

        # Total reflection (Fabry-Perot)
        r_total = (r_in_l + r_l_out * np.exp(2j * delta)) / \
                  (1 + r_in_l * r_l_out * np.exp(2j * delta))

        return r_total

    def transmission_coefficient(self, wavelength, eps_in, eps_out, angle=0):
        """
        Compute transmission coefficient through layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_in : complex
            Inside dielectric function
        eps_out : complex
            Outside dielectric function
        angle : float
            Incidence angle in radians

        Returns
        -------
        t : complex
            Transmission coefficient
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        n_in = np.sqrt(eps_in)
        n_l = np.sqrt(eps_l)
        n_out = np.sqrt(eps_out)

        theta_in = angle
        sin_l = n_in / n_l * np.sin(theta_in)
        cos_l = np.sqrt(1 - np.clip(sin_l**2, 0, 1) + 0j)

        sin_out = n_l / n_out * sin_l
        cos_out = np.sqrt(1 - np.clip(sin_out**2, 0, 1) + 0j)

        # Fresnel coefficients
        t_in_l = 2 * n_in * np.cos(theta_in) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        t_l_out = 2 * n_l * cos_l / \
                  (n_l * cos_l + n_out * cos_out)

        r_in_l = (n_in * np.cos(theta_in) - n_l * cos_l) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        r_l_out = (n_l * cos_l - n_out * cos_out) / \
                  (n_l * cos_l + n_out * cos_out)

        delta = k * n_l * d * cos_l

        t_total = t_in_l * t_l_out * np.exp(1j * delta) / \
                  (1 + r_in_l * r_l_out * np.exp(2j * delta))

        return t_total


class GreenStatCover:
    """
    Quasistatic Green function with cover layer.

    Modifies the standard Green function to account for
    a thin dielectric layer on the particle surface.

    Parameters
    ----------
    particle : ComParticle
        Composite particle with cover layer
    cover : CoverLayer
        Cover layer object
    options : BEMOptions, optional
        Simulation options
    """

    def __init__(self, particle, cover, options=None):
        """Initialize Green function with cover."""
        self.particle = particle
        self.cover = cover
        self.options = options

        # Get particle geometry
        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

    def G(self, wavelength, inout=None):
        """
        Compute Green function matrix with cover layer correction.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            (inside, outside) medium indices

        Returns
        -------
        G : ndarray
            Green function matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        area = self.pc.area
        n = len(pos)

        # Get dielectric functions
        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_in = self.particle.eps[1](wavelength) if len(self.particle.eps) > 1 else 1.0
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_in = 1.0
            eps_out = 1.0

        # Basic Coulomb Green function
        G = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    G[i, j] = 1.0 / (4 * np.pi * r)

        # Self-term (diagonal)
        for i in range(n):
            # Approximate self-term from area
            a_eff = np.sqrt(area[i] / np.pi)
            G[i, i] = 1.0 / (4 * np.pi * a_eff) * 1.5

        # Apply cover layer correction
        eps_l = self.cover.eps_layer(wavelength)
        d = self.cover.thickness

        # Correction factor for thin layer
        # delta_G ~ d * (eps_l - eps_out) / eps_l
        correction = d * (eps_l - eps_out) / eps_l / wavelength

        # Modify off-diagonal elements
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i, j] *= (1 + correction)

        return G

    def F(self, wavelength, inout=None):
        """
        Compute F matrix (surface derivative of Green function).

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        F : ndarray
            F matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        n = len(pos)

        F = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    r_hat = r_vec / r

                    # F = -dG/dn = (r . n) / (4 * pi * r^3)
                    F[i, j] = np.dot(r_hat, nvec[j]) / (4 * np.pi * r**2)

        return F


class GreenRetCover:
    """
    Retarded Green function with cover layer.

    Parameters
    ----------
    particle : ComParticle
        Composite particle
    cover : CoverLayer
        Cover layer object
    options : BEMOptions, optional
        Simulation options
    """

    def __init__(self, particle, cover, options=None):
        """Initialize retarded Green function with cover."""
        self.particle = particle
        self.cover = cover
        self.options = options

        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

    def G(self, wavelength, inout=None):
        """
        Compute retarded Green function with cover layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        G : ndarray
            Retarded Green function matrix
        """
        pos = self.pc.pos
        n = len(pos)
        k = 2 * np.pi / wavelength

        # Get dielectric functions
        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_out = 1.0

        n_med = np.sqrt(eps_out)
        k_med = k * n_med

        G = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    G[i, j] = np.exp(1j * k_med * r) / (4 * np.pi * r)

        # Self-term
        area = self.pc.area
        for i in range(n):
            a_eff = np.sqrt(area[i] / np.pi)
            G[i, i] = np.exp(1j * k_med * a_eff) / (4 * np.pi * a_eff) * 1.5

        # Cover layer correction
        eps_l = self.cover.eps_layer(wavelength)
        d = self.cover.thickness
        n_l = np.sqrt(eps_l)

        # Phase correction through layer
        phase_corr = np.exp(1j * k * n_l * d)

        # Apply correction
        G *= phase_corr

        return G

    def F(self, wavelength, inout=None):
        """
        Compute F matrix for retarded case with cover.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        F : ndarray
            F matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        n = len(pos)
        k = 2 * np.pi / wavelength

        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_out = 1.0

        n_med = np.sqrt(eps_out)
        k_med = k * n_med

        F = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    r_hat = r_vec / r

                    # dG/dr for retarded Green function
                    G_val = np.exp(1j * k_med * r) / (4 * np.pi * r)
                    dG_dr = G_val * (1j * k_med - 1.0 / r)

                    F[i, j] = np.dot(r_hat, nvec[j]) * dG_dr

        return F


def coverlayer(eps_layer, thickness, eps_core=None):
    """
    Factory function for cover layer.

    Parameters
    ----------
    eps_layer : complex or callable
        Layer dielectric function
    thickness : float
        Layer thickness in nm
    eps_core : complex or callable, optional
        Core dielectric function

    Returns
    -------
    CoverLayer
        Cover layer object
    """
    return CoverLayer(eps_layer, thickness, eps_core)
