"""
Retarded dipole excitation with layer (substrate) effects.

Includes full electromagnetic treatment with image dipole
contributions from planar interfaces.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions
from .dipole_ret import DipoleRet


class DipoleRetLayer(DipoleRet):
    """
    Retarded dipole excitation with layer (substrate) effects.

    Computes electromagnetic fields of an oscillating dipole
    including reflections from substrate interface. Important
    for calculating decay rates near surfaces.

    Parameters
    ----------
    pt : array_like
        Dipole positions (n_dip, 3) or (3,).
    dip : array_like
        Dipole moment directions (n_dip, 3) or (3,).
    layer : LayerStructure
        Layer structure defining substrate.
    options : BEMOptions, optional
        BEM simulation options.

    Examples
    --------
    >>> from pymnpbem import DipoleRetLayer, LayerStructure, EpsConst
    >>> layer = LayerStructure([EpsConst(1), EpsConst(2.25)])
    >>> dip = DipoleRetLayer([0, 0, 10], [0, 0, 1], layer=layer)
    """

    def __init__(
        self,
        pt,
        dip,
        layer=None,
        options=None
    ):
        """
        Initialize retarded dipole with layer.

        Parameters
        ----------
        pt : array_like
            Dipole positions.
        dip : array_like
            Dipole directions.
        layer : LayerStructure
            Layer structure.
        options : BEMOptions, optional
            Simulation options.
        """
        super().__init__(pt, dip, options)
        self.layer = layer

    def _image_dipole(
        self,
        pt: np.ndarray,
        dip: np.ndarray,
        z_interface: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image dipole position and moment.

        Parameters
        ----------
        pt : ndarray
            Dipole position.
        dip : ndarray
            Dipole moment.
        z_interface : float
            Z-coordinate of interface.

        Returns
        -------
        pt_image : ndarray
            Image dipole position.
        dip_image : ndarray
            Image dipole moment.
        """
        # Image position: mirror about interface
        pt_image = pt.copy()
        pt_image[2] = 2 * z_interface - pt[2]

        # Image dipole moment: parallel components same, perpendicular flipped
        dip_image = dip.copy()
        dip_image[2] = -dip[2]

        return pt_image, dip_image

    def _fresnel_coefficients(
        self,
        wavelength: float,
        kpar: float = 0.0
    ) -> Tuple[complex, complex]:
        """
        Compute Fresnel reflection coefficients.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        kpar : float
            Parallel wavevector component.

        Returns
        -------
        r_s : complex
            s-polarization reflection coefficient.
        r_p : complex
            p-polarization reflection coefficient.
        """
        if self.layer is None or len(self.layer.eps) < 2:
            return 0.0, 0.0

        # Get dielectric functions
        eps1 = self.layer.eps[0]
        eps2 = self.layer.eps[1]

        if callable(eps1):
            e1, _ = eps1(wavelength)
        else:
            e1 = eps1

        if callable(eps2):
            e2, _ = eps2(wavelength)
        else:
            e2 = eps2

        k0 = 2 * np.pi / wavelength

        # Perpendicular wavevectors
        kz1_sq = e1 * k0**2 - kpar**2
        kz2_sq = e2 * k0**2 - kpar**2

        kz1 = np.sqrt(kz1_sq + 0j)
        kz2 = np.sqrt(kz2_sq + 0j)

        # Choose correct branch (Im(kz) > 0 for evanescent)
        if np.imag(kz1) < 0:
            kz1 = -kz1
        if np.imag(kz2) < 0:
            kz2 = -kz2

        # Fresnel coefficients
        r_s = (kz1 - kz2) / (kz1 + kz2)
        r_p = (e2 * kz1 - e1 * kz2) / (e2 * kz1 + e1 * kz2)

        return r_s, r_p

    def fields(self, pos: np.ndarray, wavelength: float, eps_out: float = 1.0):
        """
        Compute electric and magnetic fields including substrate reflection.

        Parameters
        ----------
        pos : ndarray
            Field positions (n_pos, 3).
        wavelength : float
            Wavelength in nm.
        eps_out : float
            Dielectric function of surrounding medium.

        Returns
        -------
        E : ndarray
            Electric field (n_pos, n_dip, 3).
        H : ndarray
            Magnetic field (n_pos, n_dip, 3).
        """
        # Get direct fields from dipole
        E, H = super().fields(pos, wavelength, eps_out)

        if self.layer is None:
            return E, H

        # Add reflected fields
        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0

        # Get Fresnel coefficients at normal incidence (approximation)
        r_s, r_p = self._fresnel_coefficients(wavelength, 0)

        # Wave parameters
        n_med = np.sqrt(eps_out)
        k = 2 * np.pi * n_med / wavelength

        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        for i in range(self.n_dip):
            # Only add reflected field for dipoles above interface
            if self.pt[i, 2] < z_interface:
                continue

            # Image dipole
            pt_image, dip_image_para = self._image_dipole(
                self.pt[i], self.dip[i], z_interface
            )

            # Perpendicular and parallel components get different reflection
            dip_perp = self.dip[i].copy()
            dip_perp[:2] = 0  # Keep only z-component
            dip_para = self.dip[i].copy()
            dip_para[2] = 0  # Keep only x,y-components

            # Image dipole moments with Fresnel weights
            # For s-polarization (perpendicular to plane containing k and z)
            # For p-polarization (in that plane)
            # Simplified: perpendicular dipole uses r_p, parallel uses average
            dip_image = r_p * dip_perp + r_s * dip_para
            dip_image[2] = -dip_image[2]  # Flip z for image

            # Compute field from image dipole
            for j, r in enumerate(pos):
                if r[2] < z_interface:
                    continue  # No image field below interface

                r_vec = r - pt_image
                r_dist = np.linalg.norm(r_vec)

                if r_dist < 1e-10:
                    continue

                n_hat = r_vec / r_dist
                n_dot_p = np.dot(n_hat, dip_image)

                kr = k * r_dist
                G = np.exp(1j * kr) / (4 * np.pi * r_dist)

                # Near and far field contributions
                near_coef = (1.0 / r_dist**3 - 1j * k / r_dist**2) * np.exp(1j * kr)
                far_coef = k**2 * G

                E_far = far_coef * (dip_image - n_hat * n_dot_p)
                E_near = near_coef * (3 * n_hat * n_dot_p - dip_image)

                E[j, i, :] += E_far + E_near

                # Magnetic field
                n_cross_p = np.cross(n_hat, dip_image)
                H_coef = k**2 * G * (1 - 1.0 / (1j * kr))
                H[j, i, :] += H_coef * n_cross_p

        return E, H

    def decayrate(self, sig, plane='total'):
        """
        Compute decay rate enhancement including substrate effects.

        The decay rate is modified by both the nanoparticle and
        the substrate interface.

        Parameters
        ----------
        sig : Solution
            BEM solution.
        plane : str
            'total', 'parallel', or 'perpendicular'.

        Returns
        -------
        gamma : ndarray
            Decay rate enhancement for each dipole.
        """
        # Get base decay rate from BEM
        gamma_particle = super().decayrate(sig, plane)

        if self.layer is None:
            return gamma_particle

        # Add substrate contribution
        # This is a simplified model; full treatment requires
        # Sommerfeld integrals
        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0
        r_s, r_p = self._fresnel_coefficients(sig.wavelength, 0)

        gamma = np.zeros(self.n_dip)

        for i in range(self.n_dip):
            d = np.abs(self.pt[i, 2] - z_interface)  # Distance to interface

            if d < 1e-10:
                gamma[i] = gamma_particle[i]
                continue

            k = 2 * np.pi / sig.wavelength

            # Dipole orientation
            dip_perp = np.abs(self.dip[i, 2])  # Perpendicular component
            dip_para = np.sqrt(self.dip[i, 0]**2 + self.dip[i, 1]**2)

            # Near-field approximation for substrate contribution
            # gamma_sub ~ Im(r) / (k*d)^3
            gamma_sub_perp = np.imag(r_p) / (2 * (k * d)**3)
            gamma_sub_para = np.imag(r_s) / (4 * (k * d)**3)

            gamma_substrate = dip_perp**2 * gamma_sub_perp + dip_para**2 * gamma_sub_para

            gamma[i] = gamma_particle[i] + gamma_substrate

        return gamma

    def __call__(self, particle: ComParticle, wavelength: float):
        """
        Compute excitation for particle.

        Parameters
        ----------
        particle : ComParticle
            Composite particle.
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        exc : DipoleRetLayerExcitation
            Excitation object.
        """
        self._wavelength = wavelength
        return DipoleRetLayerExcitation(self, particle, wavelength)

    def __repr__(self) -> str:
        return f"DipoleRetLayer(n_dip={self.n_dip}, layer={self.layer})"


class DipoleRetLayerExcitation:
    """
    Excitation object for retarded dipole with layer.
    """

    def __init__(self, dipole, particle, wavelength):
        """Initialize excitation."""
        self.dipole = dipole
        self.particle = particle
        self.wavelength = wavelength

        if hasattr(particle, 'pos'):
            self.pos = particle.pos
        else:
            self.pos = particle.pc.pos

        self._compute_fields()

    def _compute_fields(self):
        """Compute incident fields at surface."""
        eps_out = 1.0
        if hasattr(self.particle, 'eps'):
            eps_out = self.particle.eps[0](self.wavelength)

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


class DipoleRetMirror(DipoleRetLayer):
    """
    Retarded dipole with perfect mirror substrate.

    Parameters
    ----------
    pt : array_like
        Dipole positions.
    dip : array_like
        Dipole directions.
    z_mirror : float
        Z-coordinate of mirror plane.
    options : BEMOptions, optional
        Simulation options.
    """

    def __init__(
        self,
        pt,
        dip,
        z_mirror: float = 0.0,
        options=None
    ):
        """Initialize with perfect mirror."""
        super().__init__(pt, dip, None, options)
        self.z_mirror = z_mirror
        self._r_mirror = -1.0

    def _fresnel_coefficients(
        self,
        wavelength: float,
        kpar: float = 0.0
    ) -> Tuple[complex, complex]:
        """Perfect mirror reflection."""
        return self._r_mirror, self._r_mirror

    def __repr__(self) -> str:
        return f"DipoleRetMirror(n_dip={self.n_dip}, z_mirror={self.z_mirror})"


def dipole_ret_layer(
    pt,
    dip,
    layer,
    options=None
) -> DipoleRetLayer:
    """
    Factory function for retarded dipole with layer.

    Parameters
    ----------
    pt : array_like
        Dipole positions.
    dip : array_like
        Dipole directions.
    layer : LayerStructure
        Layer structure.
    options : BEMOptions, optional
        Simulation options.

    Returns
    -------
    DipoleRetLayer
        Dipole excitation with layer effects.
    """
    return DipoleRetLayer(pt, dip, layer, options)
