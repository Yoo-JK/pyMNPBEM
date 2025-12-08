"""
Retarded plane wave excitation with layer (substrate) effects.

Includes full electromagnetic treatment with reflected fields
from planar interfaces.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions
from .planewave_ret import PlaneWaveRet


class PlaneWaveRetLayer(PlaneWaveRet):
    """
    Retarded plane wave excitation with layer (substrate) effects.

    Includes full electromagnetic reflected fields from substrate,
    including retardation effects important for larger particles.

    Parameters
    ----------
    pol : ndarray
        Light polarization directions, shape (n_exc, 3) or (3,).
    dir : ndarray
        Light propagation directions, shape (n_exc, 3) or (3,).
    layer : LayerStructure
        Layer structure defining substrate.
    medium : int
        Index of medium through which particle is excited.
    options : BEMOptions, optional
        BEM simulation options.

    Examples
    --------
    >>> from pymnpbem import PlaneWaveRetLayer, LayerStructure, EpsConst
    >>> layer = LayerStructure([EpsConst(1), EpsConst(2.25)])  # air-glass
    >>> exc = PlaneWaveRetLayer([1, 0, 0], [0, 0, 1], layer=layer)
    """

    def __init__(
        self,
        pol: np.ndarray,
        dir: np.ndarray,
        layer=None,
        medium: int = 1,
        options=None
    ):
        """
        Initialize retarded plane wave with layer.

        Parameters
        ----------
        pol : ndarray
            Polarization directions.
        dir : ndarray
            Propagation directions.
        layer : LayerStructure
            Layer structure.
        medium : int
            Index of excitation medium (1-based).
        options : BEMOptions, optional
            Simulation options.
        """
        super().__init__(pol, dir, options)
        self.layer = layer
        self.medium = medium

    def _fresnel_coefficients(
        self,
        wavelength: float,
        theta: float = 0.0
    ) -> Tuple[complex, complex]:
        """
        Compute Fresnel reflection coefficients.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        theta : float
            Angle of incidence.

        Returns
        -------
        r_s : complex
            s-polarization (TE) reflection coefficient.
        r_p : complex
            p-polarization (TM) reflection coefficient.
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

        n1 = np.sqrt(e1)
        n2 = np.sqrt(e2)

        # Snell's law
        cos_theta1 = np.cos(theta)
        sin_theta1 = np.sin(theta)
        cos_theta2 = np.sqrt(1 - (n1 / n2 * sin_theta1) ** 2 + 0j)

        # Fresnel coefficients
        # s-polarization (TE)
        r_s = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)

        # p-polarization (TM)
        r_p = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)

        return r_s, r_p

    def _decompose_polarization(self, pol: np.ndarray, dir: np.ndarray) -> Tuple[float, float]:
        """
        Decompose polarization into s and p components.

        Parameters
        ----------
        pol : ndarray
            Polarization vector.
        dir : ndarray
            Propagation direction.

        Returns
        -------
        s_comp : float
            s-polarization component amplitude.
        p_comp : float
            p-polarization component amplitude.
        """
        # s-direction: perpendicular to plane of incidence (k x z)
        z_hat = np.array([0, 0, 1])
        s_dir = np.cross(dir, z_hat)
        s_norm = np.linalg.norm(s_dir)

        if s_norm < 1e-10:
            # Normal incidence, s and p are degenerate
            return np.linalg.norm(pol[:2]), 0.0

        s_dir = s_dir / s_norm

        # p-direction: in plane of incidence, perpendicular to k
        p_dir = np.cross(s_dir, dir)

        s_comp = np.dot(pol, s_dir)
        p_comp = np.dot(pol, p_dir)

        return s_comp, p_comp

    def fields(self, pos: np.ndarray, wavelength: float, eps_out: float = 1.0):
        """
        Compute electric and magnetic fields including reflection.

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
            Electric field (n_pos, n_pol, 3).
        H : ndarray
            Magnetic field (n_pos, n_pol, 3).
        """
        # Get direct fields
        E_direct, H_direct = super().fields(pos, wavelength, eps_out)

        if self.layer is None:
            return E_direct, H_direct

        pos = np.atleast_2d(pos)
        n_pos = len(pos)

        E = E_direct.copy()
        H = H_direct.copy()

        # Wave number
        n_med = np.sqrt(eps_out)
        k = 2 * np.pi * n_med / wavelength

        # Interface position
        z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0

        for i in range(self.n_pol):
            # Get angle of incidence
            cos_theta = np.abs(self.dir[i, 2])
            theta = np.arccos(cos_theta)

            # Get Fresnel coefficients
            r_s, r_p = self._fresnel_coefficients(wavelength, theta)

            # Decompose polarization
            s_comp, p_comp = self._decompose_polarization(self.pol[i], self.dir[i])

            # Reflected wave direction
            dir_refl = self.dir[i].copy()
            dir_refl[2] = -dir_refl[2]

            # Reflected polarization
            pol_s = self.pol[i] - self.pol[i, 2] * np.array([0, 0, 1])  # in-plane
            pol_p = self.pol[i, 2] * np.array([0, 0, 1])

            pol_refl = r_s * pol_s + r_p * pol_p
            pol_refl[2] = -pol_refl[2]  # Flip z-component for reflection

            # Wave vector for reflected wave
            kvec_refl = k * dir_refl

            # Phase for reflected wave (from interface)
            # Distance traveled: down to interface and back up
            z_to_interface = 2 * (z_interface - pos[:, 2])
            phase_refl = np.exp(1j * k * np.abs(self.dir[i, 2]) * z_to_interface)

            # Add reflected field
            for j, pt in enumerate(pos):
                if pt[2] > z_interface:  # Above interface
                    phase = phase_refl[j] * np.exp(1j * np.dot(kvec_refl, pt))
                    E[j, i, :] += phase * pol_refl

                    # Magnetic field
                    k_cross_E = np.cross(dir_refl, pol_refl)
                    H[j, i, :] += n_med * phase * k_cross_E

        return E, H

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
        exc : PlaneWaveRetLayerExcitation
            Excitation object.
        """
        self._wavelength = wavelength
        self._k = 2 * np.pi / wavelength
        return PlaneWaveRetLayerExcitation(self, particle, wavelength)

    def __repr__(self) -> str:
        return f"PlaneWaveRetLayer(n_pol={self.n_pol}, layer={self.layer})"


class PlaneWaveRetLayerExcitation:
    """
    Excitation object for retarded plane wave with layer.
    """

    def __init__(self, planewave, particle, wavelength):
        """Initialize excitation."""
        self.planewave = planewave
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


class PlaneWaveRetMirror(PlaneWaveRetLayer):
    """
    Retarded plane wave with perfect mirror substrate.

    Parameters
    ----------
    pol : ndarray
        Polarization directions.
    dir : ndarray
        Propagation directions.
    z_mirror : float
        Z-coordinate of mirror plane.
    options : BEMOptions, optional
        Simulation options.
    """

    def __init__(
        self,
        pol: np.ndarray,
        dir: np.ndarray,
        z_mirror: float = 0.0,
        options=None
    ):
        """Initialize with perfect mirror."""
        super().__init__(pol, dir, None, 1, options)
        self.z_mirror = z_mirror
        self._r_mirror = -1.0

    def _fresnel_coefficients(
        self,
        wavelength: float,
        theta: float = 0.0
    ) -> Tuple[complex, complex]:
        """Perfect mirror reflection."""
        return self._r_mirror, self._r_mirror

    def __repr__(self) -> str:
        return f"PlaneWaveRetMirror(n_pol={self.n_pol}, z_mirror={self.z_mirror})"


def planewave_ret_layer(
    pol: np.ndarray,
    dir: np.ndarray,
    layer,
    options=None,
    **kwargs
) -> PlaneWaveRetLayer:
    """
    Factory function for retarded plane wave with layer.

    Parameters
    ----------
    pol : ndarray
        Polarization directions.
    dir : ndarray
        Propagation directions.
    layer : LayerStructure
        Layer structure.
    options : BEMOptions, optional
        Simulation options.

    Returns
    -------
    PlaneWaveRetLayer
        Plane wave excitation with layer effects.
    """
    medium = kwargs.pop('medium', 1)
    return PlaneWaveRetLayer(pol, dir, layer, medium, options)
