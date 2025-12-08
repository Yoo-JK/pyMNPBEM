"""
Dipole excitation with layer (substrate) effects.

Includes image dipole contributions from planar interfaces.
"""

import numpy as np
from typing import Optional, Union

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions
from .dipole_stat import DipoleStat


class DipoleStatLayer(DipoleStat):
    """
    Dipole excitation with layer (substrate) effects.

    Includes image dipole from substrate using mirror charges.

    Parameters
    ----------
    pt : ndarray
        Dipole positions, shape (n_dip, 3) or (3,).
    dip : ndarray
        Dipole moments, shape (n_dip, 3) or (3,).
    layer : LayerStructure
        Layer structure defining substrate.
    medium : int
        Index of medium containing the dipole.

    Examples
    --------
    >>> from mnpbem import DipoleStatLayer, LayerStructure
    >>> layer = LayerStructure([0], [eps_vacuum, eps_glass])
    >>> exc = DipoleStatLayer([0, 0, 10], [0, 0, 1], layer=layer)
    """

    def __init__(
        self,
        pt: np.ndarray,
        dip: np.ndarray,
        layer=None,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize dipole excitation with layer.

        Parameters
        ----------
        pt : ndarray
            Dipole positions.
        dip : ndarray
            Dipole moments.
        layer : LayerStructure
            Layer structure.
        medium : int
            Index of excitation medium (1-based).
        """
        super().__init__(pt, dip, medium, **kwargs)
        self.layer = layer

    def _fresnel_reflection(self, enei: float) -> complex:
        """
        Compute Fresnel reflection coefficient for image charge.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        complex
            Fresnel reflection coefficient.
        """
        if self.layer is None or len(self.layer.eps) < 2:
            return 0.0

        eps1 = self.layer.eps[0]
        eps2 = self.layer.eps[1]

        if callable(eps1):
            e1, _ = eps1(enei)
        else:
            e1 = eps1

        if callable(eps2):
            e2, _ = eps2(enei)
        else:
            e2 = eps2

        # Image charge coefficient
        r = (e2 - e1) / (e2 + e1)

        return r

    def _image_dipole(self, enei: float) -> tuple:
        """
        Compute image dipole position and moment.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        pt_image : ndarray
            Image dipole positions.
        dip_image : ndarray
            Image dipole moments.
        r : complex
            Reflection coefficient.
        """
        z_interface = self.layer.z[0] if self.layer is not None and len(self.layer.z) > 0 else 0

        # Mirror position about interface
        pt_image = self.pt.copy()
        pt_image[:, 2] = 2 * z_interface - pt_image[:, 2]

        # Mirror dipole moment
        # Parallel components keep sign, perpendicular component flips
        dip_image = self.dip.copy()
        dip_image[:, 2] = -dip_image[:, 2]

        r = self._fresnel_reflection(enei)

        return pt_image, dip_image, r

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential at particle boundary including image dipole.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        pos = p.pos
        nvec = p.nvec

        phip = np.zeros((p.n_faces, self.n_dip))

        # Direct dipole contribution
        for i, (r0, dip) in enumerate(zip(self.pt, self.dip)):
            dr = pos - r0
            r = np.linalg.norm(dr, axis=1)
            r = np.where(r == 0, np.inf, r)
            r3 = r ** 3
            r5 = r ** 5

            p_dot_dr = np.sum(dip * dr, axis=1)
            term1 = np.sum(dip * nvec, axis=1) / r3
            term2 = 3 * p_dot_dr * np.sum(dr * nvec, axis=1) / r5

            phip[:, i] = (1 / (4 * np.pi)) * (term1 - term2)

        # Image dipole contribution
        if self.layer is not None:
            pt_image, dip_image, r_coeff = self._image_dipole(enei)

            for i, (r0, dip) in enumerate(zip(pt_image, dip_image)):
                dr = pos - r0
                r = np.linalg.norm(dr, axis=1)
                r = np.where(r == 0, np.inf, r)
                r3 = r ** 3
                r5 = r ** 5

                p_dot_dr = np.sum(dip * dr, axis=1)
                term1 = np.sum(dip * nvec, axis=1) / r3
                term2 = 3 * p_dot_dr * np.sum(dr * nvec, axis=1) / r5

                phip[:, i] += r_coeff * (1 / (4 * np.pi)) * (term1 - term2)

        return CompStruct(p, enei, phip=phip)

    def decay_rate(self, sig: CompStruct) -> np.ndarray:
        """
        Compute decay rate enhancement including substrate effects.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Decay rate enhancement for each dipole.
        """
        # Basic decay rate from parent class
        gamma_basic = super().decay_rate(sig)

        # Add image dipole contribution
        if self.layer is not None:
            pt_image, dip_image, r_coeff = self._image_dipole(sig.enei)

            gamma_image = np.zeros(self.n_dip)
            for i, (r0, r0_img, dip, dip_img) in enumerate(zip(self.pt, pt_image, self.dip, dip_image)):
                # Distance between dipole and its image
                dr = r0 - r0_img
                r = np.linalg.norm(dr)

                if r > 1e-10:
                    # Field from image dipole at original dipole location
                    r3 = r ** 3
                    r5 = r ** 5

                    p_dot_dr = np.dot(dip_img, dr)
                    E_image = r_coeff / (4 * np.pi) * (3 * p_dot_dr * dr / r5 - dip_img / r3)

                    # Interaction with original dipole
                    gamma_image[i] = np.imag(np.dot(dip, E_image))

            gamma_basic += gamma_image

        return gamma_basic

    def __repr__(self) -> str:
        return f"DipoleStatLayer(n_dip={self.n_dip}, layer={self.layer})"


class DipoleStatMirror(DipoleStatLayer):
    """
    Dipole excitation with perfect mirror substrate.

    Parameters
    ----------
    pt : ndarray
        Dipole positions.
    dip : ndarray
        Dipole moments.
    z_mirror : float
        Z-coordinate of mirror plane.
    medium : int
        Index of medium.
    """

    def __init__(
        self,
        pt: np.ndarray,
        dip: np.ndarray,
        z_mirror: float = 0.0,
        medium: int = 1,
        **kwargs
    ):
        from ..particles import LayerStructure

        layer = LayerStructure([z_mirror], [])
        super().__init__(pt, dip, layer, medium, **kwargs)
        self.z_mirror = z_mirror
        self._r_mirror = 1.0  # Perfect conductor (for perpendicular)

    def _fresnel_reflection(self, enei: float) -> complex:
        """Perfect mirror reflection."""
        return self._r_mirror

    def __repr__(self) -> str:
        return f"DipoleStatMirror(n_dip={self.n_dip}, z_mirror={self.z_mirror})"


def dipole_layer(
    pt: np.ndarray,
    dip: np.ndarray,
    layer,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> DipoleStatLayer:
    """
    Factory function for dipole excitation with layer.

    Parameters
    ----------
    pt : ndarray
        Dipole positions.
    dip : ndarray
        Dipole moments.
    layer : LayerStructure
        Layer structure.
    options : BEMOptions or dict, optional
        Options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    DipoleStatLayer
        Dipole excitation with layer effects.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}
    medium = all_options.pop('medium', 1)

    return DipoleStatLayer(pt, dip, layer, medium=medium, **all_options)
