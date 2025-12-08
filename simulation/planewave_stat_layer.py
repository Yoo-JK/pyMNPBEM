"""
Plane wave excitation with layer (substrate) effects.

Includes reflected fields from planar interfaces.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions
from .planewave_stat import PlaneWaveStat


class PlaneWaveStatLayer(PlaneWaveStat):
    """
    Plane wave excitation with layer (substrate) effects.

    In quasistatic limit, includes reflected plane wave from substrate.

    Parameters
    ----------
    pol : ndarray
        Light polarization directions, shape (n_exc, 3) or (3,).
    direction : ndarray, optional
        Light propagation directions.
    layer : LayerStructure
        Layer structure defining substrate.
    medium : int
        Index of medium through which particle is excited.

    Examples
    --------
    >>> from mnpbem import PlaneWaveStatLayer, LayerStructure
    >>> layer = LayerStructure([0], [eps_vacuum, eps_glass])
    >>> exc = PlaneWaveStatLayer([1, 0, 0], layer=layer)
    """

    def __init__(
        self,
        pol: np.ndarray,
        direction: Optional[np.ndarray] = None,
        layer=None,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize plane wave excitation with layer.

        Parameters
        ----------
        pol : ndarray
            Polarization directions.
        direction : ndarray, optional
            Propagation directions.
        layer : LayerStructure
            Layer structure.
        medium : int
            Index of excitation medium (1-based).
        """
        super().__init__(pol, direction, medium, **kwargs)
        self.layer = layer

    def _fresnel_reflection(self, enei: float, pol_type: str = 's') -> complex:
        """
        Compute Fresnel reflection coefficient.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        pol_type : str
            Polarization type: 's' or 'p'.

        Returns
        -------
        complex
            Fresnel reflection coefficient.
        """
        if self.layer is None or len(self.layer.eps) < 2:
            return 0.0

        # Get dielectric functions
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

        n1 = np.sqrt(e1)
        n2 = np.sqrt(e2)

        # Normal incidence Fresnel coefficient
        if pol_type == 's':
            r = (n1 - n2) / (n1 + n2)
        else:  # p-polarization
            r = (n2 * n1 - n1 * n2) / (n2 * n1 + n1 * n2)
            # Simplifies to same as s for normal incidence

        return r

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential derivative at particle boundary including reflection.

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
        pos = p.pos  # Face centroids
        nvec = p.nvec  # Normal vectors

        # Direct field contribution
        phip_direct = -nvec @ self.pol.T

        # Reflected field contribution
        if self.layer is not None:
            z_interface = self.layer.z[0] if len(self.layer.z) > 0 else 0

            # Reflection coefficient
            r_s = self._fresnel_reflection(enei, 's')

            # Mirror image polarization
            pol_refl = self.pol.copy()
            pol_refl[:, 2] = -pol_refl[:, 2]  # Flip z-component

            # Phase factor based on distance from interface
            # In quasistatic limit, this is just the reflection coefficient
            phip_refl = -r_s * (nvec @ pol_refl.T)

            phip = phip_direct + phip_refl
        else:
            phip = phip_direct

        return CompStruct(p, enei, phip=phip)

    def field(self, p: ComParticle, enei: float) -> np.ndarray:
        """
        Electric field of plane wave including reflection.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Electric field at face centroids, shape (n_faces, 3, n_exc).
        """
        n_faces = p.n_faces
        field = np.zeros((n_faces, 3, self.n_exc))

        # Direct field
        for i, pol in enumerate(self.pol):
            field[:, :, i] = pol

        # Add reflected field
        if self.layer is not None:
            r_s = self._fresnel_reflection(enei, 's')

            pol_refl = self.pol.copy()
            pol_refl[:, 2] = -pol_refl[:, 2]

            for i, pol in enumerate(pol_refl):
                field[:, :, i] += r_s * pol

        return field

    def __repr__(self) -> str:
        return f"PlaneWaveStatLayer(pol={self.pol.tolist()}, layer={self.layer})"


class PlaneWaveStatMirror(PlaneWaveStatLayer):
    """
    Plane wave excitation with perfect mirror substrate.

    Simplified version of PlaneWaveStatLayer with r = -1 (perfect conductor).

    Parameters
    ----------
    pol : ndarray
        Light polarization directions.
    direction : ndarray, optional
        Light propagation directions.
    z_mirror : float
        Z-coordinate of mirror plane.
    medium : int
        Index of medium.

    Examples
    --------
    >>> exc = PlaneWaveStatMirror([1, 0, 0], z_mirror=0)
    """

    def __init__(
        self,
        pol: np.ndarray,
        direction: Optional[np.ndarray] = None,
        z_mirror: float = 0.0,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize plane wave with mirror.

        Parameters
        ----------
        pol : ndarray
            Polarization directions.
        direction : ndarray, optional
            Propagation directions.
        z_mirror : float
            Z-coordinate of mirror.
        medium : int
            Medium index.
        """
        # Create dummy layer structure
        from ..particles import LayerStructure

        # Perfect mirror has r = -1
        layer = LayerStructure([z_mirror], [])
        super().__init__(pol, direction, layer, medium, **kwargs)
        self.z_mirror = z_mirror
        self._r_mirror = -1.0  # Perfect reflection

    def _fresnel_reflection(self, enei: float, pol_type: str = 's') -> complex:
        """Perfect mirror reflection."""
        return self._r_mirror

    def __repr__(self) -> str:
        return f"PlaneWaveStatMirror(pol={self.pol.tolist()}, z_mirror={self.z_mirror})"


def planewave_layer(
    pol: np.ndarray,
    layer,
    direction: Optional[np.ndarray] = None,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> PlaneWaveStatLayer:
    """
    Factory function for plane wave with layer.

    Parameters
    ----------
    pol : ndarray
        Polarization directions.
    layer : LayerStructure
        Layer structure.
    direction : ndarray, optional
        Propagation directions.
    options : BEMOptions or dict, optional
        Options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    PlaneWaveStatLayer
        Plane wave excitation with layer effects.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}
    medium = all_options.pop('medium', 1)

    return PlaneWaveStatLayer(pol, direction, layer, medium=medium, **all_options)
