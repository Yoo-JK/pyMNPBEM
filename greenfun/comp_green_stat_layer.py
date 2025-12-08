"""
Green function for layer structure in quasistatic limit.

Uses image charge method for substrate reflection.
"""

import numpy as np
from typing import Optional, Tuple, Any

from .green_stat import GreenStat
from .comp_green_stat import CompGreenStat


class CompGreenStatLayer:
    """
    Quasistatic Green function for layer structure.

    Works for single layer (substrate) with particles in upper medium.
    The reflected part is computed using image charges.

    Parameters
    ----------
    p1 : ComParticle or Point
        First particle/points.
    p2 : ComParticle
        Second particle.
    layer : LayerStructure
        Layer structure (substrate).
    **kwargs : dict
        Additional options.

    Attributes
    ----------
    g : CompGreenStat
        Direct Green function.
    gr : CompGreenStat
        Reflected Green function (image charges).
    layer : LayerStructure
        Layer structure reference.
    """

    def __init__(self, p1, p2, layer, **kwargs):
        """Initialize layer Green function."""
        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.options = kwargs

        # Get positions
        self.pos1 = p1.pos if hasattr(p1, 'pos') else p1.pc.pos
        self.pos2 = p2.pos if hasattr(p2, 'pos') else p2.pc.pos

        # Ensure single layer (substrate)
        assert layer.n_layers == 2, "Only single layer (substrate) supported"

        # Find elements in and above layer
        self._init_indices()

        # Create image particle
        self._create_image_particle()

        # Initialize Green functions
        self.g = CompGreenStat(p1, p2, **kwargs)
        if len(self.ind1) > 0 and len(self.ind2) > 0:
            self.gr = self._create_reflected_green(**kwargs)
        else:
            self.gr = None

    def _init_indices(self) -> None:
        """Initialize indices for elements above/in layer."""
        z = self.layer.z[0] if len(self.layer.z) > 0 else 0

        # Elements in p2 at the layer interface
        self.indl = np.where(np.abs(self.pos2[:, 2] - z) < 1e-10)[0]

        # Elements in p2 not at interface
        self.ind2 = np.setdiff1d(np.arange(len(self.pos2)), self.indl)

        # Elements in p1 above layer
        self.ind1 = np.where(self.pos1[:, 2] >= z)[0]

    def _create_image_particle(self) -> None:
        """Create image particle for reflection."""
        z = self.layer.z[0] if len(self.layer.z) > 0 else 0

        # Create image positions by reflecting across z-plane
        # Only for elements not in layer
        if hasattr(self.p2, 'pc'):
            from ..particles import Particle
            pos = self.p2.pc.pos[self.ind2].copy()
            pos[:, 2] = 2 * z - pos[:, 2]  # Reflect z

            # Flip normals
            nvec = self.p2.pc.nvec[self.ind2].copy()
            nvec[:, 2] = -nvec[:, 2]

            area = self.p2.pc.area[self.ind2].copy()

            # Store image particle data
            self.p2r_pos = pos
            self.p2r_nvec = nvec
            self.p2r_area = area
        else:
            pos = self.pos2[self.ind2].copy()
            pos[:, 2] = 2 * z - pos[:, 2]
            self.p2r_pos = pos

    def _create_reflected_green(self, **kwargs) -> Any:
        """Create Green function for reflected (image) charges."""
        from ..particles import Point

        # Create point objects for image charges
        pos1 = self.pos1[self.ind1]

        # Create simple Green function for image charges
        class ImageGreen:
            """Simple Green function for image charges."""

            def __init__(self, pos1, pos2_img, nvec2_img, area2_img):
                self.pos1 = pos1
                self.pos2 = pos2_img
                self.nvec2 = nvec2_img
                self.area2 = area2_img
                self._G = None
                self._F = None

            def _compute(self):
                """Compute Green function matrices."""
                n1, n2 = len(self.pos1), len(self.pos2)
                self._G = np.zeros((n1, n2))
                self._F = np.zeros((n1, n2))

                for i in range(n1):
                    for j in range(n2):
                        r_vec = self.pos1[i] - self.pos2[j]
                        r = np.linalg.norm(r_vec)
                        if r > 1e-10:
                            # Green function: G = 1/(4*pi*r)
                            self._G[i, j] = 1.0 / (4 * np.pi * r)
                            # Surface derivative: F = (r.n)/(4*pi*r^3) * area
                            r_hat = r_vec / r
                            n_dot_r = np.dot(r_hat, self.nvec2[j])
                            self._F[i, j] = n_dot_r / (4 * np.pi * r**2) * self.area2[j]

            @property
            def G(self):
                if self._G is None:
                    self._compute()
                return self._G

            @property
            def F(self):
                if self._F is None:
                    self._compute()
                return self._F

        return ImageGreen(pos1, self.p2r_pos, self.p2r_nvec, self.p2r_area)

    def eval(self, enei: float, *keys) -> Tuple:
        """
        Evaluate Green function.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        *keys : str
            Keys for requested quantities: 'G', 'F', 'H1', 'H2', 'Gp'.

        Returns
        -------
        tuple
            Requested Green function matrices.
        """
        # Get dielectric functions
        eps1 = self.layer.eps[0](enei)[0]  # Upper medium
        eps2 = self.layer.eps[1](enei)[0]  # Lower medium (substrate)

        # Multiplication factors
        n1 = len(self.pos1)
        n2 = len(self.pos2)

        # Factor for elements in upper medium
        f1 = np.ones(n1) * 2 * eps1 / (eps2 + eps1)

        # Factor for elements at layer interface
        fl = 2 * eps2 / (eps1 + eps2)

        # Image charge factor
        f2 = -(eps2 - eps1) / (eps2 + eps1)

        results = []

        for key in keys:
            if key == 'G':
                # Direct part
                G_direct = self.g.G
                G = np.diag(f1) @ G_direct

                # Correct for contributions at layer interface
                if len(self.indl) > 0:
                    G[:, self.indl] *= fl

                # Add reflected part
                if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
                    G[np.ix_(self.ind1, self.ind2)] += f2 * self.gr.G

                results.append(G)

            elif key in ('F', 'H1', 'H2'):
                # Direct part
                F_direct = self.g.F
                F = np.diag(f1) @ F_direct

                # Correct for layer contributions
                if len(self.indl) > 0:
                    ind_not_layer = np.setdiff1d(self.ind1, self.indl)
                    if len(ind_not_layer) > 0:
                        F[np.ix_(ind_not_layer, self.indl)] *= (1 + f2)

                # Add reflected part
                if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
                    F[np.ix_(self.ind1, self.ind2)] += f2 * self.gr.F

                # For self-interaction (p1 == p2)
                if self.p1 is self.p2:
                    # Zero diagonal for layer elements
                    if len(self.indl) > 0:
                        F[np.ix_(self.indl, self.indl)] = 0

                    if key == 'H1':
                        # H1 = F + 2*pi*(I + f2*diag)
                        diag_factor = np.zeros(n1)
                        diag_factor[self.indl] = f2
                        F = F + 2 * np.pi * (np.diag(diag_factor) + np.eye(n1))
                    elif key == 'H2':
                        # H2 = F + 2*pi*(-I + f2*diag)
                        diag_factor = np.zeros(n1)
                        diag_factor[self.indl] = f2
                        F = F + 2 * np.pi * (np.diag(diag_factor) - np.eye(n1))

                results.append(F)

            elif key == 'd':
                # Distance matrix
                from scipy.spatial.distance import cdist
                results.append(cdist(self.pos1, self.pos2))

        return tuple(results) if len(results) > 1 else results[0]

    @property
    def G(self) -> np.ndarray:
        """Green function matrix (at default wavelength)."""
        return self.g.G

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F

    def potential(self, sig: np.ndarray, enei: float) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Potential at p1 positions.
        """
        G = self.eval(enei, 'G')
        return G @ sig

    def field(self, sig: np.ndarray, enei: float) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Electric field at p1 positions.
        """
        return self.g.field(sig)

    def __repr__(self) -> str:
        return f"CompGreenStatLayer(n1={len(self.pos1)}, n2={len(self.pos2)})"


class CompGreenStatMirror:
    """
    Quasistatic Green function for particles with mirror symmetry.

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry.
    **kwargs : dict
        Additional options.
    """

    def __init__(self, p, p2=None, **kwargs):
        """Initialize mirror Green function."""
        self.p = p
        self.options = kwargs

        # Create Green function connecting reduced particle to full particle
        self.g = CompGreenStat(p, p.full, **kwargs)

    @property
    def G(self) -> np.ndarray:
        """Green function matrix."""
        return self.g.G

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F

    def eval(self, *keys) -> Tuple:
        """Evaluate Green function."""
        return self.g.eval(*keys)

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """Compute potential from surface charges."""
        return self.g.potential(sig)

    def field(self, sig: np.ndarray) -> np.ndarray:
        """Compute electric field from surface charges."""
        return self.g.field(sig)

    def __repr__(self) -> str:
        return f"CompGreenStatMirror(p={self.p})"


class CompGreenRetMirror:
    """
    Retarded Green function for particles with mirror symmetry.

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry.
    **kwargs : dict
        Additional options.
    """

    def __init__(self, p, p2=None, **kwargs):
        """Initialize mirror Green function."""
        from .comp_green_ret import CompGreenRet

        self.p = p
        self.options = kwargs

        # Create Green function connecting reduced particle to full particle
        self.g = CompGreenRet(p, p.full, **kwargs)

    def set_k(self, k: complex) -> None:
        """Set wavenumber."""
        self.g.set_k(k)

    def G(self, k: complex) -> np.ndarray:
        """Green function matrix."""
        return self.g.G(k)

    def F(self, k: complex) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F(k)

    def L(self, k: complex) -> np.ndarray:
        """Magnetic Green function tensor."""
        return self.g.L(k)

    def potential(self, sig: np.ndarray, k: complex) -> np.ndarray:
        """Compute potential from surface charges."""
        return self.g.potential(sig, k)

    def field(self, sig: np.ndarray, k: complex) -> np.ndarray:
        """Compute electric field from surface charges."""
        return self.g.field(sig, k)

    def __repr__(self) -> str:
        return f"CompGreenRetMirror(p={self.p})"
