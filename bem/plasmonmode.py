"""
Plasmon mode analysis for metallic nanoparticles.

This module provides eigenmode analysis of plasmonic structures
to find resonant modes and their associated field distributions.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, List, Union


class PlasmonMode:
    """
    Plasmon mode calculator.

    Finds the plasmonic eigenmodes of a nanoparticle by solving
    the eigenvalue problem for the BEM matrix.

    For quasistatic case:
    Lambda * sigma = F * sigma

    where Lambda = (eps_in + eps_out) / (eps_in - eps_out)
    is the eigenvalue and F is the surface integral operator.

    Parameters
    ----------
    particle : ComParticle
        Composite particle
    options : BEMOptions, optional
        BEM options

    Attributes
    ----------
    eigenvalues : ndarray
        Plasmon eigenvalues Lambda
    eigenmodes : ndarray
        Plasmon eigenmodes (surface charge distributions)
    resonance_eps : ndarray
        Dielectric function at resonance for each mode
    """

    def __init__(self, particle, options=None):
        """Initialize plasmon mode calculator."""
        self.particle = particle
        self.options = options

        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

        self.pos = self.pc.pos
        self.nvec = self.pc.nvec
        self.area = self.pc.area
        self.n = len(self.pos)

        self.eigenvalues = None
        self.eigenmodes = None
        self.resonance_eps = None

    def compute(self, n_modes=None, sigma=None):
        """
        Compute plasmon eigenmodes.

        Parameters
        ----------
        n_modes : int, optional
            Number of modes to compute (default: all)
        sigma : str, optional
            Eigenvalue selection: 'LM' (largest magnitude),
            'SM' (smallest), 'LR' (largest real), etc.

        Returns
        -------
        eigenvalues : ndarray
            Plasmon eigenvalues
        eigenmodes : ndarray
            Plasmon eigenmodes
        """
        # Build the F matrix (surface integral operator)
        F = self._build_F_matrix()

        if n_modes is None or n_modes >= self.n:
            # Full eigenvalue decomposition
            eigenvalues, eigenmodes = linalg.eig(F)
        else:
            # Sparse eigenvalue decomposition
            from scipy.sparse.linalg import eigs

            # Convert sigma to scipy format
            which = sigma if sigma else 'LM'

            eigenvalues, eigenmodes = eigs(F, k=n_modes, which=which)

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenmodes = eigenmodes[:, idx]

        self.eigenvalues = eigenvalues
        self.eigenmodes = eigenmodes

        # Compute resonance dielectric functions
        # Lambda = (eps_in + eps_out) / (eps_in - eps_out)
        # For vacuum outside (eps_out = 1):
        # eps_in = (Lambda + 1) / (Lambda - 1)
        self.resonance_eps = (eigenvalues + 1) / (eigenvalues - 1)

        return eigenvalues, eigenmodes

    def _build_F_matrix(self):
        """
        Build the F matrix (double-layer potential operator).

        F_ij = integral of dG/dn over face j, evaluated at centroid i

        Returns
        -------
        F : ndarray
            F matrix (n, n)
        """
        F = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    r_vec = self.pos[i] - self.pos[j]
                    r = np.linalg.norm(r_vec)
                    r_hat = r_vec / r

                    # F_ij = (r_hat . n_j) / (4 * pi * r^2) * area_j
                    n_dot_r = np.dot(r_hat, self.nvec[j])
                    F[i, j] = n_dot_r / (4 * np.pi * r**2) * self.area[j]

        # Diagonal term from solid angle
        # F_ii = -(1 - Omega_i / (4*pi)) where Omega_i is solid angle
        # For smooth surface: Omega = 2*pi, so F_ii = -0.5
        for i in range(self.n):
            F[i, i] = -0.5

        return F

    def resonance_wavelength(self, eps_func, wavelength_range=(300, 1000), n_points=100):
        """
        Find resonance wavelengths for a given material.

        Parameters
        ----------
        eps_func : callable
            Material dielectric function eps(wavelength)
        wavelength_range : tuple
            (min, max) wavelength in nm
        n_points : int
            Number of points to sample

        Returns
        -------
        resonances : list of dict
            List of resonance information for each mode
        """
        if self.eigenvalues is None:
            self.compute()

        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

        resonances = []

        for i, (lam, eps_res) in enumerate(zip(self.eigenvalues, self.resonance_eps)):
            # Skip modes with unphysical eigenvalues
            if np.abs(lam) < 1e-6 or np.abs(lam) > 1e6:
                continue

            # Find wavelength where Re(eps) = Re(eps_res)
            eps_values = np.array([eps_func(wl) for wl in wavelengths])
            target_eps = np.real(eps_res)

            # Find zero crossings
            diff = np.real(eps_values) - target_eps
            crossings = np.where(np.diff(np.sign(diff)))[0]

            for cross in crossings:
                # Linear interpolation
                wl_res = wavelengths[cross] - diff[cross] * \
                         (wavelengths[cross+1] - wavelengths[cross]) / \
                         (diff[cross+1] - diff[cross])

                resonances.append({
                    'mode': i,
                    'eigenvalue': lam,
                    'resonance_eps': eps_res,
                    'wavelength': wl_res,
                    'mode_pattern': self.eigenmodes[:, i]
                })

        return resonances

    def mode_dipole(self, mode_index):
        """
        Compute dipole moment of a plasmon mode.

        Parameters
        ----------
        mode_index : int
            Mode index

        Returns
        -------
        dipole : ndarray
            Dipole moment vector (3,)
        """
        if self.eigenmodes is None:
            self.compute()

        sigma = self.eigenmodes[:, mode_index]

        # Dipole moment = sum of charge * position
        dipole = np.sum(sigma[:, np.newaxis] * self.area[:, np.newaxis] * self.pos, axis=0)

        return dipole

    def mode_quadrupole(self, mode_index):
        """
        Compute quadrupole moment tensor of a plasmon mode.

        Parameters
        ----------
        mode_index : int
            Mode index

        Returns
        -------
        Q : ndarray
            Quadrupole tensor (3, 3)
        """
        if self.eigenmodes is None:
            self.compute()

        sigma = self.eigenmodes[:, mode_index]

        Q = np.zeros((3, 3), dtype=complex)

        for i in range(self.n):
            r = self.pos[i]
            r2 = np.dot(r, r)
            for a in range(3):
                for b in range(3):
                    Q[a, b] += sigma[i] * self.area[i] * (3 * r[a] * r[b] - r2 * (a == b))

        return Q

    def is_bright(self, mode_index, threshold=0.1):
        """
        Check if mode is optically bright (dipole active).

        Parameters
        ----------
        mode_index : int
            Mode index
        threshold : float
            Relative dipole moment threshold

        Returns
        -------
        bool
            True if mode is bright
        """
        dipole = self.mode_dipole(mode_index)
        dipole_mag = np.linalg.norm(dipole)

        # Compare to total charge magnitude
        sigma = self.eigenmodes[:, mode_index]
        total_charge = np.sum(np.abs(sigma) * self.area)

        # Typical scale for dipole
        size = np.max(self.pos, axis=0) - np.min(self.pos, axis=0)
        char_length = np.mean(size)

        relative_dipole = dipole_mag / (total_charge * char_length)

        return relative_dipole > threshold

    def classify_modes(self):
        """
        Classify plasmon modes by their character.

        Returns
        -------
        classification : list of dict
            Mode classification information
        """
        if self.eigenmodes is None:
            self.compute()

        classification = []

        for i in range(len(self.eigenvalues)):
            dipole = self.mode_dipole(i)
            dipole_mag = np.linalg.norm(dipole)

            quadrupole = self.mode_quadrupole(i)
            quad_mag = np.linalg.norm(quadrupole)

            is_bright = self.is_bright(i)

            # Determine dominant polarization direction for bright modes
            if is_bright and dipole_mag > 0:
                pol_dir = dipole / dipole_mag
            else:
                pol_dir = None

            classification.append({
                'mode': i,
                'eigenvalue': self.eigenvalues[i],
                'type': 'bright' if is_bright else 'dark',
                'dipole_moment': dipole_mag,
                'quadrupole_moment': quad_mag,
                'polarization': pol_dir
            })

        return classification

    def visualize_mode(self, mode_index, ax=None):
        """
        Visualize a plasmon mode on the particle surface.

        Parameters
        ----------
        mode_index : int
            Mode index
        ax : matplotlib Axes, optional
            Axes to plot on

        Returns
        -------
        ax : matplotlib Axes
            Plot axes
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import cm
        from matplotlib.colors import Normalize

        if self.eigenmodes is None:
            self.compute()

        sigma = np.real(self.eigenmodes[:, mode_index])

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Create polygons from faces
        faces = self.pc.faces
        verts = self.pc.verts

        polygons = []
        for face in faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            polygon = verts[indices]
            polygons.append(polygon)

        # Normalize colors
        norm = Normalize(vmin=-np.abs(sigma).max(), vmax=np.abs(sigma).max())
        colors = cm.RdBu_r(norm(sigma))

        collection = Poly3DCollection(
            polygons,
            facecolors=colors,
            edgecolors='black',
            linewidths=0.2,
            alpha=0.9
        )
        ax.add_collection3d(collection)

        # Set axis limits
        all_pts = verts
        max_range = np.max(np.max(all_pts, axis=0) - np.min(all_pts, axis=0))
        center = np.mean(all_pts, axis=0)

        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')

        lam = self.eigenvalues[mode_index]
        ax.set_title(f'Mode {mode_index}: λ = {lam:.3f}')

        return ax


def plasmonmode(particle, n_modes=None, options=None):
    """
    Factory function for plasmon mode calculation.

    Parameters
    ----------
    particle : ComParticle
        Composite particle
    n_modes : int, optional
        Number of modes to compute
    options : BEMOptions, optional
        Options

    Returns
    -------
    PlasmonMode
        Plasmon mode calculator with computed modes
    """
    pm = PlasmonMode(particle, options)
    pm.compute(n_modes)
    return pm
