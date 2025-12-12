"""
Compound particle class with dielectric functions.
"""

import numpy as np
from typing import List, Optional, Union, Any

from .particle import Particle
from .compound import Compound
from ..misc.options import BEMOptions


class ComParticle(Compound):
    """
    Compound of particles in a dielectric environment.

    This is the main class for defining particle geometries with
    associated dielectric functions.

    Parameters
    ----------
    eps : list
        List of dielectric function objects (e.g., EpsConst, EpsDrude).
    p : list
        List of Particle objects.
    inout : ndarray
        Index array of shape (n_particles, 2) specifying
        [inside_medium_idx, outside_medium_idx] for each particle.
        Indices are 1-based (MATLAB convention).
    closed : int or list, optional
        Indices of particles that form closed surfaces.
    **kwargs : dict
        Additional options.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, ComParticle
    >>> from mnpbem.particles.shapes import trisphere
    >>>
    >>> # Define dielectric functions
    >>> eps_vacuum = EpsConst(1.0)
    >>> eps_gold = EpsTable('gold.dat')
    >>> epstab = [eps_vacuum, eps_gold]
    >>>
    >>> # Create gold sphere in vacuum
    >>> sphere = trisphere(144, 10)  # 144 points, 10 nm diameter
    >>> p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Particle],
        inout: np.ndarray,
        closed: Optional[Union[int, List[int]]] = None,
        **kwargs
    ):
        """
        Initialize compound particle.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        p : list
            List of Particle objects.
        inout : ndarray
            Medium indices for each particle.
        closed : int or list, optional
            Indices of closed surfaces (1-based).
        **kwargs : dict
            Additional options.
        """
        # Convert single particle to list
        if isinstance(p, Particle):
            p = [p]

        # Initialize base class
        super().__init__(eps, p, inout)

        # Handle closed surfaces
        if closed is None:
            self.closed = np.zeros(len(p), dtype=bool)
        elif isinstance(closed, (int, np.integer)):
            self.closed = np.zeros(len(p), dtype=bool)
            if closed > 0:
                self.closed[closed - 1] = True  # Convert to 0-based
        else:
            self.closed = np.zeros(len(p), dtype=bool)
            for idx in closed:
                if idx > 0:
                    self.closed[idx - 1] = True

        # Store options
        self.options = kwargs

    @property
    def pos(self) -> np.ndarray:
        """Face centroids of combined particle."""
        return self.pc.pos

    @property
    def nvec(self) -> np.ndarray:
        """Normal vectors of combined particle."""
        return self.pc.nvec

    @property
    def area(self) -> np.ndarray:
        """Face areas of combined particle."""
        return self.pc.area

    @property
    def n_faces(self) -> int:
        """Total number of faces."""
        return self.pc.n_faces

    @property
    def verts(self) -> np.ndarray:
        """Vertices of combined particle."""
        return self.pc.verts

    @property
    def faces(self) -> np.ndarray:
        """Faces of combined particle."""
        return self.pc.faces

    def closedparticle(self, particle_idx: int) -> bool:
        """
        Check if particle forms a closed surface.

        Parameters
        ----------
        particle_idx : int
            Index of particle (0-based).

        Returns
        -------
        bool
            True if particle is closed.
        """
        return self.closed[particle_idx]

    def lambda_factor(self, enei: float) -> np.ndarray:
        """
        Compute Lambda factor for BEM equations.

        Lambda = 2 * pi * (eps1 + eps2) / (eps1 - eps2)

        Following Garcia de Abajo, PRB 65, 115418 (2002), Eq. (23).

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Lambda factor for each face.
        """
        n_faces = self.n_faces
        Lambda = np.zeros(n_faces, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p):
            eps_in, eps_out = self.dielectric_inout(enei, i)
            n = particle.n_faces

            # Lambda = 2 * pi * (eps1 + eps2) / (eps1 - eps2)
            # Garcia de Abajo, Eq. (23)
            lam = 2 * np.pi * (eps_in + eps_out) / (eps_in - eps_out)
            Lambda[face_idx:face_idx + n] = lam

            face_idx += n

        return Lambda

    def delta_eps(self, enei: float) -> np.ndarray:
        """
        Compute dielectric function difference for each face.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            eps_in - eps_out for each face.
        """
        n_faces = self.n_faces
        deps = np.zeros(n_faces, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p):
            eps_in, eps_out = self.dielectric_inout(enei, i)
            n = particle.n_faces
            deps[face_idx:face_idx + n] = eps_in - eps_out
            face_idx += n

        return deps

    def shift(self, displacement: np.ndarray) -> 'ComParticle':
        """
        Shift all particles by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        ComParticle
            Shifted compound particle.
        """
        new_p = [p.shift(displacement) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def scale(self, factor: Union[float, np.ndarray]) -> 'ComParticle':
        """
        Scale all particles.

        Parameters
        ----------
        factor : float or array_like
            Scale factor.

        Returns
        -------
        ComParticle
            Scaled compound particle.
        """
        new_p = [p.scale(factor) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def rot(self, angle: float, axis: int = 2) -> 'ComParticle':
        """
        Rotate all particles.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : int
            Rotation axis.

        Returns
        -------
        ComParticle
            Rotated compound particle.
        """
        new_p = [p.rot(angle, axis) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def flip(self, axis: int) -> 'ComParticle':
        """
        Flip all particles along axis.

        Parameters
        ----------
        axis : int
            Axis to flip.

        Returns
        -------
        ComParticle
            Flipped compound particle.
        """
        new_p = [p.flip(axis) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def clean(self, tol: float = 1e-10) -> 'ComParticle':
        """
        Clean all particles (remove duplicate vertices).

        Parameters
        ----------
        tol : float
            Tolerance for duplicates.

        Returns
        -------
        ComParticle
            Cleaned compound particle.
        """
        new_p = [p.clean(tol) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def curvature(self) -> np.ndarray:
        """
        Compute curvature for all faces.

        Returns
        -------
        ndarray
            Curvature at each face.
        """
        return self.pc.curvature()

    def select(
        self,
        indices: np.ndarray = None,
        carfun: callable = None,
        polfun: callable = None,
        sphfun: callable = None,
        particle_idx: int = None
    ) -> 'ComParticle':
        """
        Select faces from compound particle.

        Parameters
        ----------
        indices : ndarray, optional
            Global indices of faces to select.
        carfun : callable, optional
            Function f(x, y, z) returning boolean for Cartesian selection.
        polfun : callable, optional
            Function f(phi, r, z) returning boolean for polar selection.
        sphfun : callable, optional
            Function f(phi, theta, r) returning boolean for spherical selection.
        particle_idx : int, optional
            Select all faces from specific particle (0-based index).

        Returns
        -------
        ComParticle
            Selected compound particle.
        """
        # Select specific particle
        if particle_idx is not None:
            return ComParticle(
                self.eps,
                [self.p[particle_idx]],
                self.inout[particle_idx:particle_idx+1],
                self.closed[particle_idx:particle_idx+1],
                **self.options
            )

        # Selection by indices
        if indices is not None:
            indices = np.atleast_1d(indices)
            cum_n = 0
            new_p = []
            new_inout = []
            new_closed = []

            for i, particle in enumerate(self.p):
                n = particle.n_faces
                local_mask = (indices >= cum_n) & (indices < cum_n + n)
                local_indices = indices[local_mask] - cum_n

                if len(local_indices) > 0:
                    new_p.append(particle.select(local_indices))
                    new_inout.append(self.inout[i])
                    new_closed.append(self.closed[i])

                cum_n += n

            if not new_p:
                return ComParticle(self.eps, [Particle()], np.array([[1, 1]]))

            return ComParticle(
                self.eps, new_p, np.array(new_inout), np.array(new_closed),
                **self.options
            )

        # Selection by coordinate function
        new_p = []
        new_inout = []
        new_closed = []

        for i, particle in enumerate(self.p):
            pos = particle.pos
            mask = np.ones(len(pos), dtype=bool)

            if carfun is not None:
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                mask &= carfun(x, y, z)

            if polfun is not None:
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                mask &= polfun(phi, r, z)

            if sphfun is not None:
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arccos(np.clip(z / (r + 1e-10), -1, 1))
                phi = np.arctan2(y, x)
                mask &= sphfun(phi, theta, r)

            selected_indices = np.where(mask)[0]
            if len(selected_indices) > 0:
                new_p.append(particle.select(selected_indices))
                new_inout.append(self.inout[i])
                new_closed.append(self.closed[i])

        if not new_p:
            return ComParticle(self.eps, [Particle()], np.array([[1, 1]]))

        return ComParticle(
            self.eps, new_p, np.array(new_inout), np.array(new_closed),
            **self.options
        )

    def plot(self, ax=None, values: np.ndarray = None, **kwargs):
        """
        Plot the compound particle surface.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes for plotting. If None, creates new figure.
        values : ndarray, optional
            Values to color the faces (e.g., surface charges).
        **kwargs : dict
            Additional arguments for plot (cmap, alpha, edgecolor, etc.).

        Returns
        -------
        ax : matplotlib axes
            The plot axes.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Get default plotting options
        cmap = kwargs.pop('cmap', 'viridis')
        alpha = kwargs.pop('alpha', 0.8)
        edgecolor = kwargs.pop('edgecolor', 'k')
        linewidth = kwargs.pop('linewidth', 0.1)

        # Get vertices and faces from combined particle
        verts = self.pc.verts
        faces = self.pc.faces

        # Create polygon collection
        polygons = []
        for face in faces:
            # Handle triangles and quads
            valid_idx = face[~np.isnan(face)].astype(int)
            polygon = verts[valid_idx]
            polygons.append(polygon)

        poly_collection = Poly3DCollection(
            polygons,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs
        )

        # Color by values if provided
        if values is not None:
            values = np.atleast_1d(values)
            if np.iscomplexobj(values):
                values = np.abs(values)
            poly_collection.set_array(values)
            poly_collection.set_cmap(cmap)
            plt.colorbar(poly_collection, ax=ax, shrink=0.6)
        else:
            poly_collection.set_facecolor('lightblue')

        ax.add_collection3d(poly_collection)

        # Set axis limits
        max_range = np.max(np.abs(verts)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')

        return ax

    def plot2(self, ax=None, values: np.ndarray = None, plane: str = 'xy',
              z_level: float = None, **kwargs):
        """
        Plot 2D cross-section of compound particle.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes for plotting.
        values : ndarray, optional
            Values for coloring.
        plane : str
            Cross-section plane: 'xy', 'xz', or 'yz'.
        z_level : float, optional
            Level for cross-section.
        **kwargs : dict
            Additional plot arguments.

        Returns
        -------
        ax : matplotlib axes
            The plot axes.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Get position data
        pos = self.pc.pos

        # Determine axis mapping
        if plane == 'xy':
            x_idx, y_idx = 0, 1
            xlabel, ylabel = 'X (nm)', 'Y (nm)'
        elif plane == 'xz':
            x_idx, y_idx = 0, 2
            xlabel, ylabel = 'X (nm)', 'Z (nm)'
        else:  # yz
            x_idx, y_idx = 1, 2
            xlabel, ylabel = 'Y (nm)', 'Z (nm)'

        # Plot face centroids
        x = pos[:, x_idx]
        y = pos[:, y_idx]

        cmap = kwargs.pop('cmap', 'viridis')
        s = kwargs.pop('s', 20)

        if values is not None:
            values = np.atleast_1d(values)
            if np.iscomplexobj(values):
                values = np.abs(values)
            scatter = ax.scatter(x, y, c=values, cmap=cmap, s=s, **kwargs)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(x, y, s=s, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self) -> str:
        return (f"ComParticle(n_eps={len(self.eps)}, "
                f"n_particles={len(self.p)}, "
                f"n_faces={self.n_faces})")
