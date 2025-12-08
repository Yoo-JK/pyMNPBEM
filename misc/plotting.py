"""
Visualization utilities for MNPBEM.

Provides functions for plotting particles, fields, and spectra.
"""

import numpy as np
from typing import Optional, Tuple, Union


def plot_particle(
    p,
    ax=None,
    field: np.ndarray = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    **kwargs
):
    """
    Plot a particle with optional field coloring.

    Parameters
    ----------
    p : Particle or ComParticle
        Particle to plot.
    ax : matplotlib axes, optional
        3D axes. Creates new figure if None.
    field : ndarray, optional
        Scalar field to color faces.
    cmap : str
        Colormap name.
    colorbar : bool
        Show colorbar for field.
    **kwargs : dict
        Additional plotting arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Get particle object
    if hasattr(p, 'pc'):
        particle = p.pc
    else:
        particle = p

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Default colors
    alpha = kwargs.pop('alpha', 0.8)
    edgecolor = kwargs.pop('edgecolor', 'black')
    linewidth = kwargs.pop('linewidth', 0.3)

    # Create polygon collection
    polygons = []
    for face in particle.faces:
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)
        vertices = particle.verts[indices]
        polygons.append(vertices)

    if field is not None:
        # Color by field
        from matplotlib import cm
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=field.min(), vmax=field.max())
        colormap = cm.get_cmap(cmap)
        facecolors = colormap(norm(field))

        collection = Poly3DCollection(
            polygons,
            alpha=alpha,
            facecolors=facecolors,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs
        )

        ax.add_collection3d(collection)

        if colorbar:
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.6)
    else:
        color = kwargs.pop('color', 'gold')
        collection = Poly3DCollection(
            polygons,
            alpha=alpha,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs
        )
        ax.add_collection3d(collection)

    # Set axis limits
    all_verts = particle.verts
    max_range = np.max([
        all_verts[:, 0].max() - all_verts[:, 0].min(),
        all_verts[:, 1].max() - all_verts[:, 1].min(),
        all_verts[:, 2].max() - all_verts[:, 2].min()
    ]) / 2

    mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
    mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
    mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')

    return ax


def plot_spectrum(
    wavelengths: np.ndarray,
    sca: np.ndarray = None,
    ext: np.ndarray = None,
    abs_cs: np.ndarray = None,
    ax=None,
    xlabel: str = 'Wavelength (nm)',
    ylabel: str = 'Cross section (nm²)',
    legend: bool = True,
    **kwargs
):
    """
    Plot optical spectra.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    sca : ndarray, optional
        Scattering cross section.
    ext : ndarray, optional
        Extinction cross section.
    abs_cs : ndarray, optional
        Absorption cross section.
    ax : matplotlib axes, optional
        Axes to plot on.
    xlabel, ylabel : str
        Axis labels.
    legend : bool
        Show legend.
    **kwargs : dict
        Additional plot arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if sca is not None:
        ax.plot(wavelengths, sca, label='Scattering', **kwargs)

    if ext is not None:
        ax.plot(wavelengths, ext, label='Extinction', **kwargs)

    if abs_cs is not None:
        ax.plot(wavelengths, abs_cs, label='Absorption', **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend and (sca is not None or ext is not None or abs_cs is not None):
        ax.legend()

    ax.grid(True, alpha=0.3)

    return ax


def plot_field_slice(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ax=None,
    cmap: str = 'RdBu',
    colorbar: bool = True,
    symmetric: bool = True,
    **kwargs
):
    """
    Plot a 2D field slice.

    Parameters
    ----------
    field : ndarray
        Field values, shape (len(y), len(x)).
    x, y : ndarray
        Coordinate arrays.
    ax : matplotlib axes, optional
        Axes to plot on.
    cmap : str
        Colormap name.
    colorbar : bool
        Show colorbar.
    symmetric : bool
        Use symmetric color limits.
    **kwargs : dict
        Additional arguments for pcolormesh.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if symmetric:
        vmax = np.abs(field).max()
        vmin = -vmax
    else:
        vmin, vmax = field.min(), field.max()

    im = ax.pcolormesh(x, y, field, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if colorbar:
        plt.colorbar(im, ax=ax)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_aspect('equal')

    return ax


def arrow_plot(
    pos: np.ndarray,
    vec: np.ndarray,
    ax=None,
    scale: float = 1.0,
    color: str = 'blue',
    **kwargs
):
    """
    Plot vector field as arrows.

    Parameters
    ----------
    pos : ndarray
        Arrow positions, shape (n, 3).
    vec : ndarray
        Arrow directions, shape (n, 3).
    ax : matplotlib axes, optional
        3D axes.
    scale : float
        Arrow scale factor.
    color : str
        Arrow color.
    **kwargs : dict
        Additional quiver arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2],
        vec[:, 0] * scale, vec[:, 1] * scale, vec[:, 2] * scale,
        color=color,
        **kwargs
    )

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')

    return ax


def plot_eels_map(
    loss_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ax=None,
    cmap: str = 'hot',
    particle=None,
    **kwargs
):
    """
    Plot EELS loss probability map.

    Parameters
    ----------
    loss_map : ndarray
        Loss probability map, shape (len(y), len(x)).
    x, y : ndarray
        Coordinate arrays.
    ax : matplotlib axes, optional
        Axes to plot on.
    cmap : str
        Colormap name.
    particle : Particle, optional
        Particle to overlay.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.pcolormesh(x, y, loss_map, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax, label='Loss probability')

    # Overlay particle outline if provided
    if particle is not None:
        if hasattr(particle, 'pc'):
            particle = particle.pc

        # Project to xy plane
        for face in particle.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            vertices = particle.verts[indices]

            # Close polygon
            xy = np.vstack([vertices[:, :2], vertices[0, :2]])
            ax.plot(xy[:, 0], xy[:, 1], 'w-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_aspect('equal')

    return ax


def create_colormap(name: str = 'plasmonic'):
    """
    Create custom colormaps for plasmonic simulations.

    Parameters
    ----------
    name : str
        Colormap name: 'plasmonic', 'field', 'charge'.

    Returns
    -------
    colormap
        Matplotlib colormap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if name == 'plasmonic':
        colors = ['darkblue', 'blue', 'cyan', 'white', 'yellow', 'red', 'darkred']
    elif name == 'field':
        colors = ['blue', 'cyan', 'white', 'yellow', 'red']
    elif name == 'charge':
        colors = ['blue', 'white', 'red']
    else:
        return plt.cm.get_cmap(name)

    return LinearSegmentedColormap.from_list(name, colors)
