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


def coneplot(
    pos: np.ndarray,
    vec: np.ndarray,
    ax=None,
    scale: float = 1.0,
    color: str = None,
    cmap: str = 'jet',
    magnitude: np.ndarray = None,
    normalize: bool = True,
    **kwargs
):
    """
    Plot 3D vector field using cones.

    Each vector is represented as a cone oriented along the vector
    direction with size proportional to magnitude.

    Parameters
    ----------
    pos : ndarray
        Cone positions, shape (n, 3).
    vec : ndarray
        Vector directions, shape (n, 3).
    ax : matplotlib axes, optional
        3D axes. Creates new if None.
    scale : float
        Overall scale factor for cone sizes.
    color : str, optional
        Fixed color for all cones. If None, colors by magnitude.
    cmap : str
        Colormap for magnitude coloring.
    magnitude : ndarray, optional
        Magnitude values for coloring. If None, uses vector norms.
    normalize : bool
        Normalize vectors before scaling.
    **kwargs : dict
        Additional plotting arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.

    Examples
    --------
    >>> pos = np.random.randn(100, 3)
    >>> vec = np.random.randn(100, 3)
    >>> ax = coneplot(pos, vec, scale=0.1)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    pos = np.atleast_2d(pos)
    vec = np.atleast_2d(vec)

    # Compute magnitudes
    mags = np.linalg.norm(vec, axis=1)

    if normalize:
        vec_norm = vec / (mags[:, np.newaxis] + 1e-10)
        lengths = scale * np.ones_like(mags)
    else:
        vec_norm = vec / (mags[:, np.newaxis] + 1e-10)
        lengths = scale * mags / (mags.max() + 1e-10)

    # Colors
    if color is not None:
        colors = [color] * len(pos)
    else:
        from matplotlib import cm
        from matplotlib.colors import Normalize

        if magnitude is not None:
            mag_vals = magnitude
        else:
            mag_vals = mags

        norm = Normalize(vmin=mag_vals.min(), vmax=mag_vals.max())
        colormap = cm.get_cmap(cmap)
        colors = colormap(norm(mag_vals))

    # Plot using quiver (cone-like appearance)
    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2],
        vec_norm[:, 0] * lengths,
        vec_norm[:, 1] * lengths,
        vec_norm[:, 2] * lengths,
        colors=colors,
        arrow_length_ratio=0.3,
        **kwargs
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def coneplot2(
    pos: np.ndarray,
    vec: np.ndarray,
    ax=None,
    scale: float = 1.0,
    color_by: str = 'magnitude',
    cmap: str = 'jet',
    cone_resolution: int = 10,
    **kwargs
):
    """
    Plot 3D vector field using detailed cone geometry.

    Creates actual cone meshes for higher quality visualization.

    Parameters
    ----------
    pos : ndarray
        Cone positions, shape (n, 3).
    vec : ndarray
        Vector directions, shape (n, 3).
    ax : matplotlib axes, optional
        3D axes.
    scale : float
        Scale factor for cones.
    color_by : str
        Color mode: 'magnitude', 'x', 'y', 'z', or 'component'.
    cmap : str
        Colormap name.
    cone_resolution : int
        Number of facets in cone base.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    pos = np.atleast_2d(pos)
    vec = np.atleast_2d(vec)

    mags = np.linalg.norm(vec, axis=1)

    # Determine colors
    if color_by == 'magnitude':
        color_vals = mags
    elif color_by == 'x':
        color_vals = vec[:, 0]
    elif color_by == 'y':
        color_vals = vec[:, 1]
    elif color_by == 'z':
        color_vals = vec[:, 2]
    else:
        color_vals = mags

    norm = Normalize(vmin=color_vals.min(), vmax=color_vals.max())
    colormap = cm.get_cmap(cmap)
    colors = colormap(norm(color_vals))

    # Draw each cone
    for i in range(len(pos)):
        if mags[i] < 1e-10:
            continue

        # Cone geometry
        height = scale * mags[i] / (mags.max() + 1e-10)
        radius = height * 0.3

        # Direction
        direction = vec[i] / mags[i]

        # Base circle
        theta = np.linspace(0, 2 * np.pi, cone_resolution + 1)[:-1]

        # Find perpendicular vectors
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Base vertices
        base_points = []
        for t in theta:
            point = pos[i] + radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
            base_points.append(point)

        # Tip
        tip = pos[i] + height * direction

        # Create cone faces
        faces = []
        for j in range(len(base_points)):
            face = [base_points[j], base_points[(j + 1) % len(base_points)], tip]
            faces.append(face)

        collection = Poly3DCollection(
            faces,
            alpha=0.8,
            facecolor=colors[i],
            edgecolor='none'
        )
        ax.add_collection3d(collection)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label=color_by)

    return ax


def patchcurvature(
    particle,
    curvature_type: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute surface curvature of particle mesh.

    Parameters
    ----------
    particle : Particle
        Particle object with verts and faces.
    curvature_type : str
        Type of curvature: 'mean', 'gaussian', 'principal'.

    Returns
    -------
    curvature : ndarray
        Curvature values at each face centroid.
    principal : ndarray (only if curvature_type='principal')
        Principal curvatures (k1, k2) at each face.

    Examples
    --------
    >>> sphere = trisphere(144, 10)
    >>> H, _ = patchcurvature(sphere, 'mean')
    >>> # For sphere of radius R=5, mean curvature H = 1/R = 0.2
    """
    if hasattr(particle, 'pc'):
        particle = particle.pc

    verts = particle.verts
    faces = particle.faces
    n_faces = len(faces)

    # Compute normals if not available
    if hasattr(particle, 'nvec'):
        normals = particle.nvec
    else:
        normals = np.zeros((n_faces, 3))
        for i, face in enumerate(faces):
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            if len(indices) >= 3:
                v0 = verts[indices[0]]
                v1 = verts[indices[1]]
                v2 = verts[indices[2]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                normals[i] = n / norm if norm > 0 else [0, 0, 1]

    # Build vertex-to-face adjacency
    vert_faces = [[] for _ in range(len(verts))]
    for i, face in enumerate(faces):
        valid = ~np.isnan(face)
        for idx in face[valid].astype(int):
            vert_faces[idx].append(i)

    # Compute curvature tensor at each vertex
    vertex_curvature = np.zeros((len(verts), 2))  # k1, k2

    for v_idx in range(len(verts)):
        adj_faces = vert_faces[v_idx]
        if len(adj_faces) < 3:
            continue

        vertex = verts[v_idx]

        # Average normal at vertex
        avg_normal = np.mean(normals[adj_faces], axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm < 1e-10:
            continue
        avg_normal = avg_normal / norm

        # Collect neighbor vertices
        neighbors = set()
        for f_idx in adj_faces:
            face = faces[f_idx]
            valid = ~np.isnan(face)
            for idx in face[valid].astype(int):
                if idx != v_idx:
                    neighbors.add(idx)

        neighbors = list(neighbors)
        if len(neighbors) < 3:
            continue

        # Estimate curvature using shape operator
        # Build local coordinate system
        if abs(avg_normal[2]) < 0.9:
            u_axis = np.cross(avg_normal, [0, 0, 1])
        else:
            u_axis = np.cross(avg_normal, [1, 0, 0])
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(avg_normal, u_axis)

        # Fit quadric surface
        A = []
        b = []
        for n_idx in neighbors:
            delta = verts[n_idx] - vertex
            u = np.dot(delta, u_axis)
            v = np.dot(delta, v_axis)
            w = np.dot(delta, avg_normal)

            A.append([u * u, u * v, v * v])
            b.append(w)

        A = np.array(A)
        b = np.array(b)

        if len(A) >= 3:
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                a, c, e = coeffs[0], coeffs[1], coeffs[2]

                # Principal curvatures from quadric coefficients
                H = a + e  # Mean curvature (approx)
                K = 4 * a * e - c * c  # Gaussian curvature (approx)

                disc = H * H - K
                if disc >= 0:
                    k1 = H + np.sqrt(disc)
                    k2 = H - np.sqrt(disc)
                else:
                    k1 = k2 = H

                vertex_curvature[v_idx] = [k1, k2]
            except Exception:
                pass

    # Transfer to faces
    face_curvature = np.zeros((n_faces, 2))
    for i, face in enumerate(faces):
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)
        face_curvature[i] = np.mean(vertex_curvature[indices], axis=0)

    k1, k2 = face_curvature[:, 0], face_curvature[:, 1]

    if curvature_type == 'mean':
        return (k1 + k2) / 2, face_curvature
    elif curvature_type == 'gaussian':
        return k1 * k2, face_curvature
    elif curvature_type == 'principal':
        return face_curvature, face_curvature
    else:
        return (k1 + k2) / 2, face_curvature


def plot_curvature(
    particle,
    curvature_type: str = 'mean',
    ax=None,
    cmap: str = 'coolwarm',
    **kwargs
):
    """
    Plot particle colored by surface curvature.

    Parameters
    ----------
    particle : Particle
        Particle to plot.
    curvature_type : str
        Type: 'mean', 'gaussian', 'principal'.
    ax : matplotlib axes, optional
        3D axes.
    cmap : str
        Colormap.
    **kwargs : dict
        Additional arguments for plot_particle.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    curvature, _ = patchcurvature(particle, curvature_type)
    return plot_particle(particle, ax=ax, field=curvature, cmap=cmap, **kwargs)


def streamplot3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    ax=None,
    density: float = 1.0,
    color: str = 'blue',
    linewidth: float = 1.0,
    **kwargs
):
    """
    Plot 3D streamlines of vector field.

    Parameters
    ----------
    X, Y, Z : ndarray
        3D coordinate grids.
    U, V, W : ndarray
        Vector field components.
    ax : matplotlib axes, optional
        3D axes.
    density : float
        Streamline density.
    color : str
        Line color.
    linewidth : float
        Line width.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    ax : matplotlib axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Sample starting points
    n_lines = int(density * 10)

    x_range = (X.min(), X.max())
    y_range = (Y.min(), Y.max())
    z_range = (Z.min(), Z.max())

    # Random starting points
    np.random.seed(42)
    starts = np.random.uniform(
        low=[x_range[0], y_range[0], z_range[0]],
        high=[x_range[1], y_range[1], z_range[1]],
        size=(n_lines, 3)
    )

    # Integrate streamlines
    from scipy.interpolate import RegularGridInterpolator

    x_1d = X[0, :, 0] if X.ndim == 3 else X[0, :]
    y_1d = Y[:, 0, 0] if Y.ndim == 3 else Y[:, 0]
    z_1d = Z[0, 0, :] if Z.ndim == 3 else np.array([0])

    if Z.ndim < 3:
        # 2D case, extend to 3D
        return ax

    interp_u = RegularGridInterpolator((y_1d, x_1d, z_1d), U, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((y_1d, x_1d, z_1d), V, bounds_error=False, fill_value=0)
    interp_w = RegularGridInterpolator((y_1d, x_1d, z_1d), W, bounds_error=False, fill_value=0)

    dt = 0.1
    n_steps = 100

    for start in starts:
        line = [start.copy()]
        pos = start.copy()

        for _ in range(n_steps):
            try:
                vel = np.array([
                    interp_u((pos[1], pos[0], pos[2])),
                    interp_v((pos[1], pos[0], pos[2])),
                    interp_w((pos[1], pos[0], pos[2]))
                ])

                if np.linalg.norm(vel) < 1e-10:
                    break

                pos = pos + dt * vel
                line.append(pos.copy())

                if (pos[0] < x_range[0] or pos[0] > x_range[1] or
                    pos[1] < y_range[0] or pos[1] > y_range[1] or
                    pos[2] < z_range[0] or pos[2] > z_range[1]):
                    break
            except Exception:
                break

        if len(line) > 1:
            line = np.array(line)
            ax.plot(line[:, 0], line[:, 1], line[:, 2],
                   color=color, linewidth=linewidth, **kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax
