"""
Triangulated sphere generation.

Uses pre-computed energy-minimized point distributions from MATLAB MNPBEM,
originally from http://www.maths.unsw.edu.au/school/articles/me100.html
"""

import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial import ConvexHull

from ..particle import Particle


# Path to pre-computed sphere data
_DATA_FILE = Path(__file__).parent / 'trisphere_data.npz'

# Available sphere point counts (matching MATLAB MNPBEM)
SPHERE_POINTS = [32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484,
                 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444]

# Cache for loaded sphere data
_sphere_cache = {}


def _load_sphere_data(n: int) -> np.ndarray:
    """
    Load pre-computed sphere vertices from data file.

    Parameters
    ----------
    n : int
        Number of points (must be in SPHERE_POINTS).

    Returns
    -------
    verts : ndarray
        Vertices on unit sphere, shape (n, 3).
    """
    global _sphere_cache

    key = f'sphere{n}'

    if key not in _sphere_cache:
        if not _DATA_FILE.exists():
            raise FileNotFoundError(
                f"Sphere data file not found: {_DATA_FILE}\n"
                "Run the data extraction script to generate this file."
            )

        data = np.load(_DATA_FILE)
        if key not in data:
            raise ValueError(f"Sphere data for n={n} not found in data file")

        _sphere_cache[key] = data[key]

    return _sphere_cache[key].copy()


def sphtriangulate(verts: np.ndarray) -> np.ndarray:
    """
    Triangulate points on a sphere surface using convex hull.

    This matches MATLAB's sphtriangulate function which uses
    Delaunay triangulation on sphere surface via convex hull.

    Parameters
    ----------
    verts : ndarray
        Vertices on sphere surface, shape (n, 3).

    Returns
    -------
    faces : ndarray
        Triangle face indices, shape (n_faces, 3).
    """
    hull = ConvexHull(verts)
    return hull.simplices


def trisphere(
    n: int = 144,
    diameter: float = 1.0,
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create triangulated sphere using pre-computed energy-minimized points.

    Loads pre-computed point distributions that minimize potential energy
    on the sphere surface, matching MATLAB MNPBEM's trisphere function.

    Parameters
    ----------
    n : int
        Number of vertices. Available: 32, 60, 144, 169, 225, 256,
        289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900,
        961, 1024, 1225, 1444. Other values will use closest available.
    diameter : float
        Diameter of sphere.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Triangulated sphere.

    Examples
    --------
    >>> sphere = trisphere(144, 10)  # 144 vertices, 10 nm diameter
    >>> sphere.n_verts
    144

    Notes
    -----
    The point distributions are from MATLAB MNPBEM, which uses
    energy-minimized configurations from:
    http://www.maths.unsw.edu.au/school/articles/me100.html
    """
    # Find closest available n
    available = np.array(SPHERE_POINTS)
    idx = np.argmin(np.abs(available - n))
    n_actual = available[idx]

    if n != n_actual:
        print(f"trisphere: loading sphere{n_actual} from trisphere_data.npz")

    # Load pre-computed vertices from MATLAB data
    verts = _load_sphere_data(n_actual)

    # Triangulate using convex hull (same as MATLAB sphtriangulate)
    faces = sphtriangulate(verts)

    # Scale to desired diameter (verts are on unit sphere)
    verts = verts * (diameter / 2)

    # Create particle with norm='off' equivalent
    p = Particle(verts, faces, interp='flat', compute_normals=False)

    # Add midpoints for curved interpolation (matching MATLAB: midpoints(p, 'flat'))
    p = p.midpoints('flat')

    # Rescale midpoint vertices to sphere surface
    # MATLAB: verts2 = 0.5 * diameter * bsxfun(@rdivide, p.verts2, sqrt(dot(p.verts2, p.verts2, 2)))
    if p.verts2 is not None:
        norms = np.sqrt(np.sum(p.verts2 * p.verts2, axis=1, keepdims=True))
        norms = np.where(norms > 0, norms, 1)
        p.verts2 = (diameter / 2) * p.verts2 / norms

    # Create final particle with curved boundary data
    # MATLAB: p = particle(verts2, p.faces2, varargin{:})
    if p.verts2 is not None:
        p_final = Particle(p.verts2, p.faces2, interp=interp, **kwargs)
    else:
        p_final = Particle(verts, faces, interp=interp, **kwargs)

    return p_final


def trispherescale(
    n: int = 144,
    diameter: float = 1.0,
    scale: tuple = (1.0, 1.0, 1.0),
    interp: str = 'curv',
    **kwargs
) -> Particle:
    """
    Create triangulated sphere with non-uniform scaling (spheroid/ellipsoid).

    Parameters
    ----------
    n : int
        Approximate number of vertices.
    diameter : float
        Base diameter of sphere before scaling.
    scale : tuple
        (sx, sy, sz) scale factors along each axis.
    interp : str
        Interpolation type: 'flat' or 'curv'.
    **kwargs : dict
        Additional arguments passed to Particle.

    Returns
    -------
    Particle
        Scaled sphere particle.

    Examples
    --------
    >>> # Create oblate spheroid (flattened along z)
    >>> p = trispherescale(144, 10, scale=(1, 1, 0.5))
    >>> # Create prolate spheroid (elongated along z)
    >>> p = trispherescale(144, 10, scale=(1, 1, 2.0))
    """
    # Find closest available n
    available = np.array(SPHERE_POINTS)
    idx = np.argmin(np.abs(available - n))
    n_actual = available[idx]

    if n != n_actual:
        print(f"trispherescale: loading sphere{n_actual} from trisphere_data.npz")

    # Load pre-computed vertices
    verts = _load_sphere_data(n_actual)

    # Triangulate
    faces = sphtriangulate(verts)

    # Scale to diameter and apply non-uniform scaling
    scale = np.asarray(scale)
    verts = verts * (diameter / 2) * scale

    # Create particle
    p = Particle(verts, faces, interp='flat', compute_normals=False)

    # Add midpoints
    p = p.midpoints('flat')

    # Rescale midpoint vertices to scaled ellipsoid surface
    if p.verts2 is not None:
        # Normalize by scale to get unit sphere position, then project and rescale
        verts2_normalized = p.verts2 / scale
        norms = np.sqrt(np.sum(verts2_normalized * verts2_normalized, axis=1, keepdims=True))
        norms = np.where(norms > 0, norms, 1)
        p.verts2 = (diameter / 2) * verts2_normalized / norms * scale

    # Create final particle
    if p.verts2 is not None:
        p_final = Particle(p.verts2, p.faces2, interp=interp, **kwargs)
    else:
        p_final = Particle(verts, faces, interp=interp, **kwargs)

    return p_final
