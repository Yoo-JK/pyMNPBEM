"""
Helper functions for vector and matrix operations.
"""

import numpy as np
from scipy import sparse
from typing import Union


def inner(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute inner product of two arrays along specified axis.

    Parameters
    ----------
    a : ndarray
        First array.
    b : ndarray
        Second array.
    axis : int, optional
        Axis along which to compute inner product. Default is -1.

    Returns
    -------
    ndarray
        Inner product result.
    """
    return np.sum(a * b, axis=axis)


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute outer product of two arrays.

    Parameters
    ----------
    a : ndarray
        First array of shape (..., n).
    b : ndarray
        Second array of shape (..., m).

    Returns
    -------
    ndarray
        Outer product of shape (..., n, m).
    """
    return np.einsum('...i,...j->...ij', a, b)


def matcross(a: np.ndarray) -> np.ndarray:
    """
    Create cross product matrix from vector.

    For a vector [ax, ay, az], returns the matrix M such that
    M @ v = a x v (cross product).

    Parameters
    ----------
    a : ndarray
        Vector(s) of shape (..., 3).

    Returns
    -------
    ndarray
        Cross product matrix of shape (..., 3, 3).
    """
    if a.shape[-1] != 3:
        raise ValueError("Input must have last dimension of size 3")

    shape = a.shape[:-1] + (3, 3)
    result = np.zeros(shape, dtype=a.dtype)

    result[..., 0, 1] = -a[..., 2]
    result[..., 0, 2] = a[..., 1]
    result[..., 1, 0] = a[..., 2]
    result[..., 1, 2] = -a[..., 0]
    result[..., 2, 0] = -a[..., 1]
    result[..., 2, 1] = a[..., 0]

    return result


def matmul(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication with broadcasting.

    Parameters
    ----------
    mat : ndarray
        Matrix of shape (..., m, n) or sparse matrix.
    vec : ndarray
        Vector of shape (..., n).

    Returns
    -------
    ndarray
        Result of shape (..., m).
    """
    if sparse.issparse(mat):
        return mat @ vec
    return np.einsum('...ij,...j->...i', mat, vec)


def spdiag(diag: np.ndarray) -> sparse.csr_matrix:
    """
    Create sparse diagonal matrix.

    Parameters
    ----------
    diag : ndarray
        Diagonal elements.

    Returns
    -------
    sparse.csr_matrix
        Sparse diagonal matrix.
    """
    n = len(diag)
    return sparse.diags(diag, 0, shape=(n, n), format='csr')


def vecnorm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute vector norm along specified axis.

    Parameters
    ----------
    v : ndarray
        Input vectors.
    axis : int, optional
        Axis along which to compute norm. Default is -1.

    Returns
    -------
    ndarray
        Vector norms.
    """
    return np.sqrt(np.sum(v * v, axis=axis))


def vecnormalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize vectors along specified axis.

    Parameters
    ----------
    v : ndarray
        Input vectors.
    axis : int, optional
        Axis along which to normalize. Default is -1.

    Returns
    -------
    ndarray
        Normalized vectors.
    """
    norms = vecnorm(v, axis=axis)
    # Keep dimensions for broadcasting
    norms = np.expand_dims(norms, axis=axis)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return v / norms


def bdist2(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared distances between two sets of points.

    Parameters
    ----------
    pos1 : ndarray
        First set of points, shape (n1, d).
    pos2 : ndarray
        Second set of points, shape (n2, d).

    Returns
    -------
    ndarray
        Squared distances, shape (n1, n2).
    """
    # Use broadcasting: (n1, 1, d) - (1, n2, d) -> (n1, n2, d)
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)


def pdist2(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between two sets of points.

    Parameters
    ----------
    pos1 : ndarray
        First set of points, shape (n1, d).
    pos2 : ndarray
        Second set of points, shape (n2, d).

    Returns
    -------
    ndarray
        Distances, shape (n1, n2).
    """
    return np.sqrt(bdist2(pos1, pos2))


def bradius(p) -> np.ndarray:
    """
    Compute minimal radius for spheres enclosing boundary elements.

    The radius is the maximum distance from each face centroid to any
    of its vertices.

    Parameters
    ----------
    p : particle object
        Discretized particle boundary with pos, verts, and faces attributes.

    Returns
    -------
    ndarray
        Minimal radius for spheres enclosing each boundary element, shape (n,).
    """
    n = p.n
    r = np.zeros(n)

    pos = p.pos  # centroids, shape (n, 3)
    verts = p.verts  # vertices
    faces = p.faces  # face vertex indices

    # Process each vertex of the faces
    num_verts_per_face = faces.shape[1] if len(faces.shape) > 1 else 3

    for i in range(num_verts_per_face):
        # Get vertex indices for this corner
        if len(faces.shape) > 1:
            vert_idx = faces[:, i]
        else:
            vert_idx = faces

        # Handle NaN values (for triangles in a mixed tri/quad mesh)
        valid = ~np.isnan(vert_idx) if np.issubdtype(vert_idx.dtype, np.floating) else np.ones(n, dtype=bool)
        valid_idx = vert_idx[valid].astype(int)

        if len(valid_idx) > 0:
            # Calculate distance from centroid to this vertex
            dist = np.sqrt(np.sum((pos[valid] - verts[valid_idx]) ** 2, axis=1))
            r[valid] = np.maximum(r[valid], dist)

    return r


def refinematrix(p1, p2, abs_cutoff: float = 0, rel_cutoff: float = 0,
                 memsize: int = int(2e7)) -> sparse.csr_matrix:
    """
    Compute refinement matrix for Green functions.

    Creates a sparse matrix indicating which elements need refinement:
    - 2 for diagonal elements (same position)
    - 1 for off-diagonal elements within cutoff distance
    - 0 for elements that don't need refinement

    Parameters
    ----------
    p1 : particle object
        First discretized particle boundary with pos attribute.
    p2 : particle object
        Second discretized particle boundary with pos, verts, faces attributes.
    abs_cutoff : float, optional
        Absolute distance cutoff for integration refinement. Default 0.
    rel_cutoff : float, optional
        Relative distance cutoff (in units of boundary element radius). Default 0.
    memsize : int, optional
        Maximum memory size for processing. Default 2e7.

    Returns
    -------
    sparse.csr_matrix
        Refinement matrix of shape (n1, n2).
    """
    pos1 = p1.pos
    pos2 = p2.pos
    n1 = len(pos1)
    n2 = len(pos2)

    # Boundary element radii for p2
    rad2 = bradius(p2)

    # Try to use boundary radius of p1, fall back to rad2
    try:
        rad1 = bradius(p1)
    except:
        rad1 = rad2

    # Initialize sparse matrix data
    rows = []
    cols = []
    data = []

    # Work through matrix in portions to manage memory
    chunk_size = max(1, int(memsize / n1))
    ind2 = list(range(0, n2, chunk_size))
    if ind2[-1] != n2:
        ind2.append(n2)

    for i in range(len(ind2) - 1):
        i2_start = ind2[i]
        i2_end = ind2[i + 1]
        i2_slice = slice(i2_start, i2_end)

        # Distance between positions
        d = pdist2(pos1, pos2[i2_slice])

        # Subtract radius to get approximate distance to boundary elements
        d2 = d - rad2[i2_slice]

        # Distances in units of boundary element radius
        if len(rad1) != 1:
            id_dist = d2 / rad1[:, np.newaxis]
        else:
            id_dist = d2 / rad2[i2_slice]

        # Find diagonal elements (d == 0)
        diag_row, diag_col = np.where(d == 0)
        for r, c in zip(diag_row, diag_col):
            rows.append(r)
            cols.append(c + i2_start)
            data.append(2)

        # Find off-diagonal elements for refinement
        refine_mask = ((d2 < abs_cutoff) | (id_dist < rel_cutoff)) & (d != 0)
        refine_row, refine_col = np.where(refine_mask)
        for r, c in zip(refine_row, refine_col):
            rows.append(r)
            cols.append(c + i2_start)
            data.append(1)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n1, n2))


def refinematrixlayer(p1, p2, layer, abs_cutoff: float = 0, rel_cutoff: float = 0,
                      memsize: int = int(2e7)) -> sparse.csr_matrix:
    """
    Compute refinement matrix for Green functions with layer structures.

    For layer structures, the distance is computed using the radial distance
    in the x-y plane and the minimum distance to the layer in z.

    Parameters
    ----------
    p1 : particle object
        First discretized particle boundary with pos attribute.
    p2 : particle object
        Second discretized particle boundary with pos, verts, faces attributes.
    layer : layer structure
        Layer structure with mindist() method.
    abs_cutoff : float, optional
        Absolute distance cutoff for integration refinement. Default 0.
    rel_cutoff : float, optional
        Relative distance cutoff (in units of boundary element radius). Default 0.
    memsize : int, optional
        Maximum memory size for processing. Default 2e7.

    Returns
    -------
    sparse.csr_matrix
        Refinement matrix of shape (n1, n2).
    """
    pos1 = p1.pos
    pos2 = p2.pos
    n1 = len(pos1)
    n2 = len(pos2)

    # Boundary element radii for p2
    rad2 = bradius(p2)

    # Try to use boundary radius of p1, fall back to rad2
    try:
        rad1 = bradius(p1)
    except:
        rad1 = rad2

    # Initialize sparse matrix data
    rows = []
    cols = []
    data = []

    # Work through matrix in portions to manage memory
    chunk_size = max(1, int(memsize / n1))
    ind2 = list(range(0, n2, chunk_size))
    if ind2[-1] != n2:
        ind2.append(n2)

    for i in range(len(ind2) - 1):
        i2_start = ind2[i]
        i2_end = ind2[i + 1]
        i2_slice = slice(i2_start, i2_end)

        # Radial distance in x-y plane
        r = pdist2(pos1[:, :2], pos2[i2_slice, :2])

        # Minimum distance to layer in z
        z1 = layer.mindist(pos1[:, 2])
        z2 = layer.mindist(pos2[i2_slice, 2])
        z = z1[:, np.newaxis] + z2[np.newaxis, :]

        # Total distance
        d = np.sqrt(r ** 2 + z ** 2)

        # Subtract radius to get approximate distance to boundary elements
        d2 = d - rad2[i2_slice]

        # Distances in units of boundary element radius
        if len(rad1) != 1:
            id_dist = d2 / rad1[:, np.newaxis]
        else:
            id_dist = d2 / rad2[i2_slice]

        # Find elements for refinement
        refine_mask = (d2 < abs_cutoff) | (id_dist < rel_cutoff)
        refine_row, refine_col = np.where(refine_mask)
        for r, c in zip(refine_row, refine_col):
            rows.append(r)
            cols.append(c + i2_start)
            data.append(1)

        # Check for diagonal boundary elements (same positions)
        if np.array_equal(pos1, pos2):
            # Add extra weight for diagonal elements
            diag_mask = refine_mask & (np.arange(n1)[:, np.newaxis] == np.arange(i2_start, i2_end))
            diag_row, diag_col = np.where(diag_mask)
            for r, c in zip(diag_row, diag_col):
                rows.append(r)
                cols.append(c + i2_start)
                data.append(1)  # Additional 1 to make it 2 total

    return sparse.csr_matrix((data, (rows, cols)), shape=(n1, n2))
