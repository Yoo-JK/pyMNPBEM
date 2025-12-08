"""
Adaptive Cross Approximation (ACA) for matrix compression.

ACA is used to compress dense matrices arising in BEM calculations
by approximating subblocks with low-rank matrices: A ≈ U @ V.T

This enables O(N log N) complexity instead of O(N^2) for matrix-vector
products and storage.
"""

import numpy as np
from typing import Optional, Tuple, Callable


class ACAMatrix:
    """
    Low-rank matrix representation from ACA decomposition.

    Stores the matrix as A ≈ U @ V.T where U is (m, k) and V is (n, k)
    with k << min(m, n).

    Parameters
    ----------
    U : ndarray
        Left factor (m, k)
    V : ndarray
        Right factor (n, k)

    Attributes
    ----------
    shape : tuple
        Matrix shape (m, n)
    rank : int
        Approximation rank k
    """

    def __init__(self, U, V):
        """Initialize low-rank matrix."""
        self.U = np.asarray(U)
        self.V = np.asarray(V)
        self.shape = (self.U.shape[0], self.V.shape[0])
        self.rank = self.U.shape[1] if self.U.ndim > 1 else 1

    def __matmul__(self, x):
        """Matrix-vector product: (U @ V.T) @ x = U @ (V.T @ x)."""
        x = np.asarray(x)
        if x.ndim == 1:
            return self.U @ (self.V.T @ x)
        else:
            return self.U @ (self.V.T @ x)

    def matvec(self, x):
        """Matrix-vector product."""
        return self @ x

    def rmatvec(self, x):
        """Transpose matrix-vector product: (U @ V.T).T @ x = V @ (U.T @ x)."""
        x = np.asarray(x)
        return self.V @ (self.U.T @ x)

    def todense(self):
        """Convert to dense matrix."""
        return self.U @ self.V.T

    def __repr__(self):
        return f"ACAMatrix(shape={self.shape}, rank={self.rank})"


def aca(matrix_func, m, n, eps=1e-6, max_rank=None, max_iter=None):
    """
    Adaptive Cross Approximation algorithm.

    Computes a low-rank approximation A ≈ U @ V.T by adaptively
    selecting rows and columns.

    Parameters
    ----------
    matrix_func : callable
        Function to compute matrix elements: matrix_func(i, j) -> A[i, j]
        Can also accept arrays: matrix_func(I, J) -> A[I, :][:, J]
    m : int
        Number of rows
    n : int
        Number of columns
    eps : float
        Relative accuracy tolerance
    max_rank : int, optional
        Maximum rank (default: min(m, n) // 2)
    max_iter : int, optional
        Maximum iterations

    Returns
    -------
    U : ndarray
        Left factor (m, k)
    V : ndarray
        Right factor (n, k)
    info : dict
        Information about convergence
    """
    if max_rank is None:
        max_rank = min(m, n) // 2
    if max_iter is None:
        max_iter = max_rank + 10

    # Initialize
    U_list = []
    V_list = []

    # Track which rows/columns have been used
    used_rows = set()
    used_cols = set()

    # Estimate of Frobenius norm for convergence
    norm_est = 0.0

    # Initial pivot selection (random or first)
    pivot_row = 0

    for k in range(max_iter):
        # Get row pivot_row of current residual
        # R[pivot_row, :] = A[pivot_row, :] - sum_i U[pivot_row, i] * V[:, i]
        row = get_row(matrix_func, pivot_row, n)
        for i, (u, v) in enumerate(zip(U_list, V_list)):
            row = row - u[pivot_row] * v

        # Find pivot column (max element)
        pivot_col = np.argmax(np.abs(row))

        # Check if row is essentially zero
        row_max = np.abs(row[pivot_col])
        if row_max < eps * max(1.0, np.sqrt(norm_est)):
            # Try another row
            candidates = [i for i in range(m) if i not in used_rows]
            if not candidates:
                break
            pivot_row = candidates[k % len(candidates)]
            continue

        # Normalize to get v
        v = row / row[pivot_col]

        # Get column pivot_col of current residual
        col = get_col(matrix_func, pivot_col, m)
        for i, (u, vv) in enumerate(zip(U_list, V_list)):
            col = col - u * vv[pivot_col]

        # This gives u directly (already normalized by pivot)
        u = col

        # Store
        U_list.append(u)
        V_list.append(v)
        used_rows.add(pivot_row)
        used_cols.add(pivot_col)

        # Update norm estimate
        uv_norm = np.linalg.norm(u) * np.linalg.norm(v)
        norm_est += uv_norm**2

        # Check convergence
        if uv_norm < eps * np.sqrt(norm_est):
            break

        # Check rank limit
        if len(U_list) >= max_rank:
            break

        # Select next pivot row (maximum residual)
        # Simple strategy: find row with max u component
        pivot_row = np.argmax(np.abs(u))
        if pivot_row in used_rows:
            # Find next best
            u_sorted = np.argsort(np.abs(u))[::-1]
            for idx in u_sorted:
                if idx not in used_rows:
                    pivot_row = idx
                    break

    # Assemble matrices
    if U_list:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
    else:
        U = np.zeros((m, 0))
        V = np.zeros((n, 0))

    info = {
        'rank': len(U_list),
        'iterations': k + 1,
        'converged': len(U_list) < max_rank,
        'norm_estimate': np.sqrt(norm_est)
    }

    return U, V, info


def get_row(matrix_func, i, n):
    """Get row i of matrix."""
    try:
        # Try to get entire row at once
        return matrix_func(i, slice(None))
    except:
        # Fall back to element-wise
        return np.array([matrix_func(i, j) for j in range(n)])


def get_col(matrix_func, j, m):
    """Get column j of matrix."""
    try:
        # Try to get entire column at once
        return matrix_func(slice(None), j)
    except:
        # Fall back to element-wise
        return np.array([matrix_func(i, j) for i in range(m)])


def aca_full(A, eps=1e-6, max_rank=None):
    """
    ACA on a fully available matrix.

    Useful for testing and when matrix is already computed.

    Parameters
    ----------
    A : ndarray
        Dense matrix (m, n)
    eps : float
        Tolerance
    max_rank : int, optional
        Maximum rank

    Returns
    -------
    aca_mat : ACAMatrix
        Low-rank approximation
    """
    m, n = A.shape

    def matrix_func(i, j):
        return A[i, j]

    U, V, info = aca(matrix_func, m, n, eps, max_rank)
    return ACAMatrix(U, V)


def aca_partial(matrix_func, row_indices, col_indices, eps=1e-6, max_rank=None):
    """
    ACA for a submatrix.

    Parameters
    ----------
    matrix_func : callable
        Function to compute full matrix elements
    row_indices : array_like
        Row indices of submatrix
    col_indices : array_like
        Column indices of submatrix
    eps : float
        Tolerance
    max_rank : int, optional
        Maximum rank

    Returns
    -------
    aca_mat : ACAMatrix
        Low-rank approximation of submatrix
    """
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)

    m = len(row_indices)
    n = len(col_indices)

    def sub_matrix_func(i, j):
        if isinstance(i, slice):
            i = row_indices
        else:
            i = row_indices[i]
        if isinstance(j, slice):
            j = col_indices
        else:
            j = col_indices[j]
        return matrix_func(i, j)

    U, V, info = aca(sub_matrix_func, m, n, eps, max_rank)
    return ACAMatrix(U, V)


class ACAGreen:
    """
    Green function matrix with ACA compression.

    Automatically compresses far-field interactions while
    keeping near-field interactions exact.

    Parameters
    ----------
    particle : Particle or ComParticle
        Particle geometry
    near_field_distance : float
        Distance threshold for near-field (exact) treatment
    eps : float
        ACA tolerance
    """

    def __init__(self, particle, near_field_distance=None, eps=1e-6):
        """Initialize ACA-compressed Green function."""
        self.particle = particle

        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

        self.pos = self.pc.pos
        self.n = len(self.pos)

        if near_field_distance is None:
            # Estimate from particle size
            diameter = np.max(self.pos, axis=0) - np.min(self.pos, axis=0)
            near_field_distance = 2 * np.mean(diameter)

        self.near_field_distance = near_field_distance
        self.eps = eps

        # Build cluster tree and identify near/far interactions
        self._build_structure()

    def _build_structure(self):
        """Build hierarchical structure for ACA."""
        # Simple implementation: identify near and far pairs

        # Compute pairwise distances
        pos = self.pos
        n = self.n

        # Near-field mask
        self.near_mask = np.zeros((n, n), dtype=bool)

        for i in range(n):
            dist = np.linalg.norm(pos - pos[i], axis=1)
            self.near_mask[i, :] = dist < self.near_field_distance

        # Far-field pairs (complementary)
        self.far_mask = ~self.near_mask

        # Count
        self.n_near = np.sum(self.near_mask)
        self.n_far = np.sum(self.far_mask)

    def compute(self, wavelength, green_type='stat'):
        """
        Compute compressed Green function matrix.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        green_type : str
            'stat' for quasistatic, 'ret' for retarded

        Returns
        -------
        G_compressed : CompressedGreenMatrix
            Compressed Green function
        """
        n = self.n
        pos = self.pos

        if green_type == 'stat':
            # Quasistatic: G = 1 / (4 * pi * r)
            def green_func(i, j):
                if np.isscalar(i) and np.isscalar(j):
                    if i == j:
                        # Self-term approximation
                        area = self.pc.area[i]
                        a_eff = np.sqrt(area / np.pi)
                        return 1.5 / (4 * np.pi * a_eff)
                    r = np.linalg.norm(pos[i] - pos[j])
                    return 1.0 / (4 * np.pi * r)
                else:
                    # Array case - simplified
                    raise NotImplementedError("Array indexing in green_func")
        else:
            # Retarded: G = exp(ikr) / (4 * pi * r)
            k = 2 * np.pi / wavelength

            def green_func(i, j):
                if np.isscalar(i) and np.isscalar(j):
                    if i == j:
                        area = self.pc.area[i]
                        a_eff = np.sqrt(area / np.pi)
                        return np.exp(1j * k * a_eff) * 1.5 / (4 * np.pi * a_eff)
                    r = np.linalg.norm(pos[i] - pos[j])
                    return np.exp(1j * k * r) / (4 * np.pi * r)
                else:
                    raise NotImplementedError("Array indexing in green_func")

        # Compute near-field exactly
        G_near = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                if self.near_mask[i, j]:
                    G_near[i, j] = green_func(i, j)

        # Compress far-field with ACA
        # For simplicity, we'll compute full far-field and then compress
        # In production, ACA would be applied block-wise
        if self.n_far > 0:
            G_far_full = np.zeros((n, n), dtype=complex)
            for i in range(n):
                for j in range(n):
                    if self.far_mask[i, j]:
                        G_far_full[i, j] = green_func(i, j)

            # Compress
            G_far_aca = aca_full(G_far_full, self.eps)
        else:
            G_far_aca = None

        return CompressedGreenMatrix(G_near, G_far_aca, self.near_mask)


class CompressedGreenMatrix:
    """
    Compressed Green function matrix.

    Stores near-field exactly and far-field as low-rank.

    Parameters
    ----------
    G_near : ndarray
        Near-field interactions (sparse/masked)
    G_far : ACAMatrix or None
        Far-field interactions (low-rank)
    near_mask : ndarray
        Boolean mask for near-field
    """

    def __init__(self, G_near, G_far, near_mask):
        """Initialize compressed matrix."""
        self.G_near = G_near
        self.G_far = G_far
        self.near_mask = near_mask
        self.shape = G_near.shape

    def __matmul__(self, x):
        """Matrix-vector product."""
        result = self.G_near @ x
        if self.G_far is not None:
            result += self.G_far @ x
        return result

    def matvec(self, x):
        """Matrix-vector product."""
        return self @ x

    def todense(self):
        """Convert to dense matrix."""
        result = self.G_near.copy()
        if self.G_far is not None:
            result += self.G_far.todense() * (~self.near_mask)
        return result

    @property
    def compression_ratio(self):
        """Compute compression ratio."""
        n = self.shape[0]
        full_storage = n * n

        near_storage = np.sum(self.near_mask)
        if self.G_far is not None:
            far_storage = self.G_far.rank * 2 * n
        else:
            far_storage = 0

        compressed_storage = near_storage + far_storage

        return full_storage / max(compressed_storage, 1)
