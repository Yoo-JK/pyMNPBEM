"""
Hierarchical matrices (H-matrices) for BEM acceleration.

H-matrices provide a hierarchical data structure for efficient
representation and manipulation of dense matrices arising from
integral equations. Combined with ACA, they enable O(N log N)
complexity for matrix operations.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .aca import ACAMatrix, aca


class ClusterTree:
    """
    Binary cluster tree for hierarchical partitioning.

    Recursively partitions a set of points into clusters
    for H-matrix construction.

    Parameters
    ----------
    points : ndarray
        Point coordinates (n, 3)
    indices : array_like, optional
        Point indices (default: 0 to n-1)
    leaf_size : int
        Maximum number of points in a leaf cluster
    """

    def __init__(self, points, indices=None, leaf_size=32):
        """Initialize cluster tree."""
        self.points = np.asarray(points)

        if indices is None:
            indices = np.arange(len(points))
        self.indices = np.asarray(indices)

        self.leaf_size = leaf_size
        self.n_points = len(self.indices)

        # Compute bounding box
        pts = self.points[self.indices]
        self.bbox_min = np.min(pts, axis=0)
        self.bbox_max = np.max(pts, axis=0)
        self.center = 0.5 * (self.bbox_min + self.bbox_max)
        self.diameter = np.linalg.norm(self.bbox_max - self.bbox_min)

        # Build tree
        self.is_leaf = len(self.indices) <= leaf_size
        self.children = []

        if not self.is_leaf:
            self._split()

    def _split(self):
        """Split cluster into two children."""
        pts = self.points[self.indices]

        # Split along longest dimension
        dims = self.bbox_max - self.bbox_min
        split_dim = np.argmax(dims)

        # Find median
        coords = pts[:, split_dim]
        median = np.median(coords)

        # Partition
        mask = coords <= median
        left_idx = self.indices[mask]
        right_idx = self.indices[~mask]

        # Handle edge case where all points are on one side
        if len(left_idx) == 0:
            left_idx = self.indices[:len(self.indices)//2]
            right_idx = self.indices[len(self.indices)//2:]
        elif len(right_idx) == 0:
            left_idx = self.indices[:len(self.indices)//2]
            right_idx = self.indices[len(self.indices)//2:]

        # Create children
        if len(left_idx) > 0:
            self.children.append(
                ClusterTree(self.points, left_idx, self.leaf_size)
            )
        if len(right_idx) > 0:
            self.children.append(
                ClusterTree(self.points, right_idx, self.leaf_size)
            )

    def depth(self):
        """Return tree depth."""
        if self.is_leaf:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def n_leaves(self):
        """Return number of leaf clusters."""
        if self.is_leaf:
            return 1
        return sum(child.n_leaves() for child in self.children)

    def leaves(self):
        """Generator for leaf clusters."""
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child.leaves()

    def all_clusters(self):
        """Generator for all clusters."""
        yield self
        for child in self.children:
            yield from child.all_clusters()


def admissibility(cluster1, cluster2, eta=2.0):
    """
    Check admissibility condition for two clusters.

    Two clusters are admissible (far-field) if:
    min(diam1, diam2) <= eta * dist(cluster1, cluster2)

    Parameters
    ----------
    cluster1 : ClusterTree
        First cluster
    cluster2 : ClusterTree
        Second cluster
    eta : float
        Admissibility parameter

    Returns
    -------
    bool
        True if clusters are admissible (can use low-rank)
    """
    dist = np.linalg.norm(cluster1.center - cluster2.center)
    min_diam = min(cluster1.diameter, cluster2.diameter)

    return min_diam <= eta * dist


class HMatrix:
    """
    Hierarchical matrix representation.

    Stores a matrix hierarchically with:
    - Dense blocks for near-field (inadmissible) interactions
    - Low-rank blocks for far-field (admissible) interactions

    Parameters
    ----------
    row_tree : ClusterTree
        Cluster tree for rows
    col_tree : ClusterTree
        Cluster tree for columns
    matrix_func : callable
        Function to compute matrix elements
    eta : float
        Admissibility parameter
    eps : float
        ACA tolerance
    """

    def __init__(self, row_tree, col_tree, matrix_func, eta=2.0, eps=1e-6):
        """Initialize H-matrix."""
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.matrix_func = matrix_func
        self.eta = eta
        self.eps = eps

        self.shape = (row_tree.n_points, col_tree.n_points)

        # Build block structure
        self.blocks = []
        self._build_blocks(row_tree, col_tree)

    def _build_blocks(self, row_cluster, col_cluster):
        """Recursively build H-matrix blocks."""
        # Check if both are leaves
        if row_cluster.is_leaf and col_cluster.is_leaf:
            # Create dense block
            block = HMatrixBlock(
                row_cluster.indices,
                col_cluster.indices,
                self.matrix_func,
                block_type='dense'
            )
            self.blocks.append(block)
            return

        # Check admissibility
        if admissibility(row_cluster, col_cluster, self.eta):
            # Create low-rank block via ACA
            block = HMatrixBlock(
                row_cluster.indices,
                col_cluster.indices,
                self.matrix_func,
                block_type='lowrank',
                eps=self.eps
            )
            self.blocks.append(block)
            return

        # Recursively subdivide
        row_children = row_cluster.children if row_cluster.children else [row_cluster]
        col_children = col_cluster.children if col_cluster.children else [col_cluster]

        for rc in row_children:
            for cc in col_children:
                self._build_blocks(rc, cc)

    def __matmul__(self, x):
        """Matrix-vector product."""
        x = np.asarray(x)
        result = np.zeros(self.shape[0], dtype=complex if np.iscomplexobj(x) else float)

        for block in self.blocks:
            block.matvec_add(x, result)

        return result

    def matvec(self, x):
        """Matrix-vector product."""
        return self @ x

    def todense(self):
        """Convert to dense matrix."""
        A = np.zeros(self.shape, dtype=complex)

        for block in self.blocks:
            row_idx = block.row_indices
            col_idx = block.col_indices
            A[np.ix_(row_idx, col_idx)] = block.todense()

        return A

    @property
    def n_blocks(self):
        """Number of blocks."""
        return len(self.blocks)

    @property
    def compression_ratio(self):
        """Estimate compression ratio."""
        full = self.shape[0] * self.shape[1]
        compressed = sum(block.storage for block in self.blocks)
        return full / max(compressed, 1)

    def memory_usage(self):
        """Estimate memory usage in bytes."""
        return sum(block.storage for block in self.blocks) * 16  # complex128

    def lu(self) -> Tuple['HMatrixLU', dict]:
        """
        Compute LU decomposition of H-matrix.

        This provides an approximate LU factorization for use as a
        preconditioner or direct solver. For H-matrices, the LU factors
        are also H-matrices.

        Returns
        -------
        lu : HMatrixLU
            LU decomposition object with solve method.
        info : dict
            Decomposition information.

        Notes
        -----
        For H-matrices, exact LU is expensive. This implementation
        converts to dense and uses standard LU, which loses the H-matrix
        structure but provides exact factorization. For large problems,
        consider using iterative solvers instead.
        """
        from scipy.linalg import lu_factor

        # Convert to dense and compute LU
        A_dense = self.todense()
        lu_piv = lu_factor(A_dense)

        lu_decomp = HMatrixLU(lu_piv, self.shape)

        info = {
            'method': 'dense_lu',
            'shape': self.shape,
            'compression_ratio': self.compression_ratio
        }

        return lu_decomp, info

    def inv(self) -> np.ndarray:
        """
        Compute inverse of H-matrix.

        Returns the inverse as a dense matrix. For H-matrices,
        maintaining structure in the inverse is complex, so this
        returns a dense result.

        Returns
        -------
        A_inv : ndarray
            Inverse matrix (dense).

        Warnings
        --------
        For large matrices, this is expensive (O(n^3)).
        Consider using solve() with iterative methods instead.
        """
        A_dense = self.todense()
        return np.linalg.inv(A_dense)

    def solve(self, b: np.ndarray, method: str = 'gmres',
              tol: float = 1e-8, maxiter: int = None) -> Tuple[np.ndarray, dict]:
        """
        Solve H-matrix linear system H @ x = b.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector or matrix.
        method : str
            Solver method:
            - 'gmres': GMRES iterative solver (default)
            - 'lu': Direct LU solver (converts to dense)
            - 'cg': Conjugate gradient (for SPD matrices)
        tol : float
            Solver tolerance for iterative methods.
        maxiter : int, optional
            Maximum iterations for iterative methods.

        Returns
        -------
        x : ndarray
            Solution vector.
        info : dict
            Solver information including convergence status.

        Examples
        --------
        >>> H = HMatrix(...)
        >>> x, info = H.solve(b, method='gmres')
        >>> print(f"Converged: {info['converged']}")
        """
        from scipy.sparse.linalg import gmres, cg, LinearOperator

        b = np.asarray(b)
        n = self.shape[0]

        if maxiter is None:
            maxiter = min(n, 500)

        if method == 'lu':
            # Direct solver via LU
            lu_decomp, _ = self.lu()
            x = lu_decomp.solve(b)
            info = {'method': 'lu', 'converged': True}
            return x, info

        # Iterative methods
        op = LinearOperator(self.shape, matvec=self.matvec, dtype=complex)

        if method == 'cg':
            # Conjugate gradient (assumes SPD)
            x, flag = cg(op, b, tol=tol, maxiter=maxiter)
        else:
            # Default: GMRES
            x, flag = gmres(op, b, tol=tol, maxiter=maxiter)

        info = {
            'method': method,
            'converged': flag == 0,
            'flag': flag
        }

        return x, info

    def truncate(self, eps: float = None) -> 'HMatrix':
        """
        Truncate low-rank blocks to reduce rank.

        Recompresses low-rank blocks with a new tolerance.

        Parameters
        ----------
        eps : float, optional
            New truncation tolerance. If None, uses original eps.

        Returns
        -------
        H_truncated : HMatrix
            Truncated H-matrix (modified in place and returned).
        """
        if eps is None:
            eps = self.eps

        for block in self.blocks:
            if block.block_type == 'lowrank' and block.rank > 0:
                # SVD truncation
                U, s, Vt = np.linalg.svd(block.U @ block.V.T, full_matrices=False)

                # Find truncation rank
                total = np.sum(s)
                cumsum = np.cumsum(s)
                keep = np.searchsorted(cumsum, (1 - eps) * total) + 1
                keep = max(1, min(keep, len(s)))

                # Truncate
                block.U = U[:, :keep] * s[:keep]
                block.V = Vt[:keep, :].T
                block.rank = keep
                block.storage = keep * (block.shape[0] + block.shape[1])

        return self


class HMatrixLU:
    """
    LU decomposition of H-matrix.

    Stores LU factors for solving linear systems.
    """

    def __init__(self, lu_piv, shape):
        """
        Initialize from LU factorization.

        Parameters
        ----------
        lu_piv : tuple
            LU factorization from scipy.linalg.lu_factor.
        shape : tuple
            Matrix shape.
        """
        self.lu_piv = lu_piv
        self.shape = shape

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system using LU factors.

        Parameters
        ----------
        b : ndarray
            Right-hand side.

        Returns
        -------
        x : ndarray
            Solution.
        """
        from scipy.linalg import lu_solve
        return lu_solve(self.lu_piv, b)


class HMatrixBlock:
    """
    Single block in H-matrix.

    Can be either dense or low-rank.

    Parameters
    ----------
    row_indices : array_like
        Row indices for this block
    col_indices : array_like
        Column indices for this block
    matrix_func : callable
        Function to compute matrix elements
    block_type : str
        'dense' or 'lowrank'
    eps : float
        ACA tolerance (for lowrank blocks)
    """

    def __init__(self, row_indices, col_indices, matrix_func,
                 block_type='dense', eps=1e-6):
        """Initialize block."""
        self.row_indices = np.asarray(row_indices)
        self.col_indices = np.asarray(col_indices)
        self.block_type = block_type

        m = len(row_indices)
        n = len(col_indices)
        self.shape = (m, n)

        if block_type == 'dense':
            # Compute dense block
            self.data = self._compute_dense(matrix_func)
            self.storage = m * n
        else:
            # Compute low-rank approximation
            self.U, self.V = self._compute_lowrank(matrix_func, eps)
            self.rank = self.U.shape[1] if self.U.ndim > 1 else 0
            self.storage = self.rank * (m + n)

    def _compute_dense(self, matrix_func):
        """Compute dense block."""
        m, n = self.shape
        A = np.zeros((m, n), dtype=complex)

        for i, ri in enumerate(self.row_indices):
            for j, cj in enumerate(self.col_indices):
                A[i, j] = matrix_func(ri, cj)

        return A

    def _compute_lowrank(self, matrix_func, eps):
        """Compute low-rank approximation via ACA."""
        m, n = self.shape

        # Define local matrix function
        def local_func(i, j):
            ri = self.row_indices[i]
            cj = self.col_indices[j]
            return matrix_func(ri, cj)

        U, V, info = aca(local_func, m, n, eps)
        return U, V

    def matvec_add(self, x, result):
        """Add block contribution to result: result += A @ x."""
        x_local = x[self.col_indices]

        if self.block_type == 'dense':
            y_local = self.data @ x_local
        else:
            # Low-rank: U @ (V.T @ x)
            y_local = self.U @ (self.V.T @ x_local)

        np.add.at(result, self.row_indices, y_local)

    def todense(self):
        """Convert block to dense."""
        if self.block_type == 'dense':
            return self.data
        else:
            return self.U @ self.V.T


class HMatrixGreen:
    """
    H-matrix representation of Green function.

    Parameters
    ----------
    particle : Particle or ComParticle
        Particle geometry
    leaf_size : int
        Leaf cluster size
    eta : float
        Admissibility parameter
    eps : float
        ACA tolerance
    """

    def __init__(self, particle, leaf_size=32, eta=2.0, eps=1e-6):
        """Initialize H-matrix Green function."""
        self.particle = particle

        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

        self.pos = self.pc.pos
        self.area = self.pc.area
        self.n = len(self.pos)

        self.leaf_size = leaf_size
        self.eta = eta
        self.eps = eps

        # Build cluster tree
        self.tree = ClusterTree(self.pos, leaf_size=leaf_size)

    def compute(self, wavelength, green_type='stat'):
        """
        Compute H-matrix Green function.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        green_type : str
            'stat' or 'ret'

        Returns
        -------
        H : HMatrix
            H-matrix representation
        """
        pos = self.pos
        area = self.area

        if green_type == 'stat':
            def green_func(i, j):
                if i == j:
                    a_eff = np.sqrt(area[i] / np.pi)
                    return 1.5 / (4 * np.pi * a_eff)
                r = np.linalg.norm(pos[i] - pos[j])
                return 1.0 / (4 * np.pi * r)
        else:
            k = 2 * np.pi / wavelength

            def green_func(i, j):
                if i == j:
                    a_eff = np.sqrt(area[i] / np.pi)
                    return np.exp(1j * k * a_eff) * 1.5 / (4 * np.pi * a_eff)
                r = np.linalg.norm(pos[i] - pos[j])
                return np.exp(1j * k * r) / (4 * np.pi * r)

        return HMatrix(
            self.tree, self.tree, green_func,
            eta=self.eta, eps=self.eps
        )


def hmatrix_solve(H, b, tol=1e-8, maxiter=None):
    """
    Solve H-matrix system H @ x = b using iterative method.

    Uses GMRES with H-matrix as operator.

    Parameters
    ----------
    H : HMatrix
        H-matrix
    b : ndarray
        Right-hand side
    tol : float
        Solver tolerance
    maxiter : int, optional
        Maximum iterations

    Returns
    -------
    x : ndarray
        Solution
    info : dict
        Solver information
    """
    from scipy.sparse.linalg import gmres, LinearOperator

    n = H.shape[0]
    if maxiter is None:
        maxiter = n

    # Create linear operator
    op = LinearOperator(H.shape, matvec=H.matvec)

    # Solve
    x, flag = gmres(op, b, tol=tol, maxiter=maxiter)

    info = {
        'converged': flag == 0,
        'flag': flag
    }

    return x, info
