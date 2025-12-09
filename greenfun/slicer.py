"""
Matrix slicer utility for efficient Green's function computation.

Provides tools for slicing and manipulating large matrices in
block-wise fashion for memory-efficient BEM computations.
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Iterator
from scipy import sparse


class Slicer:
    """
    Matrix slicer for block-wise operations.

    Enables efficient slicing and iteration over large matrices
    in blocks for memory-efficient Green's function computation.

    Parameters
    ----------
    shape : tuple
        Full matrix shape (m, n).
    block_size : int or tuple
        Block size for slicing. If int, same for rows and cols.
    mode : str
        Slicing mode: 'rows', 'cols', 'blocks'.

    Attributes
    ----------
    shape : tuple
        Full matrix shape.
    block_shape : tuple
        Block size (row_block, col_block).
    n_row_blocks : int
        Number of row blocks.
    n_col_blocks : int
        Number of column blocks.

    Examples
    --------
    >>> slicer = Slicer((1000, 1000), block_size=100)
    >>> for row_slice, col_slice in slicer.iter_blocks():
    ...     block = matrix[row_slice, col_slice]
    ...     # Process block
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        block_size: Union[int, Tuple[int, int]] = 100,
        mode: str = 'blocks'
    ):
        """Initialize slicer."""
        self.shape = shape

        if isinstance(block_size, int):
            self.block_shape = (block_size, block_size)
        else:
            self.block_shape = block_size

        self.mode = mode

        # Compute number of blocks
        self.n_row_blocks = (shape[0] + self.block_shape[0] - 1) // self.block_shape[0]
        self.n_col_blocks = (shape[1] + self.block_shape[1] - 1) // self.block_shape[1]

    def row_slice(self, i: int) -> slice:
        """
        Get slice for row block i.

        Parameters
        ----------
        i : int
            Row block index.

        Returns
        -------
        slice
            Slice object for rows.
        """
        start = i * self.block_shape[0]
        end = min((i + 1) * self.block_shape[0], self.shape[0])
        return slice(start, end)

    def col_slice(self, j: int) -> slice:
        """
        Get slice for column block j.

        Parameters
        ----------
        j : int
            Column block index.

        Returns
        -------
        slice
            Slice object for columns.
        """
        start = j * self.block_shape[1]
        end = min((j + 1) * self.block_shape[1], self.shape[1])
        return slice(start, end)

    def block_slice(self, i: int, j: int) -> Tuple[slice, slice]:
        """
        Get slices for block (i, j).

        Parameters
        ----------
        i : int
            Row block index.
        j : int
            Column block index.

        Returns
        -------
        tuple
            (row_slice, col_slice).
        """
        return self.row_slice(i), self.col_slice(j)

    def iter_rows(self) -> Iterator[slice]:
        """Iterate over row slices."""
        for i in range(self.n_row_blocks):
            yield self.row_slice(i)

    def iter_cols(self) -> Iterator[slice]:
        """Iterate over column slices."""
        for j in range(self.n_col_blocks):
            yield self.col_slice(j)

    def iter_blocks(self) -> Iterator[Tuple[slice, slice]]:
        """
        Iterate over all block slices.

        Yields
        ------
        tuple
            (row_slice, col_slice) for each block.
        """
        for i in range(self.n_row_blocks):
            for j in range(self.n_col_blocks):
                yield self.block_slice(i, j)

    def iter_diagonal_blocks(self) -> Iterator[Tuple[slice, slice]]:
        """
        Iterate over diagonal blocks only.

        Yields
        ------
        tuple
            (row_slice, col_slice) for diagonal blocks.
        """
        n = min(self.n_row_blocks, self.n_col_blocks)
        for i in range(n):
            yield self.block_slice(i, i)

    def iter_offdiagonal_blocks(self) -> Iterator[Tuple[slice, slice, int, int]]:
        """
        Iterate over off-diagonal blocks.

        Yields
        ------
        tuple
            (row_slice, col_slice, i, j) for off-diagonal blocks.
        """
        for i in range(self.n_row_blocks):
            for j in range(self.n_col_blocks):
                if i != j:
                    yield (*self.block_slice(i, j), i, j)

    def block_indices(self) -> np.ndarray:
        """
        Get array of block indices.

        Returns
        -------
        ndarray
            Array of shape (n_blocks, 2) with (i, j) block indices.
        """
        indices = []
        for i in range(self.n_row_blocks):
            for j in range(self.n_col_blocks):
                indices.append([i, j])
        return np.array(indices)

    def __len__(self) -> int:
        """Total number of blocks."""
        return self.n_row_blocks * self.n_col_blocks

    def __repr__(self) -> str:
        return (f"Slicer(shape={self.shape}, block_shape={self.block_shape}, "
                f"n_blocks={len(self)})")


def slicer(
    shape: Tuple[int, int],
    block_size: Union[int, Tuple[int, int]] = 100
) -> Slicer:
    """
    Create a matrix slicer.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    block_size : int or tuple
        Block size for slicing.

    Returns
    -------
    Slicer
        Slicer object.
    """
    return Slicer(shape, block_size)


def slice_matrix(
    matrix: np.ndarray,
    row_slice: slice,
    col_slice: Optional[slice] = None
) -> np.ndarray:
    """
    Extract a slice from a matrix.

    Parameters
    ----------
    matrix : ndarray
        Input matrix.
    row_slice : slice
        Row slice.
    col_slice : slice, optional
        Column slice. If None, returns full rows.

    Returns
    -------
    ndarray
        Sliced matrix.
    """
    if col_slice is None:
        return matrix[row_slice]
    return matrix[row_slice, col_slice]


def apply_blockwise(
    matrix: np.ndarray,
    func,
    block_size: int = 100,
    output_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """
    Apply a function to matrix block-by-block.

    Parameters
    ----------
    matrix : ndarray
        Input matrix.
    func : callable
        Function to apply to each block. Should take a 2D array.
    block_size : int
        Block size.
    output_shape : tuple, optional
        Shape of output. If None, same as input.

    Returns
    -------
    ndarray
        Result matrix.
    """
    if output_shape is None:
        output_shape = matrix.shape

    result = np.zeros(output_shape, dtype=matrix.dtype)
    s = Slicer(matrix.shape, block_size)

    for row_slice, col_slice in s.iter_blocks():
        block = matrix[row_slice, col_slice]
        result[row_slice, col_slice] = func(block)

    return result


def blockwise_multiply(
    A: np.ndarray,
    B: np.ndarray,
    block_size: int = 100
) -> np.ndarray:
    """
    Block-wise matrix multiplication for memory efficiency.

    Parameters
    ----------
    A : ndarray
        First matrix (m, k).
    B : ndarray
        Second matrix (k, n).
    block_size : int
        Block size for computation.

    Returns
    -------
    ndarray
        Product matrix (m, n).
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match"

    C = np.zeros((m, n), dtype=np.result_type(A, B))

    s_A = Slicer((m, k), block_size)
    s_B = Slicer((k, n), block_size)

    for i in range(s_A.n_row_blocks):
        row_slice = s_A.row_slice(i)
        for j in range(s_B.n_col_blocks):
            col_slice = s_B.col_slice(j)
            # Accumulate over k blocks
            for l in range(s_A.n_col_blocks):
                k_slice = s_A.col_slice(l)
                C[row_slice, col_slice] += A[row_slice, k_slice] @ B[k_slice, col_slice]

    return C


class BlockMatrix:
    """
    Block matrix representation for memory-efficient storage.

    Stores a large matrix as a collection of blocks for efficient
    operations on large Green's function matrices.

    Parameters
    ----------
    shape : tuple
        Full matrix shape.
    block_size : int
        Block size.
    dtype : dtype, optional
        Data type.

    Attributes
    ----------
    blocks : dict
        Dictionary mapping (i, j) to block arrays.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        block_size: int = 100,
        dtype: np.dtype = np.complex128
    ):
        """Initialize block matrix."""
        self.shape = shape
        self.block_size = block_size
        self.dtype = dtype
        self.blocks = {}

        self.slicer = Slicer(shape, block_size)

    def set_block(self, i: int, j: int, block: np.ndarray):
        """Set block (i, j)."""
        self.blocks[(i, j)] = block.astype(self.dtype)

    def get_block(self, i: int, j: int) -> np.ndarray:
        """Get block (i, j). Returns zeros if not set."""
        if (i, j) in self.blocks:
            return self.blocks[(i, j)]
        else:
            row_slice = self.slicer.row_slice(i)
            col_slice = self.slicer.col_slice(j)
            block_shape = (row_slice.stop - row_slice.start,
                          col_slice.stop - col_slice.start)
            return np.zeros(block_shape, dtype=self.dtype)

    def todense(self) -> np.ndarray:
        """Convert to dense matrix."""
        result = np.zeros(self.shape, dtype=self.dtype)

        for (i, j), block in self.blocks.items():
            row_slice, col_slice = self.slicer.block_slice(i, j)
            result[row_slice, col_slice] = block

        return result

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix-vector or matrix-matrix product."""
        if other.ndim == 1:
            result = np.zeros(self.shape[0], dtype=self.dtype)
            for (i, j), block in self.blocks.items():
                row_slice = self.slicer.row_slice(i)
                col_slice = self.slicer.col_slice(j)
                result[row_slice] += block @ other[col_slice]
            return result
        else:
            result = np.zeros((self.shape[0], other.shape[1]), dtype=self.dtype)
            for (i, j), block in self.blocks.items():
                row_slice = self.slicer.row_slice(i)
                col_slice = self.slicer.col_slice(j)
                result[row_slice] += block @ other[col_slice]
            return result

    def __repr__(self) -> str:
        n_blocks = len(self.blocks)
        return f"BlockMatrix(shape={self.shape}, n_blocks={n_blocks})"


def partition_indices(
    n: int,
    n_parts: int
) -> List[slice]:
    """
    Partition indices into roughly equal parts.

    Parameters
    ----------
    n : int
        Total number of indices.
    n_parts : int
        Number of partitions.

    Returns
    -------
    list
        List of slice objects.
    """
    part_size = n // n_parts
    remainder = n % n_parts

    slices = []
    start = 0

    for i in range(n_parts):
        end = start + part_size + (1 if i < remainder else 0)
        slices.append(slice(start, end))
        start = end

    return slices
