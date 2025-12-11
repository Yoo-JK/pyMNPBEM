"""
Composite particle with mirror symmetry.

This module provides particles that exploit mirror symmetry
for computational efficiency.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any

from .particle import Particle
from .comparticle import ComParticle
from .compstruct import CompStruct


class ComParticleMirror(ComParticle):
    """
    Composite particle with mirror symmetry.

    Exploits mirror symmetry in x, y, or xy plane to reduce
    computational cost by a factor of 2 or 4.

    Parameters
    ----------
    eps : list
        List of dielectric function objects.
    p : list
        List of Particle objects.
    inout : array_like
        Index array specifying inside/outside media for each particle.
    sym : str
        Symmetry type: 'x', 'y', or 'xy'.
    closed : int or list, optional
        Closed surface specification.
    **kwargs : dict
        Additional options.

    Attributes
    ----------
    sym : str
        Symmetry type.
    symtable : ndarray
        Table of symmetry values.
    pfull : ComParticle
        Full particle expanded with mirror symmetry.

    Examples
    --------
    >>> from pymnpbem import EpsConst, EpsTable, trisphere
    >>> from pymnpbem.particles import ComParticleMirror
    >>>
    >>> eps = [EpsConst(1), EpsTable('gold.dat')]
    >>> # Create half-sphere (x > 0 only)
    >>> sphere = trisphere(144, 10)  # Will be mirrored in x
    >>> p = ComParticleMirror(eps, [sphere], [[2, 1]], sym='x')
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Particle],
        inout: np.ndarray,
        sym: str = 'x',
        closed: Optional[Union[int, List]] = None,
        **kwargs
    ):
        """Initialize composite particle with mirror symmetry."""
        # Store symmetry info
        self.sym = sym
        self._init_symtable()

        # Initialize base ComParticle with original particles
        super().__init__(eps, p, inout, closed=closed, **kwargs)

        # Create full particle by mirroring
        self.pfull = self._create_full_particle(eps, p, inout, closed, **kwargs)

    def _init_symtable(self) -> None:
        """Initialize symmetry table based on symmetry type."""
        if self.sym in ('x', 'y'):
            # Two-fold symmetry: [original, mirrored]
            # Columns: [1, symmetry_factor]
            self.symtable = np.array([
                [1, 1],    # '+' symmetric
                [1, -1]    # '-' antisymmetric
            ])
        elif self.sym == 'xy':
            # Four-fold symmetry
            # Columns: [1, x_factor, y_factor, xy_factor]
            self.symtable = np.array([
                [1, 1, 1, 1],      # '++' symmetric in both
                [1, 1, -1, -1],    # '+-' symmetric in x, antisymmetric in y
                [1, -1, 1, -1],    # '-+' antisymmetric in x, symmetric in y
                [1, -1, -1, 1]     # '--' antisymmetric in both
            ])
        else:
            raise ValueError(f"Unknown symmetry type: {self.sym}")

    def _create_full_particle(
        self,
        eps: List[Any],
        p: List[Particle],
        inout: np.ndarray,
        closed: Optional[Union[int, List]],
        **kwargs
    ) -> ComParticle:
        """Create full particle by applying mirror operations."""
        inout = np.atleast_2d(inout)

        # Start with original particles
        particles = list(p)
        inout_full = inout.copy()

        # Apply x-mirror if needed
        if self.sym in ('x', 'xy'):
            for particle in list(particles):
                # Flip in x (axis=0)
                mirrored = particle.flip(0)
                particles.append(mirrored)
                inout_full = np.vstack([inout_full, inout_full[-len(p):]])

        # Apply y-mirror if needed
        if self.sym in ('y', 'xy'):
            n_current = len(particles)
            for i, particle in enumerate(particles[:n_current]):
                # Flip in y (axis=1)
                mirrored = particle.flip(1)
                particles.append(mirrored)
                inout_full = np.vstack([inout_full, inout_full[i:i+1]])

        # Create full ComParticle
        return ComParticle(eps, particles, inout_full, closed=closed, **kwargs)

    @property
    def full(self) -> ComParticle:
        """Return full particle expanded with mirror symmetry."""
        return self.pfull

    def symvalue(self, key: str) -> np.ndarray:
        """
        Get symmetry values for given key.

        Parameters
        ----------
        key : str
            Symmetry key: '+', '-' for x/y symmetry,
            '++', '+-', '-+', '--' for xy symmetry.

        Returns
        -------
        ndarray
            Symmetry value array.
        """
        if isinstance(key, (list, tuple)):
            return np.vstack([self.symvalue(k) for k in key])

        sym_map = {
            '+': np.array([1, 1]),
            '-': np.array([1, -1]),
            '++': np.array([1, 1, 1, 1]),
            '+-': np.array([1, 1, -1, -1]),
            '-+': np.array([1, -1, 1, -1]),
            '--': np.array([1, -1, -1, 1])
        }

        if key not in sym_map:
            raise ValueError(f"Unknown symmetry key: {key}")
        return sym_map[key]

    def symindex(self, tab: np.ndarray) -> int:
        """
        Find index of symmetry values in symmetry table.

        Parameters
        ----------
        tab : ndarray
            Symmetry values to find.

        Returns
        -------
        int
            Index in symmetry table, or -1 if not found.
        """
        tab = np.asarray(tab)
        for i, row in enumerate(self.symtable):
            if np.allclose(row, tab):
                return i
        return -1

    def expand_scalar(self, values: np.ndarray, symval: np.ndarray) -> np.ndarray:
        """
        Expand scalar values using mirror symmetry.

        Parameters
        ----------
        values : ndarray
            Values for reduced particle.
        symval : ndarray
            Symmetry values.

        Returns
        -------
        ndarray
            Expanded values for full particle.
        """
        expanded = [values]
        for k in range(1, symval.shape[1] if symval.ndim > 1 else len(symval)):
            factor = symval[-1, k] if symval.ndim > 1 else symval[k]
            expanded.append(factor * values)
        return np.concatenate(expanded, axis=0)

    def expand_vector(self, values: np.ndarray, symval: np.ndarray) -> np.ndarray:
        """
        Expand vector values using mirror symmetry.

        Parameters
        ----------
        values : ndarray
            Vector values for reduced particle, shape (n, 3) or (n, 3, ...).
        symval : ndarray
            Symmetry values.

        Returns
        -------
        ndarray
            Expanded vector values for full particle.
        """
        expanded = [values]
        n_sym = symval.shape[1] if symval.ndim > 1 else len(symval)

        for k in range(1, n_sym):
            val_copy = values.copy()
            # Apply symmetry factor to each component
            for l in range(3):
                factor = symval[l, k] if symval.ndim > 1 else symval[k]
                if values.ndim == 2:
                    val_copy[:, l] = factor * values[:, l]
                else:
                    val_copy[:, l, ...] = factor * values[:, l, ...]
            expanded.append(val_copy)

        return np.concatenate(expanded, axis=0)

    def reduce_scalar(self, values: np.ndarray, symval: np.ndarray) -> np.ndarray:
        """
        Reduce scalar values from full to symmetric representation.

        Parameters
        ----------
        values : ndarray
            Values for full particle.
        symval : ndarray
            Symmetry values.

        Returns
        -------
        ndarray
            Reduced values.
        """
        n = len(values) // self.n_sym_factor
        reduced = np.zeros(n, dtype=values.dtype)

        for k in range(self.n_sym_factor):
            factor = symval[-1, k] if symval.ndim > 1 else symval[k]
            reduced += factor * values[k*n:(k+1)*n]

        return reduced / self.n_sym_factor

    @property
    def n_sym_factor(self) -> int:
        """Number of symmetry copies (2 for x/y, 4 for xy)."""
        return 4 if self.sym == 'xy' else 2

    def __repr__(self) -> str:
        return (f"ComParticleMirror(n_particles={len(self.p)}, "
                f"n_faces={self.n_faces}, sym='{self.sym}')")


class CompStructMirror:
    """
    Structure for composite data with mirror symmetry.

    Stores values for the reduced (symmetric) particle and provides
    methods to expand to the full particle.

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry.
    enei : float
        Wavelength in nm.
    fun : callable, optional
        Expansion function.

    Attributes
    ----------
    p : ComParticleMirror
        Particle reference.
    enei : float
        Wavelength.
    val : list
        List of CompStruct objects for each symmetry configuration.
    fun : callable
        Expansion function.
    """

    def __init__(
        self,
        p: ComParticleMirror,
        enei: float,
        fun: Optional[callable] = None,
        **kwargs
    ):
        """Initialize mirror structure."""
        self.p = p
        self.enei = enei
        self.fun = fun if fun is not None else self._default_expand
        self.val = []
        self._data = kwargs

    def _default_expand(self, obj: 'CompStructMirror') -> CompStruct:
        """Default expansion function."""
        return self.expand()

    def add_symmetry_value(self, struct: CompStruct, symval: np.ndarray) -> None:
        """
        Add a CompStruct with its symmetry values.

        Parameters
        ----------
        struct : CompStruct
            Structure to add.
        symval : ndarray
            Associated symmetry values.
        """
        struct.symval = symval
        self.val.append(struct)

    def expand(self) -> CompStruct:
        """
        Expand structure to full particle size.

        Returns
        -------
        CompStruct
            Expanded structure for full particle.
        """
        if not self.val:
            raise ValueError("No symmetry values added")

        # Get field names from first structure
        result = CompStruct(self.p.full, self.enei)

        for name in self.val[0].fields:
            if name == 'symval':
                continue

            value = self.val[0].get(name)
            symval = self.val[0].symval

            # Expand based on field type
            if name in ('phi', 'phip', 'phi1', 'phi2', 'phi1p', 'phi2p',
                       'sig', 'sig1', 'sig2'):
                # Scalar fields
                expanded = self.p.expand_scalar(value, symval)
            elif name in ('a1', 'a1p', 'a2', 'a2p', 'e', 'h', 'h1', 'h2'):
                # Vector fields
                expanded = self.p.expand_vector(value, symval)
            else:
                # Unknown field - try scalar expansion
                expanded = self.p.expand_scalar(value, symval)

            result.set(name, expanded)

        return result

    def full(self) -> CompStruct:
        """Expand to full particle (alias for expand)."""
        return self.fun(self)

    def get(self, name: str) -> Any:
        """Get field value."""
        return self._data.get(name)

    def set(self, name: str, value: Any) -> None:
        """Set field value."""
        self._data[name] = value

    def __repr__(self) -> str:
        return f"CompStructMirror(enei={self.enei}, n_vals={len(self.val)})"
