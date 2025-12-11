"""
Compound structure for storing BEM solution data.

This module provides the CompStruct class that stores BEM solution data
including surface charges and currents for both quasistatic and retarded
simulations.

For retarded BEM (full Maxwell equations), the solution contains:
- sig1, sig2: Surface charges inside/outside the particle boundary
- h1, h2: Surface currents (3D vectors) inside/outside the boundary

References
----------
Garcia de Abajo & Howie, PRB 65, 115418 (2002)
"""

import numpy as np
from typing import Optional, Any, Dict, Union, List

from .comparticle import ComParticle


class CompStruct:
    """
    Structure for storing BEM solution data.

    Used to pass results between BEM solver and excitation classes.
    Supports both quasistatic (scalar charges) and retarded (charges + currents)
    BEM solutions.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    enei : float
        Light wavelength in nm.
    **kwargs : dict
        Named data arrays (e.g., sig=surface_charges).

    Attributes
    ----------
    p : ComParticle
        Associated particle.
    enei : float
        Wavelength.

    Quasistatic BEM fields:
    sig : ndarray, optional
        Surface charges, shape (n_faces,) or (n_faces, n_exc).
    phip : ndarray, optional
        External potential at boundary.

    Retarded BEM fields (Garcia de Abajo & Howie, PRB 65, 115418):
    sig1 : ndarray, optional
        Surface charges on inner boundary, shape (n_faces,) or (n_faces, n_exc).
    sig2 : ndarray, optional
        Surface charges on outer boundary.
    h1 : ndarray, optional
        Surface currents on inner boundary, shape (n_faces, 3) or (n_faces, 3, n_exc).
    h2 : ndarray, optional
        Surface currents on outer boundary.
    k : complex, optional
        Wavenumber used for the solution.

    Excitation fields:
    phi1, phi2 : ndarray, optional
        Scalar potentials inside/outside (for excitation).
    phi1p, phi2p : ndarray, optional
        Surface derivatives of scalar potentials.
    a1, a2 : ndarray, optional
        Vector potentials inside/outside, shape (n_faces, 3, n_exc).
    a1p, a2p : ndarray, optional
        Surface derivatives of vector potentials.
    e : ndarray, optional
        Electric field at boundary.
    h : ndarray, optional
        Magnetic field at boundary (note: different from h1/h2 surface currents).
    """

    def __init__(
        self,
        p: ComParticle,
        enei: float,
        **kwargs
    ):
        """
        Initialize compound structure.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Light wavelength in nm.
        **kwargs : dict
            Named data arrays.
        """
        self.p = p
        self.enei = enei
        self._data: Dict[str, np.ndarray] = {}

        for key, value in kwargs.items():
            self._data[key] = np.asarray(value)
            setattr(self, key, self._data[key])

    def __getitem__(self, key: str) -> np.ndarray:
        """Get data by key."""
        return self._data.get(key, getattr(self, key, None))

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set data by key."""
        self._data[key] = np.asarray(value)
        setattr(self, key, self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        """Get data with default."""
        return self._data.get(key, default)

    def __add__(self, other: 'CompStruct') -> 'CompStruct':
        """Add two compound structures."""
        if self.p != other.p:
            raise ValueError("Cannot add CompStruct with different particles")
        if self.enei != other.enei:
            raise ValueError("Cannot add CompStruct with different wavelengths")

        result = CompStruct(self.p, self.enei)
        for key in set(self._data.keys()) | set(other._data.keys()):
            if key in self._data and key in other._data:
                result[key] = self._data[key] + other._data[key]
            elif key in self._data:
                result[key] = self._data[key].copy()
            else:
                result[key] = other._data[key].copy()
        return result

    def __sub__(self, other: 'CompStruct') -> 'CompStruct':
        """Subtract two compound structures."""
        if self.p != other.p:
            raise ValueError("Cannot subtract CompStruct with different particles")
        if self.enei != other.enei:
            raise ValueError("Cannot subtract CompStruct with different wavelengths")

        result = CompStruct(self.p, self.enei)
        for key in set(self._data.keys()) | set(other._data.keys()):
            if key in self._data and key in other._data:
                result[key] = self._data[key] - other._data[key]
            elif key in self._data:
                result[key] = self._data[key].copy()
            else:
                result[key] = -other._data[key]
        return result

    def __mul__(self, scalar: float) -> 'CompStruct':
        """Multiply by scalar."""
        result = CompStruct(self.p, self.enei)
        for key, value in self._data.items():
            result[key] = value * scalar
        return result

    def __rmul__(self, scalar: float) -> 'CompStruct':
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> 'CompStruct':
        """Negate."""
        return self.__mul__(-1)

    def keys(self):
        """Return data keys."""
        return self._data.keys()

    def set(self, **kwargs) -> 'CompStruct':
        """
        Set multiple data fields at once (MATLAB compatibility).

        Parameters
        ----------
        **kwargs : dict
            Named data arrays to set.

        Returns
        -------
        CompStruct
            Self for method chaining.
        """
        for key, value in kwargs.items():
            self[key] = value
        return self

    def has_retarded_solution(self) -> bool:
        """
        Check if this contains a retarded BEM solution.

        Returns
        -------
        bool
            True if sig1/sig2/h1/h2 fields are present.
        """
        return all(key in self._data for key in ['sig1', 'sig2', 'h1', 'h2'])

    def has_quasistatic_solution(self) -> bool:
        """
        Check if this contains a quasistatic BEM solution.

        Returns
        -------
        bool
            True if sig field is present (and no retarded fields).
        """
        return 'sig' in self._data and not self.has_retarded_solution()

    @property
    def n_exc(self) -> int:
        """
        Number of excitations in the solution.

        Returns
        -------
        int
            Number of excitation columns.
        """
        # Check various fields to determine n_exc
        for key in ['sig', 'sig1', 'sig2']:
            if key in self._data:
                arr = self._data[key]
                if arr.ndim == 1:
                    return 1
                elif arr.ndim == 2:
                    return arr.shape[1]
        return 1

    def copy(self) -> 'CompStruct':
        """
        Create a deep copy of this CompStruct.

        Returns
        -------
        CompStruct
            Deep copy.
        """
        result = CompStruct(self.p, self.enei)
        for key, value in self._data.items():
            result[key] = value.copy()
        return result

    def __repr__(self) -> str:
        data_keys = list(self._data.keys())
        sol_type = "retarded" if self.has_retarded_solution() else "quasistatic"
        return f"CompStruct(enei={self.enei}, type={sol_type}, data={data_keys})"
