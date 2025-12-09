"""
Compound point class for field evaluation points.
"""

import numpy as np
from typing import List, Any

from .point import Point
from .compound import Compound


class ComPoint(Compound):
    """
    Compound of points in a dielectric environment.

    Used for defining field evaluation points with associated
    dielectric properties.

    Parameters
    ----------
    eps : list
        List of dielectric function objects.
    p : list
        List of Point objects.
    inout : ndarray
        Index to medium eps for each point set.

    Examples
    --------
    >>> from mnpbem import EpsConst, ComPoint
    >>> from mnpbem.particles import Point
    >>>
    >>> eps_vacuum = EpsConst(1.0)
    >>> points = Point(np.random.rand(100, 3) * 20 - 10)
    >>> cp = ComPoint([eps_vacuum], [points], [1])
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Point],
        inout: np.ndarray
    ):
        """
        Initialize compound point.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        p : list
            List of Point objects.
        inout : ndarray
            Medium indices for each point set.
        """
        # Convert single point to list
        if isinstance(p, Point):
            p = [p]

        super().__init__(eps, p, inout)

    @property
    def pos(self) -> np.ndarray:
        """All point positions."""
        return self.pc.pos

    @property
    def n_points(self) -> int:
        """Total number of points."""
        return self.pc.n_points

    def shift(self, displacement: np.ndarray) -> 'ComPoint':
        """
        Shift all points by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        ComPoint
            Shifted compound point.
        """
        new_p = [p.shift(displacement) for p in self.p]
        return ComPoint(self.eps, new_p, self.inout)

    def flip(self, axis: int) -> 'ComPoint':
        """
        Flip all points along axis.

        Parameters
        ----------
        axis : int
            Axis to flip.

        Returns
        -------
        ComPoint
            Flipped compound point.
        """
        new_p = []
        for pt in self.p:
            new_pos = pt.pos.copy()
            new_pos[:, axis] = -new_pos[:, axis]
            new_p.append(Point(new_pos, pt.vec.copy()))
        return ComPoint(self.eps, new_p, self.inout)

    def select(
        self,
        indices: np.ndarray = None,
        carfun: callable = None,
        polfun: callable = None,
        sphfun: callable = None
    ) -> 'ComPoint':
        """
        Select points in compound point object.

        Parameters
        ----------
        indices : ndarray, optional
            Indices of points to select (global indices).
        carfun : callable, optional
            Function f(x, y, z) returning boolean for Cartesian selection.
        polfun : callable, optional
            Function f(phi, r, z) returning boolean for polar selection.
        sphfun : callable, optional
            Function f(phi, theta, r) returning boolean for spherical selection.

        Returns
        -------
        ComPoint
            Selected compound points.

        Examples
        --------
        >>> # Select by indices
        >>> cp_selected = cp.select(indices=[0, 1, 2])
        >>> # Select by Cartesian condition
        >>> cp_selected = cp.select(carfun=lambda x, y, z: z > 0)
        >>> # Select by spherical condition
        >>> cp_selected = cp.select(sphfun=lambda phi, theta, r: r < 10)
        """
        if indices is not None:
            # Selection by global index
            indices = np.atleast_1d(indices)

            # Create mapping from global to local indices
            cum_n = 0
            new_p = []
            new_inout = []

            for i, pt in enumerate(self.p):
                n = pt.n_points
                # Find which global indices belong to this point set
                local_mask = (indices >= cum_n) & (indices < cum_n + n)
                local_indices = indices[local_mask] - cum_n

                if len(local_indices) > 0:
                    new_p.append(pt.select(indices=local_indices))
                    new_inout.append(self.inout[i])

                cum_n += n

            if not new_p:
                # Return empty
                return ComPoint(self.eps, [Point(np.zeros((0, 3)))], np.array([1]))

            return ComPoint(self.eps, new_p, np.array(new_inout))

        else:
            # Selection by coordinate function
            new_p = [pt.select(carfun=carfun, polfun=polfun, sphfun=sphfun)
                     for pt in self.p]

            # Keep only non-empty point sets
            new_inout = []
            filtered_p = []
            for i, pt in enumerate(new_p):
                if pt.n_points > 0:
                    filtered_p.append(pt)
                    new_inout.append(self.inout[i])

            if not filtered_p:
                return ComPoint(self.eps, [Point(np.zeros((0, 3)))], np.array([1]))

            return ComPoint(self.eps, filtered_p, np.array(new_inout))

    def __repr__(self) -> str:
        return (f"ComPoint(n_eps={len(self.eps)}, "
                f"n_point_sets={len(self.p)}, "
                f"n_points={self.n_points})")
