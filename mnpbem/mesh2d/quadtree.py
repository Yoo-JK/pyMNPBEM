"""
Quadtree data structure for spatial subdivision.
"""

import numpy as np
from typing import List, Tuple, Optional


class QuadTree:
    """
    Quadtree for efficient 2D spatial queries.

    Used for adaptive mesh refinement and point location.

    Parameters
    ----------
    bounds : tuple
        (x_min, y_min, x_max, y_max) bounding box.
    max_points : int
        Maximum points before subdivision.
    max_depth : int
        Maximum tree depth.

    Examples
    --------
    >>> qt = QuadTree((0, 0, 10, 10))
    >>> qt.insert([5, 5])
    >>> qt.query_range((4, 4, 6, 6))
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        max_points: int = 4,
        max_depth: int = 10,
        depth: int = 0
    ):
        """
        Initialize quadtree node.

        Parameters
        ----------
        bounds : tuple
            (x_min, y_min, x_max, y_max).
        max_points : int
            Subdivision threshold.
        max_depth : int
            Maximum depth.
        depth : int
            Current depth.
        """
        self.bounds = bounds
        self.max_points = max_points
        self.max_depth = max_depth
        self.depth = depth

        self.points = []
        self.children = None  # NW, NE, SW, SE

    @property
    def x_min(self):
        return self.bounds[0]

    @property
    def y_min(self):
        return self.bounds[1]

    @property
    def x_max(self):
        return self.bounds[2]

    @property
    def y_max(self):
        return self.bounds[3]

    @property
    def center(self):
        return ((self.x_min + self.x_max) / 2,
                (self.y_min + self.y_max) / 2)

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is within bounds."""
        x, y = point[:2]
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def subdivide(self) -> None:
        """Split node into four children."""
        cx, cy = self.center

        self.children = [
            QuadTree((self.x_min, cy, cx, self.y_max),
                     self.max_points, self.max_depth, self.depth + 1),  # NW
            QuadTree((cx, cy, self.x_max, self.y_max),
                     self.max_points, self.max_depth, self.depth + 1),  # NE
            QuadTree((self.x_min, self.y_min, cx, cy),
                     self.max_points, self.max_depth, self.depth + 1),  # SW
            QuadTree((cx, self.y_min, self.x_max, cy),
                     self.max_points, self.max_depth, self.depth + 1),  # SE
        ]

        # Move existing points to children
        for point in self.points:
            for child in self.children:
                if child.contains(point):
                    child.insert(point)
                    break

        self.points = []

    def insert(self, point: np.ndarray) -> bool:
        """
        Insert a point into the quadtree.

        Parameters
        ----------
        point : ndarray
            Point coordinates (x, y) or (x, y, ...).

        Returns
        -------
        bool
            True if inserted successfully.
        """
        point = np.asarray(point)

        if not self.contains(point):
            return False

        if self.children is not None:
            for child in self.children:
                if child.insert(point):
                    return True
            return False

        self.points.append(point)

        if len(self.points) > self.max_points and self.depth < self.max_depth:
            self.subdivide()

        return True

    def insert_many(self, points: np.ndarray) -> None:
        """Insert multiple points."""
        for point in points:
            self.insert(point)

    def query_range(
        self,
        bounds: Tuple[float, float, float, float]
    ) -> List[np.ndarray]:
        """
        Find all points within a bounding box.

        Parameters
        ----------
        bounds : tuple
            (x_min, y_min, x_max, y_max) query box.

        Returns
        -------
        list
            Points within the query box.
        """
        result = []

        # Check if bounds intersect
        if not self._intersects(bounds):
            return result

        # Check points in this node
        for point in self.points:
            x, y = point[:2]
            if (bounds[0] <= x <= bounds[2] and
                bounds[1] <= y <= bounds[3]):
                result.append(point)

        # Check children
        if self.children is not None:
            for child in self.children:
                result.extend(child.query_range(bounds))

        return result

    def query_radius(
        self,
        center: np.ndarray,
        radius: float
    ) -> List[np.ndarray]:
        """
        Find all points within a radius of a point.

        Parameters
        ----------
        center : ndarray
            Query center point.
        radius : float
            Search radius.

        Returns
        -------
        list
            Points within radius.
        """
        # First get points in bounding box
        bounds = (center[0] - radius, center[1] - radius,
                  center[0] + radius, center[1] + radius)
        candidates = self.query_range(bounds)

        # Filter by actual distance
        result = []
        for point in candidates:
            if np.linalg.norm(point[:2] - center[:2]) <= radius:
                result.append(point)

        return result

    def _intersects(self, bounds: Tuple[float, float, float, float]) -> bool:
        """Check if bounds intersect this node."""
        return not (bounds[2] < self.x_min or
                    bounds[0] > self.x_max or
                    bounds[3] < self.y_min or
                    bounds[1] > self.y_max)

    def all_points(self) -> List[np.ndarray]:
        """Get all points in the tree."""
        result = list(self.points)

        if self.children is not None:
            for child in self.children:
                result.extend(child.all_points())

        return result

    def __len__(self) -> int:
        """Number of points in tree."""
        count = len(self.points)
        if self.children is not None:
            for child in self.children:
                count += len(child)
        return count
