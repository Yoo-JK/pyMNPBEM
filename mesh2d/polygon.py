"""
Polygon and edge profile classes for 2D mesh generation.

These classes provide tools for defining and manipulating 2D
polygonal boundaries for particle geometry creation.
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Callable
from scipy import interpolate


class Polygon:
    """
    2D polygon class for boundary definition.

    A polygon consists of vertices connected by edges. Can be used
    to define particle cross-sections for extrusion or revolution.

    Parameters
    ----------
    verts : ndarray
        Polygon vertices, shape (n_verts, 2).
    closed : bool
        If True, polygon is closed (first = last vertex).

    Attributes
    ----------
    verts : ndarray
        Polygon vertices.
    n_verts : int
        Number of vertices.
    closed : bool
        Whether polygon is closed.

    Examples
    --------
    >>> # Create a square
    >>> square = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> print(square.area)
    1.0
    """

    def __init__(self, verts: np.ndarray, closed: bool = True):
        """
        Initialize polygon.

        Parameters
        ----------
        verts : ndarray
            Vertices, shape (n, 2).
        closed : bool
            Whether to close the polygon.
        """
        self.verts = np.atleast_2d(verts).astype(float)

        # Ensure 2D coordinates
        if self.verts.shape[1] != 2:
            raise ValueError("Vertices must be 2D (n, 2)")

        self.closed = closed

        # Close polygon if needed
        if closed and not np.allclose(self.verts[0], self.verts[-1]):
            self.verts = np.vstack([self.verts, self.verts[0]])

    @property
    def n_verts(self) -> int:
        """Number of vertices."""
        return len(self.verts) - 1 if self.closed else len(self.verts)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.verts) - 1 if self.closed else len(self.verts) - 1

    @property
    def edges(self) -> np.ndarray:
        """
        Get edge vectors.

        Returns
        -------
        ndarray
            Edge vectors, shape (n_edges, 2).
        """
        return np.diff(self.verts, axis=0)

    @property
    def edge_lengths(self) -> np.ndarray:
        """Edge lengths."""
        return np.linalg.norm(self.edges, axis=1)

    @property
    def perimeter(self) -> float:
        """Polygon perimeter."""
        return np.sum(self.edge_lengths)

    @property
    def area(self) -> float:
        """
        Polygon area using shoelace formula.

        Returns
        -------
        float
            Polygon area (positive if CCW, negative if CW).
        """
        if not self.closed:
            return 0.0

        x = self.verts[:-1, 0]
        y = self.verts[:-1, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @property
    def centroid(self) -> np.ndarray:
        """
        Polygon centroid.

        Returns
        -------
        ndarray
            Centroid coordinates (2,).
        """
        if not self.closed:
            return np.mean(self.verts, axis=0)

        x = self.verts[:-1, 0]
        y = self.verts[:-1, 1]
        A = self.area

        if np.abs(A) < 1e-10:
            return np.mean(self.verts[:-1], axis=0)

        cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
        cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)

        return np.array([cx, cy])

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bounding box.

        Returns
        -------
        min_corner : ndarray
            Minimum x, y.
        max_corner : ndarray
            Maximum x, y.
        """
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Test if points are inside polygon.

        Parameters
        ----------
        points : ndarray
            Test points, shape (n, 2) or (2,).

        Returns
        -------
        ndarray
            Boolean array.
        """
        from .inpoly import inpoly
        return inpoly(np.atleast_2d(points), self.verts)

    def sample(self, n_points: int = 100) -> np.ndarray:
        """
        Sample points along polygon boundary.

        Parameters
        ----------
        n_points : int
            Number of sample points.

        Returns
        -------
        ndarray
            Sample points, shape (n_points, 2).
        """
        # Parametric sampling
        edge_lens = self.edge_lengths
        cum_lens = np.concatenate([[0], np.cumsum(edge_lens)])
        total_len = cum_lens[-1]

        t_vals = np.linspace(0, total_len, n_points, endpoint=False)
        points = []

        for t in t_vals:
            # Find which edge
            idx = np.searchsorted(cum_lens[1:], t)
            idx = min(idx, len(edge_lens) - 1)

            # Local parameter
            t_local = (t - cum_lens[idx]) / edge_lens[idx]
            p = self.verts[idx] + t_local * self.edges[idx]
            points.append(p)

        return np.array(points)

    def translate(self, offset: np.ndarray) -> 'Polygon':
        """
        Translate polygon.

        Parameters
        ----------
        offset : ndarray
            Translation vector (2,).

        Returns
        -------
        Polygon
            Translated polygon.
        """
        offset = np.asarray(offset)
        return Polygon(self.verts + offset, closed=self.closed)

    def scale(self, factor: Union[float, np.ndarray], center: Optional[np.ndarray] = None) -> 'Polygon':
        """
        Scale polygon.

        Parameters
        ----------
        factor : float or ndarray
            Scale factor(s).
        center : ndarray, optional
            Scale center (default: centroid).

        Returns
        -------
        Polygon
            Scaled polygon.
        """
        if center is None:
            center = self.centroid

        center = np.asarray(center)
        verts_centered = self.verts - center
        verts_scaled = verts_centered * factor + center

        return Polygon(verts_scaled, closed=self.closed)

    def rotate(self, angle: float, center: Optional[np.ndarray] = None) -> 'Polygon':
        """
        Rotate polygon.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        center : ndarray, optional
            Rotation center (default: centroid).

        Returns
        -------
        Polygon
            Rotated polygon.
        """
        if center is None:
            center = self.centroid

        center = np.asarray(center)

        # Rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])

        verts_centered = self.verts - center
        verts_rotated = verts_centered @ R.T + center

        return Polygon(verts_rotated, closed=self.closed)

    def reverse(self) -> 'Polygon':
        """
        Reverse vertex order (change orientation).

        Returns
        -------
        Polygon
            Reversed polygon.
        """
        return Polygon(self.verts[::-1], closed=self.closed)

    def smooth(self, n_points: int = 100, method: str = 'spline') -> 'Polygon':
        """
        Smooth polygon using interpolation.

        Parameters
        ----------
        n_points : int
            Number of output points.
        method : str
            Interpolation method: 'spline', 'linear'.

        Returns
        -------
        Polygon
            Smoothed polygon.
        """
        if method == 'spline':
            # Parametric spline interpolation
            t = np.linspace(0, 1, len(self.verts))
            t_new = np.linspace(0, 1, n_points)

            if self.closed:
                # Periodic spline
                tck_x = interpolate.splrep(t, self.verts[:, 0], per=True)
                tck_y = interpolate.splrep(t, self.verts[:, 1], per=True)
            else:
                tck_x = interpolate.splrep(t, self.verts[:, 0])
                tck_y = interpolate.splrep(t, self.verts[:, 1])

            x_new = interpolate.splev(t_new, tck_x)
            y_new = interpolate.splev(t_new, tck_y)

            return Polygon(np.column_stack([x_new, y_new]), closed=self.closed)
        else:
            # Linear resampling
            return Polygon(self.sample(n_points), closed=self.closed)

    def to_3d(self, z: float = 0.0, axis: int = 2) -> np.ndarray:
        """
        Convert to 3D coordinates.

        Parameters
        ----------
        z : float
            Coordinate along extrusion axis.
        axis : int
            Extrusion axis (0, 1, or 2).

        Returns
        -------
        ndarray
            3D vertices, shape (n_verts, 3).
        """
        verts_3d = np.zeros((len(self.verts), 3))

        if axis == 2:
            verts_3d[:, 0] = self.verts[:, 0]
            verts_3d[:, 1] = self.verts[:, 1]
            verts_3d[:, 2] = z
        elif axis == 1:
            verts_3d[:, 0] = self.verts[:, 0]
            verts_3d[:, 1] = z
            verts_3d[:, 2] = self.verts[:, 1]
        else:
            verts_3d[:, 0] = z
            verts_3d[:, 1] = self.verts[:, 0]
            verts_3d[:, 2] = self.verts[:, 1]

        return verts_3d

    def __repr__(self) -> str:
        return f"Polygon(n_verts={self.n_verts}, area={self.area:.3f})"


class EdgeProfile:
    """
    Edge profile for rounded corners and edge treatments.

    Defines how edges are rounded or modified when creating
    3D particle meshes from 2D cross-sections.

    Parameters
    ----------
    mode : str
        Profile mode: 'round', 'chamfer', 'fillet', 'custom'.
    radius : float
        Rounding radius.
    n_segments : int
        Number of segments for rounding.
    func : callable, optional
        Custom profile function for 'custom' mode.

    Attributes
    ----------
    mode : str
        Profile mode.
    radius : float
        Rounding radius.

    Examples
    --------
    >>> # Round profile with 5 nm radius
    >>> profile = EdgeProfile('round', radius=5, n_segments=8)
    >>> points = profile.apply([0, 0], [1, 0], [0.5, 0.5])
    """

    def __init__(
        self,
        mode: str = 'round',
        radius: float = 1.0,
        n_segments: int = 8,
        func: Optional[Callable] = None
    ):
        """
        Initialize edge profile.

        Parameters
        ----------
        mode : str
            Profile mode.
        radius : float
            Rounding radius.
        n_segments : int
            Number of segments.
        func : callable, optional
            Custom function f(t) -> r for custom mode.
        """
        self.mode = mode
        self.radius = radius
        self.n_segments = n_segments
        self.func = func

    def apply(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """
        Apply edge profile between two points.

        Parameters
        ----------
        p1 : ndarray
            Start point.
        p2 : ndarray
            End point.
        normal : ndarray
            Edge normal direction.

        Returns
        -------
        ndarray
            Profile points, shape (n_segments, 2) or (n_segments, 3).
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        normal = np.asarray(normal)
        normal = normal / np.linalg.norm(normal)

        if self.mode == 'round':
            return self._round_profile(p1, p2, normal)
        elif self.mode == 'chamfer':
            return self._chamfer_profile(p1, p2, normal)
        elif self.mode == 'fillet':
            return self._fillet_profile(p1, p2, normal)
        elif self.mode == 'custom' and self.func is not None:
            return self._custom_profile(p1, p2, normal)
        else:
            # Linear profile
            t = np.linspace(0, 1, self.n_segments)
            return np.outer(1 - t, p1) + np.outer(t, p2)

    def _round_profile(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """Generate rounded edge profile."""
        # Quarter circle arc
        theta = np.linspace(0, np.pi / 2, self.n_segments)

        # Edge direction
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        edge_dir = edge / edge_len

        points = []
        for t in theta:
            # Parametric point on arc
            offset = self.radius * (1 - np.cos(t)) * normal
            along = self.radius * np.sin(t) * edge_dir

            point = p1 + along + offset
            points.append(point)

        return np.array(points)

    def _chamfer_profile(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """Generate chamfered (45-degree cut) edge profile."""
        # Straight diagonal cut
        t = np.linspace(0, 1, self.n_segments)

        p1_offset = p1 + self.radius * normal
        p2_offset = p2 + self.radius * normal

        points = np.outer(1 - t, p1_offset) + np.outer(t, p2_offset)
        return points

    def _fillet_profile(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """Generate fillet (quarter circle) profile."""
        # Same as round but with different parameterization
        return self._round_profile(p1, p2, normal)

    def _custom_profile(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """Generate custom profile using user function."""
        t = np.linspace(0, 1, self.n_segments)

        edge = p2 - p1
        edge_len = np.linalg.norm(edge)

        points = []
        for ti in t:
            # User function gives radius as function of t
            r = self.func(ti) * self.radius
            point = p1 + ti * edge + r * normal
            points.append(point)

        return np.array(points)

    def __repr__(self) -> str:
        return f"EdgeProfile(mode='{self.mode}', radius={self.radius})"


def polygon_from_function(
    func: Callable,
    t_range: Tuple[float, float] = (0, 2 * np.pi),
    n_points: int = 100,
    closed: bool = True
) -> Polygon:
    """
    Create polygon from parametric function.

    Parameters
    ----------
    func : callable
        Function f(t) -> (x, y).
    t_range : tuple
        Parameter range.
    n_points : int
        Number of points.
    closed : bool
        Whether polygon is closed.

    Returns
    -------
    Polygon
        Generated polygon.

    Examples
    --------
    >>> # Create ellipse
    >>> ellipse = polygon_from_function(
    ...     lambda t: (2 * np.cos(t), np.sin(t)),
    ...     n_points=50
    ... )
    """
    t = np.linspace(t_range[0], t_range[1], n_points)
    points = np.array([func(ti) for ti in t])
    return Polygon(points, closed=closed)


def circle(radius: float = 1.0, center: Tuple[float, float] = (0, 0), n_points: int = 64) -> Polygon:
    """
    Create circular polygon.

    Parameters
    ----------
    radius : float
        Circle radius.
    center : tuple
        Center coordinates.
    n_points : int
        Number of points.

    Returns
    -------
    Polygon
        Circular polygon.
    """
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return Polygon(np.column_stack([x, y]), closed=True)


def ellipse(
    a: float = 1.0,
    b: float = 0.5,
    center: Tuple[float, float] = (0, 0),
    angle: float = 0.0,
    n_points: int = 64
) -> Polygon:
    """
    Create elliptical polygon.

    Parameters
    ----------
    a : float
        Semi-major axis.
    b : float
        Semi-minor axis.
    center : tuple
        Center coordinates.
    angle : float
        Rotation angle in radians.
    n_points : int
        Number of points.

    Returns
    -------
    Polygon
        Elliptical polygon.
    """
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    if angle != 0:
        c, s = np.cos(angle), np.sin(angle)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        x, y = x_rot, y_rot

    x += center[0]
    y += center[1]

    return Polygon(np.column_stack([x, y]), closed=True)


def rectangle(
    width: float = 1.0,
    height: float = 1.0,
    center: Tuple[float, float] = (0, 0)
) -> Polygon:
    """
    Create rectangular polygon.

    Parameters
    ----------
    width : float
        Rectangle width.
    height : float
        Rectangle height.
    center : tuple
        Center coordinates.

    Returns
    -------
    Polygon
        Rectangular polygon.
    """
    w, h = width / 2, height / 2
    cx, cy = center
    verts = [
        [cx - w, cy - h],
        [cx + w, cy - h],
        [cx + w, cy + h],
        [cx - w, cy + h]
    ]
    return Polygon(verts, closed=True)


def rounded_rectangle(
    width: float = 1.0,
    height: float = 1.0,
    radius: float = 0.1,
    center: Tuple[float, float] = (0, 0),
    n_corner: int = 8
) -> Polygon:
    """
    Create rectangle with rounded corners.

    Parameters
    ----------
    width : float
        Rectangle width.
    height : float
        Rectangle height.
    radius : float
        Corner radius.
    center : tuple
        Center coordinates.
    n_corner : int
        Points per corner.

    Returns
    -------
    Polygon
        Rounded rectangle polygon.
    """
    w, h = width / 2 - radius, height / 2 - radius
    cx, cy = center

    # Corner centers
    corners = [
        (cx + w, cy + h, 0),           # top-right
        (cx - w, cy + h, np.pi / 2),   # top-left
        (cx - w, cy - h, np.pi),       # bottom-left
        (cx + w, cy - h, 3 * np.pi / 2)  # bottom-right
    ]

    points = []
    for ccx, ccy, start_angle in corners:
        theta = np.linspace(start_angle, start_angle + np.pi / 2, n_corner)
        x = ccx + radius * np.cos(theta)
        y = ccy + radius * np.sin(theta)
        points.extend(zip(x, y))

    return Polygon(points, closed=True)


def regular_polygon(n_sides: int, radius: float = 1.0, center: Tuple[float, float] = (0, 0)) -> Polygon:
    """
    Create regular polygon.

    Parameters
    ----------
    n_sides : int
        Number of sides.
    radius : float
        Circumradius.
    center : tuple
        Center coordinates.

    Returns
    -------
    Polygon
        Regular polygon.
    """
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return Polygon(np.column_stack([x, y]), closed=True)
