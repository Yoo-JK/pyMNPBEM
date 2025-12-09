"""
Polygon classes for geometry definition.

Provides 2D and 3D polygon handling for particle shape creation.
"""

import numpy as np
from typing import Optional, List, Tuple, Union


class Polygon:
    """
    2D polygon for boundary definition.

    Used for creating particle shapes via extrusion or revolution.

    Parameters
    ----------
    x : ndarray
        X-coordinates of vertices.
    y : ndarray
        Y-coordinates of vertices (or z for edge profiles).
    closed : bool
        Whether polygon is closed.

    Examples
    --------
    >>> poly = Polygon([0, 1, 1, 0], [0, 0, 1, 1])
    >>> poly.close()
    >>> area = poly.area()
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        closed: bool = False
    ):
        """Initialize polygon."""
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.closed = closed

        if len(self.x) != len(self.y):
            raise ValueError("x and y must have same length")

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.x)

    @property
    def vertices(self) -> np.ndarray:
        """Vertices as (n, 2) array."""
        return np.column_stack([self.x, self.y])

    def close(self) -> 'Polygon':
        """
        Close polygon by connecting last to first vertex.

        Returns
        -------
        Polygon
            Self for chaining.
        """
        if not self.closed:
            if not np.allclose([self.x[0], self.y[0]], [self.x[-1], self.y[-1]]):
                self.x = np.append(self.x, self.x[0])
                self.y = np.append(self.y, self.y[0])
            self.closed = True
        return self

    def area(self) -> float:
        """
        Compute polygon area using shoelace formula.

        Returns
        -------
        float
            Polygon area.
        """
        x, y = self.x, self.y
        n = len(x)

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]

        return abs(area) / 2.0

    def perimeter(self) -> float:
        """
        Compute polygon perimeter.

        Returns
        -------
        float
            Perimeter length.
        """
        x, y = self.x, self.y
        n = len(x)

        perim = 0.0
        for i in range(n - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            perim += np.sqrt(dx**2 + dy**2)

        if self.closed:
            dx = x[0] - x[-1]
            dy = y[0] - y[-1]
            perim += np.sqrt(dx**2 + dy**2)

        return perim

    def centroid(self) -> Tuple[float, float]:
        """
        Compute polygon centroid.

        Returns
        -------
        tuple
            (cx, cy) centroid coordinates.
        """
        x, y = self.x, self.y
        n = len(x)
        A = self.area()

        if A < 1e-10:
            return np.mean(x), np.mean(y)

        cx, cy = 0.0, 0.0
        for i in range(n):
            j = (i + 1) % n
            factor = x[i] * y[j] - x[j] * y[i]
            cx += (x[i] + x[j]) * factor
            cy += (y[i] + y[j]) * factor

        return cx / (6 * A), cy / (6 * A)

    def shift(self, dx: float, dy: float) -> 'Polygon':
        """
        Shift polygon by offset.

        Parameters
        ----------
        dx : float
            X-offset.
        dy : float
            Y-offset.

        Returns
        -------
        Polygon
            New shifted polygon.
        """
        return Polygon(self.x + dx, self.y + dy, self.closed)

    def scale(self, sx: float, sy: Optional[float] = None) -> 'Polygon':
        """
        Scale polygon.

        Parameters
        ----------
        sx : float
            X scale factor.
        sy : float, optional
            Y scale factor (default: same as sx).

        Returns
        -------
        Polygon
            New scaled polygon.
        """
        if sy is None:
            sy = sx
        return Polygon(self.x * sx, self.y * sy, self.closed)

    def rotate(self, angle: float, center: Optional[Tuple[float, float]] = None) -> 'Polygon':
        """
        Rotate polygon.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        center : tuple, optional
            Center of rotation.

        Returns
        -------
        Polygon
            New rotated polygon.
        """
        if center is None:
            center = self.centroid()

        cx, cy = center
        x = self.x - cx
        y = self.y - cy

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        x_new = x * cos_a - y * sin_a + cx
        y_new = x * sin_a + y * cos_a + cy

        return Polygon(x_new, y_new, self.closed)

    def flip(self, axis: str = 'x') -> 'Polygon':
        """
        Flip polygon about axis.

        Parameters
        ----------
        axis : str
            'x' or 'y'.

        Returns
        -------
        Polygon
            New flipped polygon.
        """
        if axis == 'x':
            return Polygon(-self.x, self.y, self.closed)
        else:
            return Polygon(self.x, -self.y, self.closed)

    def reverse(self) -> 'Polygon':
        """Reverse vertex order."""
        return Polygon(self.x[::-1], self.y[::-1], self.closed)

    def dist(self, point: Tuple[float, float]) -> float:
        """
        Compute minimum distance to polygon boundary.

        Parameters
        ----------
        point : tuple
            (x, y) point.

        Returns
        -------
        float
            Minimum distance.
        """
        px, py = point
        min_dist = np.inf

        for i in range(len(self.x) - 1):
            dist = self._point_segment_dist(
                px, py,
                self.x[i], self.y[i],
                self.x[i + 1], self.y[i + 1]
            )
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_segment_dist(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> float:
        """Compute distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def midpoints(self, n: Optional[int] = None) -> np.ndarray:
        """
        Get midpoints of edges.

        Parameters
        ----------
        n : int, optional
            Number of points per edge.

        Returns
        -------
        ndarray
            Midpoint coordinates (n, 2).
        """
        if n is None:
            n = 1

        midpts = []
        for i in range(len(self.x) - 1):
            for j in range(n):
                t = (j + 0.5) / n
                mx = self.x[i] + t * (self.x[i + 1] - self.x[i])
                my = self.y[i] + t * (self.y[i + 1] - self.y[i])
                midpts.append([mx, my])

        return np.array(midpts)

    def interp(self, n: int) -> 'Polygon':
        """
        Interpolate polygon to have n points.

        Parameters
        ----------
        n : int
            Number of output points.

        Returns
        -------
        Polygon
            Interpolated polygon.
        """
        # Compute arc length
        s = np.zeros(len(self.x))
        for i in range(1, len(self.x)):
            dx = self.x[i] - self.x[i - 1]
            dy = self.y[i] - self.y[i - 1]
            s[i] = s[i - 1] + np.sqrt(dx**2 + dy**2)

        # Interpolate
        s_new = np.linspace(0, s[-1], n)
        x_new = np.interp(s_new, s, self.x)
        y_new = np.interp(s_new, s, self.y)

        return Polygon(x_new, y_new, self.closed)

    def plot(self, ax=None, **kwargs):
        """Plot polygon."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        x_plot = np.append(self.x, self.x[0]) if self.closed else self.x
        y_plot = np.append(self.y, self.y[0]) if self.closed else self.y

        ax.plot(x_plot, y_plot, **kwargs)
        ax.set_aspect('equal')

        return ax

    def __repr__(self) -> str:
        return f"Polygon(n_vertices={self.n_vertices}, closed={self.closed})"


class EdgeProfile:
    """
    Edge profile for particle shape definition.

    Defines a 2D profile (r, z) that can be revolved or extruded
    to create 3D particle shapes.

    Parameters
    ----------
    r : ndarray
        Radial coordinates.
    z : ndarray
        Vertical coordinates.

    Examples
    --------
    >>> # Create rounded edge profile
    >>> profile = EdgeProfile.rounded(height=10, radius=2)
    """

    def __init__(self, r: np.ndarray, z: np.ndarray):
        """Initialize edge profile."""
        self.r = np.asarray(r, dtype=float)
        self.z = np.asarray(z, dtype=float)

    @property
    def n_points(self) -> int:
        """Number of profile points."""
        return len(self.r)

    @classmethod
    def rounded(cls, height: float, radius: float, n: int = 20) -> 'EdgeProfile':
        """
        Create rounded edge profile.

        Parameters
        ----------
        height : float
            Profile height.
        radius : float
            Edge radius.
        n : int
            Number of points.

        Returns
        -------
        EdgeProfile
            Rounded profile.
        """
        # Quarter circle at top
        theta = np.linspace(np.pi / 2, 0, n // 2)
        r_top = radius * np.cos(theta)
        z_top = height / 2 - radius + radius * np.sin(theta)

        # Straight section
        r_mid = np.array([radius, radius])
        z_mid = np.array([height / 2 - radius, -height / 2 + radius])

        # Quarter circle at bottom
        theta = np.linspace(0, -np.pi / 2, n // 2)
        r_bot = radius * np.cos(theta)
        z_bot = -height / 2 + radius + radius * np.sin(theta)

        r = np.concatenate([r_top, r_mid, r_bot])
        z = np.concatenate([z_top, z_mid, z_bot])

        return cls(r, z)

    @classmethod
    def chamfered(cls, height: float, chamfer: float, n: int = 10) -> 'EdgeProfile':
        """
        Create chamfered edge profile.

        Parameters
        ----------
        height : float
            Profile height.
        chamfer : float
            Chamfer size.
        n : int
            Number of points.

        Returns
        -------
        EdgeProfile
            Chamfered profile.
        """
        r = np.array([0, chamfer, chamfer, 0])
        z = np.array([height / 2, height / 2 - chamfer,
                     -height / 2 + chamfer, -height / 2])

        return cls(r, z)

    def hshift(self, offset: float) -> 'EdgeProfile':
        """
        Horizontal (radial) shift.

        Parameters
        ----------
        offset : float
            Radial offset.

        Returns
        -------
        EdgeProfile
            Shifted profile.
        """
        return EdgeProfile(self.r + offset, self.z)

    def vshift(self, offset: float) -> 'EdgeProfile':
        """
        Vertical shift.

        Parameters
        ----------
        offset : float
            Vertical offset.

        Returns
        -------
        EdgeProfile
            Shifted profile.
        """
        return EdgeProfile(self.r, self.z + offset)

    def scale(self, sr: float, sz: Optional[float] = None) -> 'EdgeProfile':
        """
        Scale profile.

        Parameters
        ----------
        sr : float
            Radial scale.
        sz : float, optional
            Vertical scale (default: same as sr).

        Returns
        -------
        EdgeProfile
            Scaled profile.
        """
        if sz is None:
            sz = sr
        return EdgeProfile(self.r * sr, self.z * sz)

    def flip(self) -> 'EdgeProfile':
        """Flip profile vertically."""
        return EdgeProfile(self.r, -self.z)

    def to_polygon(self) -> Polygon:
        """Convert to Polygon."""
        return Polygon(self.r, self.z)

    def __repr__(self) -> str:
        return f"EdgeProfile(n_points={self.n_points})"


class Polygon3:
    """
    3D polygon for complex particle geometries.

    Represents a 3D polygon that can be used as a base
    for extrusion or for direct mesh generation.

    Parameters
    ----------
    vertices : ndarray
        3D vertices (n, 3).
    faces : ndarray, optional
        Face connectivity.
    closed : bool
        Whether polygon is closed.

    Examples
    --------
    >>> # Create hexagonal base
    >>> angles = np.linspace(0, 2*np.pi, 7)[:-1]
    >>> verts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    >>> poly3 = Polygon3(verts)
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
        closed: bool = True
    ):
        """Initialize 3D polygon."""
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = faces
        self.closed = closed

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must be (n, 3) array")

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @classmethod
    def from_2d(cls, poly: Polygon, z: float = 0) -> 'Polygon3':
        """
        Create 3D polygon from 2D polygon.

        Parameters
        ----------
        poly : Polygon
            2D polygon.
        z : float
            Z-coordinate.

        Returns
        -------
        Polygon3
            3D polygon.
        """
        n = poly.n_vertices
        verts = np.column_stack([poly.x, poly.y, np.full(n, z)])
        return cls(verts, closed=poly.closed)

    @classmethod
    def plate(cls, size: Tuple[float, float], center: Tuple[float, float, float] = (0, 0, 0)) -> 'Polygon3':
        """
        Create rectangular plate.

        Parameters
        ----------
        size : tuple
            (width, height) of plate.
        center : tuple
            Center position.

        Returns
        -------
        Polygon3
            Rectangular polygon.
        """
        w, h = size
        cx, cy, cz = center

        verts = np.array([
            [cx - w / 2, cy - h / 2, cz],
            [cx + w / 2, cy - h / 2, cz],
            [cx + w / 2, cy + h / 2, cz],
            [cx - w / 2, cy + h / 2, cz]
        ])

        return cls(verts)

    @classmethod
    def hribbon(cls, width: float, length: float, z: float = 0) -> 'Polygon3':
        """
        Create horizontal ribbon.

        Parameters
        ----------
        width : float
            Ribbon width.
        length : float
            Ribbon length.
        z : float
            Z-position.

        Returns
        -------
        Polygon3
            Ribbon polygon.
        """
        return cls.plate((width, length), (0, 0, z))

    @classmethod
    def vribbon(cls, width: float, height: float, y: float = 0) -> 'Polygon3':
        """
        Create vertical ribbon.

        Parameters
        ----------
        width : float
            Ribbon width.
        height : float
            Ribbon height.
        y : float
            Y-position.

        Returns
        -------
        Polygon3
            Ribbon polygon.
        """
        verts = np.array([
            [-width / 2, y, -height / 2],
            [width / 2, y, -height / 2],
            [width / 2, y, height / 2],
            [-width / 2, y, height / 2]
        ])
        return cls(verts)

    def shift(self, offset: Union[float, Tuple[float, float, float]]) -> 'Polygon3':
        """
        Shift polygon.

        Parameters
        ----------
        offset : float or tuple
            Shift offset.

        Returns
        -------
        Polygon3
            Shifted polygon.
        """
        if isinstance(offset, (int, float)):
            offset = (0, 0, offset)
        return Polygon3(self.vertices + np.array(offset), self.faces, self.closed)

    def scale(self, factor: Union[float, Tuple[float, float, float]]) -> 'Polygon3':
        """
        Scale polygon.

        Parameters
        ----------
        factor : float or tuple
            Scale factor(s).

        Returns
        -------
        Polygon3
            Scaled polygon.
        """
        if isinstance(factor, (int, float)):
            factor = (factor, factor, factor)
        return Polygon3(self.vertices * np.array(factor), self.faces, self.closed)

    def rotate(self, angle: float, axis: str = 'z') -> 'Polygon3':
        """
        Rotate polygon.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : str
            Rotation axis ('x', 'y', or 'z').

        Returns
        -------
        Polygon3
            Rotated polygon.
        """
        c, s = np.cos(angle), np.sin(angle)

        if axis == 'z':
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif axis == 'y':
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # x
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        return Polygon3(self.vertices @ R.T, self.faces, self.closed)

    def flip(self, axis: str = 'z') -> 'Polygon3':
        """
        Flip polygon.

        Parameters
        ----------
        axis : str
            Flip axis.

        Returns
        -------
        Polygon3
            Flipped polygon.
        """
        new_verts = self.vertices.copy()
        if axis == 'x':
            new_verts[:, 0] *= -1
        elif axis == 'y':
            new_verts[:, 1] *= -1
        else:
            new_verts[:, 2] *= -1
        return Polygon3(new_verts, self.faces, self.closed)

    def shiftbnd(self, amount: float) -> 'Polygon3':
        """
        Shift boundary inward/outward.

        Parameters
        ----------
        amount : float
            Shift amount (positive = outward).

        Returns
        -------
        Polygon3
            Shifted polygon.
        """
        # Compute normal for each vertex
        normals = np.zeros_like(self.vertices)

        for i in range(len(self.vertices)):
            i_prev = (i - 1) % len(self.vertices)
            i_next = (i + 1) % len(self.vertices)

            v1 = self.vertices[i] - self.vertices[i_prev]
            v2 = self.vertices[i_next] - self.vertices[i]

            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n /= n_norm

            normals[i] = n

        return Polygon3(self.vertices + amount * normals, self.faces, self.closed)

    def __repr__(self) -> str:
        return f"Polygon3(n_vertices={self.n_vertices})"
