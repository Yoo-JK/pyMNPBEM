"""
Discretized particle surface class.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field

from ..misc.helpers import vecnorm, vecnormalize


@dataclass
class QuadFace:
    """Quadrature rules for boundary element integration."""
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


class Particle:
    """
    Faces and vertices of a discretized particle surface.

    The particle faces can be either triangles or quadrilaterals, or both.

    Parameters
    ----------
    verts : ndarray
        Vertices of boundary elements, shape (n_verts, 3).
    faces : ndarray
        Faces of boundary elements, shape (n_faces, 3) or (n_faces, 4).
        For triangles, use 3 columns. For quads, use 4 columns.
        Use NaN to indicate missing vertices for mixed meshes.
    interp : str, optional
        Interpolation type: 'flat' or 'curv'. Default is 'flat'.

    Attributes
    ----------
    verts : ndarray
        Vertices of the particle, shape (n_verts, 3).
    faces : ndarray
        Face indices, shape (n_faces, 4). Triangles have NaN in 4th column.
    pos : ndarray
        Centroids of faces, shape (n_faces, 3).
    nvec : ndarray
        Normal vectors at centroids, shape (n_faces, 3).
    tvec1, tvec2 : ndarray
        Tangent vectors at centroids, shape (n_faces, 3).
    area : ndarray
        Area of faces, shape (n_faces,).
    """

    def __init__(
        self,
        verts: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        interp: str = 'flat',
        compute_normals: bool = True,
        **kwargs
    ):
        """
        Initialize particle surface.

        Parameters
        ----------
        verts : ndarray, optional
            Vertices of boundary elements.
        faces : ndarray, optional
            Faces of boundary elements.
        interp : str, optional
            'flat' or 'curv' particle boundaries.
        compute_normals : bool, optional
            Whether to compute normal vectors. Default True.
        """
        self.interp = interp
        self.verts2 = None
        self.faces2 = None

        if verts is None or faces is None:
            self.verts = np.array([]).reshape(0, 3)
            self.faces = np.array([]).reshape(0, 4)
            self.pos = np.array([]).reshape(0, 3)
            self.nvec = np.array([]).reshape(0, 3)
            self.tvec1 = np.array([]).reshape(0, 3)
            self.tvec2 = np.array([]).reshape(0, 3)
            self.area = np.array([])
            return

        self.verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces)

        # Ensure faces has 4 columns (pad with NaN for triangles)
        if faces.ndim == 1:
            faces = faces.reshape(1, -1)
        if faces.shape[1] == 3:
            self.faces = np.column_stack([faces, np.full(len(faces), np.nan)])
        else:
            self.faces = faces.astype(float)

        # Compute auxiliary information
        if compute_normals:
            self._compute_normals()
        else:
            n = len(self.faces)
            self.pos = np.zeros((n, 3))
            self.nvec = np.zeros((n, 3))
            self.tvec1 = np.zeros((n, 3))
            self.tvec2 = np.zeros((n, 3))
            self.area = np.zeros(n)

    def _compute_normals(self) -> None:
        """Compute face centroids, normals, tangents, and areas."""
        n_faces = len(self.faces)
        self.pos = np.zeros((n_faces, 3))
        self.nvec = np.zeros((n_faces, 3))
        self.tvec1 = np.zeros((n_faces, 3))
        self.tvec2 = np.zeros((n_faces, 3))
        self.area = np.zeros(n_faces)

        for i, face in enumerate(self.faces):
            # Get valid vertex indices
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            vertices = self.verts[indices]

            # Check if face uses curved format [v0, m01, v1, m12, v2, m20, ...]
            # where even indices are corners and odd indices are midpoints
            # A curved triangle has 6 vertices, curved quad has 8
            is_curved = len(indices) >= 6

            if is_curved:
                # Extract corner vertices (even indices)
                corner_indices = indices[::2]
                corner_verts = self.verts[corner_indices]
                # Centroid of corners only
                self.pos[i] = corner_verts.mean(axis=0)
            else:
                # Flat face - use all vertices for centroid
                self.pos[i] = vertices.mean(axis=0)

            if len(indices) >= 3:
                if is_curved:
                    # Use corner vertices for normal computation
                    v0, v1, v2 = corner_verts[0], corner_verts[1], corner_verts[2]
                else:
                    # Use first 3 vertices
                    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

                e1 = v1 - v0
                e2 = v2 - v0

                # Normal vector (cross product)
                normal = np.cross(e1, e2)
                norm_mag = np.linalg.norm(normal)

                if norm_mag > 1e-12:
                    self.nvec[i] = normal / norm_mag

                    # Tangent vectors
                    self.tvec1[i] = vecnormalize(e1)
                    self.tvec2[i] = np.cross(self.nvec[i], self.tvec1[i])

                # Area computation
                if is_curved:
                    # For curved triangles, compute area using corners
                    n_corners = len(corner_indices)
                    if n_corners == 3:
                        self.area[i] = 0.5 * norm_mag
                    elif n_corners >= 4:
                        v3 = corner_verts[3]
                        e3 = v3 - v0
                        area1 = 0.5 * np.linalg.norm(np.cross(e1, e2))
                        area2 = 0.5 * np.linalg.norm(np.cross(e2, e3))
                        self.area[i] = area1 + area2
                elif len(indices) == 3:
                    # Triangle area = 0.5 * |e1 x e2|
                    self.area[i] = 0.5 * norm_mag
                else:
                    # Quadrilateral: sum of two triangles
                    v3 = vertices[3]
                    e3 = v3 - v0
                    area1 = 0.5 * np.linalg.norm(np.cross(e1, e2))
                    area2 = 0.5 * np.linalg.norm(np.cross(e2, e3))
                    self.area[i] = area1 + area2

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    @property
    def n_verts(self) -> int:
        """Number of vertices."""
        return len(self.verts)

    def is_triangle(self) -> np.ndarray:
        """Return boolean array indicating which faces are triangles."""
        return np.isnan(self.faces[:, 3])

    def is_quad(self) -> np.ndarray:
        """Return boolean array indicating which faces are quadrilaterals."""
        return ~np.isnan(self.faces[:, 3])

    def shift(self, displacement: np.ndarray) -> 'Particle':
        """
        Shift particle by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        Particle
            Shifted particle (new object).
        """
        displacement = np.asarray(displacement)
        new_verts = self.verts + displacement
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 + displacement
            p.faces2 = self.faces2.copy()
        return p

    def scale(self, factor: Union[float, np.ndarray]) -> 'Particle':
        """
        Scale particle by given factor.

        Parameters
        ----------
        factor : float or array_like
            Scale factor. Can be scalar or [sx, sy, sz].

        Returns
        -------
        Particle
            Scaled particle (new object).
        """
        factor = np.asarray(factor)
        new_verts = self.verts * factor
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 * factor
            p.faces2 = self.faces2.copy()
        return p

    def rot(self, angle: float, axis: int = 2) -> 'Particle':
        """
        Rotate particle around specified axis.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : int
            Axis of rotation: 0=x, 1=y, 2=z.

        Returns
        -------
        Particle
            Rotated particle (new object).
        """
        c, s = np.cos(angle), np.sin(angle)

        if axis == 0:  # x-axis
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:  # y-axis
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # z-axis
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        new_verts = self.verts @ R.T
        p = Particle(new_verts, self.faces.copy(), self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2 @ R.T
            p.faces2 = self.faces2.copy()
        return p

    def flip(self, axis: int) -> 'Particle':
        """
        Flip particle along specified axis.

        Parameters
        ----------
        axis : int
            Axis to flip: 0=x, 1=y, 2=z.

        Returns
        -------
        Particle
            Flipped particle (new object).
        """
        new_verts = self.verts.copy()
        new_verts[:, axis] = -new_verts[:, axis]

        # Reverse face winding to maintain outward normals
        new_faces = self.faces.copy()
        new_faces[:, :3] = new_faces[:, [0, 2, 1]]

        p = Particle(new_verts, new_faces, self.interp)
        if self.verts2 is not None:
            p.verts2 = self.verts2.copy()
            p.verts2[:, axis] = -p.verts2[:, axis]
            p.faces2 = self.faces2.copy()
        return p

    def flipfaces(self) -> 'Particle':
        """
        Flip face orientations (reverse normals).

        Returns
        -------
        Particle
            Particle with flipped faces.
        """
        new_faces = self.faces.copy()

        for i in range(len(new_faces)):
            valid = ~np.isnan(new_faces[i])
            n_valid = np.sum(valid)

            if n_valid >= 6:
                # Curved face format: [v0, m01, v1, m12, v2, m20, ...]
                # To flip: reverse the corner sequence while adjusting midpoints
                indices = new_faces[i][valid].copy()
                n_corners = n_valid // 2

                # Extract corners and midpoints
                corners = indices[::2]  # v0, v1, v2, ...
                midpoints = indices[1::2]  # m01, m12, m20, ...

                # Reverse corners (keep v0 first, reverse the rest)
                # v0, v1, v2 -> v0, v2, v1
                new_corners = np.concatenate([[corners[0]], corners[1:][::-1]])

                # Reverse midpoints: m01, m12, m20 -> m20, m12, m01 -> shift to m02, m21, m10
                # Actually for v0->v2->v1->v0: midpoints are m02, m21, m10 = m20, m12, m01 reversed
                new_midpoints = midpoints[::-1]

                # Interleave: [v0, m02, v2, m21, v1, m10]
                new_indices = np.zeros_like(indices)
                new_indices[::2] = new_corners
                new_indices[1::2] = new_midpoints

                new_faces[i][valid] = new_indices
            elif n_valid >= 3:
                # Flat face: swap positions 1 and 2 (v0, v1, v2 -> v0, v2, v1)
                indices = new_faces[i][valid].copy()
                if n_valid == 3:
                    indices[1], indices[2] = indices[2], indices[1]
                elif n_valid == 4:
                    # Quad: reverse order but keep v0 first: v0,v1,v2,v3 -> v0,v3,v2,v1
                    indices[1:] = indices[1:][::-1]
                new_faces[i][valid] = indices

        return Particle(self.verts.copy(), new_faces, self.interp)

    def orient_normals(self, outward: bool = True) -> 'Particle':
        """
        Orient normals consistently outward or inward for closed surfaces.

        Checks each face's normal against the direction from origin to face
        centroid and flips if needed.

        Parameters
        ----------
        outward : bool
            If True, orient normals outward. If False, orient inward.

        Returns
        -------
        Particle
            Particle with consistently oriented normals.
        """
        # Create a copy of the particle
        p = Particle(self.verts.copy(), self.faces.copy(), self.interp,
                     compute_normals=False)
        p.pos = self.pos.copy()
        p.nvec = self.nvec.copy()
        p.tvec1 = self.tvec1.copy()
        p.tvec2 = self.tvec2.copy()
        p.area = self.area.copy()

        # For each face, check if normal points in desired direction
        # relative to centroid direction
        for i in range(len(p.faces)):
            pos_dir = p.pos[i]
            pos_norm = np.linalg.norm(pos_dir)
            if pos_norm > 1e-12:
                pos_dir = pos_dir / pos_norm

            # Check if normal is aligned with position direction
            dot = np.dot(p.nvec[i], pos_dir)

            # If outward and dot < 0, or inward and dot > 0, flip normal
            if (outward and dot < 0) or (not outward and dot > 0):
                p.nvec[i] = -p.nvec[i]
                # Also flip tangent to maintain right-hand rule
                p.tvec2[i] = -p.tvec2[i]

        return p

    def clean(self, tol: float = 1e-10) -> 'Particle':
        """
        Remove duplicate vertices and update face indices.

        Parameters
        ----------
        tol : float
            Tolerance for considering vertices as duplicates.

        Returns
        -------
        Particle
            Cleaned particle.
        """
        from scipy.spatial import cKDTree

        # Find unique vertices
        tree = cKDTree(self.verts)
        groups = tree.query_ball_tree(tree, tol)

        # Map each vertex to its representative (first in group)
        vertex_map = np.arange(len(self.verts))
        for group in groups:
            if len(group) > 1:
                rep = min(group)
                for idx in group:
                    vertex_map[idx] = rep

        # Get unique vertices
        unique_mask = np.array([vertex_map[i] == i for i in range(len(self.verts))])
        unique_verts = self.verts[unique_mask]

        # Create mapping from old to new indices
        old_to_new = np.zeros(len(self.verts), dtype=int)
        new_idx = 0
        for i in range(len(self.verts)):
            if unique_mask[i]:
                old_to_new[i] = new_idx
                new_idx += 1
            else:
                old_to_new[i] = old_to_new[vertex_map[i]]

        # Update face indices
        new_faces = self.faces.copy()
        valid = ~np.isnan(new_faces)
        new_faces[valid] = old_to_new[new_faces[valid].astype(int)]

        return Particle(unique_verts, new_faces, self.interp)

    def midpoints(self, mode: str = 'flat') -> 'Particle':
        """
        Add midpoints to edges for curved interpolation.

        Parameters
        ----------
        mode : str
            'flat' or 'curv'.

        Returns
        -------
        Particle
            Particle with midpoint vertices added.
        """
        # For triangular faces, add midpoints on each edge
        edges = []
        edge_midpoints = {}

        # Collect all edges
        for i, face in enumerate(self.faces):
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)
            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                if edge not in edge_midpoints:
                    midpoint = 0.5 * (self.verts[v1] + self.verts[v2])
                    edge_midpoints[edge] = len(self.verts) + len(edge_midpoints)
                    edges.append(midpoint)

        if len(edges) == 0:
            return self

        # Create extended vertex list
        new_verts = np.vstack([self.verts, np.array(edges)])

        # Create extended faces with midpoint indices
        # faces2 format: [v0, v01, v1, v12, v2, v20, v3, v30, ...]
        new_faces = []
        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)
            new_face = []
            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                mid_idx = edge_midpoints[edge]
                new_face.extend([indices[j], mid_idx])
            # Pad to uniform length
            while len(new_face) < 8:
                new_face.append(np.nan)
            new_faces.append(new_face[:8])

        p = Particle(self.verts.copy(), self.faces.copy(), self.interp, compute_normals=False)
        p.verts2 = new_verts
        p.faces2 = np.array(new_faces)

        return p

    def curved(self, **kwargs) -> 'Particle':
        """
        Make curved particle boundary.

        Curved boundaries use higher-order interpolation with midpoint
        vertices for more accurate representation of curved surfaces.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to midpoints().

        Returns
        -------
        Particle
            Particle with curved interpolation enabled.

        Notes
        -----
        This method adds midpoint vertices to edges (if not already present)
        and sets the interpolation mode to 'curv'.

        Examples
        --------
        >>> sphere = trisphere(144, 10)
        >>> sphere_curved = sphere.curved()
        """
        # Add midpoints if not already present
        if self.verts2 is None or len(kwargs) > 0:
            result = self.midpoints(**kwargs)
        else:
            result = Particle(self.verts.copy(), self.faces.copy(), self.interp)
            result.verts2 = self.verts2.copy() if self.verts2 is not None else None
            result.faces2 = self.faces2.copy() if self.faces2 is not None else None

        # Set curved interpolation flag
        result.interp = 'curv'

        # Recompute normals for curved surface
        result._compute_normals()

        return result

    def flat(self) -> 'Particle':
        """
        Make flat particle boundary.

        Flat boundaries use linear interpolation between vertices,
        which is faster but less accurate for curved surfaces.

        Returns
        -------
        Particle
            Particle with flat interpolation enabled.

        Notes
        -----
        This method sets the interpolation mode to 'flat'. Midpoint
        vertices (if present) are preserved but not used.

        Examples
        --------
        >>> sphere_curved = trisphere(144, 10).curved()
        >>> sphere_flat = sphere_curved.flat()
        """
        result = Particle(self.verts.copy(), self.faces.copy(), 'flat')
        result.verts2 = self.verts2.copy() if self.verts2 is not None else None
        result.faces2 = self.faces2.copy() if self.faces2 is not None else None

        # Recompute normals for flat surface
        result._compute_normals()

        return result

    def curvature(self) -> np.ndarray:
        """
        Compute mean curvature at face centroids.

        Returns
        -------
        ndarray
            Mean curvature at each face.
        """
        # Simplified curvature estimation using discrete differential geometry
        curv = np.zeros(self.n_faces)

        # Build vertex-face adjacency
        vertex_faces = [[] for _ in range(self.n_verts)]
        for i, face in enumerate(self.faces):
            valid = ~np.isnan(face)
            for idx in face[valid].astype(int):
                vertex_faces[idx].append(i)

        # Estimate curvature from normal variation
        for i in range(self.n_faces):
            valid = ~np.isnan(self.faces[i])
            indices = self.faces[i][valid].astype(int)

            # Get neighboring faces
            neighbor_faces = set()
            for idx in indices:
                neighbor_faces.update(vertex_faces[idx])
            neighbor_faces.discard(i)

            if len(neighbor_faces) > 0:
                neighbor_normals = self.nvec[list(neighbor_faces)]
                # Curvature from normal deviation
                dot_products = np.dot(neighbor_normals, self.nvec[i])
                curv[i] = np.arccos(np.clip(dot_products, -1, 1)).mean()

        return curv

    def edges(self) -> np.ndarray:
        """
        Extract all edges from the mesh.

        Returns
        -------
        ndarray
            Edge list, shape (n_edges, 2). Each row contains two vertex indices.
        """
        edges = set()

        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)

            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                edges.add(edge)

        return np.array(list(edges))

    def border(self) -> np.ndarray:
        """
        Find boundary edges (edges that belong to only one face).

        Returns
        -------
        ndarray
            Boundary edge list, shape (n_boundary_edges, 2).
        """
        from collections import Counter

        edge_count = Counter()

        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            n = len(indices)

            for j in range(n):
                v1, v2 = indices[j], indices[(j + 1) % n]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] += 1

        # Boundary edges have count == 1
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        return np.array(boundary_edges) if boundary_edges else np.array([]).reshape(0, 2)

    def quad(self, order: int = 3) -> QuadFace:
        """
        Get quadrature rules for face integration.

        Parameters
        ----------
        order : int
            Quadrature order (number of points).

        Returns
        -------
        QuadFace
            Quadrature points and weights.
        """
        # Gaussian quadrature for triangles
        if order == 1:
            # Centroid rule
            points = np.array([[1/3, 1/3, 1/3]])
            weights = np.array([1.0])
        elif order == 3:
            # 3-point rule
            points = np.array([
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ])
            weights = np.array([1/3, 1/3, 1/3])
        elif order == 4:
            # 4-point rule
            points = np.array([
                [1/3, 1/3, 1/3],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6]
            ])
            weights = np.array([-27/48, 25/48, 25/48, 25/48])
        else:
            # Default to 7-point rule
            points = np.array([
                [1/3, 1/3, 1/3],
                [0.059715871789770, 0.470142064105115, 0.470142064105115],
                [0.470142064105115, 0.059715871789770, 0.470142064105115],
                [0.470142064105115, 0.470142064105115, 0.059715871789770],
                [0.797426985353087, 0.101286507323456, 0.101286507323456],
                [0.101286507323456, 0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.101286507323456, 0.797426985353087]
            ])
            weights = np.array([0.225, 0.132394152788506, 0.132394152788506,
                               0.132394152788506, 0.125939180544827, 0.125939180544827,
                               0.125939180544827])

        return QuadFace(points=points, weights=weights)

    def quadpol(self, face_idx: int, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quadrature points and weights for a specific face.

        Parameters
        ----------
        face_idx : int
            Face index.
        order : int
            Quadrature order.

        Returns
        -------
        points : ndarray
            Quadrature points in 3D space, shape (n_pts, 3).
        weights : ndarray
            Quadrature weights multiplied by face area.
        """
        quad = self.quad(order)
        face = self.faces[face_idx]
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)
        vertices = self.verts[indices]

        if len(indices) == 3:
            # Triangle: barycentric interpolation
            points = quad.points @ vertices
            weights = quad.weights * self.area[face_idx]
        else:
            # Quadrilateral: bilinear interpolation
            # Split into two triangles
            v0, v1, v2, v3 = vertices
            tri1 = np.array([v0, v1, v2])
            tri2 = np.array([v0, v2, v3])

            pts1 = quad.points @ tri1
            pts2 = quad.points @ tri2

            points = np.vstack([pts1, pts2])
            weights = np.concatenate([quad.weights, quad.weights]) * self.area[face_idx] / 2

        return points, weights

    def totriangles(self) -> 'Particle':
        """
        Convert all quadrilateral faces to triangles.

        Returns
        -------
        Particle
            Particle with only triangular faces.
        """
        new_faces = []

        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)

            if len(indices) == 3:
                new_faces.append(indices)
            elif len(indices) == 4:
                # Split quad into two triangles
                new_faces.append([indices[0], indices[1], indices[2]])
                new_faces.append([indices[0], indices[2], indices[3]])

        new_faces = np.array(new_faces)
        return Particle(self.verts.copy(), new_faces, self.interp)

    def index34(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get separate indices for triangular and quadrilateral faces.

        Returns
        -------
        tri_idx : ndarray
            Indices of triangular faces.
        quad_idx : ndarray
            Indices of quadrilateral faces.
        """
        is_tri = self.is_triangle()
        tri_idx = np.where(is_tri)[0]
        quad_idx = np.where(~is_tri)[0]
        return tri_idx, quad_idx

    def select(self, face_indices: np.ndarray) -> 'Particle':
        """
        Select a subset of faces.

        Parameters
        ----------
        face_indices : ndarray
            Indices of faces to keep.

        Returns
        -------
        Particle
            New particle with selected faces.
        """
        face_indices = np.asarray(face_indices)
        new_faces = self.faces[face_indices]

        # Find used vertices
        used_verts = set()
        for face in new_faces:
            valid = ~np.isnan(face)
            used_verts.update(face[valid].astype(int))

        used_verts = sorted(list(used_verts))

        # Create vertex mapping
        vert_map = {old: new for new, old in enumerate(used_verts)}

        # Update face indices
        updated_faces = new_faces.copy()
        for i, face in enumerate(updated_faces):
            valid = ~np.isnan(face)
            updated_faces[i, valid] = [vert_map[int(v)] for v in face[valid]]

        new_verts = self.verts[used_verts]

        return Particle(new_verts, updated_faces, self.interp)

    def plot(self, ax=None, **kwargs):
        """
        Plot the particle surface.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs : dict
            Keyword arguments for plot3D or plot_trisurf.

        Returns
        -------
        ax : matplotlib axes
            The axes object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Get default kwargs
        color = kwargs.pop('color', 'gold')
        alpha = kwargs.pop('alpha', 0.7)
        edgecolor = kwargs.pop('edgecolor', 'black')
        linewidth = kwargs.pop('linewidth', 0.5)

        # Create polygon collection
        polygons = []
        for face in self.faces:
            valid = ~np.isnan(face)
            indices = face[valid].astype(int)
            vertices = self.verts[indices]
            polygons.append(vertices)

        collection = Poly3DCollection(polygons, alpha=alpha, facecolor=color,
                                       edgecolor=edgecolor, linewidth=linewidth, **kwargs)
        ax.add_collection3d(collection)

        # Set axis limits
        all_verts = self.verts
        max_range = np.max([
            all_verts[:, 0].max() - all_verts[:, 0].min(),
            all_verts[:, 1].max() - all_verts[:, 1].min(),
            all_verts[:, 2].max() - all_verts[:, 2].min()
        ]) / 2

        mid_x = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
        mid_y = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
        mid_z = (all_verts[:, 2].max() + all_verts[:, 2].min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')

        return ax

    def plot2(self, ax=None, **kwargs):
        """
        Alternative plot using triangulation.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        ax : matplotlib axes
            The axes object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Convert to triangles for plot_trisurf
        p_tri = self.totriangles()

        color = kwargs.pop('color', 'gold')
        alpha = kwargs.pop('alpha', 0.7)

        ax.plot_trisurf(p_tri.verts[:, 0], p_tri.verts[:, 1], p_tri.verts[:, 2],
                        triangles=p_tri.faces[:, :3].astype(int),
                        color=color, alpha=alpha, **kwargs)

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')

        return ax

    def flip(self) -> 'Particle':
        """
        Flip normal vectors by reversing face vertex order.

        Returns
        -------
        Particle
            Particle with flipped normals.
        """
        new_faces = self.faces.copy()

        for i in range(len(new_faces)):
            valid = ~np.isnan(new_faces[i])
            indices = new_faces[i][valid]
            # Reverse order (keeping same shape)
            new_faces[i][valid] = indices[::-1]

        p = Particle(self.verts.copy(), new_faces, self.interp)
        return p

    def clean(self, tol: float = 1e-10) -> 'Particle':
        """
        Clean mesh by removing duplicate vertices and degenerate faces.

        Parameters
        ----------
        tol : float
            Tolerance for identifying duplicate vertices.

        Returns
        -------
        Particle
            Cleaned particle.
        """
        from scipy.spatial import cKDTree

        # Find unique vertices
        tree = cKDTree(self.verts)
        pairs = tree.query_pairs(tol)

        # Build mapping from old to new indices
        mapping = np.arange(self.n_verts)
        for i, j in pairs:
            # Map higher index to lower
            if i < j:
                mapping[j] = i
            else:
                mapping[i] = j

        # Compress mapping to contiguous indices
        unique_verts, inverse = np.unique(mapping, return_inverse=True)
        new_mapping = inverse

        # Apply mapping to faces
        new_faces = self.faces.copy()
        valid = ~np.isnan(new_faces)
        new_faces[valid] = new_mapping[new_faces[valid].astype(int)]

        # Remove degenerate faces (faces with repeated vertices)
        good_faces = []
        for face in new_faces:
            valid_mask = ~np.isnan(face)
            indices = face[valid_mask].astype(int)
            if len(np.unique(indices)) == len(indices):
                good_faces.append(face)

        if len(good_faces) == 0:
            return Particle()

        new_faces = np.array(good_faces)
        new_verts = self.verts[unique_verts]

        return Particle(new_verts, new_faces, self.interp)

    def shift(self, offset: np.ndarray) -> 'Particle':
        """
        Shift particle by offset vector.

        Parameters
        ----------
        offset : array_like
            Translation vector (3,).

        Returns
        -------
        Particle
            Shifted particle.
        """
        offset = np.asarray(offset)
        new_verts = self.verts + offset
        return Particle(new_verts, self.faces.copy(), self.interp)

    def scale(self, factor: Union[float, np.ndarray]) -> 'Particle':
        """
        Scale particle by factor.

        Parameters
        ----------
        factor : float or array_like
            Scale factor (scalar or (3,) for anisotropic).

        Returns
        -------
        Particle
            Scaled particle.
        """
        factor = np.asarray(factor)
        new_verts = self.verts * factor
        return Particle(new_verts, self.faces.copy(), self.interp)

    def rotate(self, axis: np.ndarray, angle: float) -> 'Particle':
        """
        Rotate particle around axis by angle.

        Parameters
        ----------
        axis : array_like
            Rotation axis (3,).
        angle : float
            Rotation angle in radians.

        Returns
        -------
        Particle
            Rotated particle.
        """
        from scipy.spatial.transform import Rotation

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        rot = Rotation.from_rotvec(angle * axis)
        new_verts = rot.apply(self.verts)

        return Particle(new_verts, self.faces.copy(), self.interp)

    def centroid(self) -> np.ndarray:
        """
        Compute centroid of particle.

        Returns
        -------
        ndarray
            Centroid position (3,).
        """
        # Weighted by face area
        return np.sum(self.pos * self.area[:, np.newaxis], axis=0) / np.sum(self.area)

    def center(self) -> 'Particle':
        """
        Center particle at origin.

        Returns
        -------
        Particle
            Centered particle.
        """
        c = self.centroid()
        return self.shift(-c)

    def volume(self) -> float:
        """
        Compute volume enclosed by particle surface.

        Uses divergence theorem: V = (1/3) * integral(r . n dA)

        Returns
        -------
        float
            Volume in nm^3.
        """
        # V = (1/3) * sum(pos . nvec * area)
        return np.abs(np.sum(np.sum(self.pos * self.nvec, axis=1) * self.area) / 3)

    def surface_area(self) -> float:
        """
        Compute total surface area.

        Returns
        -------
        float
            Surface area in nm^2.
        """
        return np.sum(self.area)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bounding box.

        Returns
        -------
        min_coords : ndarray
            Minimum coordinates (3,).
        max_coords : ndarray
            Maximum coordinates (3,).
        """
        return np.min(self.verts, axis=0), np.max(self.verts, axis=0)

    def diameter(self) -> float:
        """
        Compute approximate diameter (max extent).

        Returns
        -------
        float
            Diameter in nm.
        """
        min_c, max_c = self.bounding_box()
        return np.max(max_c - min_c)

    @staticmethod
    def vertcat(*particles) -> 'Particle':
        """
        Vertically concatenate multiple particles.

        Parameters
        ----------
        *particles : Particle
            Particles to concatenate.

        Returns
        -------
        Particle
            Combined particle.
        """
        if len(particles) == 0:
            return Particle()
        if len(particles) == 1:
            return particles[0]

        result = particles[0]
        for p in particles[1:]:
            result = result + p

        return result

    def __add__(self, other: 'Particle') -> 'Particle':
        """Concatenate two particles."""
        if self.n_faces == 0:
            return other
        if other.n_faces == 0:
            return self

        # Offset face indices for second particle
        n_verts1 = len(self.verts)
        new_faces2 = other.faces.copy()
        valid = ~np.isnan(new_faces2)
        new_faces2[valid] = new_faces2[valid] + n_verts1

        new_verts = np.vstack([self.verts, other.verts])
        new_faces = np.vstack([self.faces, new_faces2])

        return Particle(new_verts, new_faces, self.interp)

    def deriv(self, func: np.ndarray, direction: str = 'n') -> np.ndarray:
        """
        Compute surface derivative of a function defined on faces.

        This method computes directional derivatives of scalar or vector
        fields defined on the particle surface.

        Parameters
        ----------
        func : ndarray
            Function values at face centroids. Shape (n_faces,) for scalar
            or (n_faces, 3) for vector.
        direction : str
            Derivative direction:
            - 'n': normal direction (outward derivative)
            - 't1': first tangent direction
            - 't2': second tangent direction
            - 'grad': surface gradient (returns vector)

        Returns
        -------
        ndarray
            Derivative values. Shape depends on input and direction:
            - Scalar input with 'n', 't1', 't2': shape (n_faces,)
            - Scalar input with 'grad': shape (n_faces, 3)
            - Vector input: shape (n_faces, 3)

        Examples
        --------
        >>> # Compute normal derivative of potential
        >>> phi = np.sin(particle.pos[:, 0])  # Scalar field
        >>> dphi_dn = particle.deriv(phi, 'n')
        >>>
        >>> # Compute surface gradient
        >>> grad_phi = particle.deriv(phi, 'grad')
        """
        func = np.atleast_1d(func)

        if len(func) != self.n_faces:
            raise ValueError(f"Function length {len(func)} != n_faces {self.n_faces}")

        # Build face adjacency for gradient computation
        # Find faces sharing vertices
        vertex_faces = [[] for _ in range(self.n_verts)]
        for i, face in enumerate(self.faces):
            valid = ~np.isnan(face)
            for v in face[valid].astype(int):
                vertex_faces[v].append(i)

        if direction == 'grad':
            # Surface gradient using finite differences
            grad = np.zeros((self.n_faces, 3))

            for i in range(self.n_faces):
                # Get neighboring faces
                valid = ~np.isnan(self.faces[i])
                verts_i = self.faces[i][valid].astype(int)

                neighbors = set()
                for v in verts_i:
                    neighbors.update(vertex_faces[v])
                neighbors.discard(i)
                neighbors = list(neighbors)

                if len(neighbors) == 0:
                    continue

                # Least squares fit for gradient
                # f(r) = f(r0) + grad_f . (r - r0)
                dr = self.pos[neighbors] - self.pos[i]  # (n_neigh, 3)
                df = func[neighbors] - func[i]  # (n_neigh,) or (n_neigh, 3)

                if func.ndim == 1:
                    # Scalar: solve grad_f from least squares
                    # df = dr @ grad_f  =>  grad_f = (dr^T dr)^-1 dr^T df
                    try:
                        grad[i] = np.linalg.lstsq(dr, df, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
                else:
                    # Vector: compute gradient for each component
                    for j in range(3):
                        df_j = func[neighbors, j] - func[i, j]
                        try:
                            grad[i, j] = np.linalg.lstsq(dr, df_j, rcond=None)[0][j]
                        except np.linalg.LinAlgError:
                            pass

            # Project onto surface (remove normal component)
            normal_comp = np.sum(grad * self.nvec, axis=1, keepdims=True)
            grad = grad - normal_comp * self.nvec

            return grad

        elif direction == 'n':
            # Normal derivative (requires neighboring face values)
            deriv = np.zeros(self.n_faces)

            for i in range(self.n_faces):
                # Get neighbors and compute derivative
                valid = ~np.isnan(self.faces[i])
                verts_i = self.faces[i][valid].astype(int)

                neighbors = set()
                for v in verts_i:
                    neighbors.update(vertex_faces[v])
                neighbors.discard(i)
                neighbors = list(neighbors)

                if len(neighbors) == 0:
                    continue

                # Average difference weighted by distance
                dr = self.pos[neighbors] - self.pos[i]
                dist = np.linalg.norm(dr, axis=1)

                # Project onto normal direction
                normal_dist = np.dot(dr, self.nvec[i])

                # Value difference
                if func.ndim == 1:
                    df = func[neighbors] - func[i]
                else:
                    df = np.linalg.norm(func[neighbors] - func[i], axis=1)

                # Derivative estimate
                valid_mask = np.abs(normal_dist) > 1e-10
                if np.any(valid_mask):
                    deriv[i] = np.mean(df[valid_mask] / normal_dist[valid_mask])

            return deriv

        elif direction in ['t1', 't2']:
            # Tangent derivative
            tvec = self.tvec1 if direction == 't1' else self.tvec2
            deriv = np.zeros(self.n_faces)

            for i in range(self.n_faces):
                valid = ~np.isnan(self.faces[i])
                verts_i = self.faces[i][valid].astype(int)

                neighbors = set()
                for v in verts_i:
                    neighbors.update(vertex_faces[v])
                neighbors.discard(i)
                neighbors = list(neighbors)

                if len(neighbors) == 0:
                    continue

                dr = self.pos[neighbors] - self.pos[i]
                tang_dist = np.dot(dr, tvec[i])

                if func.ndim == 1:
                    df = func[neighbors] - func[i]
                else:
                    df = np.dot(func[neighbors] - func[i], tvec[i])

                valid_mask = np.abs(tang_dist) > 1e-10
                if np.any(valid_mask):
                    deriv[i] = np.mean(df[valid_mask] / tang_dist[valid_mask])

            return deriv

        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'n', 't1', 't2', or 'grad'")

    def interp(self, values: np.ndarray, points: np.ndarray,
               method: str = 'nearest') -> np.ndarray:
        """
        Interpolate values from faces to arbitrary surface points.

        Parameters
        ----------
        values : ndarray
            Values at face centroids, shape (n_faces,) or (n_faces, m).
        points : ndarray
            Query points on surface, shape (n_points, 3).
        method : str
            Interpolation method:
            - 'nearest': nearest neighbor (default)
            - 'linear': linear interpolation from nearest faces
            - 'idw': inverse distance weighting

        Returns
        -------
        ndarray
            Interpolated values at query points.

        Examples
        --------
        >>> # Interpolate field to new points
        >>> new_points = np.array([[0, 0, 5], [1, 0, 5]])
        >>> interp_values = particle.interp(field_values, new_points)
        """
        values = np.atleast_1d(values)
        points = np.atleast_2d(points)

        if len(values) != self.n_faces:
            raise ValueError(f"Values length {len(values)} != n_faces {self.n_faces}")

        n_points = len(points)

        if method == 'nearest':
            # Find nearest face centroid for each point
            from scipy.spatial import cKDTree
            tree = cKDTree(self.pos)
            _, indices = tree.query(points)

            if values.ndim == 1:
                return values[indices]
            else:
                return values[indices, :]

        elif method == 'linear' or method == 'idw':
            # Use k nearest neighbors with distance weighting
            from scipy.spatial import cKDTree
            tree = cKDTree(self.pos)

            k = min(4, self.n_faces)  # 4 nearest neighbors
            distances, indices = tree.query(points, k=k)

            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)

            if method == 'idw':
                # Inverse distance weighting
                weights = 1.0 / distances
            else:
                # Linear (barycentric-like)
                weights = 1.0 / distances

            # Normalize weights
            weights = weights / weights.sum(axis=1, keepdims=True)

            if values.ndim == 1:
                result = np.sum(weights * values[indices], axis=1)
            else:
                result = np.zeros((n_points, values.shape[1]))
                for j in range(values.shape[1]):
                    result[:, j] = np.sum(weights * values[indices, j], axis=1)

            return result

        else:
            raise ValueError(f"Unknown method: {method}. Use 'nearest', 'linear', or 'idw'")

    def __repr__(self) -> str:
        return f"Particle(n_verts={self.n_verts}, n_faces={self.n_faces}, interp='{self.interp}')"
