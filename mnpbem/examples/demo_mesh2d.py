"""
Demo: 2D mesh generation

This example demonstrates the 2D meshing capabilities:
- Creating polygonal boundaries
- Delaunay triangulation with constraints
- Mesh refinement
- Mesh smoothing
- Quality metrics

These 2D meshes can be used as bases for extruded 3D structures.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem.mesh2d import (
    mesh2d, mesh_polygon, delaunay_constrained,
    refine, smoothmesh
)
from mnpbem.mesh2d.quality import mesh_stats, quality


def create_polygon(shape='circle', n_points=50, size=10):
    """Create different polygon shapes."""

    if shape == 'circle':
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = size * np.cos(theta)
        y = size * np.sin(theta)

    elif shape == 'square':
        n_side = n_points // 4
        t = np.linspace(0, 1, n_side, endpoint=False)
        edges = [
            np.column_stack([t * size - size/2, np.full(n_side, -size/2)]),
            np.column_stack([np.full(n_side, size/2), t * size - size/2]),
            np.column_stack([size/2 - t * size, np.full(n_side, size/2)]),
            np.column_stack([np.full(n_side, -size/2), size/2 - t * size]),
        ]
        polygon = np.vstack(edges)
        return polygon

    elif shape == 'star':
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        r = size * (1 + 0.3 * np.cos(5 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif shape == 'ellipse':
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = size * np.cos(theta)
        y = size * 0.5 * np.sin(theta)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return np.column_stack([x, y])


def main():
    """Demonstrate 2D mesh generation."""

    print("MNPBEM Demo: 2D mesh generation")
    print("=" * 60)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # 1. Simple circle mesh
    print("Creating circle mesh...")
    ax = axes[0, 0]

    polygon_circle = create_polygon('circle', n_points=60, size=10)
    verts, faces = mesh_polygon(polygon_circle, max_area=2.0)

    ax.triplot(verts[:, 0], verts[:, 1], faces, 'b-', linewidth=0.5)
    ax.plot(polygon_circle[:, 0], polygon_circle[:, 1], 'r-', linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(f'Circle: {len(faces)} triangles')

    stats = mesh_stats(verts, faces)
    print(f"  Vertices: {stats['n_vertices']}, Triangles: {stats['n_triangles']}")
    print(f"  Mean quality: {stats['mean_quality']:.3f}")

    # 2. Square mesh
    print("Creating square mesh...")
    ax = axes[0, 1]

    polygon_square = create_polygon('square', n_points=40, size=10)
    verts, faces = mesh_polygon(polygon_square, max_area=2.0)

    ax.triplot(verts[:, 0], verts[:, 1], faces, 'b-', linewidth=0.5)
    ax.plot(np.append(polygon_square[:, 0], polygon_square[0, 0]),
            np.append(polygon_square[:, 1], polygon_square[0, 1]),
            'r-', linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(f'Square: {len(faces)} triangles')

    # 3. Star mesh
    print("Creating star mesh...")
    ax = axes[0, 2]

    polygon_star = create_polygon('star', n_points=100, size=10)
    verts, faces = mesh_polygon(polygon_star, max_area=1.5)

    ax.triplot(verts[:, 0], verts[:, 1], faces, 'b-', linewidth=0.5)
    ax.plot(np.append(polygon_star[:, 0], polygon_star[0, 0]),
            np.append(polygon_star[:, 1], polygon_star[0, 1]),
            'r-', linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(f'Star: {len(faces)} triangles')

    # 4. Mesh refinement
    print("Demonstrating refinement...")
    ax = axes[1, 0]

    # Start with coarse mesh
    polygon = create_polygon('circle', n_points=30, size=10)
    verts_coarse, faces_coarse = mesh_polygon(polygon, max_area=10.0)

    # Refine
    verts_fine, faces_fine = refine(verts_coarse, faces_coarse,
                                     polygon=polygon, max_edge=2.0)

    ax.triplot(verts_fine[:, 0], verts_fine[:, 1], faces_fine,
               'b-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Refined: {len(faces_coarse)} → {len(faces_fine)} triangles')

    # 5. Mesh smoothing
    print("Demonstrating smoothing...")
    ax = axes[1, 1]

    # Create mesh
    polygon = create_polygon('ellipse', n_points=60, size=10)
    verts_orig, faces_orig = mesh_polygon(polygon, max_area=2.0)

    # Smooth
    verts_smooth = smoothmesh(verts_orig, faces_orig, polygon=polygon,
                              n_iter=5, method='laplacian')

    # Plot both
    ax.triplot(verts_orig[:, 0], verts_orig[:, 1], faces_orig,
               'r-', linewidth=0.3, alpha=0.3, label='Original')
    ax.triplot(verts_smooth[:, 0], verts_smooth[:, 1], faces_orig,
               'b-', linewidth=0.5, label='Smoothed')
    ax.set_aspect('equal')
    ax.set_title('Mesh smoothing')
    ax.legend()

    # 6. Quality visualization
    print("Visualizing mesh quality...")
    ax = axes[1, 2]

    polygon = create_polygon('star', n_points=100, size=10)
    verts, faces = mesh_polygon(polygon, max_area=1.5)

    # Compute quality
    q = quality(verts, faces)

    # Color triangles by quality
    from matplotlib.collections import PolyCollection
    from matplotlib import cm

    triangles = verts[faces]
    colors = cm.viridis(q)

    collection = PolyCollection(triangles, facecolors=colors,
                                edgecolors='black', linewidths=0.3)
    ax.add_collection(collection)

    ax.set_xlim(verts[:, 0].min() - 1, verts[:, 0].max() + 1)
    ax.set_ylim(verts[:, 1].min() - 1, verts[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.set_title(f'Quality (mean: {q.mean():.2f})')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cm.viridis,
                           norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Quality')

    fig.suptitle('2D Mesh Generation Examples', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_mesh2d_result.png', dpi=150)
    print("\nFigure saved to demo_mesh2d_result.png")

    plt.show()


if __name__ == '__main__':
    main()
