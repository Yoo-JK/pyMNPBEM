"""
Demo: Surface curvature and mesh quality analysis.

Shows importance of mesh quality for accurate simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere, trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet
from pymnpbem.misc.plotting import patchcurvature


def main():
    """Mesh quality and curvature analysis."""
    print("Demo: Surface curvature and mesh quality")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm

    # Create meshes with different resolutions
    n_verts_list = [50, 100, 200, 400]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    wavelengths = np.linspace(400, 700, 60)

    extinction_curves = []

    for i, n_verts in enumerate(n_verts_list):
        print(f"\nMesh resolution: {n_verts} vertices")
        ax = axes[i]

        sphere = trisphere(n_verts, diameter)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

        # Compute curvature
        curvature, face_curvature = patchcurvature(sphere, curvature_type='mean')

        # Theoretical curvature for sphere
        theoretical_curv = 2 / diameter

        # Statistics
        mean_curv = np.mean(np.abs(face_curvature))
        std_curv = np.std(face_curvature)
        curv_error = np.abs(mean_curv - theoretical_curv) / theoretical_curv * 100

        print(f"  Faces: {p.n_faces}")
        print(f"  Mean curvature: {mean_curv:.4f} (theoretical: {theoretical_curv:.4f})")
        print(f"  Curvature error: {curv_error:.1f}%")

        # Plot curvature histogram
        ax.hist(face_curvature, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(theoretical_curv, color='r', linestyle='--', linewidth=2,
                   label=f'Theoretical (1/R = {theoretical_curv:.4f})')
        ax.axvline(mean_curv, color='g', linestyle='-', linewidth=2,
                   label=f'Mean ({mean_curv:.4f})')

        ax.set_xlabel('Mean curvature (1/nm)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'N = {n_verts} ({p.n_faces} faces)\nError: {curv_error:.1f}%', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Compute extinction spectrum
        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        extinction_curves.append(result['ext'][:, 0])

    plt.suptitle(f'Mesh Quality Analysis (d={diameter} nm sphere)', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_curvature_effect_hist.png', dpi=150)

    # Plot extinction convergence
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_verts_list)))

    for i, (n_verts, ext) in enumerate(zip(n_verts_list, extinction_curves)):
        ax2.plot(wavelengths, ext, color=colors[i], linewidth=2, label=f'N = {n_verts}')

    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax2.set_title('Extinction Convergence with Mesh Resolution', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_curvature_effect_convergence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
