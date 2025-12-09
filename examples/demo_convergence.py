"""
Demo: Mesh convergence study.

Shows how results converge with increasing mesh resolution.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet
from pymnpbem.mie import MieRet


def main():
    """Mesh convergence study."""
    print("Demo: Mesh convergence study")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 60  # nm
    wavelengths = np.linspace(450, 700, 50)

    # Compute Mie theory reference
    print("Computing Mie theory reference...")
    ext_mie = np.zeros(len(wavelengths))
    for j, wl in enumerate(wavelengths):
        eps_in = eps_gold(wl)[0]
        mie = MieRet(diameter, eps_in, 1.0, wl)
        ext_mie[j] = mie.ext()

    # BEM with different mesh resolutions
    n_verts_list = [50, 100, 150, 200, 300, 400]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_verts_list)))
    errors = []

    for i, n_verts in enumerate(n_verts_list):
        print(f"\nMesh: {n_verts} vertices...")

        sphere = trisphere(n_verts, diameter)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
        print(f"  Faces: {p.n_faces}")

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext_bem = result['ext'][:, 0]

        # Compute error relative to Mie
        rel_error = np.mean(np.abs(ext_bem - ext_mie) / ext_mie) * 100
        errors.append(rel_error)
        print(f"  Mean relative error: {rel_error:.2f}%")

        ax1.plot(wavelengths, ext_bem, color=colors[i], linewidth=2,
                 label=f'N = {n_verts} ({rel_error:.1f}%)')

    # Plot Mie reference
    ax1.plot(wavelengths, ext_mie, 'k--', linewidth=3, label='Mie (exact)')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'BEM Convergence (d={diameter} nm)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Error vs mesh size
    ax2.semilogy(n_verts_list, errors, 'ko-', linewidth=2, markersize=10)
    ax2.set_xlabel('Number of vertices', fontsize=12)
    ax2.set_ylabel('Mean relative error (%)', fontsize=12)
    ax2.set_title('Convergence to Mie Theory', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add threshold lines
    ax2.axhline(5, color='r', linestyle='--', alpha=0.5, label='5% error')
    ax2.axhline(1, color='g', linestyle='--', alpha=0.5, label='1% error')
    ax2.legend(fontsize=10)

    plt.suptitle('BEM Mesh Convergence Study', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_convergence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
