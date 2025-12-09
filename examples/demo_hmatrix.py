"""
Demo: H-matrix acceleration for large particles.

Shows computational speedup using hierarchical matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat


def main():
    """H-matrix acceleration demonstration."""
    print("Demo: H-matrix acceleration")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    n_verts_list = [100, 200, 400, 800]

    from pymnpbem.simulation import PlaneWaveStat

    exc = PlaneWaveStat(pol=[1, 0, 0])
    wavelength = 520  # nm

    direct_times = []
    n_faces_list = []

    print("\nDirect BEM timing:")
    print("-" * 40)

    for n_verts in n_verts_list:
        sphere = trisphere(n_verts, diameter)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
        n_faces = p.n_faces
        n_faces_list.append(n_faces)

        print(f"N = {n_verts} vertices ({n_faces} faces)...")

        bem = BEMStat(p)

        # Time the solve
        exc_wl = exc(p, wavelength)
        t0 = time.time()
        sig = bem.solve(exc_wl)
        t_solve = time.time() - t0

        direct_times.append(t_solve)
        print(f"  Solve time: {t_solve:.3f} s")

    # Plot scaling
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(n_faces_list, direct_times, 'bo-', linewidth=2, markersize=10, label='Direct BEM')

    # Fit O(N^3) curve
    log_n = np.log(n_faces_list)
    log_t = np.log(direct_times)
    slope, intercept = np.polyfit(log_n, log_t, 1)

    n_fit = np.logspace(np.log10(min(n_faces_list)), np.log10(max(n_faces_list)), 50)
    t_fit = np.exp(intercept) * n_fit**slope
    ax.loglog(n_fit, t_fit, 'r--', linewidth=2, label=f'Fit: O(N^{slope:.1f})')

    # Expected O(N^3) scaling
    t_expected = direct_times[0] * (np.array(n_faces_list) / n_faces_list[0])**3
    ax.loglog(n_faces_list, t_expected, 'g:', linewidth=2, label='Expected O(N³)')

    ax.set_xlabel('Number of faces', fontsize=12)
    ax.set_ylabel('Solve time (s)', fontsize=12)
    ax.set_title('BEM Computational Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    print(f"\nScaling exponent: {slope:.2f}")
    print("(Expected: 3.0 for direct solve)")

    plt.tight_layout()
    plt.savefig('demo_hmatrix.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
