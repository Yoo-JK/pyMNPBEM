"""
Demo: Mirror symmetry for computational efficiency.

Shows use of BEMStatMirror for symmetric particles.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from pymnpbem.particles import ComParticle, ComParticleMirror
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat, BEMStatMirror
from pymnpbem.simulation import PlaneWaveStat


def main():
    """Mirror symmetry for gold nanosphere dimer."""
    print("Demo: Mirror symmetry acceleration")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 30  # nm
    gap = 2  # nm

    # Create symmetric dimer
    sphere1 = trisphere(100, diameter).shift([-(diameter + gap)/2, 0, 0])
    sphere2 = trisphere(100, diameter).shift([(diameter + gap)/2, 0, 0])

    p_full = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])
    print(f"Full dimer: {p_full.n_faces} faces")

    # Create half-dimer with mirror
    p_half = ComParticleMirror([eps_vacuum, eps_gold], [sphere1], [[2, 1]])
    print(f"Half dimer (with mirror): {sphere1.n_faces} faces")

    wavelengths = np.linspace(400, 800, 50)
    exc = PlaneWaveStat(pol=[1, 0, 0])  # Along dimer axis

    # Full calculation
    print("\nFull BEM calculation...")
    bem_full = BEMStat(p_full)
    t0 = time.time()
    ext_full = []
    for wl in wavelengths:
        exc_wl = exc(p_full, wl)
        sig = bem_full.solve(exc_wl)
        ext_full.append(np.abs(sig.phip).sum())
    t_full = time.time() - t0

    # Mirror calculation
    print("Mirror BEM calculation...")
    bem_mirror = BEMStatMirror(p_half)
    t0 = time.time()
    ext_mirror = []
    for wl in wavelengths:
        exc_wl = exc(p_half, wl)
        sig = bem_mirror.solve(exc_wl)
        ext_mirror.append(np.abs(sig.phip).sum())
    t_mirror = time.time() - t0

    ext_full = np.array(ext_full)
    ext_mirror = np.array(ext_mirror)

    # Normalize
    ext_full /= ext_full.max()
    ext_mirror /= ext_mirror.max()

    print(f"\nFull calculation time: {t_full:.2f} s")
    print(f"Mirror calculation time: {t_mirror:.2f} s")
    print(f"Speedup: {t_full/t_mirror:.1f}×")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_full, 'b-', linewidth=2, label='Full BEM')
    ax.plot(wavelengths, ext_mirror, 'r--', linewidth=2, label='Mirror BEM')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized extinction', fontsize=12)
    ax.set_title('Mirror Symmetry Acceleration', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_symmetry.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
