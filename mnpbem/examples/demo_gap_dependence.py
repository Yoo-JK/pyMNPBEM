"""
Demo: Gap-dependent plasmon coupling in dimer.

Shows LSPR shift with varying gap size.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Gap-dependent plasmon coupling."""
    print("Demo: Gap-dependent plasmon coupling")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    gaps = [2, 5, 10, 20, 40, 80]  # nm

    wavelengths = np.linspace(450, 850, 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gaps)))
    peak_wavelengths = []

    for i, gap in enumerate(gaps):
        print(f"\nGap: {gap} nm")

        # Create dimer
        sphere1 = trisphere(200, diameter).shift([-(diameter + gap)/2, 0, 0])
        sphere2 = trisphere(200, diameter).shift([(diameter + gap)/2, 0, 0])

        p = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])  # Longitudinal
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax1.plot(wavelengths, ext, color=colors[i], linewidth=2,
                 label=f'gap = {gap} nm')

        peak_wl = wavelengths[np.argmax(ext)]
        peak_wavelengths.append(peak_wl)
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    # Single sphere reference
    p_single = ComParticle([eps_vacuum, eps_gold], [trisphere(200, diameter)], [[2, 1]])
    bem_single = BEMRet(p_single)
    spec_single = SpectrumRet(bem_single, PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1]),
                               wavelengths, show_progress=False)
    result_single = spec_single.compute()
    peak_single = wavelengths[np.argmax(result_single['ext'][:, 0])]

    # Peak wavelength vs gap
    ax2.semilogx(gaps, peak_wavelengths, 'ko-', linewidth=2, markersize=10)
    ax2.axhline(peak_single, color='r', linestyle='--', label=f'Single sphere ({peak_single:.0f} nm)')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Extinction Spectra (d={diameter} nm)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Gap (nm)', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('Coupling vs Gap Size', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Gap-Dependent Plasmon Coupling in Gold Nanosphere Dimer', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_gap_dependence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
