"""
Demo: Aspect ratio dependence of nanorod LSPR.

Shows tunability of longitudinal mode with aspect ratio.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Aspect ratio dependence of nanorod LSPR."""
    print("Demo: Aspect ratio dependence")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 20  # nm (fixed)
    aspect_ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    wavelengths = np.linspace(500, 1200, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(aspect_ratios)))
    peak_wavelengths = []

    for i, ar in enumerate(aspect_ratios):
        length = diameter * ar
        print(f"\nAspect ratio: {ar:.1f} (L={length:.0f} nm, d={diameter} nm)")

        n_verts = int(200 * ar / 2)
        rod = trirod(n_verts, length, diameter)
        p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])  # Along rod
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax1.plot(wavelengths, ext, color=colors[i], linewidth=2,
                 label=f'AR = {ar:.1f}')

        peak_wl = wavelengths[np.argmax(ext)]
        peak_wavelengths.append(peak_wl)
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    # Plot peak wavelength vs aspect ratio
    ax2.plot(aspect_ratios, peak_wavelengths, 'ko-', linewidth=2, markersize=8)

    # Linear fit
    coeffs = np.polyfit(aspect_ratios, peak_wavelengths, 1)
    ar_fit = np.linspace(min(aspect_ratios), max(aspect_ratios), 50)
    wl_fit = np.polyval(coeffs, ar_fit)
    ax2.plot(ar_fit, wl_fit, 'r--', linewidth=2, label=f'Linear fit: λ = {coeffs[0]:.0f}×AR + {coeffs[1]:.0f}')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Extinction Spectra (d={diameter} nm)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Aspect ratio', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('LSPR Peak vs Aspect Ratio', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Aspect Ratio Tunability of Gold Nanorod LSPR', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_aspect_ratio.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
