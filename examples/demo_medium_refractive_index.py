"""
Demo: Effect of surrounding medium refractive index.

Shows LSPR shift with different embedding media.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Effect of surrounding medium on LSPR."""
    print("Demo: Medium refractive index effect")
    print("=" * 60)

    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    wavelengths = np.linspace(400, 800, 100)

    # Different media
    refractive_indices = [1.0, 1.33, 1.5, 1.7, 2.0]  # Air, water, glass, etc.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.cool(np.linspace(0, 1, len(refractive_indices)))
    peak_wavelengths = []

    for i, n in enumerate(refractive_indices):
        eps_medium = EpsConst(n**2)
        print(f"\nRefractive index: n = {n:.2f} (ε = {n**2:.2f})")

        sphere = trisphere(200, diameter)
        p = ComParticle([eps_medium, eps_gold], [sphere], [[2, 1]])

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax1.plot(wavelengths, ext, color=colors[i], linewidth=2,
                 label=f'n = {n:.2f}')

        peak_wl = wavelengths[np.argmax(ext)]
        peak_wavelengths.append(peak_wl)
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    # Peak wavelength vs refractive index
    ax2.plot(refractive_indices, peak_wavelengths, 'ko-', linewidth=2, markersize=10)

    # Calculate sensitivity (nm/RIU)
    coeffs = np.polyfit(refractive_indices, peak_wavelengths, 1)
    n_fit = np.linspace(min(refractive_indices), max(refractive_indices), 50)
    wl_fit = np.polyval(coeffs, n_fit)
    ax2.plot(n_fit, wl_fit, 'r--', linewidth=2,
             label=f'Sensitivity: {coeffs[0]:.0f} nm/RIU')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Extinction Spectra (d={diameter} nm)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Refractive index', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('LSPR Sensitivity', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    print(f"\nLSPR sensitivity: {coeffs[0]:.0f} nm/RIU")

    plt.suptitle('Refractive Index Sensitivity of Gold Nanosphere LSPR', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_medium_refractive_index.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
