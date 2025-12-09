"""
Demo: Linear chain of nanospheres.

Shows collective modes in 1D chain.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Collective modes of linear nanosphere chain."""
    print("Demo: Linear nanosphere chain")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 30  # nm
    gap = 3  # nm
    n_spheres_list = [2, 3, 4, 5]

    wavelengths = np.linspace(450, 850, 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_spheres_list)))

    peak_wavelengths = []

    for i, n_spheres in enumerate(n_spheres_list):
        print(f"\nChain of {n_spheres} spheres")

        # Create chain along x-axis
        spheres = []
        for j in range(n_spheres):
            x = (j - (n_spheres - 1) / 2) * (diameter + gap)
            sphere = trisphere(144, diameter).shift([x, 0, 0])
            spheres.append(sphere)

        p = ComParticle([eps_vacuum, eps_gold], spheres, [[2, 1]] * n_spheres)

        bem = BEMRet(p)

        # Longitudinal mode (polarization along chain)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax1.plot(wavelengths, ext, color=colors[i], linewidth=2,
                 label=f'N = {n_spheres}')

        peak_wl = wavelengths[np.argmax(ext)]
        peak_wavelengths.append(peak_wl)
        print(f"  Longitudinal mode peak: {peak_wl:.1f} nm")

    # Peak wavelength vs chain length
    ax2.plot(n_spheres_list, peak_wavelengths, 'ko-', linewidth=2, markersize=10)
    ax2.set_xlabel('Number of spheres', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('Chain Length Effect on LSPR', fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Extinction Spectra (d={diameter} nm, gap={gap} nm)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.suptitle('Collective Plasmon Modes in Linear Nanosphere Chain', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_chain.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
