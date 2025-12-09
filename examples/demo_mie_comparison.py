"""
Demo: Comparison of BEM with analytical Mie theory.

Validates BEM implementation for spherical particles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat, BEMRet
from pymnpbem.simulation import PlaneWaveStat, PlaneWaveRet, SpectrumStat, SpectrumRet
from pymnpbem.mie import MieStat, MieRet


def main():
    """BEM vs Mie theory comparison."""
    print("Demo: BEM validation with Mie theory")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Test for different particle sizes
    diameters = [20, 60, 100]  # nm
    wavelengths = np.linspace(400, 800, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, d in enumerate(diameters):
        print(f"\nDiameter: {d} nm")
        ax = axes[i]

        # BEM calculation
        if d <= 30:  # Quasistatic
            n_verts = 144
            sphere = trisphere(n_verts, d)
            p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
            bem = BEMStat(p)
            exc = PlaneWaveStat(pol=[1, 0, 0])
            spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
            _, ext_bem = spec.compute()
            ext_bem = ext_bem[:, 0]

            # Mie calculation (quasistatic)
            ext_mie = np.zeros(len(wavelengths))
            for j, wl in enumerate(wavelengths):
                eps_in = eps_gold(wl)[0]
                mie = MieStat(d, eps_in, 1.0)
                ext_mie[j] = mie.ext()
        else:  # Retarded
            n_verts = 400
            sphere = trisphere(n_verts, d)
            p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
            bem = BEMRet(p)
            exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
            spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
            result = spec.compute()
            ext_bem = result['ext'][:, 0]

            # Mie calculation (retarded)
            ext_mie = np.zeros(len(wavelengths))
            for j, wl in enumerate(wavelengths):
                eps_in = eps_gold(wl)[0]
                mie = MieRet(d, eps_in, 1.0, wl)
                ext_mie[j] = mie.ext()

        # Calculate error
        rel_error = np.mean(np.abs(ext_bem - ext_mie) / ext_mie) * 100

        # Plot
        ax.plot(wavelengths, ext_bem, 'b-', linewidth=2, label='BEM')
        ax.plot(wavelengths, ext_mie, 'r--', linewidth=2, label='Mie')
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Extinction (nm²)', fontsize=11)
        ax.set_title(f'd = {d} nm\nError: {rel_error:.1f}%', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        print(f"  Mean relative error: {rel_error:.2f}%")

    plt.suptitle('BEM vs Mie Theory for Gold Nanospheres', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_mie_comparison.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
