"""
Demo: Size-dependent retardation effects.

Shows transition from quasistatic to retarded regime.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat, BEMRet
from pymnpbem.simulation import PlaneWaveStat, PlaneWaveRet, SpectrumStat, SpectrumRet


def main():
    """Size-dependent LSPR with retardation effects."""
    print("Demo: Size-dependent retardation effects")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    wavelengths = np.linspace(400, 900, 100)
    diameters = [20, 40, 60, 80, 100, 150]  # nm

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, d in enumerate(diameters):
        print(f"\nDiameter: {d} nm")
        ax = axes[idx]

        n_verts = max(144, int(d * 4))
        sphere = trisphere(n_verts, d)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

        # Quasistatic BEM
        print("  Computing quasistatic...")
        bem_stat = BEMStat(p)
        exc_stat = PlaneWaveStat(pol=[1, 0, 0])
        spec_stat = SpectrumStat(bem_stat, exc_stat, wavelengths, show_progress=False)
        _, ext_stat = spec_stat.compute()
        ext_stat = ext_stat[:, 0]

        # Retarded BEM
        print("  Computing retarded...")
        bem_ret = BEMRet(p)
        exc_ret = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec_ret = SpectrumRet(bem_ret, exc_ret, wavelengths, show_progress=False)
        result_ret = spec_ret.compute()
        ext_ret = result_ret['ext'][:, 0]

        # Normalize for comparison
        ext_stat_norm = ext_stat / ext_stat.max()
        ext_ret_norm = ext_ret / ext_ret.max()

        ax.plot(wavelengths, ext_stat_norm, 'b-', linewidth=2, label='Quasistatic')
        ax.plot(wavelengths, ext_ret_norm, 'r--', linewidth=2, label='Retarded')

        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Normalized extinction', fontsize=10)
        ax.set_title(f'd = {d} nm', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Find peaks
        peak_stat = wavelengths[np.argmax(ext_stat)]
        peak_ret = wavelengths[np.argmax(ext_ret)]
        print(f"  Quasistatic peak: {peak_stat:.1f} nm")
        print(f"  Retarded peak: {peak_ret:.1f} nm")
        print(f"  Red shift: {peak_ret - peak_stat:.1f} nm")

    plt.suptitle('Size-Dependent Retardation Effects in Gold Nanospheres', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_planewave_ret_size.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
