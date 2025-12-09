"""
Demo: Scattering vs absorption cross sections.

Shows size-dependent transition from absorption to scattering dominated.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Scattering vs absorption for different particle sizes."""
    print("Demo: Scattering vs absorption cross sections")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    wavelengths = np.linspace(400, 800, 80)
    diameters = [20, 40, 60, 80, 100]  # nm

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, d in enumerate(diameters):
        print(f"\nDiameter: {d} nm")
        ax = axes[idx]

        n_verts = max(144, int(d * 4))
        sphere = trisphere(n_verts, d)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()

        ext = result['ext'][:, 0]
        sca = result['sca'][:, 0]
        abs_cs = ext - sca  # Absorption = Extinction - Scattering

        ax.plot(wavelengths, ext, 'k-', linewidth=2, label='Extinction')
        ax.plot(wavelengths, sca, 'b--', linewidth=2, label='Scattering')
        ax.plot(wavelengths, abs_cs, 'r:', linewidth=2, label='Absorption')

        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Cross section (nm²)', fontsize=10)

        # Calculate scattering efficiency at peak
        peak_idx = np.argmax(ext)
        sca_ratio = sca[peak_idx] / ext[peak_idx] * 100
        ax.set_title(f'd = {d} nm (Sca/Ext = {sca_ratio:.0f}%)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        print(f"  Scattering ratio at peak: {sca_ratio:.1f}%")

    # Hide last subplot if not used
    if len(diameters) < 6:
        axes[-1].axis('off')

    plt.suptitle('Size-Dependent Scattering vs Absorption in Gold Nanospheres', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_scattering_absorption.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
