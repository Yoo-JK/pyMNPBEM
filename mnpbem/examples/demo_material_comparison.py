"""
Demo: LSPR comparison across different materials.

Shows plasmon resonances for various metals and semiconductors.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude, EpsTable
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """LSPR comparison for different materials."""
    print("Demo: Material comparison for LSPR")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)

    # Different materials
    materials = {
        'Gold (Au)': EpsDrude('Au'),
        'Silver (Ag)': EpsDrude('Ag'),
        'Copper (Cu)': EpsDrude('Cu'),
        'Aluminum (Al)': EpsDrude('Al')
    }

    diameter = 40  # nm
    wavelengths = np.linspace(300, 900, 120)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'Gold (Au)': 'gold', 'Silver (Ag)': 'silver',
              'Copper (Cu)': '#B87333', 'Aluminum (Al)': 'gray'}

    results = {}

    for name, eps_mat in materials.items():
        print(f"\nMaterial: {name}")

        sphere = trisphere(200, diameter)
        p = ComParticle([eps_vacuum, eps_mat], [sphere], [[2, 1]])

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()

        ext = result['ext'][:, 0]
        sca = result['sca'][:, 0]

        results[name] = {'ext': ext, 'sca': sca}

        color = colors[name]
        ax1.plot(wavelengths, ext, color=color, linewidth=2, label=name)

        # Normalize for comparison
        ext_norm = ext / ext.max()
        ax2.plot(wavelengths, ext_norm, color=color, linewidth=2, label=name)

        peak_wl = wavelengths[np.argmax(ext)]
        print(f"  Peak wavelength: {peak_wl:.1f} nm")
        print(f"  Max extinction: {ext.max():.0f} nm²")

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Extinction Cross Section (d={diameter} nm)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Normalized extinction', fontsize=12)
    ax2.set_title('Normalized Extinction (for shape comparison)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('LSPR Comparison Across Different Materials', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_material_comparison.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
