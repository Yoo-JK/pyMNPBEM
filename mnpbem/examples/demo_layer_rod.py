"""
Demo: Nanorod on different substrates.

Shows substrate-dependent LSPR shift.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, LayerStructure
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStatLayer
from pymnpbem.simulation import PlaneWaveStatLayer, SpectrumStatLayer


def main():
    """Gold nanorod on different substrates."""
    print("Demo: Nanorod on different substrates")
    print("=" * 60)

    eps_air = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Different substrates
    substrates = {
        'Air (n=1.0)': EpsConst(1.0),
        'Glass (n=1.5)': EpsConst(2.25),
        'TiO2 (n=2.5)': EpsConst(6.25),
        'Si3N4 (n=2.0)': EpsConst(4.0)
    }

    length = 60  # nm
    diameter = 20  # nm
    gap = 2  # nm above substrate

    wavelengths = np.linspace(500, 950, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(substrates)))

    for i, (name, eps_sub) in enumerate(substrates.items()):
        print(f"\nSubstrate: {name}")

        layer = LayerStructure([eps_air, eps_sub])

        # Rod lying flat, just above substrate
        z_center = diameter/2 + gap
        rod = trirod(200, length, diameter).shift([0, 0, z_center])

        p = ComParticle([eps_air, eps_gold], [rod], [[2, 1]])

        bem = BEMStatLayer(p, layer)
        exc = PlaneWaveStatLayer(pol=[1, 0, 0], layer=layer)  # Along rod axis
        spec = SpectrumStatLayer(bem, exc, wavelengths)
        _, ext = spec.compute()

        ax.plot(wavelengths, ext[:, 0], color=colors[i], linewidth=2, label=name)

        peak_wl = wavelengths[np.argmax(ext[:, 0])]
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Nanorod on Different Substrates (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_layer_rod.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
