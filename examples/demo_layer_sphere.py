"""
Demo: Nanoparticle on substrate - Layer structure simulation.

Shows effect of substrate on plasmon resonance.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, LayerStructure
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStatLayer
from pymnpbem.simulation import PlaneWaveStatLayer, SpectrumStatLayer


def main():
    """Gold nanosphere on glass substrate."""
    print("Demo: Gold nanosphere on glass substrate")
    print("=" * 60)

    eps_air = EpsConst(1.0)
    eps_glass = EpsConst(2.25)  # n=1.5
    eps_gold = EpsDrude('Au')

    # Create layer structure: air above, glass below
    layer = LayerStructure([eps_air, eps_glass])
    print("Layer structure: air | glass")

    diameter = 40  # nm
    gap = 2  # nm above substrate

    # Sphere above substrate
    sphere = trisphere(200, diameter)
    sphere = sphere.shift([0, 0, diameter/2 + gap])

    p = ComParticle([eps_air, eps_gold], [sphere], [[2, 1]])
    print(f"Sphere: d={diameter} nm, gap={gap} nm above substrate")

    wavelengths = np.linspace(400, 800, 100)

    # With substrate
    print("\nComputing spectrum with substrate...")
    bem_layer = BEMStatLayer(p, layer)
    exc_layer = PlaneWaveStatLayer(pol=[1, 0, 0], layer=layer)
    spec_layer = SpectrumStatLayer(bem_layer, exc_layer, wavelengths)
    sca_layer, ext_layer = spec_layer.compute()

    # Without substrate (free-standing)
    from pymnpbem.bem import BEMStat
    from pymnpbem.simulation import PlaneWaveStat, SpectrumStat

    print("Computing spectrum without substrate...")
    p_free = ComParticle([eps_air, eps_gold], [trisphere(200, diameter)], [[2, 1]])
    bem_free = BEMStat(p_free)
    exc_free = PlaneWaveStat(pol=[1, 0, 0])
    spec_free = SpectrumStat(bem_free, exc_free, wavelengths)
    _, ext_free = spec_free.compute()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_layer[:, 0], 'r-', linewidth=2, label='On glass substrate')
    ax.plot(wavelengths, ext_free[:, 0], 'b--', linewidth=2, label='Free-standing')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Effect of Glass Substrate (d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak_layer = wavelengths[np.argmax(ext_layer[:, 0])]
    peak_free = wavelengths[np.argmax(ext_free[:, 0])]
    print(f"\nWith substrate: peak at {peak_layer:.1f} nm")
    print(f"Free-standing: peak at {peak_free:.1f} nm")
    print(f"Red shift: {peak_layer - peak_free:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_layer_sphere.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
