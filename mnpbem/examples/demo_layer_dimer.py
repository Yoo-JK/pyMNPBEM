"""
Demo: Nanosphere dimer on substrate.

Shows substrate effect on coupled plasmon modes.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, LayerStructure
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStatLayer
from pymnpbem.simulation import PlaneWaveStatLayer, SpectrumStatLayer


def main():
    """Gold nanosphere dimer on glass substrate."""
    print("Demo: Dimer on glass substrate")
    print("=" * 60)

    eps_air = EpsConst(1.0)
    eps_glass = EpsConst(2.25)  # n=1.5
    eps_gold = EpsDrude('Au')

    # Layer structure: air above, glass below (interface at z=0)
    layer = LayerStructure([eps_air, eps_glass])

    diameter = 40  # nm
    gap_surface = 2  # nm above substrate
    gap_dimer = 5  # nm between spheres

    # Create dimer above substrate
    z_center = diameter/2 + gap_surface
    sphere1 = trisphere(200, diameter).shift([-(diameter + gap_dimer)/2, 0, z_center])
    sphere2 = trisphere(200, diameter).shift([(diameter + gap_dimer)/2, 0, z_center])

    p = ComParticle([eps_air, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])
    print(f"Dimer: d={diameter} nm, gap={gap_dimer} nm, {gap_surface} nm above substrate")

    wavelengths = np.linspace(400, 800, 100)

    # With substrate - longitudinal
    print("\nComputing longitudinal mode with substrate...")
    bem_layer = BEMStatLayer(p, layer)
    exc_long = PlaneWaveStatLayer(pol=[1, 0, 0], layer=layer)
    spec_long = SpectrumStatLayer(bem_layer, exc_long, wavelengths)
    _, ext_long = spec_long.compute()

    # With substrate - transverse
    print("Computing transverse mode with substrate...")
    exc_trans = PlaneWaveStatLayer(pol=[0, 1, 0], layer=layer)
    spec_trans = SpectrumStatLayer(bem_layer, exc_trans, wavelengths)
    _, ext_trans = spec_trans.compute()

    # Without substrate for comparison
    from pymnpbem.bem import BEMStat
    from pymnpbem.simulation import PlaneWaveStat, SpectrumStat

    print("Computing without substrate...")
    sphere1_free = trisphere(200, diameter).shift([-(diameter + gap_dimer)/2, 0, 0])
    sphere2_free = trisphere(200, diameter).shift([(diameter + gap_dimer)/2, 0, 0])
    p_free = ComParticle([eps_air, eps_gold], [sphere1_free, sphere2_free], [[2, 1], [2, 1]])
    bem_free = BEMStat(p_free)
    exc_free = PlaneWaveStat(pol=[1, 0, 0])
    spec_free = SpectrumStat(bem_free, exc_free, wavelengths)
    _, ext_free = spec_free.compute()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, ext_long[:, 0], 'r-', linewidth=2,
            label='On substrate - Longitudinal')
    ax.plot(wavelengths, ext_trans[:, 0], 'b-', linewidth=2,
            label='On substrate - Transverse')
    ax.plot(wavelengths, ext_free[:, 0], 'k--', linewidth=2,
            label='Free-standing - Longitudinal')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Dimer on Glass Substrate (d={diameter} nm, gap={gap_dimer} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak_long = wavelengths[np.argmax(ext_long[:, 0])]
    peak_trans = wavelengths[np.argmax(ext_trans[:, 0])]
    peak_free = wavelengths[np.argmax(ext_free[:, 0])]

    print(f"\nOn substrate - Longitudinal peak: {peak_long:.1f} nm")
    print(f"On substrate - Transverse peak: {peak_trans:.1f} nm")
    print(f"Free-standing peak: {peak_free:.1f} nm")
    print(f"Substrate red shift: {peak_long - peak_free:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_layer_dimer.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
