"""
Demo: Plane wave excitation of a nanoparticle dimer (quasistatic).

Shows gap plasmon mode in coupled nanoparticles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanosphere dimer - gap plasmon mode."""
    print("Demo: Gold nanosphere dimer - Gap plasmon")
    print("=" * 60)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create two spheres with gap
    diameter = 30  # nm
    gap = 2  # nm gap between spheres

    sphere1 = trisphere(100, diameter)
    sphere2 = trisphere(100, diameter)

    # Position spheres
    offset = (diameter + gap) / 2
    sphere1 = sphere1.shift([-offset, 0, 0])
    sphere2 = sphere2.shift([+offset, 0, 0])

    print(f"Dimer: 2 spheres, d={diameter} nm, gap={gap} nm")

    # Create particle (both spheres share same materials)
    p = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])

    wavelengths = np.linspace(400, 900, 120)

    # Polarization along dimer axis (x) - excites gap mode
    exc_gap = PlaneWaveStat(pol=[1, 0, 0])
    # Polarization perpendicular (y) - weaker coupling
    exc_perp = PlaneWaveStat(pol=[0, 1, 0])

    bem = BEMStat(p)

    print("\nComputing gap mode spectrum (x-pol)...")
    spec_gap = SpectrumStat(bem, exc_gap, wavelengths)
    _, ext_gap = spec_gap.compute()

    print("Computing perpendicular mode spectrum (y-pol)...")
    spec_perp = SpectrumStat(bem, exc_perp, wavelengths)
    _, ext_perp = spec_perp.compute()

    # Compare with single sphere
    single = trisphere(100, diameter)
    p_single = ComParticle([eps_vacuum, eps_gold], [single], [[2, 1]])
    bem_single = BEMStat(p_single)
    spec_single = SpectrumStat(bem_single, exc_gap, wavelengths)
    _, ext_single = spec_single.compute()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_gap[:, 0], 'r-', linewidth=2, label='Dimer (gap mode, x-pol)')
    ax.plot(wavelengths, ext_perp[:, 0], 'b-', linewidth=2, label='Dimer (y-pol)')
    ax.plot(wavelengths, 2 * ext_single[:, 0], 'k--', linewidth=1.5, label='2× Single sphere')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanosphere Dimer (d={diameter} nm, gap={gap} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    gap_peak = wavelengths[np.argmax(ext_gap[:, 0])]
    single_peak = wavelengths[np.argmax(ext_single[:, 0])]
    print(f"\nGap plasmon resonance: {gap_peak:.1f} nm")
    print(f"Single sphere resonance: {single_peak:.1f} nm")
    print(f"Red shift: {gap_peak - single_peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_dimer.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
