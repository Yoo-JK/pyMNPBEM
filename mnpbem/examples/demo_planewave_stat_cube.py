"""
Demo: Plane wave excitation of a nanocube (quasistatic).

Shows corner and edge plasmon modes in cubic particles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import tricube
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanocube plasmon resonance."""
    print("Demo: Gold nanocube - Corner plasmon modes")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create nanocube
    edge_length = 40  # nm
    cube = tricube(200, edge_length)
    print(f"Nanocube edge length: {edge_length} nm")

    p = ComParticle([eps_vacuum, eps_gold], [cube], [[2, 1]])

    wavelengths = np.linspace(450, 750, 100)

    # Face-on polarization
    exc = PlaneWaveStat(pol=[1, 0, 0])
    bem = BEMStat(p)

    print("\nComputing spectrum...")
    spec = SpectrumStat(bem, exc, wavelengths)
    sca, ext = spec.compute()
    abs_cs = ext - sca

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext[:, 0], 'b-', linewidth=2, label='Extinction')
    ax.plot(wavelengths, sca[:, 0], 'r--', linewidth=2, label='Scattering')
    ax.plot(wavelengths, abs_cs[:, 0], 'g:', linewidth=2, label='Absorption')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanocube (edge={edge_length} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak = wavelengths[np.argmax(ext[:, 0])]
    print(f"\nPlasmon resonance: {peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_cube.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
