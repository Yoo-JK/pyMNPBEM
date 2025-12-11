"""
Demo: Plane wave excitation of a nanodisk (quasistatic).

Shows in-plane and out-of-plane plasmon modes.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trinanodisk
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanodisk plasmon modes."""
    print("Demo: Gold nanodisk - In-plane vs out-of-plane")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create nanodisk
    diameter = 60  # nm
    thickness = 10  # nm
    disk = trinanodisk(200, diameter, thickness)
    print(f"Nanodisk: diameter={diameter} nm, thickness={thickness} nm")
    print(f"Aspect ratio: {diameter/thickness:.1f}")

    p = ComParticle([eps_vacuum, eps_gold], [disk], [[2, 1]])

    wavelengths = np.linspace(400, 1000, 150)

    bem = BEMStat(p)

    # In-plane polarization (disk is in xy-plane)
    print("\nComputing in-plane mode (x-pol)...")
    exc_in = PlaneWaveStat(pol=[1, 0, 0])
    spec_in = SpectrumStat(bem, exc_in, wavelengths, show_progress=False)
    _, ext_in = spec_in.compute()

    # Out-of-plane polarization
    print("Computing out-of-plane mode (z-pol)...")
    exc_out = PlaneWaveStat(pol=[0, 0, 1])
    spec_out = SpectrumStat(bem, exc_out, wavelengths, show_progress=False)
    _, ext_out = spec_out.compute()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_in[:, 0], 'r-', linewidth=2, label='In-plane (x-pol)')
    ax.plot(wavelengths, ext_out[:, 0], 'b-', linewidth=2, label='Out-of-plane (z-pol)')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanodisk (d={diameter} nm, t={thickness} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    in_peak = wavelengths[np.argmax(ext_in[:, 0])]
    out_peak = wavelengths[np.argmax(ext_out[:, 0])]
    print(f"\nIn-plane resonance: {in_peak:.1f} nm")
    print(f"Out-of-plane resonance: {out_peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_disk.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
