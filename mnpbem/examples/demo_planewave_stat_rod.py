"""
Demo: Plane wave excitation of a gold nanorod (quasistatic).

This example shows how longitudinal and transverse plasmon modes
of a nanorod can be excited with different polarizations.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanorod with different polarizations."""
    print("Demo: Gold nanorod - Longitudinal and transverse modes")
    print("=" * 60)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create nanorod (cylinder with hemispherical caps)
    length = 60  # nm
    diameter = 20  # nm
    rod = trirod(144, length, diameter)
    print(f"Created rod: length={length} nm, diameter={diameter} nm")
    print(f"Aspect ratio: {length/diameter:.1f}")

    # Create particle
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    # Define polarizations
    # Longitudinal: along rod axis (z)
    # Transverse: perpendicular to rod (x)
    pol_long = [0, 0, 1]  # z-polarized
    pol_trans = [1, 0, 0]  # x-polarized

    wavelengths = np.linspace(400, 1000, 150)

    # Compute spectra for both polarizations
    print("\nComputing longitudinal mode spectrum...")
    bem = BEMStat(p)
    exc_long = PlaneWaveStat(pol=pol_long)
    spec_long = SpectrumStat(bem, exc_long, wavelengths)
    _, ext_long = spec_long.compute()

    print("Computing transverse mode spectrum...")
    exc_trans = PlaneWaveStat(pol=pol_trans)
    spec_trans = SpectrumStat(bem, exc_trans, wavelengths)
    _, ext_trans = spec_trans.compute()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_long[:, 0], 'r-', linewidth=2, label='Longitudinal (z-pol)')
    ax.plot(wavelengths, ext_trans[:, 0], 'b-', linewidth=2, label='Transverse (x-pol)')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanorod (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Find resonances
    long_peak = wavelengths[np.argmax(ext_long[:, 0])]
    trans_peak = wavelengths[np.argmax(ext_trans[:, 0])]
    print(f"\nLongitudinal plasmon resonance: {long_peak:.1f} nm")
    print(f"Transverse plasmon resonance: {trans_peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_rod.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
