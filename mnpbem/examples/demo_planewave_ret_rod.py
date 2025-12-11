"""
Demo: Plane wave excitation of gold nanorod (retarded).

Full electromagnetic calculation for nanorod.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Retarded calculation for gold nanorod."""
    print("Demo: Gold nanorod - Retarded BEM")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 100  # nm
    diameter = 30  # nm
    rod = trirod(400, length, diameter)
    print(f"Nanorod: L={length} nm, d={diameter} nm, AR={length/diameter:.1f}")

    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    wavelengths = np.linspace(500, 1200, 100)
    bem = BEMRet(p)

    # Longitudinal polarization
    print("\nComputing longitudinal mode...")
    exc_long = PlaneWaveRet(pol=[0, 0, 1], direction=[1, 0, 0])
    spec_long = SpectrumRet(bem, exc_long, wavelengths)
    result_long = spec_long.compute()

    # Transverse polarization
    print("Computing transverse mode...")
    exc_trans = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
    spec_trans = SpectrumRet(bem, exc_trans, wavelengths)
    result_trans = spec_trans.compute()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Longitudinal
    ax1.plot(wavelengths, result_long['ext'][:, 0], 'b-', linewidth=2, label='Extinction')
    ax1.plot(wavelengths, result_long['sca'][:, 0], 'r--', linewidth=2, label='Scattering')
    ax1.plot(wavelengths, result_long['abs'][:, 0], 'g:', linewidth=2, label='Absorption')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Cross section (nm²)', fontsize=12)
    ax1.set_title('Longitudinal Mode', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Transverse
    ax2.plot(wavelengths, result_trans['ext'][:, 0], 'b-', linewidth=2, label='Extinction')
    ax2.plot(wavelengths, result_trans['sca'][:, 0], 'r--', linewidth=2, label='Scattering')
    ax2.plot(wavelengths, result_trans['abs'][:, 0], 'g:', linewidth=2, label='Absorption')
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Cross section (nm²)', fontsize=12)
    ax2.set_title('Transverse Mode', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    long_peak = wavelengths[np.argmax(result_long['ext'][:, 0])]
    trans_peak = wavelengths[np.argmax(result_trans['ext'][:, 0])]
    print(f"\nLongitudinal resonance: {long_peak:.1f} nm")
    print(f"Transverse resonance: {trans_peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_ret_rod.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
