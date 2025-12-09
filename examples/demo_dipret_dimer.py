"""
Demo: Dipole emission near nanosphere dimer.

Shows enhanced emission in dimer gap (hot spot).
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import DipoleRet


def main():
    """Dipole emission in dimer hot spot."""
    print("Demo: Dipole emission near dimer")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    gap = 5  # nm

    # Create dimer
    sphere1 = trisphere(200, diameter).shift([-(diameter + gap)/2, 0, 0])
    sphere2 = trisphere(200, diameter).shift([(diameter + gap)/2, 0, 0])

    p_dimer = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])

    # Single sphere for comparison
    p_single = ComParticle([eps_vacuum, eps_gold], [trisphere(200, diameter)], [[2, 1]])

    print(f"Dimer: d={diameter} nm, gap={gap} nm")

    bem_dimer = BEMRet(p_dimer)
    bem_single = BEMRet(p_single)

    wavelengths = np.linspace(450, 750, 60)

    # Dipole at center of gap (hot spot)
    pos_gap = np.array([[0, 0, 0]])  # Center of dimer
    pol_x = np.array([[1, 0, 0]])  # Parallel to dimer axis
    pol_y = np.array([[0, 1, 0]])  # Perpendicular

    # Dipole near single sphere
    pos_single = np.array([[diameter/2 + gap/2, 0, 0]])  # Same distance

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Configuration 1: Parallel polarization in gap
    print("\nDipole parallel to dimer axis in gap...")
    enh_dimer_x = []
    enh_single = []

    for wl in wavelengths:
        # Dimer
        dip = DipoleRet(position=pos_gap, moment=pol_x)
        exc = dip(p_dimer, wl)
        sig = bem_dimer.solve(exc)
        enh_dimer_x.append(dip.total_decay_rate(sig))

        # Single sphere
        dip = DipoleRet(position=pos_single, moment=pol_x)
        exc = dip(p_single, wl)
        sig = bem_single.solve(exc)
        enh_single.append(dip.total_decay_rate(sig))

    enh_dimer_x = np.array(enh_dimer_x)
    enh_single = np.array(enh_single)

    ax1.plot(wavelengths, enh_dimer_x, 'r-', linewidth=2, label='Dimer gap - parallel')
    ax1.plot(wavelengths, enh_single, 'b--', linewidth=2, label='Single sphere')

    # Configuration 2: Perpendicular polarization in gap
    print("Dipole perpendicular in gap...")
    enh_dimer_y = []

    for wl in wavelengths:
        dip = DipoleRet(position=pos_gap, moment=pol_y)
        exc = dip(p_dimer, wl)
        sig = bem_dimer.solve(exc)
        enh_dimer_y.append(dip.total_decay_rate(sig))

    enh_dimer_y = np.array(enh_dimer_y)
    ax1.plot(wavelengths, enh_dimer_y, 'g:', linewidth=2, label='Dimer gap - perpendicular')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Decay rate enhancement', fontsize=12)
    ax1.set_title('Decay Rate Enhancement', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Calculate enhancement factor relative to single
    peak_idx = np.argmax(enh_dimer_x)
    gap_enhancement = enh_dimer_x.max() / enh_single[peak_idx]
    print(f"\nGap enhancement factor: {gap_enhancement:.1f}×")

    # Plot quantum efficiency
    # For simplicity, assume same radiative decay pattern
    qe_dimer_x = 1 / enh_dimer_x  # Inverse for visualization (simplified)
    qe_single = 1 / enh_single

    ax2.plot(wavelengths, qe_dimer_x / qe_dimer_x.max(), 'r-', linewidth=2,
             label='Dimer gap')
    ax2.plot(wavelengths, qe_single / qe_single.max(), 'b--', linewidth=2,
             label='Single sphere')

    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Relative quantum efficiency', fontsize=12)
    ax2.set_title('Quantum Efficiency (relative)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Dipole Emission Near Dimer (d={diameter} nm, gap={gap} nm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_dipret_dimer.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
