"""
Demo: Dipole in nanoparticle gap (quasistatic).

Shows giant field enhancement in plasmonic hot spots.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import DipoleStat


def main():
    """Dipole enhancement in dimer gap."""
    print("Demo: Dipole in nanoparticle dimer gap")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 30  # nm
    gap = 2  # nm

    # Create dimer
    sphere1 = trisphere(100, diameter).shift([-(diameter + gap)/2, 0, 0])
    sphere2 = trisphere(100, diameter).shift([(diameter + gap)/2, 0, 0])

    p = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])
    print(f"Dimer: d={diameter} nm, gap={gap} nm")

    # Dipole at gap center
    dipole_pos = np.array([[0, 0, 0]])
    dipole = DipoleStat(pt=dipole_pos, dip=np.array([[1, 0, 0]]))  # Along dimer axis

    wavelengths = np.linspace(400, 800, 100)
    bem = BEMStat(p)

    enhancement = []
    print("Computing gap enhancement...")

    for wl in wavelengths:
        exc = dipole(p, wl)
        sig = bem.solve(exc)
        # Simple enhancement estimate
        enh = np.abs(1 + sig.phip.sum() / (exc.phip.sum() + 1e-30))**2
        enhancement.append(enh)

    enhancement = np.array(enhancement)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, enhancement, 'r-', linewidth=2)
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Enhancement factor', fontsize=12)
    ax.set_title(f'Gap Enhancement (d={diameter} nm, gap={gap} nm)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    peak_wl = wavelengths[np.argmax(enhancement)]
    peak_enh = np.max(enhancement)
    print(f"\nPeak enhancement: {peak_enh:.0f}× at {peak_wl:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_dipstat_gap.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
