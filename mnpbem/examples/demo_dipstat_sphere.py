"""
Demo: Dipole excitation near gold nanosphere (quasistatic).

Shows enhancement of dipole emission near plasmonic particles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import DipoleStat


def main():
    """Dipole emission enhancement near gold nanosphere."""
    print("Demo: Dipole near gold nanosphere")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    sphere = trisphere(200, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    # Place dipole near the sphere
    gap = 5  # nm from surface
    dipole_pos = np.array([[diameter/2 + gap, 0, 0]])

    # Radial dipole (pointing toward sphere)
    dipole_rad = DipoleStat(pt=dipole_pos, dip=np.array([[1, 0, 0]]))
    # Tangential dipole (perpendicular to radial)
    dipole_tan = DipoleStat(pt=dipole_pos, dip=np.array([[0, 1, 0]]))

    wavelengths = np.linspace(400, 700, 80)
    bem = BEMStat(p)

    enhancement_rad = []
    enhancement_tan = []

    print("Computing dipole enhancement...")
    for wl in wavelengths:
        # Radial
        exc_rad = dipole_rad(p, wl)
        sig_rad = bem.solve(exc_rad)
        enh_r = np.abs(1 + sig_rad.phip.sum() / exc_rad.phip.sum())**2
        enhancement_rad.append(enh_r)

        # Tangential
        exc_tan = dipole_tan(p, wl)
        sig_tan = bem.solve(exc_tan)
        enh_t = np.abs(1 + sig_tan.phip.sum() / exc_tan.phip.sum())**2
        enhancement_tan.append(enh_t)

    enhancement_rad = np.array(enhancement_rad)
    enhancement_tan = np.array(enhancement_tan)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, enhancement_rad, 'r-', linewidth=2, label='Radial dipole')
    ax.plot(wavelengths, enhancement_tan, 'b-', linewidth=2, label='Tangential dipole')
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Free space')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Enhancement factor', fontsize=12)
    ax.set_title(f'Dipole Enhancement near Au Sphere (d={diameter} nm, gap={gap} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    peak_wl = wavelengths[np.argmax(enhancement_rad)]
    peak_enh = np.max(enhancement_rad)
    print(f"\nMax radial enhancement: {peak_enh:.1f}× at {peak_wl:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_dipstat_sphere.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
