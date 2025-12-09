"""
Demo: Position-dependent dipole enhancement.

Shows enhancement as function of dipole position near particle.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, Point
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import DipoleStat


def main():
    """Position-dependent enhancement near gold nanosphere."""
    print("Demo: Position-dependent dipole enhancement")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    sphere = trisphere(200, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    bem = BEMStat(p)

    wavelength = 520  # nm (near resonance)
    print(f"Wavelength: {wavelength} nm")

    # Scan dipole along x-axis
    distances = np.linspace(1, 30, 30)  # nm from surface
    orientations = ['radial', 'tangential']

    fig, ax = plt.subplots(figsize=(10, 6))

    for ori in orientations:
        print(f"\nOrientation: {ori}")
        enhancement = []

        for d in distances:
            r = diameter/2 + d
            pos = np.array([[r, 0, 0]])

            if ori == 'radial':
                pol = np.array([[1, 0, 0]])  # Along r direction
            else:
                pol = np.array([[0, 1, 0]])  # Perpendicular to r

            dip = DipoleStat(position=pos, moment=pol)
            exc = dip(p, wavelength)
            sig = bem.solve(exc)

            # Calculate enhancement from induced potential
            enh = 1 + np.abs(sig.phip.sum())
            enhancement.append(enh)

        enhancement = np.array(enhancement)
        ax.semilogy(distances, enhancement, linewidth=2, label=ori.capitalize())
        print(f"  Max enhancement: {enhancement.max():.1f}×")

    ax.set_xlabel('Distance from surface (nm)', fontsize=12)
    ax.set_ylabel('Enhancement factor', fontsize=12)
    ax.set_title(f'Dipole Enhancement vs Position (d={diameter} nm, λ={wavelength} nm)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_dipstat_position.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
