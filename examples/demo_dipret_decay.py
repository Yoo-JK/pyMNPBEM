"""
Demo: Dipole decay rate near plasmonic nanoparticle (retarded).

Shows Purcell enhancement of spontaneous emission.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import DipoleRet, DecayRateSpectrum


def main():
    """Purcell enhancement near gold nanosphere."""
    print("Demo: Purcell enhancement - Retarded BEM")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 60  # nm
    sphere = trisphere(300, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    # Dipole positions at different distances
    distances = [5, 10, 20, 40]  # nm from surface
    wavelengths = np.linspace(450, 700, 60)

    bem = BEMRet(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(distances)))

    for i, d in enumerate(distances):
        print(f"Computing distance = {d} nm...")
        dipole_pos = np.array([[diameter/2 + d, 0, 0]])
        dipole_dip = np.array([[1, 0, 0]])  # Radial

        exc = DipoleRet(pt=dipole_pos, dip=dipole_dip)

        purcell = []
        for wl in wavelengths:
            exc_wl = exc(p, wl)
            sig = bem.solve(exc_wl)
            # Estimate Purcell factor from imaginary part of Green function
            k = 2 * np.pi / wl
            pf = 1 + 6 * np.pi / k**3 * np.abs(sig.phip.sum())
            purcell.append(pf)

        ax.plot(wavelengths, purcell, color=colors[i], linewidth=2,
                label=f'd={d} nm')

    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Purcell factor', fontsize=12)
    ax.set_title(f'Purcell Enhancement near Au Sphere (D={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_dipret_decay.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
