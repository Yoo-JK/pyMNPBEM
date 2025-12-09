"""
Demo: EELS spectrum at different positions.

Shows energy-loss spectrum at tip vs side of nanorod.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import EELSRet


def main():
    """EELS spectrum at different positions around nanorod."""
    print("Demo: Position-dependent EELS spectrum")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 80  # nm
    diameter = 25  # nm
    rod = trirod(300, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    print(f"Nanorod: L={length} nm, d={diameter} nm")

    bem = BEMRet(p)
    beam_energy = 100e3  # 100 keV

    # Energy range (converted to wavelength for BEM)
    energies = np.linspace(0.5, 3.0, 100)  # eV
    wavelengths = 1239.8 / energies  # Convert eV to nm

    # Beam positions
    positions = {
        'Tip (x)': np.array([length/2 + 5, 0, 0]),
        'Side (y)': np.array([0, diameter/2 + 5, 0]),
        'Corner': np.array([length/2 - 10, diameter/2 + 5, 0])
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['r', 'b', 'g']

    for i, (name, pos) in enumerate(positions.items()):
        print(f"\nPosition: {name} at {pos}")

        loss_spectrum = []
        for wl in wavelengths:
            eels = EELSRet(position=pos, beam_energy=beam_energy, direction=[0, 0, 1])
            exc = eels(p, wl)
            sig = bem.solve(exc)
            loss = eels.loss_probability(sig)
            loss_spectrum.append(loss)

        loss_spectrum = np.array(loss_spectrum)
        loss_spectrum /= loss_spectrum.max()  # Normalize

        ax.plot(energies, loss_spectrum, color=colors[i], linewidth=2, label=name)

        # Find peak
        peak_idx = np.argmax(loss_spectrum)
        peak_energy = energies[peak_idx]
        print(f"  Peak energy: {peak_energy:.2f} eV ({wavelengths[peak_idx]:.0f} nm)")

    ax.set_xlabel('Energy loss (eV)', fontsize=12)
    ax.set_ylabel('Normalized loss probability', fontsize=12)
    ax.set_title(f'EELS Spectrum at Different Positions (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_eels_spectrum.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
