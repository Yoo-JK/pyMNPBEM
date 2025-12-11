"""
Demo: Comparison of different plasmonic materials (quasistatic).

Shows plasmon resonances for gold, silver, and aluminum.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Compare Au, Ag, Al plasmon resonances."""
    print("Demo: Material comparison - Au, Ag, Al")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    materials = {
        'Gold': EpsDrude('Au'),
        'Silver': EpsDrude('Ag'),
        'Aluminum': EpsDrude('Al')
    }

    diameter = 30  # nm
    wavelengths = np.linspace(200, 800, 200)
    exc = PlaneWaveStat(pol=[1, 0, 0])

    results = {}

    for name, eps_mat in materials.items():
        print(f"Computing {name}...")
        sphere = trisphere(144, diameter)
        p = ComParticle([eps_vacuum, eps_mat], [sphere], [[2, 1]])
        bem = BEMStat(p)
        spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
        _, ext = spec.compute()
        results[name] = ext[:, 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Gold': 'gold', 'Silver': 'silver', 'Aluminum': 'steelblue'}

    for name, ext in results.items():
        peak = wavelengths[np.argmax(ext)]
        ax.plot(wavelengths, ext, color=colors[name], linewidth=2,
                label=f'{name} (peak={peak:.0f} nm)')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Plasmonic Materials Comparison (d={diameter} nm sphere)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(200, 800)

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_materials.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
