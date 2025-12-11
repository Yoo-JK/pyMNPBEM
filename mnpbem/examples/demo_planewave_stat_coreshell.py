"""
Demo: Plane wave excitation of core-shell nanoparticle (quasistatic).

Shows tunable plasmon resonance by varying shell thickness.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanoshell with silica core."""
    print("Demo: Core-shell nanoparticle - Tunable resonance")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_silica = EpsConst(2.04)  # Silica core
    eps_gold = EpsDrude('Au')

    # Core-shell geometry
    core_radius = 30  # nm
    shell_thicknesses = [5, 10, 15, 20]  # nm

    wavelengths = np.linspace(400, 1200, 200)
    exc = PlaneWaveStat(pol=[1, 0, 0])

    results = {}

    for shell_t in shell_thicknesses:
        outer_radius = core_radius + shell_t
        print(f"\nProcessing: core={core_radius} nm, shell={shell_t} nm")

        # Create core and shell
        core = trisphere(100, 2 * core_radius)
        shell = trisphere(144, 2 * outer_radius)

        # Core-shell: inner surface is silica-gold, outer is gold-vacuum
        # [vacuum, gold, silica] with interfaces defined
        p = ComParticle(
            [eps_vacuum, eps_gold, eps_silica],
            [shell, core],
            [[2, 1], [3, 2]]  # shell: gold-vacuum, core: silica-gold
        )

        bem = BEMStat(p)
        spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
        _, ext = spec.compute()
        results[shell_t] = ext[:, 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(shell_thicknesses)))

    for i, shell_t in enumerate(shell_thicknesses):
        peak = wavelengths[np.argmax(results[shell_t])]
        ax.plot(wavelengths, results[shell_t], color=colors[i], linewidth=2,
                label=f'Shell={shell_t} nm (peak={peak:.0f} nm)')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanoshell (Silica core r={core_radius} nm)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_coreshell.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
