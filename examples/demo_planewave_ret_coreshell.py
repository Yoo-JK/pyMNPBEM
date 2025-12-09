"""
Demo: Retarded BEM for core-shell nanoparticle.

Shows LSPR tunability with shell thickness.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Retarded BEM for silica-gold core-shell."""
    print("Demo: Retarded BEM - Core-shell nanoparticle")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_silica = EpsConst(2.13)  # n=1.46
    eps_gold = EpsDrude('Au')

    wavelengths = np.linspace(400, 1000, 100)

    core_diameter = 50  # nm
    shell_thicknesses = [5, 10, 20, 30]  # nm

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(shell_thicknesses)))

    for i, shell_t in enumerate(shell_thicknesses):
        outer_d = core_diameter + 2 * shell_t
        print(f"\nCore: {core_diameter} nm, Shell: {shell_t} nm, Total: {outer_d} nm")

        # Create core-shell: silica core, gold shell
        core = trisphere(144, core_diameter)
        shell = trisphere(200, outer_d)

        # Inner boundary: core surface (vacuum inside silica shell)
        # Outer boundary: shell surface (gold to vacuum)
        # This is silica@Au (silica core, gold shell)
        p = ComParticle(
            [eps_vacuum, eps_silica, eps_gold],
            [core, shell],
            [[2, 1], [3, 1]]  # core: silica-vacuum, shell: gold-vacuum
        )

        bem = BEMRet(p)
        exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax.plot(wavelengths, ext, color=colors[i], linewidth=2,
                label=f'Shell = {shell_t} nm')

        peak_wl = wavelengths[np.argmax(ext)]
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Core-Shell (SiO₂@Au, core={core_diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_planewave_ret_coreshell.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
