"""
Demo: Mie theory comparison

This example demonstrates Mie theory calculations for spherical particles:
- Quasistatic Mie theory (small particles)
- Full retarded Mie theory (arbitrary size)
- Comparison of efficiency factors

Shows how plasmon resonance shifts and broadens with increasing size.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import EpsConst, EpsTable
from mnpbem.mie import MieStat, MieRet, mie_efficiencies


def main():
    """Demonstrate Mie theory calculations."""

    print("MNPBEM Demo: Mie theory calculations")
    print("=" * 60)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')

    # Wavelengths
    wavelengths = np.linspace(400, 900, 200)

    # Different particle sizes
    diameters = [10, 30, 60, 100, 150]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(diameters)))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print("Computing Mie scattering for different sizes...")

    # Scattering cross sections
    ax1 = axes[0, 0]
    for i, d in enumerate(diameters):
        if d <= 30:
            # Use quasistatic for small particles
            mie = MieStat(eps_gold, eps_vacuum, d)
            label = f'd={d} nm (QS)'
        else:
            # Use retarded for larger particles
            mie = MieRet(eps_gold, eps_vacuum, d)
            label = f'd={d} nm (Ret)'

        sca = mie.sca(wavelengths)
        ax1.plot(wavelengths, sca, color=colors[i], linewidth=2, label=label)
        print(f"  Diameter {d} nm: peak at {wavelengths[np.argmax(sca)]:.1f} nm")

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Scattering cross section (nm²)')
    ax1.set_title('Scattering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Extinction cross sections
    ax2 = axes[0, 1]
    for i, d in enumerate(diameters):
        if d <= 30:
            mie = MieStat(eps_gold, eps_vacuum, d)
        else:
            mie = MieRet(eps_gold, eps_vacuum, d)

        ext = mie.ext(wavelengths)
        ax2.plot(wavelengths, ext, color=colors[i], linewidth=2, label=f'd={d} nm')

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Extinction cross section (nm²)')
    ax2.set_title('Extinction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Efficiency factors (Q = C / pi*a^2)
    ax3 = axes[1, 0]
    for i, d in enumerate(diameters):
        mie = MieRet(eps_gold, eps_vacuum, d)
        sca = mie.sca(wavelengths)
        # Convert to efficiency
        Q_sca = sca / (np.pi * (d/2)**2)
        ax3.plot(wavelengths, Q_sca, color=colors[i], linewidth=2, label=f'd={d} nm')

    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Scattering efficiency Q_sca')
    ax3.set_title('Scattering Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Quasistatic vs Retarded comparison for 60nm
    ax4 = axes[1, 1]
    d = 60
    mie_qs = MieStat(eps_gold, eps_vacuum, d)
    mie_ret = MieRet(eps_gold, eps_vacuum, d)

    ax4.plot(wavelengths, mie_qs.sca(wavelengths), 'b-', linewidth=2,
             label='Quasistatic')
    ax4.plot(wavelengths, mie_ret.sca(wavelengths), 'r--', linewidth=2,
             label='Retarded')

    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Scattering cross section (nm²)')
    ax4.set_title(f'QS vs Retarded (d={d} nm)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Gold nanosphere Mie theory', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_mie_result.png', dpi=150)
    print("\nFigure saved to demo_mie_result.png")

    plt.show()


if __name__ == '__main__':
    main()
