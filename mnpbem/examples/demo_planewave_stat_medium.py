"""
Demo: Effect of surrounding medium on plasmon resonance (quasistatic).

Shows refractive index sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Refractive index sensitivity of gold nanosphere."""
    print("Demo: Refractive index sensitivity")
    print("=" * 60)

    eps_gold = EpsDrude('Au')

    # Different surrounding media
    media = {
        'Air (n=1.0)': EpsConst(1.0),
        'Water (n=1.33)': EpsConst(1.33**2),
        'Ethanol (n=1.36)': EpsConst(1.36**2),
        'Glass (n=1.5)': EpsConst(1.5**2),
        'TiO2 (n=2.4)': EpsConst(2.4**2),
    }

    diameter = 40  # nm
    wavelengths = np.linspace(400, 900, 150)
    exc = PlaneWaveStat(pol=[1, 0, 0])

    results = {}
    peaks = []
    n_values = []

    for name, eps_med in media.items():
        n = np.sqrt(eps_med.value).real
        n_values.append(n)
        print(f"Computing: {name}...")

        sphere = trisphere(144, diameter)
        p = ComParticle([eps_med, eps_gold], [sphere], [[2, 1]])
        bem = BEMStat(p)
        spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
        _, ext = spec.compute()
        results[name] = ext[:, 0]
        peaks.append(wavelengths[np.argmax(ext[:, 0])])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(media)))
    for i, (name, ext) in enumerate(results.items()):
        ax1.plot(wavelengths, ext, color=colors[i], linewidth=2, label=name)

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax1.set_title(f'Gold Nanosphere in Different Media (d={diameter} nm)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Sensitivity plot
    ax2.plot(n_values, peaks, 'bo-', markersize=10, linewidth=2)
    ax2.set_xlabel('Refractive index', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('Refractive Index Sensitivity', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Calculate sensitivity
    coeffs = np.polyfit(n_values, peaks, 1)
    sensitivity = coeffs[0]
    print(f"\nSensitivity: {sensitivity:.1f} nm/RIU")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_medium.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
