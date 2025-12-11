"""
Demo: Size-dependent plasmon resonance (quasistatic).

Shows how plasmon resonance shifts with particle size.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Size dependence of gold nanosphere plasmon."""
    print("Demo: Size-dependent plasmon resonance")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameters = [10, 20, 40, 60, 80, 100]  # nm
    wavelengths = np.linspace(400, 700, 100)
    exc = PlaneWaveStat(pol=[1, 0, 0])

    results = {}
    peaks = []

    for d in diameters:
        print(f"Computing d={d} nm...")
        sphere = trisphere(144, d)
        p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
        bem = BEMStat(p)
        spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
        _, ext = spec.compute()
        results[d] = ext[:, 0]
        peaks.append(wavelengths[np.argmax(ext[:, 0])])

    # Plot spectra
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(diameters)))
    for i, d in enumerate(diameters):
        # Normalize for comparison
        ext_norm = results[d] / results[d].max()
        ax1.plot(wavelengths, ext_norm, color=colors[i], linewidth=2, label=f'd={d} nm')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Normalized extinction', fontsize=12)
    ax1.set_title('Normalized Extinction Spectra', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot peak position vs size
    ax2.plot(diameters, peaks, 'bo-', markersize=10, linewidth=2)
    ax2.set_xlabel('Diameter (nm)', fontsize=12)
    ax2.set_ylabel('Peak wavelength (nm)', fontsize=12)
    ax2.set_title('Plasmon Resonance vs Size', fontsize=14)
    ax2.grid(True, alpha=0.3)

    print("\nResonance positions:")
    for d, p in zip(diameters, peaks):
        print(f"  d={d} nm: {p:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_size.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
