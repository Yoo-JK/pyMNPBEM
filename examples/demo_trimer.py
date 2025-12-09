"""
Demo: Nanosphere trimer plasmon modes.

Shows coupled modes in triangular arrangement.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Plasmon modes of gold nanosphere trimer."""
    print("Demo: Nanosphere trimer")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    gap = 5  # nm

    # Triangular arrangement in xy plane
    # Distance from center to each sphere center
    d_center = (diameter + gap) / np.sqrt(3)

    angles = [0, 2*np.pi/3, 4*np.pi/3]
    spheres = []
    for angle in angles:
        x = d_center * np.cos(angle)
        y = d_center * np.sin(angle)
        sphere = trisphere(200, diameter).shift([x, y, 0])
        spheres.append(sphere)

    p = ComParticle([eps_vacuum, eps_gold], spheres, [[2, 1], [2, 1], [2, 1]])
    print(f"Trimer: d={diameter} nm, gap={gap} nm")
    print(f"Total faces: {p.n_faces}")

    bem = BEMRet(p)
    wavelengths = np.linspace(450, 800, 80)

    # Different polarizations
    polarizations = {
        'X-polarized': [1, 0, 0],
        'Y-polarized': [0, 1, 0],
        'Circular (RCP)': [1, 1j, 0]  # Right circular polarization
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['r', 'b', 'g']

    for i, (name, pol) in enumerate(polarizations.items()):
        print(f"\nPolarization: {name}")

        exc = PlaneWaveRet(pol=pol, direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax.plot(wavelengths, ext, color=colors[i], linewidth=2, label=name)

        peak_wl = wavelengths[np.argmax(ext)]
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanosphere Trimer (d={diameter} nm, gap={gap} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_trimer.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
