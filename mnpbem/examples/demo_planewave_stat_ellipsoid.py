"""
Demo: Plane wave excitation of gold nanoellipsoid (quasistatic).

Shows depolarization effects in non-spherical particles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import triellipsoid
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanoellipsoid - Polarization-dependent response."""
    print("Demo: Gold nanoellipsoid - Anisotropic response")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create prolate ellipsoid (elongated along z)
    axes = (10, 10, 30)  # (a, b, c) semi-axes in nm
    ellipsoid = triellipsoid(200, axes)
    print(f"Ellipsoid axes: a={axes[0]}, b={axes[1]}, c={axes[2]} nm")
    print(f"Aspect ratio: {axes[2]/axes[0]:.1f}")

    p = ComParticle([eps_vacuum, eps_gold], [ellipsoid], [[2, 1]])

    wavelengths = np.linspace(400, 1000, 150)

    # Three polarizations
    pols = {
        'x': [1, 0, 0],
        'y': [0, 1, 0],
        'z': [0, 0, 1]
    }

    bem = BEMStat(p)
    results = {}

    for name, pol in pols.items():
        print(f"Computing {name}-polarization...")
        exc = PlaneWaveStat(pol=pol)
        spec = SpectrumStat(bem, exc, wavelengths, show_progress=False)
        _, ext = spec.compute()
        results[name] = ext[:, 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'x': 'r', 'y': 'g', 'z': 'b'}

    for name, ext in results.items():
        ax.plot(wavelengths, ext, colors[name] + '-', linewidth=2, label=f'{name}-polarization')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanoellipsoid (axes: {axes[0]}×{axes[1]}×{axes[2]} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Find resonances
    for name, ext in results.items():
        peak = wavelengths[np.argmax(ext)]
        print(f"\n{name}-polarization resonance: {peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_ellipsoid.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
