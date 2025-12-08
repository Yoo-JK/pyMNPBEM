"""
Demo: Different nanoparticle shapes

This example demonstrates the various nanoparticle shapes available:
- Sphere (trisphere)
- Rod (trirod)
- Cube (tricube)
- Torus (tritorus)
- Ellipsoid (triellipsoid)
- Cone (tricone)
- Nanodisk (trinanodisk)
- Plate (triplate)

Compares optical spectra of different shapes.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import bemoptions, EpsConst, EpsTable, ComParticle, bemsolver, planewave
from mnpbem.particles.shapes import (
    trisphere, trirod, tricube, tritorus,
    triellipsoid, tricone, trinanodisk, triplate
)


def compute_spectrum(particle, bem, exc, wavelengths):
    """Compute scattering spectrum for a particle."""
    sca = np.zeros(len(wavelengths))
    for i, wl in enumerate(wavelengths):
        sig = bem.solve(exc(particle, wl))
        sca[i] = exc.sca(sig)[0]
    return sca


def main():
    """Compare optical spectra of different shapes."""

    print("MNPBEM Demo: Different nanoparticle shapes")
    print("=" * 60)

    # Options
    op = bemoptions(sim='stat', waitbar=0)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    epstab = [eps_vacuum, eps_gold]

    # Create different shapes (all with similar volume)
    print("Creating particles...")

    shapes = {}

    # Sphere (d=30 nm)
    shapes['Sphere'] = trisphere(144, 30)

    # Rod (d=15, L=60 nm, aspect ratio 4)
    shapes['Rod'] = trirod(15, 60, mesh='normal')

    # Cube (edge=24 nm)
    shapes['Cube'] = tricube(24, mesh='fine')

    # Torus (R=20, r=5 nm)
    shapes['Torus'] = tritorus(20, 5)

    # Ellipsoid (axes: 40x20x20 nm, prolate)
    shapes['Ellipsoid'] = triellipsoid([40, 20, 20], n=50)

    # Cone (base d=30, height=30 nm)
    shapes['Cone'] = tricone(30, 30)

    # Nanodisk (d=40, height=10 nm)
    shapes['Nanodisk'] = trinanodisk(40, 10)

    # Plate (40x40x5 nm)
    shapes['Plate'] = triplate([40, 40, 5])

    # Print info
    for name, shape in shapes.items():
        print(f"  {name}: {shape.n_faces} faces")

    # Wavelengths
    wavelengths = np.linspace(400, 900, 80)

    # Plane wave excitation
    exc = planewave([[1, 0, 0]], [[0, 0, 1]], op)

    # Compute spectra
    print(f"\nComputing spectra for {len(shapes)} shapes...")
    spectra = {}

    for name, shape in shapes.items():
        print(f"  Processing {name}...")
        p = ComParticle(epstab, [shape], [[2, 1]], closed=1)
        bem = bemsolver(p, op)
        spectra[name] = compute_spectrum(p, bem, exc, wavelengths)

    print("Calculations complete!")

    # Plot results
    print("\nPlotting results...")

    # Color scheme
    colors = plt.cm.tab10(range(len(shapes)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # All spectra comparison
    ax1 = axes[0, 0]
    for i, (name, sca) in enumerate(spectra.items()):
        ax1.plot(wavelengths, sca, color=colors[i], linewidth=2, label=name)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Scattering cross section (nm²)')
    ax1.set_title('All shapes comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Normalized spectra
    ax2 = axes[0, 1]
    for i, (name, sca) in enumerate(spectra.items()):
        sca_norm = sca / np.max(sca)
        ax2.plot(wavelengths, sca_norm, color=colors[i], linewidth=2, label=name)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Normalized scattering')
    ax2.set_title('Normalized spectra (peak = 1)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Isotropic vs anisotropic shapes
    ax3 = axes[1, 0]
    isotropic = ['Sphere', 'Cube', 'Torus']
    for i, name in enumerate(isotropic):
        if name in spectra:
            ax3.plot(wavelengths, spectra[name], linewidth=2, label=name)

    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Scattering cross section (nm²)')
    ax3.set_title('Isotropic shapes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Elongated shapes (rod, ellipsoid)
    ax4 = axes[1, 1]
    elongated = ['Sphere', 'Rod', 'Ellipsoid']
    for name in elongated:
        if name in spectra:
            ax4.plot(wavelengths, spectra[name], linewidth=2, label=name)

    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Scattering cross section (nm²)')
    ax4.set_title('Elongated shapes (longitudinal mode)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Gold nanoparticle shapes - Optical spectra', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_shapes_result.png', dpi=150)
    print("Figure saved to demo_shapes_result.png")

    # Print resonance wavelengths
    print("\nResonance wavelengths:")
    for name, sca in spectra.items():
        peak_wl = wavelengths[np.argmax(sca)]
        peak_val = np.max(sca)
        print(f"  {name:12s}: {peak_wl:.1f} nm (peak: {peak_val:.1f} nm²)")

    plt.show()


if __name__ == '__main__':
    main()
