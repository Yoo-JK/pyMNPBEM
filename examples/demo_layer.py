"""
Demo: Nanoparticle on substrate

This example demonstrates simulations of nanoparticles near a dielectric
or metallic substrate using:
- PlaneWaveStatLayer: Plane wave with Fresnel reflection
- PlaneWaveStatMirror: Perfect mirror approximation

The substrate modifies the local field and shifts the plasmon resonance.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import bemoptions, EpsConst, EpsTable, ComParticle, bemsolver
from mnpbem.particles.shapes import trisphere
from mnpbem.simulation import PlaneWaveStatLayer, PlaneWaveStatMirror


def main():
    """Simulate nanoparticle on substrate."""

    print("MNPBEM Demo: Nanoparticle on substrate")
    print("=" * 60)

    # Options
    op = bemoptions(sim='stat', waitbar=0)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    eps_glass = EpsConst(2.25)  # n=1.5 glass

    epstab = [eps_vacuum, eps_gold]

    # Gold nanosphere
    diameter = 20  # nm
    sphere = trisphere(144, diameter)

    # Position above substrate (center at z = diameter/2 + gap)
    gap = 2  # nm gap between sphere and substrate
    z_offset = diameter/2 + gap

    # Shift sphere upward
    sphere_verts = sphere.verts.copy()
    sphere_verts[:, 2] += z_offset

    from mnpbem import Particle
    sphere_shifted = Particle(sphere_verts, sphere.faces)

    # Create composite particle
    p = ComParticle(epstab, [sphere_shifted], [[2, 1]], closed=1)

    print(f"Particle: {p}")
    print(f"Sphere center height: {z_offset} nm above substrate")

    # BEM solver
    bem = bemsolver(p, op)

    # Wavelengths
    wavelengths = np.linspace(450, 750, 60)

    # Create different excitations
    print("\nSetting up excitations...")

    # 1. Free space (no substrate)
    from mnpbem.simulation import PlaneWaveStat
    exc_free = PlaneWaveStat(
        pol=[[1, 0, 0]],  # x-polarization
        dir=[[0, 0, -1]],  # Coming from +z
        options=op
    )

    # 2. Glass substrate (Fresnel reflection)
    exc_glass = PlaneWaveStatLayer(
        pol=[[1, 0, 0]],
        dir=[[0, 0, -1]],
        eps_substrate=eps_glass,
        options=op
    )

    # 3. Perfect mirror substrate
    exc_mirror = PlaneWaveStatMirror(
        pol=[[1, 0, 0]],
        dir=[[0, 0, -1]],
        options=op
    )

    # Compute spectra
    sca_free = np.zeros(len(wavelengths))
    sca_glass = np.zeros(len(wavelengths))
    sca_mirror = np.zeros(len(wavelengths))

    print(f"Computing spectra for {len(wavelengths)} wavelengths...")

    for i, wl in enumerate(wavelengths):
        # Free space
        sig = bem.solve(exc_free(p, wl))
        sca_free[i] = exc_free.sca(sig)[0]

        # Glass substrate
        sig = bem.solve(exc_glass(p, wl))
        sca_glass[i] = exc_glass.sca(sig)[0]

        # Mirror substrate
        sig = bem.solve(exc_mirror(p, wl))
        sca_mirror[i] = exc_mirror.sca(sig)[0]

        if (i + 1) % 15 == 0:
            print(f"  Processed {i + 1}/{len(wavelengths)} wavelengths")

    print("Calculation complete!")

    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))

    plt.plot(wavelengths, sca_free, 'b-', linewidth=2, label='Free space')
    plt.plot(wavelengths, sca_glass, 'g-', linewidth=2, label='Glass substrate')
    plt.plot(wavelengths, sca_mirror, 'r-', linewidth=2, label='Mirror substrate')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering cross section (nm²)')
    plt.title(f'Gold nanosphere (d={diameter} nm) on substrate\n'
              f'Gap = {gap} nm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_layer_result.png', dpi=150)
    print("Figure saved to demo_layer_result.png")

    # Print peak positions
    print(f"\nPeak wavelengths:")
    print(f"  Free space: {wavelengths[np.argmax(sca_free)]:.1f} nm")
    print(f"  Glass:      {wavelengths[np.argmax(sca_glass)]:.1f} nm")
    print(f"  Mirror:     {wavelengths[np.argmax(sca_mirror)]:.1f} nm")

    plt.show()


if __name__ == '__main__':
    main()
