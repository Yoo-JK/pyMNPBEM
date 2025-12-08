"""
Demo: Retarded BEM scattering of larger nanosphere

This example computes the scattering cross section for a gold nanosphere
using the full retarded BEM solver, and compares with retarded Mie theory.

For larger particles (diameter > ~30 nm), retardation effects become
important and the quasistatic approximation breaks down.

This is a Python port of the MATLAB demo demospecret1.m
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import (
    bemoptions, EpsConst, EpsTable,
    ComParticle, bemsolver, planewave
)
from mnpbem.particles.shapes import trisphere
from mnpbem.mie import MieRet


def main():
    """Run the retarded gold nanosphere scattering simulation."""

    print("MNPBEM Demo: Retarded BEM scattering of nanosphere")
    print("=" * 60)

    # Options for BEM simulation - use retarded solver
    op = bemoptions(sim='ret', waitbar=0)

    # Dielectric functions
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    epstab = [eps_vacuum, eps_gold]

    # Diameter of sphere - larger for retardation effects
    diameter = 80  # nm

    # Initialize sphere with finer mesh for larger particle
    sphere = trisphere(256, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    print(f"Particle: {p}")
    print(f"Number of faces: {p.n_faces}")

    # Set up BEM solver (retarded)
    bem = bemsolver(p, op)
    print(f"BEM solver: {bem}")

    # Plane wave excitation
    exc = planewave(
        [[1, 0, 0]],  # x-polarization
        [[0, 0, 1]],  # z-direction
        op
    )

    # Light wavelengths
    wavelengths = np.linspace(400, 800, 50)

    # Allocate arrays
    sca_bem = np.zeros(len(wavelengths))
    ext_bem = np.zeros(len(wavelengths))

    print(f"\nComputing retarded BEM spectra for {len(wavelengths)} wavelengths...")

    # Loop over wavelengths
    for i, enei in enumerate(wavelengths):
        # Get excitation
        exc_pot = exc(p, enei)

        # Solve BEM equations
        sig = bem.solve(exc_pot)

        # Compute cross sections
        sca_bem[i] = exc.sca(sig)[0]
        ext_bem[i] = exc.ext(sig)[0]

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(wavelengths)} wavelengths")

    print("Retarded BEM calculation complete!")

    # Retarded Mie theory for comparison
    print("\nComputing retarded Mie theory reference...")
    mie = MieRet(eps_gold, eps_vacuum, diameter)
    sca_mie = mie.sca(wavelengths)
    ext_mie = mie.ext(wavelengths)

    # Plot results
    print("\nPlotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scattering
    ax1 = axes[0]
    ax1.plot(wavelengths, sca_bem, 'b-', linewidth=2, label='Retarded BEM')
    ax1.plot(wavelengths, sca_mie, 'r--', linewidth=2, label='Retarded Mie')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Scattering cross section (nm²)')
    ax1.set_title('Scattering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Extinction
    ax2 = axes[1]
    ax2.plot(wavelengths, ext_bem, 'b-', linewidth=2, label='Retarded BEM')
    ax2.plot(wavelengths, ext_mie, 'r--', linewidth=2, label='Retarded Mie')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Extinction cross section (nm²)')
    ax2.set_title('Extinction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Gold nanosphere, diameter = {diameter} nm (retarded)', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_specret1_result.png', dpi=150)
    print("Figure saved to demo_specret1_result.png")

    plt.show()


if __name__ == '__main__':
    main()
