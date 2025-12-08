"""
Demo: Electron Energy Loss Spectroscopy (EELS)

This example demonstrates EELS simulations of a metallic nanoparticle.
A fast electron passing near a nanoparticle can excite plasmon modes,
which can be detected as energy losses in the electron beam.

EELS is a powerful technique in transmission electron microscopy for
mapping plasmonic fields at nanometer resolution.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import bemoptions, EpsConst, EpsTable, ComParticle, bemsolver
from mnpbem.particles.shapes import trisphere, trirod
from mnpbem.simulation import EELSStat


def main():
    """Demonstrate EELS simulation."""

    print("MNPBEM Demo: Electron Energy Loss Spectroscopy")
    print("=" * 60)

    # Options
    op = bemoptions(sim='stat', waitbar=0)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_silver = EpsTable('silver.dat')
    epstab = [eps_vacuum, eps_silver]

    # Silver nanorod
    print("Creating silver nanorod...")
    diameter = 20  # nm
    length = 60  # nm
    rod = trirod(diameter, length, mesh='fine')
    p = ComParticle(epstab, [rod], [[2, 1]], closed=1)

    print(f"Particle: {p}")
    print(f"Rod dimensions: {diameter} x {length} nm")

    # BEM solver
    bem = bemsolver(p, op)

    # Electron parameters
    electron_velocity = 0.5  # fraction of speed of light
    impact_param = 15  # nm from rod axis

    # Create EELS excitation
    # Electron trajectory along z passing at y = impact_param
    eels_exc = EELSStat(
        impact_param=[0, impact_param, 0],  # Impact position
        velocity=electron_velocity,
        direction=[0, 0, 1],  # Along z
        options=op
    )

    # Energy range (eV)
    energies = np.linspace(1.5, 4.0, 60)
    # Convert to wavelengths
    from mnpbem.misc.units import eV2nm
    wavelengths = eV2nm(energies)

    # Compute EELS spectrum
    print(f"\nComputing EELS spectrum for {len(energies)} energies...")
    loss_prob = np.zeros(len(energies))

    for i, wl in enumerate(wavelengths):
        # Get excitation potential
        exc_pot = eels_exc(p, wl)

        # Solve BEM
        sig = bem.solve(exc_pot)

        # Compute loss probability
        loss_prob[i] = eels_exc.loss(sig)

        if (i + 1) % 15 == 0:
            print(f"  Processed {i + 1}/{len(energies)} energies")

    print("EELS calculation complete!")

    # Create EELS map (spatial mapping at fixed energy)
    print("\nCreating EELS spatial map...")

    # Grid of impact parameters
    x_grid = np.linspace(-40, 40, 41)
    y_grid = np.linspace(-25, 25, 26)

    # Fixed energy (plasmon resonance)
    peak_idx = np.argmax(loss_prob)
    peak_energy = energies[peak_idx]
    peak_wl = wavelengths[peak_idx]
    print(f"Mapping at plasmon peak: {peak_energy:.2f} eV ({peak_wl:.1f} nm)")

    loss_map = np.zeros((len(y_grid), len(x_grid)))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            # Create EELS excitation at this position
            eels_local = EELSStat(
                impact_param=[x, y, 0],
                velocity=electron_velocity,
                direction=[0, 0, 1],
                options=op
            )

            # Solve
            exc_pot = eels_local(p, peak_wl)
            sig = bem.solve(exc_pot)
            loss_map[i, j] = eels_local.loss(sig)

        if (i + 1) % 5 == 0:
            print(f"  Row {i + 1}/{len(y_grid)}")

    print("EELS mapping complete!")

    # Plot results
    print("\nPlotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # EELS spectrum
    ax1 = axes[0]
    ax1.plot(energies, loss_prob * 1e6, 'b-', linewidth=2)
    ax1.axvline(peak_energy, color='r', linestyle='--', alpha=0.5,
                label=f'Peak: {peak_energy:.2f} eV')
    ax1.set_xlabel('Energy loss (eV)')
    ax1.set_ylabel('Loss probability (×10⁻⁶)')
    ax1.set_title(f'EELS spectrum\nImpact parameter = {impact_param} nm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # EELS map
    ax2 = axes[1]
    im = ax2.pcolormesh(x_grid, y_grid, loss_map * 1e6, cmap='hot')
    plt.colorbar(im, ax=ax2, label='Loss probability (×10⁻⁶)')

    # Overlay rod outline (simplified rectangle)
    rect_x = [-length/2, length/2, length/2, -length/2, -length/2]
    rect_y = [-diameter/2, -diameter/2, diameter/2, diameter/2, -diameter/2]
    ax2.plot(rect_x, rect_y, 'w-', linewidth=2)

    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_title(f'EELS map at {peak_energy:.2f} eV')
    ax2.set_aspect('equal')

    fig.suptitle(f'Silver nanorod EELS simulation', fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_eels_result.png', dpi=150)
    print("Figure saved to demo_eels_result.png")

    plt.show()


if __name__ == '__main__':
    main()
