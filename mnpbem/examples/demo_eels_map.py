"""
Demo: EELS spatial mapping of nanostructure.

Shows 2D EELS loss probability map.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import EELSRet


def main():
    """2D EELS mapping of gold nanorod."""
    print("Demo: EELS spatial mapping")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 80  # nm
    diameter = 30  # nm
    rod = trirod(300, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    print(f"Nanorod: L={length} nm, d={diameter} nm")

    bem = BEMRet(p)

    # Beam energy
    beam_energy = 100e3  # 100 keV
    wavelength = 700  # nm (near longitudinal resonance)

    print(f"Beam energy: {beam_energy/1e3:.0f} keV")
    print(f"Energy loss at λ = {wavelength} nm")

    # Create 2D grid of beam positions
    nx, ny = 40, 25
    x = np.linspace(-length/2 - 20, length/2 + 20, nx)
    y = np.linspace(-diameter/2 - 15, diameter/2 + 15, ny)
    X, Y = np.meshgrid(x, y)

    # Beam travels along z
    positions = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])

    print(f"\nComputing EELS map ({nx}×{ny} = {X.size} positions)...")

    loss_map = np.zeros(X.size)

    # Check which positions are inside the particle
    inside = np.zeros(X.size, dtype=bool)
    for i, pos in enumerate(positions):
        # Approximate inside check for rod
        if np.abs(pos[0]) < length/2 - diameter/4:
            if np.sqrt(pos[1]**2) < diameter/2:
                inside[i] = True
        else:
            # Hemispherical caps
            if pos[0] > 0:
                center = np.array([length/2 - diameter/2, 0, 0])
            else:
                center = np.array([-length/2 + diameter/2, 0, 0])
            if np.linalg.norm(pos - center) < diameter/2:
                inside[i] = True

    # Compute EELS for outside positions
    for i, pos in enumerate(positions):
        if inside[i]:
            loss_map[i] = np.nan
            continue

        if i % 50 == 0:
            print(f"  Position {i+1}/{X.size}...")

        eels = EELSRet(position=pos, beam_energy=beam_energy, direction=[0, 0, 1])
        exc = eels(p, wavelength)
        sig = bem.solve(exc)

        # Get loss probability
        loss = eels.loss_probability(sig)
        loss_map[i] = loss

    loss_grid = loss_map.reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.pcolormesh(X, Y, loss_grid, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax, label='Loss probability (arb. units)')

    # Draw rod outline
    theta = np.linspace(-np.pi/2, np.pi/2, 50)
    # Left cap
    ax.plot(-length/2 + diameter/2 + diameter/2*np.cos(theta + np.pi/2),
            diameter/2*np.sin(theta + np.pi/2), 'w-', linewidth=2)
    ax.plot([-length/2 + diameter/2, length/2 - diameter/2], [diameter/2, diameter/2], 'w-', linewidth=2)
    ax.plot([-length/2 + diameter/2, length/2 - diameter/2], [-diameter/2, -diameter/2], 'w-', linewidth=2)
    # Right cap
    ax.plot(length/2 - diameter/2 + diameter/2*np.cos(theta - np.pi/2),
            diameter/2*np.sin(theta - np.pi/2), 'w-', linewidth=2)

    ax.set_xlabel('X (nm)', fontsize=12)
    ax.set_ylabel('Y (nm)', fontsize=12)
    ax.set_title(f'EELS Map at λ={wavelength} nm (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('demo_eels_map.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
