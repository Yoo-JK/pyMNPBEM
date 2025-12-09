"""
Demo: Plane wave excitation of a gold nanosphere (quasistatic).

This example demonstrates the basic workflow for computing the optical
response of a gold nanosphere using the quasistatic BEM solver.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import pyMNPBEM modules
from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsTable, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat, SpectrumStat


def main():
    """Gold nanosphere scattering spectrum."""
    print("Demo: Gold nanosphere - Quasistatic plane wave excitation")
    print("=" * 60)

    # 1. Define materials
    eps_vacuum = EpsConst(1.0)  # Vacuum/air
    eps_gold = EpsDrude('Au')   # Gold with Drude model

    # 2. Create particle geometry
    diameter = 20  # nm
    n_vertices = 144
    sphere = trisphere(n_vertices, diameter)
    print(f"Created sphere: diameter={diameter} nm, {sphere.n_faces} faces")

    # 3. Create composite particle
    # inout defines which medium is inside (2=gold) and outside (1=vacuum)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])
    print(f"Particle has {p.n_faces} faces")

    # 4. Define excitation
    # Plane wave with x-polarization
    exc = PlaneWaveStat(pol=[1, 0, 0])

    # 5. Define wavelength range
    wavelengths = np.linspace(400, 800, 100)

    # 6. Compute spectrum
    print("\nComputing optical spectrum...")
    bem = BEMStat(p)
    spectrum = SpectrumStat(bem, exc, wavelengths)
    sca, ext = spectrum.compute()

    # 7. Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext[:, 0], 'b-', linewidth=2, label='Extinction')
    ax.plot(wavelengths, sca[:, 0], 'r--', linewidth=2, label='Scattering')
    ax.plot(wavelengths, ext[:, 0] - sca[:, 0], 'g:', linewidth=2, label='Absorption')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross section (nm²)', fontsize=12)
    ax.set_title(f'Gold Nanosphere (d={diameter} nm) - Quasistatic', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Find and print resonance
    peak_idx = np.argmax(ext[:, 0])
    print(f"\nPlasmon resonance at {wavelengths[peak_idx]:.1f} nm")
    print(f"Peak extinction: {ext[peak_idx, 0]:.2f} nm²")

    plt.tight_layout()
    plt.savefig('demo_planewave_stat_sphere.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
