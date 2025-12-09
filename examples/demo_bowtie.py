"""
Demo: Bowtie nanoantenna simulation.

Shows strong field enhancement in gap of bowtie structure.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import tricube
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def make_triangle(n_verts, size, z_thickness=20):
    """Create a triangular prism."""
    # Use a cube and cut it (simplified)
    from pymnpbem.particles.shapes import tricube
    cube = tricube(n_verts, size)

    # For a bowtie, we can use triangular prisms
    # Here we use cubes as approximation
    return cube


def main():
    """Bowtie nanoantenna simulation."""
    print("Demo: Bowtie nanoantenna")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Bowtie parameters (simplified as two triangular regions)
    size = 40  # nm
    gap = 5  # nm

    # Create bowtie using rotated cubes as approximation
    # In reality, would use proper triangular mesh
    cube1 = tricube(200, size).shift([-(size/2 + gap/2), 0, 0])
    cube2 = tricube(200, size).shift([(size/2 + gap/2), 0, 0])

    p = ComParticle([eps_vacuum, eps_gold], [cube1, cube2], [[2, 1], [2, 1]])
    print(f"Bowtie (cube approximation): size={size} nm, gap={gap} nm")

    bem = BEMRet(p)
    wavelengths = np.linspace(500, 1000, 80)

    # Polarization along bowtie axis (gap direction)
    print("\nComputing spectrum (polarization along gap)...")
    exc_long = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
    spec = SpectrumRet(bem, exc_long, wavelengths, show_progress=True)
    result = spec.compute()
    ext_long = result['ext'][:, 0]

    # Polarization perpendicular
    print("Computing spectrum (perpendicular polarization)...")
    exc_perp = PlaneWaveRet(pol=[0, 1, 0], direction=[0, 0, 1])
    spec_perp = SpectrumRet(bem, exc_perp, wavelengths, show_progress=False)
    result_perp = spec_perp.compute()
    ext_perp = result_perp['ext'][:, 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, ext_long, 'r-', linewidth=2, label='Pol. along gap')
    ax.plot(wavelengths, ext_perp, 'b--', linewidth=2, label='Pol. perpendicular')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Bowtie Nanoantenna (size={size} nm, gap={gap} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak_long = wavelengths[np.argmax(ext_long)]
    peak_perp = wavelengths[np.argmax(ext_perp)]
    print(f"\nLongitudinal mode peak: {peak_long:.1f} nm")
    print(f"Transverse mode peak: {peak_perp:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_bowtie.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
