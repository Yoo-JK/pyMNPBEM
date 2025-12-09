"""
Demo: Plasmon eigenmode analysis.

Shows intrinsic plasmon modes of a nanostructure.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere, trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStatEig, PlasmonMode


def main():
    """Plasmon eigenmode analysis of gold nanorod."""
    print("Demo: Plasmon eigenmode analysis")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Create nanorod
    length = 50  # nm
    diameter = 20  # nm
    rod = trirod(200, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])
    print(f"Nanorod: L={length} nm, d={diameter} nm")

    # Eigenvalue solver
    bem_eig = BEMStatEig(p)

    # Find plasmon modes
    print("\nComputing plasmon eigenmodes...")
    n_modes = 6
    eigvals, eigvecs = bem_eig.compute(n_modes=n_modes)

    # Convert eigenvalues to wavelengths
    # Lambda = 2*pi*(eps1 + eps2)/(eps1 - eps2)
    wavelength_ref = 550  # nm reference
    eps_in = eps_gold(wavelength_ref)[0]
    eps_out = 1.0

    # Approximate resonance wavelengths from eigenvalues
    print("\nPlasmon modes:")
    for i, ev in enumerate(eigvals[:n_modes]):
        # The eigenvalue relates to the dielectric function at resonance
        print(f"  Mode {i+1}: eigenvalue = {ev:.4f}")

    # Plot mode charge distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i in range(min(n_modes, 6)):
        ax = axes[i]

        # Get mode charge distribution
        sig = eigvecs[:, i].real

        # Plot particle colored by charge
        from pymnpbem.misc.plotting import plot_particle
        plot_particle(p, ax=ax, field=sig, cmap='coolwarm', colorbar=False)

        ax.set_title(f'Mode {i+1}', fontsize=12)

    plt.suptitle('Plasmon Eigenmodes of Gold Nanorod', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_plasmon_modes.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
