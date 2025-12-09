"""
Demo: Near-field calculation around nanoparticle.

Shows electric field enhancement and distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, Point
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat
from pymnpbem.misc import igrid


def main():
    """Near-field distribution around gold nanosphere."""
    print("Demo: Near-field enhancement")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 40  # nm
    sphere = trisphere(200, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    bem = BEMStat(p)

    # Excitation
    exc = PlaneWaveStat(pol=[1, 0, 0])
    wavelength = 520  # nm (near resonance)

    print(f"Wavelength: {wavelength} nm")

    # Solve BEM
    exc_wl = exc(p, wavelength)
    sig = bem.solve(exc_wl)

    # Create grid for field evaluation
    extent = diameter * 1.5
    X, Y, pts = igrid((-extent, extent), (-extent, extent), z=0, n_points=60)

    # Remove points inside particle
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    inside = r < diameter/2

    print("Computing near-field...")

    # Compute field at grid points
    from pymnpbem.misc.meshfield import field_at_points
    E = field_at_points(p, sig, pts, field_type='field')

    # Field enhancement |E/E0|^2
    E_mag = np.sqrt(np.abs(E[:, 0])**2 + np.abs(E[:, 1])**2 + np.abs(E[:, 2])**2)
    E0 = 1.0  # Incident field amplitude
    enhancement = (E_mag / E0)**2
    enhancement[inside] = np.nan

    # Reshape for plotting
    enh_grid = enhancement.reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.pcolormesh(X, Y, enh_grid, cmap='hot', shading='auto',
                       vmin=0, vmax=np.nanpercentile(enh_grid, 99))
    plt.colorbar(im, ax=ax, label='|E/E₀|²')

    # Draw particle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(diameter/2 * np.cos(theta), diameter/2 * np.sin(theta),
            'w-', linewidth=2)

    ax.set_xlabel('X (nm)', fontsize=12)
    ax.set_ylabel('Y (nm)', fontsize=12)
    ax.set_title(f'Near-Field Enhancement (λ={wavelength} nm)', fontsize=14)
    ax.set_aspect('equal')

    max_enh = np.nanmax(enh_grid)
    print(f"\nMax enhancement: {max_enh:.1f}×")

    plt.tight_layout()
    plt.savefig('demo_nearfield.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
