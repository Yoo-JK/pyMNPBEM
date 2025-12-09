"""
Demo: 3D near-field visualization.

Shows electric field distribution in multiple planes.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import PlaneWaveStat
from pymnpbem.misc import igrid


def main():
    """3D near-field distribution around gold nanorod."""
    print("Demo: 3D near-field visualization")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 60  # nm
    diameter = 20  # nm
    rod = trirod(200, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    bem = BEMStat(p)

    # Excitation along rod axis
    exc = PlaneWaveStat(pol=[1, 0, 0])
    wavelength = 700  # nm (longitudinal mode)

    print(f"Nanorod: L={length} nm, d={diameter} nm")
    print(f"Wavelength: {wavelength} nm")

    # Solve BEM
    exc_wl = exc(p, wavelength)
    sig = bem.solve(exc_wl)

    # Create grids for three planes
    extent_x = length * 0.8
    extent_y = diameter * 1.5
    n_points = 40

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    planes = [
        {'name': 'XY plane (z=0)', 'x': (-extent_x, extent_x), 'y': (-extent_y, extent_y), 'z': 0},
        {'name': 'XZ plane (y=0)', 'x': (-extent_x, extent_x), 'y': (-extent_y, extent_y), 'z': None, 'xz': True},
        {'name': 'YZ plane (x=0)', 'x': (-extent_y, extent_y), 'y': (-extent_y, extent_y), 'z': None, 'yz': True}
    ]

    from pymnpbem.misc.meshfield import field_at_points

    for i, plane in enumerate(planes):
        ax = axes[i]

        if plane.get('xz'):
            X, Z, pts = igrid(plane['x'], plane['y'], y=0, n_points=n_points)
            xlabel, ylabel = 'X (nm)', 'Z (nm)'
        elif plane.get('yz'):
            Y, Z, pts = igrid(plane['x'], plane['y'], x=0, n_points=n_points)
            X, Y = Y, Z  # Rename for plotting
            xlabel, ylabel = 'Y (nm)', 'Z (nm)'
        else:
            X, Y, pts = igrid(plane['x'], plane['y'], z=plane['z'], n_points=n_points)
            xlabel, ylabel = 'X (nm)', 'Y (nm)'

        # Check inside particle (simplified)
        if plane.get('xz') or not plane.get('yz'):
            inside = (np.abs(pts[:, 0]) < length/2) & (np.sqrt(pts[:, 1]**2 + pts[:, 2]**2) < diameter/2)
        else:
            inside = (np.abs(pts[:, 0]) < diameter/2) & (np.sqrt(pts[:, 1]**2 + pts[:, 2]**2) < diameter/2)

        print(f"\nComputing field for {plane['name']}...")
        E = field_at_points(p, sig, pts, field_type='field')

        # Field enhancement
        E_mag = np.sqrt(np.abs(E[:, 0])**2 + np.abs(E[:, 1])**2 + np.abs(E[:, 2])**2)
        enhancement = E_mag**2
        enhancement[inside] = np.nan

        enh_grid = enhancement.reshape(X.shape)

        im = ax.pcolormesh(X, Y, enh_grid, cmap='hot', shading='auto',
                          vmin=0, vmax=np.nanpercentile(enh_grid, 98))
        plt.colorbar(im, ax=ax, label='|E/E₀|²')

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(plane['name'], fontsize=12)
        ax.set_aspect('equal')

    plt.suptitle(f'Near-Field Enhancement (L={length} nm, d={diameter} nm, λ={wavelength} nm)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_field_3d.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
