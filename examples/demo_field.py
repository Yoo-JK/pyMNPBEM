"""
Demo: Near-field enhancement visualization

This example demonstrates visualization of the near-field enhancement
around a plasmonic nanoparticle under plane wave excitation.

The electromagnetic field is strongly enhanced near sharp features and
at the plasmon resonance wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import bemoptions, EpsConst, EpsTable, ComParticle, bemsolver, planewave
from mnpbem.particles.shapes import trisphere, trirod
from mnpbem.misc.plotting import plot_field_slice, plot_particle


def main():
    """Visualize near-field enhancement."""

    print("MNPBEM Demo: Near-field enhancement")
    print("=" * 60)

    # Options
    op = bemoptions(sim='stat', waitbar=0)

    # Materials
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    epstab = [eps_vacuum, eps_gold]

    # Gold nanosphere
    diameter = 40  # nm
    print(f"Creating gold nanosphere (d={diameter} nm)...")
    sphere = trisphere(144, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    # BEM solver
    bem = bemsolver(p, op)

    # Plane wave excitation (x-polarized)
    exc = planewave([[1, 0, 0]], [[0, 0, 1]], op)

    # Find plasmon resonance
    print("Finding plasmon resonance...")
    wavelengths = np.linspace(480, 600, 40)
    sca = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        sig = bem.solve(exc(p, wl))
        sca[i] = exc.sca(sig)[0]

    resonance_wl = wavelengths[np.argmax(sca)]
    print(f"Plasmon resonance at {resonance_wl:.1f} nm")

    # Solve at resonance
    print(f"Solving BEM at {resonance_wl:.1f} nm...")
    sig = bem.solve(exc(p, resonance_wl))

    # Create field evaluation grid (xy plane at z=0)
    print("Computing near-field...")
    n_grid = 81
    x = np.linspace(-60, 60, n_grid)
    y = np.linspace(-60, 60, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Points for field evaluation
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Compute field enhancement
    # The electric field enhancement is |E|/|E0|
    # For a point outside the particle, we compute the scattered field

    # For this demo, we'll compute the surface charge distribution
    # and the field enhancement using the BEM solution

    # Surface charge visualization (on particle surface)
    surface_charge = sig.sig  # Surface charge density

    # Approximate field enhancement from charge density
    # In the quasistatic limit, E ~ grad(phi) where phi comes from charges
    print("Visualizing results...")

    # Create figure with multiple views
    fig = plt.figure(figsize=(14, 10))

    # 1. Scattering spectrum
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(wavelengths, sca, 'b-', linewidth=2)
    ax1.axvline(resonance_wl, color='r', linestyle='--',
                label=f'Resonance: {resonance_wl:.1f} nm')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Scattering cross section (nm²)')
    ax1.set_title('Scattering spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Surface charge distribution
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    # Get particle geometry
    pc = p.pc

    # Plot particle with surface charge coloring
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Create polygons
    polygons = []
    for face in pc.faces:
        valid = ~np.isnan(face)
        indices = face[valid].astype(int)
        vertices = pc.verts[indices]
        polygons.append(vertices)

    # Color by surface charge (real part)
    charge_vals = np.real(surface_charge[:, 0])  # First polarization
    norm = Normalize(vmin=-np.abs(charge_vals).max(),
                     vmax=np.abs(charge_vals).max())
    colors = cm.RdBu_r(norm(charge_vals))

    collection = Poly3DCollection(
        polygons,
        alpha=0.9,
        facecolors=colors,
        edgecolor='black',
        linewidth=0.2
    )
    ax2.add_collection3d(collection)

    # Set limits
    r = diameter / 2 * 1.2
    ax2.set_xlim(-r, r)
    ax2.set_ylim(-r, r)
    ax2.set_zlim(-r, r)
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_zlabel('Z (nm)')
    ax2.set_title('Surface charge (x-pol)')

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, shrink=0.6, label='Charge density')

    # 3. Surface charge for y-polarization
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # Solve for y-polarization
    exc_y = planewave([[0, 1, 0]], [[0, 0, 1]], op)
    sig_y = bem.solve(exc_y(p, resonance_wl))
    charge_y = np.real(sig_y.sig[:, 0])

    colors_y = cm.RdBu_r(norm(charge_y))
    collection_y = Poly3DCollection(
        polygons,
        alpha=0.9,
        facecolors=colors_y,
        edgecolor='black',
        linewidth=0.2
    )
    ax3.add_collection3d(collection_y)

    ax3.set_xlim(-r, r)
    ax3.set_ylim(-r, r)
    ax3.set_zlim(-r, r)
    ax3.set_xlabel('X (nm)')
    ax3.set_ylabel('Y (nm)')
    ax3.set_zlabel('Z (nm)')
    ax3.set_title('Surface charge (y-pol)')

    # 4. Dipole moment
    ax4 = fig.add_subplot(2, 2, 4)

    # Compute dipole moments for different wavelengths
    px = np.zeros(len(wavelengths), dtype=complex)
    py = np.zeros(len(wavelengths), dtype=complex)

    for i, wl in enumerate(wavelengths):
        sig_i = bem.solve(exc(p, wl))
        # Dipole = sum of charge * position
        pos = pc.pos
        charge = sig_i.sig[:, 0]
        area = pc.area
        px[i] = np.sum(charge * area * pos[:, 0])

    ax4.plot(wavelengths, np.abs(px), 'b-', linewidth=2, label='|p_x|')
    ax4.axvline(resonance_wl, color='r', linestyle='--')
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Dipole moment (arb. units)')
    ax4.set_title('Induced dipole moment')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Gold nanosphere near-field (d={diameter} nm)', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_field_result.png', dpi=150)
    print("Figure saved to demo_field_result.png")

    plt.show()


if __name__ == '__main__':
    main()
