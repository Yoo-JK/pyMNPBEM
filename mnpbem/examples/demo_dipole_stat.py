"""
Demo: Quasistatic dipole near nanoparticle.

Computes decay rate enhancement (Purcell factor) for a dipole
near a spherical nanoparticle.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import pyMNPBEM modules
import sys
sys.path.insert(0, '..')

from pyMNPBEM.particles import ComParticle
from pyMNPBEM.particles.shapes import trisphere
from pyMNPBEM.material import EpsConst, EpsDrude
from pyMNPBEM.bem import BEMStat
from pyMNPBEM.simulation import DipoleStat


def demo_dipole_stat_basic():
    """
    Basic dipole near sphere example.

    Computes Purcell factor for dipole at various distances from
    a gold nanosphere.
    """
    print("=" * 60)
    print("Demo: Quasistatic dipole near gold nanosphere")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)  # Vacuum
    eps_gold = EpsDrude('Au')  # Gold (Drude model)

    # Create gold sphere (radius = 30 nm)
    radius = 30  # nm
    sphere = trisphere(144, radius)

    # Composite particle
    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    print(f"Particle: gold sphere, R = {radius} nm")
    print(f"Number of boundary elements: {p.n_faces}")

    # BEM solver
    bem = BEMStat(p)

    # Dipole positions: along z-axis, outside the sphere
    n_pos = 10
    z_positions = np.linspace(radius + 5, radius + 50, n_pos)

    # Wavelength
    wavelength = 520  # nm (near plasmon resonance)

    print(f"\nWavelength: {wavelength} nm")
    print(f"Computing decay rate at {n_pos} positions...")

    purcell_z = np.zeros(n_pos)  # z-oriented dipole
    purcell_x = np.zeros(n_pos)  # x-oriented dipole

    for i, z in enumerate(z_positions):
        # z-oriented dipole
        dip_z = DipoleStat(
            pt=np.array([[0, 0, z]]),
            dip=np.array([[0, 0, 1]])
        )
        exc = dip_z(p, wavelength)
        sig = bem.solve(exc)
        purcell_z[i] = dip_z.decay_rate(sig)[0]

        # x-oriented dipole
        dip_x = DipoleStat(
            pt=np.array([[0, 0, z]]),
            dip=np.array([[1, 0, 0]])
        )
        exc = dip_x(p, wavelength)
        sig = bem.solve(exc)
        purcell_x[i] = dip_x.decay_rate(sig)[0]

    # Print results
    print("\nResults:")
    print("-" * 40)
    print(f"{'Distance (nm)':>15} {'Purcell (z)':>12} {'Purcell (x)':>12}")
    print("-" * 40)
    for i, z in enumerate(z_positions):
        dist = z - radius
        print(f"{dist:>15.1f} {purcell_z[i]:>12.2f} {purcell_x[i]:>12.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    distances = z_positions - radius

    ax.semilogy(distances, purcell_z, 'b-o', label='z-oriented dipole')
    ax.semilogy(distances, purcell_x, 'r-s', label='x-oriented dipole')

    ax.set_xlabel('Distance from surface (nm)')
    ax.set_ylabel('Purcell factor')
    ax.set_title(f'Decay rate enhancement near {radius} nm Au sphere\n'
                 f'λ = {wavelength} nm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dipole_stat_purcell.png', dpi=150)
    print("\nPlot saved: dipole_stat_purcell.png")

    return distances, purcell_z, purcell_x


def demo_dipole_stat_spectrum():
    """
    Spectral response of dipole near nanoparticle.

    Computes Purcell factor spectrum.
    """
    print("\n" + "=" * 60)
    print("Demo: Dipole Purcell factor spectrum")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold sphere
    radius = 20  # nm
    sphere = trisphere(100, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    # BEM solver
    bem = BEMStat(p)

    # Dipole at fixed position
    z_dip = radius + 10  # 10 nm above surface
    dip = DipoleStat(
        pt=np.array([[0, 0, z_dip]]),
        dip=np.array([[0, 0, 1]])  # z-oriented
    )

    # Wavelength range
    wavelengths = np.linspace(400, 700, 50)

    print(f"Particle: gold sphere, R = {radius} nm")
    print(f"Dipole position: {z_dip} nm (z-axis)")
    print(f"Computing spectrum from {wavelengths[0]} to {wavelengths[-1]} nm...")

    purcell = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        exc = dip(p, wl)
        sig = bem.solve(exc)
        purcell[i] = dip.decay_rate(sig)[0]

    # Find resonance
    i_max = np.argmax(purcell)
    wl_res = wavelengths[i_max]

    print(f"\nResonance wavelength: {wl_res:.1f} nm")
    print(f"Maximum Purcell factor: {purcell[i_max]:.1f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(wavelengths, purcell, 'b-', linewidth=2)
    ax.axvline(wl_res, color='r', linestyle='--', alpha=0.5,
               label=f'Resonance: {wl_res:.0f} nm')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Purcell factor')
    ax.set_title(f'Purcell factor spectrum\n'
                 f'{radius} nm Au sphere, dipole at {z_dip-radius:.0f} nm from surface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dipole_stat_spectrum.png', dpi=150)
    print("Plot saved: dipole_stat_spectrum.png")

    return wavelengths, purcell


def demo_dipole_stat_map():
    """
    Spatial map of Purcell factor around nanoparticle.
    """
    print("\n" + "=" * 60)
    print("Demo: Purcell factor spatial map")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold sphere
    radius = 25  # nm
    sphere = trisphere(100, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    # BEM solver
    bem = BEMStat(p)

    # Wavelength at resonance
    wavelength = 520  # nm

    # Create grid for dipole positions
    n_grid = 15
    x_range = np.linspace(-60, 60, n_grid)
    z_range = np.linspace(-60, 60, n_grid)

    print(f"Computing Purcell factor on {n_grid}x{n_grid} grid...")

    purcell_map = np.zeros((n_grid, n_grid))

    for i, z in enumerate(z_range):
        for j, x in enumerate(x_range):
            # Skip points inside sphere
            r = np.sqrt(x**2 + z**2)
            if r < radius + 2:
                purcell_map[i, j] = np.nan
                continue

            # z-oriented dipole
            dip = DipoleStat(
                pt=np.array([[x, 0, z]]),
                dip=np.array([[0, 0, 1]])
            )
            exc = dip(p, wavelength)
            sig = bem.solve(exc)
            purcell_map[i, j] = dip.decay_rate(sig)[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    X, Z = np.meshgrid(x_range, z_range)

    # Use log scale for better visualization
    im = ax.pcolormesh(X, Z, np.log10(purcell_map),
                       cmap='hot', shading='auto')

    # Draw sphere
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            'w-', linewidth=2)
    ax.fill(radius * np.cos(theta), radius * np.sin(theta),
            color='gray', alpha=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(Purcell factor)')

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('z (nm)')
    ax.set_title(f'Purcell factor map (z-dipole)\n'
                 f'{radius} nm Au sphere, λ = {wavelength} nm')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('dipole_stat_map.png', dpi=150)
    print("Plot saved: dipole_stat_map.png")

    return X, Z, purcell_map


if __name__ == '__main__':
    # Run all demos
    demo_dipole_stat_basic()
    demo_dipole_stat_spectrum()
    demo_dipole_stat_map()

    plt.show()
