"""
Demo: Retarded dipole near nanoparticle.

Computes decay rate enhancement with full electromagnetic retardation.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')

from pyMNPBEM.particles import ComParticle
from pyMNPBEM.particles.shapes import trisphere, trirod
from pyMNPBEM.material import EpsConst, EpsDrude
from pyMNPBEM.bem import BEMRet
from pyMNPBEM.simulation import DipoleRet


def demo_dipole_ret_sphere():
    """
    Retarded dipole near a larger sphere where retardation matters.
    """
    print("=" * 60)
    print("Demo: Retarded dipole near gold sphere")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Larger sphere where retardation is important
    radius = 80  # nm
    sphere = trisphere(200, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    print(f"Particle: gold sphere, R = {radius} nm")
    print(f"Number of boundary elements: {p.n_faces}")

    # Retarded BEM solver
    bem = BEMRet(p)

    # Dipole position
    z_dip = radius + 20  # 20 nm above surface

    # Wavelength range
    wavelengths = np.linspace(500, 800, 40)

    print(f"\nDipole position: z = {z_dip} nm")
    print(f"Computing spectrum from {wavelengths[0]} to {wavelengths[-1]} nm...")

    purcell_z = np.zeros(len(wavelengths))
    purcell_x = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        # z-oriented dipole
        dip_z = DipoleRet(
            pt=np.array([[0, 0, z_dip]]),
            dip=np.array([[0, 0, 1]])
        )
        exc = dip_z(p, wl)
        sig = bem.solve(exc)
        purcell_z[i] = dip_z.decayrate(sig)[0]

        # x-oriented dipole
        dip_x = DipoleRet(
            pt=np.array([[0, 0, z_dip]]),
            dip=np.array([[1, 0, 0]])
        )
        exc = dip_x(p, wl)
        sig = bem.solve(exc)
        purcell_x[i] = dip_x.decayrate(sig)[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(wavelengths, purcell_z, 'b-', linewidth=2, label='z-dipole')
    ax.plot(wavelengths, purcell_x, 'r--', linewidth=2, label='x-dipole')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Purcell factor (retarded)')
    ax.set_title(f'Retarded Purcell factor spectrum\n'
                 f'{radius} nm Au sphere, dipole at {z_dip-radius:.0f} nm from surface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dipole_ret_spectrum.png', dpi=150)
    print("\nPlot saved: dipole_ret_spectrum.png")

    return wavelengths, purcell_z, purcell_x


def demo_dipole_ret_nanorod():
    """
    Dipole near a gold nanorod - shows anisotropic response.
    """
    print("\n" + "=" * 60)
    print("Demo: Retarded dipole near gold nanorod")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold nanorod (length = 80 nm, diameter = 20 nm)
    length = 80  # nm
    diameter = 20  # nm
    rod = trirod(diameter, length, n=100)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [rod], [[2, 1]], closed=1)

    print(f"Particle: gold nanorod, L = {length} nm, D = {diameter} nm")
    print(f"Number of boundary elements: {p.n_faces}")

    # BEM solver
    bem = BEMRet(p)

    # Dipole positions: at the end and at the side
    pos_end = np.array([[0, 0, length / 2 + 10]])  # Near tip
    pos_side = np.array([[diameter / 2 + 10, 0, 0]])  # Near side

    # Wavelength range
    wavelengths = np.linspace(500, 900, 50)

    print(f"\nComputing spectrum...")

    purcell_end_z = np.zeros(len(wavelengths))
    purcell_side_x = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        # Dipole at end, z-oriented (parallel to rod)
        dip_end = DipoleRet(pt=pos_end, dip=np.array([[0, 0, 1]]))
        exc = dip_end(p, wl)
        sig = bem.solve(exc)
        purcell_end_z[i] = dip_end.decayrate(sig)[0]

        # Dipole at side, x-oriented (perpendicular to rod)
        dip_side = DipoleRet(pt=pos_side, dip=np.array([[1, 0, 0]]))
        exc = dip_side(p, wl)
        sig = bem.solve(exc)
        purcell_side_x[i] = dip_side.decayrate(sig)[0]

    # Find resonances
    i_end = np.argmax(purcell_end_z)
    i_side = np.argmax(purcell_side_x)

    print(f"\nLongitudinal resonance: {wavelengths[i_end]:.0f} nm")
    print(f"Transverse resonance: {wavelengths[i_side]:.0f} nm")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(wavelengths, purcell_end_z, 'b-', linewidth=2,
            label=f'Tip dipole (z) - Long. mode')
    ax.plot(wavelengths, purcell_side_x, 'r--', linewidth=2,
            label=f'Side dipole (x) - Trans. mode')

    ax.axvline(wavelengths[i_end], color='b', linestyle=':', alpha=0.5)
    ax.axvline(wavelengths[i_side], color='r', linestyle=':', alpha=0.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Purcell factor')
    ax.set_title(f'Nanorod Purcell spectrum\n'
                 f'L = {length} nm, D = {diameter} nm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dipole_ret_nanorod.png', dpi=150)
    print("Plot saved: dipole_ret_nanorod.png")

    return wavelengths, purcell_end_z, purcell_side_x


def demo_dipole_ret_farfield():
    """
    Compute far-field radiation pattern from dipole near nanoparticle.
    """
    print("\n" + "=" * 60)
    print("Demo: Far-field radiation pattern")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold sphere
    radius = 50  # nm
    sphere = trisphere(144, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    # BEM solver
    bem = BEMRet(p)

    # Dipole position and wavelength
    z_dip = radius + 15
    wavelength = 540  # nm

    print(f"Au sphere: R = {radius} nm")
    print(f"Dipole at z = {z_dip} nm, λ = {wavelength} nm")

    # Solve BEM
    dip = DipoleRet(
        pt=np.array([[0, 0, z_dip]]),
        dip=np.array([[0, 0, 1]])
    )
    exc = dip(p, wavelength)
    sig = bem.solve(exc)

    # Far-field directions
    theta = np.linspace(0, np.pi, 180)
    phi = 0

    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    print("Computing far-field pattern...")

    # Compute far-field (simplified)
    E_ff = dip.farfield(sig, directions)
    intensity = np.sum(np.abs(E_ff)**2, axis=-1)

    # Normalize
    intensity /= np.max(intensity)

    # Plot in polar coordinates
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.plot(theta, intensity, 'b-', linewidth=2)
    ax.plot(-theta, intensity, 'b-', linewidth=2)

    ax.set_title(f'Far-field radiation pattern\n'
                 f'z-dipole near {radius} nm Au sphere')
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(45)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('dipole_ret_farfield.png', dpi=150)
    print("Plot saved: dipole_ret_farfield.png")

    return theta, intensity


if __name__ == '__main__':
    demo_dipole_ret_sphere()
    demo_dipole_ret_nanorod()
    demo_dipole_ret_farfield()

    plt.show()
