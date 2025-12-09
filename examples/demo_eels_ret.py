"""
Demo: Retarded EELS calculations.

Electron Energy Loss Spectroscopy simulations with full retardation.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')

from pyMNPBEM.particles import ComParticle
from pyMNPBEM.particles.shapes import trisphere, trirod
from pyMNPBEM.material import EpsConst, EpsDrude
from pyMNPBEM.bem import BEMRet
from pyMNPBEM.simulation import EELSRet


def demo_eels_sphere():
    """
    EELS spectrum of gold nanosphere.
    """
    print("=" * 60)
    print("Demo: EELS spectrum of gold nanosphere")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold sphere
    radius = 30  # nm
    sphere = trisphere(200, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    print(f"Particle: gold sphere, R = {radius} nm")
    print(f"Number of elements: {p.n_faces}")

    # BEM solver
    bem = BEMRet(p)

    # Electron beam parameters
    velocity = 0.5  # v/c
    impact_param = radius + 5  # nm (aloof configuration)

    # Energy range (in eV, converted to wavelength)
    energies = np.linspace(1.5, 4.0, 50)  # eV
    wavelengths = 1239.84 / energies  # nm

    print(f"\nElectron velocity: {velocity}c")
    print(f"Impact parameter: {impact_param} nm (aloof)")
    print(f"Computing EELS spectrum...")

    loss_prob = np.zeros(len(energies))

    for i, wl in enumerate(wavelengths):
        eels = EELSRet(
            impact=np.array([[impact_param, 0, 0]]),
            velocity=velocity
        )
        exc = eels(p, wl)
        sig = bem.solve(exc)
        loss_prob[i] = eels.loss(sig)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(energies, loss_prob, 'b-', linewidth=2)

    # Find peak
    i_peak = np.argmax(loss_prob)
    ax.axvline(energies[i_peak], color='r', linestyle='--', alpha=0.5,
               label=f'Peak: {energies[i_peak]:.2f} eV')

    ax.set_xlabel('Energy loss (eV)')
    ax.set_ylabel('Loss probability (arb. units)')
    ax.set_title(f'EELS spectrum\n'
                 f'{radius} nm Au sphere, impact = {impact_param} nm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eels_ret_sphere.png', dpi=150)
    print(f"\nPlasmon peak at {energies[i_peak]:.2f} eV")
    print("Plot saved: eels_ret_sphere.png")

    return energies, loss_prob


def demo_eels_nanorod():
    """
    EELS spectrum of gold nanorod at different positions.
    """
    print("\n" + "=" * 60)
    print("Demo: EELS spectrum of gold nanorod")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold nanorod
    length = 60  # nm
    diameter = 15  # nm
    rod = trirod(diameter, length, n=150)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [rod], [[2, 1]], closed=1)

    print(f"Nanorod: L = {length} nm, D = {diameter} nm")

    # BEM solver
    bem = BEMRet(p)

    # Electron beam parameters
    velocity = 0.7

    # Two impact positions: near tip and near center
    impact_tip = np.array([[0, 0, length / 2 + 5]])  # Near tip
    impact_center = np.array([[diameter / 2 + 5, 0, 0]])  # Near center

    # Energy range
    energies = np.linspace(1.0, 3.5, 50)
    wavelengths = 1239.84 / energies

    print(f"\nComputing EELS at tip and center positions...")

    loss_tip = np.zeros(len(energies))
    loss_center = np.zeros(len(energies))

    for i, wl in enumerate(wavelengths):
        # Near tip
        eels_tip = EELSRet(impact=impact_tip, velocity=velocity)
        exc = eels_tip(p, wl)
        sig = bem.solve(exc)
        loss_tip[i] = eels_tip.loss(sig)

        # Near center
        eels_center = EELSRet(impact=impact_center, velocity=velocity)
        exc = eels_center(p, wl)
        sig = bem.solve(exc)
        loss_center[i] = eels_center.loss(sig)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(energies, loss_tip, 'b-', linewidth=2, label='Near tip')
    ax.plot(energies, loss_center, 'r--', linewidth=2, label='Near center')

    ax.set_xlabel('Energy loss (eV)')
    ax.set_ylabel('Loss probability (arb. units)')
    ax.set_title(f'EELS spectrum of Au nanorod\n'
                 f'L = {length} nm, D = {diameter} nm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eels_ret_nanorod.png', dpi=150)
    print("Plot saved: eels_ret_nanorod.png")

    return energies, loss_tip, loss_center


def demo_eels_map():
    """
    EELS spatial map around nanoparticle.
    """
    print("\n" + "=" * 60)
    print("Demo: EELS spatial map")
    print("=" * 60)

    # Materials
    eps_env = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    # Gold sphere
    radius = 25  # nm
    sphere = trisphere(144, radius)

    epstab = [eps_env, eps_gold]
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    # BEM solver
    bem = BEMRet(p)

    # Fixed energy (at plasmon resonance)
    energy = 2.4  # eV
    wavelength = 1239.84 / energy
    velocity = 0.6

    # Scan grid
    n_scan = 20
    x_range = np.linspace(-50, 50, n_scan)
    y_range = np.linspace(-50, 50, n_scan)

    print(f"Energy: {energy} eV (λ = {wavelength:.0f} nm)")
    print(f"Computing EELS map on {n_scan}x{n_scan} grid...")

    eels_map = np.zeros((n_scan, n_scan))

    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            r = np.sqrt(x**2 + y**2)

            # Skip inside particle
            if r < radius + 2:
                eels_map[i, j] = np.nan
                continue

            eels = EELSRet(
                impact=np.array([[x, y, 0]]),
                velocity=velocity
            )
            exc = eels(p, wavelength)
            sig = bem.solve(exc)
            eels_map[i, j] = eels.loss(sig)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    X, Y = np.meshgrid(x_range, y_range)

    im = ax.pcolormesh(X, Y, eels_map, cmap='hot', shading='auto')

    # Draw particle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            'w-', linewidth=2)
    ax.fill(radius * np.cos(theta), radius * np.sin(theta),
            color='gray', alpha=0.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('EELS intensity')

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title(f'EELS map at {energy} eV\n{radius} nm Au sphere')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('eels_ret_map.png', dpi=150)
    print("Plot saved: eels_ret_map.png")

    return X, Y, eels_map


if __name__ == '__main__':
    demo_eels_sphere()
    demo_eels_nanorod()
    demo_eels_map()

    plt.show()
