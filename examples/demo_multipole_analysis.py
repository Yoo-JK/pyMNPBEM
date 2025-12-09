"""
Demo: Multipole decomposition of extinction.

Shows dipole, quadrupole contributions to extinction.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet


def multipole_moments(p, sig, order=2):
    """Calculate multipole moments from surface charge distribution.

    Parameters
    ----------
    p : ComParticle
        Particle
    sig : Surface charges
    order : int
        Maximum multipole order (1=dipole, 2=quadrupole)

    Returns
    -------
    dict : Multipole moments
    """
    # Get face centers and areas
    centers = p.face_centers
    areas = p.areas
    charges = sig.phip.flatten()

    moments = {}

    # Dipole moment (order 1)
    if order >= 1:
        p_dip = np.zeros(3, dtype=complex)
        for i in range(3):
            p_dip[i] = np.sum(charges * centers[:, i] * areas)
        moments['dipole'] = p_dip
        moments['dipole_mag'] = np.abs(np.sqrt(np.sum(np.abs(p_dip)**2)))

    # Quadrupole moment (order 2)
    if order >= 2:
        Q = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                Q[i, j] = np.sum(charges * centers[:, i] * centers[:, j] * areas)
        # Make traceless
        trace = np.trace(Q)
        for i in range(3):
            Q[i, i] -= trace / 3
        moments['quadrupole'] = Q
        moments['quadrupole_mag'] = np.abs(np.sqrt(np.sum(np.abs(Q)**2)))

    return moments


def main():
    """Multipole decomposition for gold nanosphere."""
    print("Demo: Multipole decomposition")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 80  # nm (large enough for quadrupole)
    sphere = trisphere(400, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    print(f"Sphere diameter: {diameter} nm")

    bem = BEMRet(p)
    exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])

    wavelengths = np.linspace(400, 700, 60)

    dipole_contrib = []
    quadrupole_contrib = []
    total_ext = []

    print("\nComputing multipole contributions...")
    for wl in wavelengths:
        exc_wl = exc(p, wl)
        sig = bem.solve(exc_wl)

        # Get multipole moments
        moments = multipole_moments(p, sig, order=2)

        dipole_contrib.append(moments['dipole_mag'])
        quadrupole_contrib.append(moments['quadrupole_mag'])

        # Total extinction
        ext = np.abs(sig.phip).sum()
        total_ext.append(ext)

    dipole_contrib = np.array(dipole_contrib)
    quadrupole_contrib = np.array(quadrupole_contrib)
    total_ext = np.array(total_ext)

    # Normalize
    dipole_norm = dipole_contrib / dipole_contrib.max()
    quadrupole_norm = quadrupole_contrib / quadrupole_contrib.max()
    total_norm = total_ext / total_ext.max()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(wavelengths, dipole_norm, 'b-', linewidth=2, label='Dipole')
    ax1.plot(wavelengths, quadrupole_norm, 'r--', linewidth=2, label='Quadrupole')
    ax1.plot(wavelengths, total_norm, 'k:', linewidth=2, label='Total')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Normalized magnitude', fontsize=12)
    ax1.set_title('Multipole Contributions', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Ratio plot
    ratio = quadrupole_contrib / dipole_contrib
    ax2.plot(wavelengths, ratio * 100, 'g-', linewidth=2)
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Quadrupole/Dipole ratio (%)', fontsize=12)
    ax2.set_title('Relative Quadrupole Contribution', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Multipole Analysis (d={diameter} nm)', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_multipole_analysis.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
