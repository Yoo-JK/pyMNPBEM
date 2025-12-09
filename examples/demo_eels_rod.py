"""
Demo: EELS spectroscopy of gold nanorod.

Shows spatial distribution of longitudinal and transverse modes.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import EELSStat


def main():
    """EELS mapping of gold nanorod plasmon modes."""
    print("Demo: EELS mapping of gold nanorod")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 60  # nm
    diameter = 20  # nm
    rod = trirod(200, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    bem = BEMStat(p)

    # EELS spectra at tip vs side
    wavelengths = np.linspace(400, 900, 100)

    # Tip position
    tip_impact = length/2 + 3
    # Side position
    side_impact = diameter/2 + 3

    print("Computing EELS at tip...")
    loss_tip = []
    for wl in wavelengths:
        exc = EELSStat(impact=tip_impact, direction=[0, 0, 1], velocity=0.5)
        exc_wl = exc(p, wl)
        sig = bem.solve(exc_wl)
        loss_tip.append(np.abs(sig.phip).sum())

    print("Computing EELS at side...")
    loss_side = []
    for wl in wavelengths:
        exc = EELSStat(impact=side_impact, direction=[0, 0, 1], velocity=0.5)
        exc_wl = exc(p, wl)
        sig = bem.solve(exc_wl)
        loss_side.append(np.abs(sig.phip).sum())

    # Normalize
    loss_tip = np.array(loss_tip) / np.max(loss_tip)
    loss_side = np.array(loss_side) / np.max(loss_side)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, loss_tip, 'r-', linewidth=2, label='Tip (longitudinal mode)')
    ax.plot(wavelengths, loss_side, 'b-', linewidth=2, label='Side (transverse mode)')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized loss probability', fontsize=12)
    ax.set_title(f'EELS of Gold Nanorod (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    tip_peak = wavelengths[np.argmax(loss_tip)]
    side_peak = wavelengths[np.argmax(loss_side)]
    print(f"\nTip mode resonance: {tip_peak:.1f} nm")
    print(f"Side mode resonance: {side_peak:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_eels_rod.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
