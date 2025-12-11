"""
Demo: Electron energy loss spectroscopy (EELS) of nanosphere.

Shows spatial mapping of plasmon modes.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMStat
from pymnpbem.simulation import EELSStat


def main():
    """EELS spectrum and mapping for gold nanosphere."""
    print("Demo: EELS spectroscopy of gold nanosphere")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 30  # nm
    sphere = trisphere(144, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    bem = BEMStat(p)

    # EELS at different impact parameters
    impacts = [0, diameter/4, diameter/2, diameter/2 + 5]
    wavelengths = np.linspace(400, 700, 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(impacts)))

    print("Computing EELS spectra at different impact parameters...")
    for i, b in enumerate(impacts):
        # Electron beam passing at impact parameter b
        beam_pos = np.array([[b, 0, -50]])  # Start position
        beam_dir = np.array([0, 0, 1])  # Direction

        exc = EELSStat(impact=b, direction=beam_dir, velocity=0.5)

        loss = []
        for wl in wavelengths:
            exc_wl = exc(p, wl)
            sig = bem.solve(exc_wl)
            # Approximate loss probability
            loss_prob = np.abs(sig.phip).sum() * 1e-4
            loss.append(loss_prob)

        ax1.plot(wavelengths, loss, color=colors[i], linewidth=2,
                label=f'b={b:.0f} nm')

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Loss probability (arb.)', fontsize=12)
    ax1.set_title('EELS Spectra at Different Impact Parameters', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # EELS map at resonance wavelength
    print("\nComputing EELS map...")
    resonance_wl = 520  # nm (approximate)
    x = np.linspace(-25, 25, 30)
    y = np.linspace(-25, 25, 30)
    X, Y = np.meshgrid(x, y)

    eels_map = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            b = np.sqrt(X[j, i]**2 + Y[j, i]**2)
            if b < diameter/2:  # Inside particle
                eels_map[j, i] = np.nan
            else:
                exc = EELSStat(impact=b, direction=[0, 0, 1], velocity=0.5)
                exc_wl = exc(p, resonance_wl)
                sig = bem.solve(exc_wl)
                eels_map[j, i] = np.abs(sig.phip).sum()

    im = ax2.pcolormesh(X, Y, eels_map, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax2, label='Loss probability')

    # Draw particle outline
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(diameter/2 * np.cos(theta), diameter/2 * np.sin(theta),
             'w-', linewidth=2)

    ax2.set_xlabel('X (nm)', fontsize=12)
    ax2.set_ylabel('Y (nm)', fontsize=12)
    ax2.set_title(f'EELS Map at {resonance_wl} nm', fontsize=14)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('demo_eels_sphere.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
