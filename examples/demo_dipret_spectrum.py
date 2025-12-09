"""
Demo: Dipole emission spectrum with retardation.

Shows wavelength-dependent enhancement and emission modification.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle, Point
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import DipoleRet


def main():
    """Dipole emission spectrum near gold nanosphere."""
    print("Demo: Retarded dipole emission spectrum")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 60  # nm
    sphere = trisphere(200, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    bem = BEMRet(p)

    wavelengths = np.linspace(450, 750, 60)
    distance = 5  # nm from surface

    # Dipole position
    r = diameter/2 + distance
    pos = np.array([[r, 0, 0]])

    # Two orientations
    orientations = {
        'radial': np.array([[1, 0, 0]]),
        'tangential': np.array([[0, 1, 0]])
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ori_name, pol in orientations.items():
        print(f"\nOrientation: {ori_name}")

        tot_decay = []
        rad_decay = []

        for wl in wavelengths:
            dip = DipoleRet(position=pos, moment=pol)
            exc = dip(p, wl)
            sig = bem.solve(exc)

            # Get decay rates
            gamma_tot = dip.total_decay_rate(sig)
            gamma_rad = dip.radiative_decay_rate(sig)

            tot_decay.append(gamma_tot)
            rad_decay.append(gamma_rad)

        tot_decay = np.array(tot_decay)
        rad_decay = np.array(rad_decay)
        nonrad_decay = tot_decay - rad_decay

        # Plot enhancement
        ax1.plot(wavelengths, tot_decay, linewidth=2, label=f'{ori_name} - total')
        ax1.plot(wavelengths, rad_decay, '--', linewidth=2, label=f'{ori_name} - radiative')

        # Quantum efficiency
        qe = rad_decay / tot_decay
        ax2.plot(wavelengths, qe, linewidth=2, label=ori_name.capitalize())

        # Find peak
        peak_wl = wavelengths[np.argmax(tot_decay)]
        print(f"  Peak enhancement wavelength: {peak_wl:.1f} nm")
        print(f"  Max total enhancement: {tot_decay.max():.1f}×")

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Decay rate enhancement', fontsize=12)
    ax1.set_title(f'Decay Rate Enhancement (d={diameter} nm, gap={distance} nm)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Quantum efficiency', fontsize=12)
    ax2.set_title(f'Quantum Efficiency (d={diameter} nm, gap={distance} nm)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.suptitle('Retarded Dipole Emission Near Gold Nanosphere', fontsize=14)
    plt.tight_layout()
    plt.savefig('demo_dipret_spectrum.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
