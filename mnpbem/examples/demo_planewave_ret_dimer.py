"""
Demo: Retarded BEM for gold nanosphere dimer.

Shows coupled plasmon resonances with retardation effects.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Retarded BEM for gold nanosphere dimer."""
    print("Demo: Retarded BEM - Gold nanosphere dimer")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 50  # nm
    gap = 5  # nm

    # Create dimer
    sphere1 = trisphere(200, diameter).shift([-(diameter + gap)/2, 0, 0])
    sphere2 = trisphere(200, diameter).shift([(diameter + gap)/2, 0, 0])

    p = ComParticle([eps_vacuum, eps_gold], [sphere1, sphere2], [[2, 1], [2, 1]])
    print(f"Dimer: d={diameter} nm, gap={gap} nm, {p.n_faces} faces")

    wavelengths = np.linspace(400, 900, 100)

    # Polarization along dimer axis (longitudinal mode)
    print("\nComputing longitudinal mode (pol || dimer axis)...")
    exc_long = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
    bem = BEMRet(p)
    spec_long = SpectrumRet(bem, exc_long, wavelengths, show_progress=True)
    result_long = spec_long.compute()
    ext_long = result_long['ext'][:, 0]

    # Polarization perpendicular to dimer axis (transverse mode)
    print("\nComputing transverse mode (pol ⊥ dimer axis)...")
    exc_trans = PlaneWaveRet(pol=[0, 1, 0], direction=[0, 0, 1])
    spec_trans = SpectrumRet(bem, exc_trans, wavelengths, show_progress=True)
    result_trans = spec_trans.compute()
    ext_trans = result_trans['ext'][:, 0]

    # Single sphere for reference
    print("\nComputing single sphere reference...")
    p_single = ComParticle([eps_vacuum, eps_gold], [trisphere(200, diameter)], [[2, 1]])
    bem_single = BEMRet(p_single)
    exc_single = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
    spec_single = SpectrumRet(bem_single, exc_single, wavelengths, show_progress=False)
    result_single = spec_single.compute()
    ext_single = result_single['ext'][:, 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, ext_long, 'r-', linewidth=2, label='Dimer - Longitudinal')
    ax.plot(wavelengths, ext_trans, 'b-', linewidth=2, label='Dimer - Transverse')
    ax.plot(wavelengths, ext_single * 2, 'k--', linewidth=2, label='2× Single sphere')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Retarded BEM: Gold Dimer (d={diameter} nm, gap={gap} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    peak_long = wavelengths[np.argmax(ext_long)]
    peak_trans = wavelengths[np.argmax(ext_trans)]
    print(f"\nLongitudinal mode peak: {peak_long:.1f} nm")
    print(f"Transverse mode peak: {peak_trans:.1f} nm")
    print(f"Mode splitting: {peak_long - peak_trans:.1f} nm")

    plt.tight_layout()
    plt.savefig('demo_planewave_ret_dimer.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
