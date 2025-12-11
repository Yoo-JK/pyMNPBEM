"""
Demo: Plane wave excitation of gold nanosphere (retarded).

Shows full electromagnetic solution for larger particles.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trisphere
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet
from pymnpbem.mie import MieRet


def main():
    """Retarded BEM vs Mie theory for gold nanosphere."""
    print("Demo: Retarded BEM - Comparison with Mie theory")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    diameter = 100  # nm (large enough for retardation effects)
    sphere = trisphere(400, diameter)
    print(f"Sphere: d={diameter} nm, {sphere.n_faces} faces")

    p = ComParticle([eps_vacuum, eps_gold], [sphere], [[2, 1]])

    wavelengths = np.linspace(400, 800, 80)

    # BEM calculation
    exc = PlaneWaveRet(pol=[1, 0, 0], direction=[0, 0, 1])
    bem = BEMRet(p)

    print("\nComputing BEM spectrum...")
    spec = SpectrumRet(bem, exc, wavelengths)
    result = spec.compute()
    ext_bem = result['ext'][:, 0]
    sca_bem = result['sca'][:, 0]

    # Mie theory calculation
    print("Computing Mie theory...")
    ext_mie = np.zeros(len(wavelengths))
    sca_mie = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        eps_in = eps_gold(wl)[0]
        eps_out = eps_vacuum(wl)[0]
        mie = MieRet(diameter, eps_in, eps_out, wl)
        ext_mie[i] = mie.ext()
        sca_mie[i] = mie.sca()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, ext_bem, 'b-', linewidth=2, label='BEM Extinction')
    ax.plot(wavelengths, sca_bem, 'r-', linewidth=2, label='BEM Scattering')
    ax.plot(wavelengths, ext_mie, 'b--', linewidth=2, label='Mie Extinction')
    ax.plot(wavelengths, sca_mie, 'r--', linewidth=2, label='Mie Scattering')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross section (nm²)', fontsize=12)
    ax.set_title(f'Retarded BEM vs Mie Theory (d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Compute error
    error = np.mean(np.abs(ext_bem - ext_mie) / ext_mie) * 100
    print(f"\nMean relative error: {error:.1f}%")

    plt.tight_layout()
    plt.savefig('demo_planewave_ret_sphere.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
