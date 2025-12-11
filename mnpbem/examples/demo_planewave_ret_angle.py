"""
Demo: Angle-dependent extinction for non-spherical particles.

Shows how extinction varies with incident angle.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymnpbem.particles import ComParticle
from pymnpbem.particles.shapes import trirod
from pymnpbem.material import EpsConst, EpsDrude
from pymnpbem.bem import BEMRet
from pymnpbem.simulation import PlaneWaveRet, SpectrumRet


def main():
    """Angle-dependent extinction of gold nanorod."""
    print("Demo: Angle-dependent extinction")
    print("=" * 60)

    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsDrude('Au')

    length = 60  # nm
    diameter = 20  # nm
    rod = trirod(300, length, diameter)
    p = ComParticle([eps_vacuum, eps_gold], [rod], [[2, 1]])

    print(f"Nanorod: L={length} nm, d={diameter} nm (aspect ratio = {length/diameter:.1f})")

    bem = BEMRet(p)
    wavelengths = np.linspace(450, 900, 80)

    # Different polarization angles (rod along x-axis)
    angles = [0, 30, 45, 60, 90]  # degrees

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    for i, angle in enumerate(angles):
        print(f"\nAngle: {angle}°")

        # Polarization rotated in xy plane, beam along z
        theta = np.radians(angle)
        pol = [np.cos(theta), np.sin(theta), 0]

        exc = PlaneWaveRet(pol=pol, direction=[0, 0, 1])
        spec = SpectrumRet(bem, exc, wavelengths, show_progress=False)
        result = spec.compute()
        ext = result['ext'][:, 0]

        ax.plot(wavelengths, ext, color=colors[i], linewidth=2, label=f'θ = {angle}°')

        peak_wl = wavelengths[np.argmax(ext)]
        print(f"  Peak wavelength: {peak_wl:.1f} nm")

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Extinction cross section (nm²)', fontsize=12)
    ax.set_title(f'Angle-Dependent Extinction (L={length} nm, d={diameter} nm)', fontsize=14)
    ax.legend(fontsize=11, title='Polarization angle')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_planewave_ret_angle.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
