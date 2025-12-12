#!/usr/bin/env python
"""
Test script for retarded BEM implementation.

Compares BEMStat and BEMRet results for a gold nanosphere.
Expected: Both should show peak around 520-530nm for 50nm gold sphere.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/pyMNPBEM')

# Test basic imports
print("=" * 60)
print("Testing pyMNPBEM BEM implementations")
print("=" * 60)

try:
    from mnpbem.particles import ComParticle
    from mnpbem.particles.shapes import trisphere
    from mnpbem.material import EpsConst, EpsTable
    from mnpbem.bem import BEMStat, BEMRet
    from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Create gold nanosphere (50 nm diameter)
print("\nCreating 50 nm gold nanosphere...")
diameter = 50  # nm
n_faces = 144

try:
    sphere = trisphere(n_faces, diameter)
    print(f"  Sphere created with {sphere.n_faces} faces")
except Exception as e:
    print(f"  Error creating sphere: {e}")
    sys.exit(1)

# Create materials
print("\nSetting up materials...")
try:
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    print(f"  Gold data range: {eps_gold.enei_min:.1f} - {eps_gold.enei_max:.1f} nm")
except Exception as e:
    print(f"  Error loading materials: {e}")
    sys.exit(1)

# Create composite particle
# inout = [2, 1] means: inside=material 2 (gold), outside=material 1 (vacuum)
print("\nCreating composite particle...")
try:
    epstab = [eps_vacuum, eps_gold]  # Index 0=vacuum, 1=gold
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)
    print(f"  Total faces: {p.n_faces}")
    print(f"  inout: {p.inout}")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# Create BEM solvers
print("\nInitializing BEM solvers...")
try:
    bem_stat = BEMStat(p)
    bem_ret = BEMRet(p)
    print("  BEMStat and BEMRet initialized")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# Create plane wave excitations
print("\nCreating plane wave excitations...")
try:
    # Polarization along x, propagation along z
    pol = [[1, 0, 0]]
    dir = [[0, 0, 1]]

    exc_stat = PlaneWaveStat(pol, dir)
    exc_ret = PlaneWaveRet(pol, dir)
    print("  PlaneWaveStat and PlaneWaveRet initialized")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# Test wavelengths
wavelengths = np.linspace(400, 700, 31)

print("\n" + "=" * 60)
print("Computing spectra...")
print("=" * 60)

ext_stat = []
ext_ret = []

for i, wl in enumerate(wavelengths):
    try:
        # BEMStat
        exc_s = exc_stat(p, wl)
        sig_s = bem_stat.solve(exc_s)
        e_s = exc_stat.extinction(sig_s)
        ext_stat.append(e_s[0] if hasattr(e_s, '__len__') else e_s)

        # BEMRet
        exc_r = exc_ret(p, wl)
        sig_r = bem_ret.solve(exc_r)
        e_r = exc_ret.extinction(sig_r)
        ext_ret.append(e_r[0] if hasattr(e_r, '__len__') else e_r)

        if i % 10 == 0 or i == len(wavelengths) - 1:
            print(f"  {wl:.0f} nm: BEMStat ext={ext_stat[-1]:.1f}, BEMRet ext={ext_ret[-1]:.1f}")

    except Exception as e:
        print(f"  Error at {wl} nm: {e}")
        import traceback
        traceback.print_exc()
        ext_stat.append(0)
        ext_ret.append(0)

ext_stat = np.array(ext_stat)
ext_ret = np.array(ext_ret)

# Find peaks
print("\n" + "=" * 60)
print("Results summary:")
print("=" * 60)

if np.any(ext_stat > 0):
    peak_idx_stat = np.argmax(ext_stat)
    print(f"BEMStat: Peak at {wavelengths[peak_idx_stat]:.0f} nm, Extinction = {ext_stat[peak_idx_stat]:.1f} nm^2")
else:
    print("BEMStat: No valid data")

if np.any(ext_ret > 0):
    peak_idx_ret = np.argmax(ext_ret)
    print(f"BEMRet:  Peak at {wavelengths[peak_idx_ret]:.0f} nm, Extinction = {ext_ret[peak_idx_ret]:.1f} nm^2")
else:
    print("BEMRet: No valid data")

# Expected: Peak around 520-530 nm for 50nm gold sphere
print("\nExpected: Peak around 520-530 nm for 50nm gold nanosphere in vacuum")

# Check if results are reasonable
peak_expected = 525  # nm (approximate)
if np.any(ext_stat > 0):
    peak_stat = wavelengths[np.argmax(ext_stat)]
    if 490 < peak_stat < 560:
        print(f"BEMStat peak ({peak_stat:.0f} nm) is in expected range (490-560 nm) - OK")
    else:
        print(f"BEMStat peak ({peak_stat:.0f} nm) is OUTSIDE expected range - CHECK")

if np.any(ext_ret > 0):
    peak_ret = wavelengths[np.argmax(ext_ret)]
    if 490 < peak_ret < 560:
        print(f"BEMRet peak ({peak_ret:.0f} nm) is in expected range (490-560 nm) - OK")
    else:
        print(f"BEMRet peak ({peak_ret:.0f} nm) is OUTSIDE expected range - CHECK")

print("\nDone!")
