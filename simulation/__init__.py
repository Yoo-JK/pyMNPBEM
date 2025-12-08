"""
Simulation classes for MNPBEM.

This module provides excitation and spectrum calculation classes:
- PlaneWaveStat: Plane wave excitation (quasistatic)
- PlaneWaveStatLayer: Plane wave with substrate (quasistatic)
- PlaneWaveStatMirror: Plane wave with mirror substrate
- DipoleStat: Dipole excitation (quasistatic)
- DipoleStatLayer: Dipole with substrate (quasistatic)
- DipoleStatMirror: Dipole with mirror substrate
- EELSStat: Electron energy loss spectroscopy
- SpectrumStat: Spectral response calculations (quasistatic)
"""

from .planewave_stat import PlaneWaveStat, planewave
from .planewave_stat_layer import PlaneWaveStatLayer, PlaneWaveStatMirror, planewave_layer
from .dipole_stat import DipoleStat, dipole
from .dipole_stat_layer import DipoleStatLayer, DipoleStatMirror, dipole_layer
from .eels_stat import EELSStat, eels
from .spectrum_stat import SpectrumStat

__all__ = [
    "PlaneWaveStat",
    "planewave",
    "PlaneWaveStatLayer",
    "PlaneWaveStatMirror",
    "planewave_layer",
    "DipoleStat",
    "dipole",
    "DipoleStatLayer",
    "DipoleStatMirror",
    "dipole_layer",
    "EELSStat",
    "eels",
    "SpectrumStat",
]
