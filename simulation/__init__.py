"""
Simulation classes for MNPBEM.

This module provides excitation and spectrum calculation classes:

Quasistatic:
- PlaneWaveStat: Plane wave excitation (quasistatic)
- PlaneWaveStatLayer: Plane wave with substrate (quasistatic)
- PlaneWaveStatMirror: Plane wave with mirror substrate
- DipoleStat: Dipole excitation (quasistatic)
- DipoleStatLayer: Dipole with substrate (quasistatic)
- DipoleStatMirror: Dipole with mirror substrate
- EELSStat: Electron energy loss spectroscopy
- SpectrumStat: Spectral response calculations (quasistatic)

Retarded:
- PlaneWaveRet: Plane wave excitation (retarded)
- DipoleRet: Dipole excitation (retarded)
- SpectrumRet: Spectral response calculations (retarded)
- ElectronBeam: Electron beam excitation
"""

from .planewave_stat import PlaneWaveStat, planewave
from .planewave_stat_layer import PlaneWaveStatLayer, PlaneWaveStatMirror, planewave_layer
from .dipole_stat import DipoleStat, dipole
from .dipole_stat_layer import DipoleStatLayer, DipoleStatMirror, dipole_layer
from .eels_stat import EELSStat, eels
from .spectrum_stat import SpectrumStat
from .planewave_ret import PlaneWaveRet, planewave_ret
from .dipole_ret import DipoleRet, dipole_ret
from .spectrum_ret import SpectrumRet, DecayRateSpectrum, spectrum_ret
from .electronbeam import ElectronBeam, ElectronBeamRet, electronbeam

__all__ = [
    # Quasistatic
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
    # Retarded
    "PlaneWaveRet",
    "planewave_ret",
    "DipoleRet",
    "dipole_ret",
    "SpectrumRet",
    "DecayRateSpectrum",
    "spectrum_ret",
    "ElectronBeam",
    "ElectronBeamRet",
    "electronbeam",
]
