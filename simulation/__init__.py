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
- EELSStat: Electron energy loss spectroscopy (quasistatic)
- SpectrumStat: Spectral response calculations (quasistatic)

Retarded:
- PlaneWaveRet: Plane wave excitation (retarded)
- PlaneWaveRetLayer: Plane wave with substrate (retarded)
- PlaneWaveRetMirror: Plane wave with mirror substrate (retarded)
- DipoleRet: Dipole excitation (retarded)
- DipoleRetLayer: Dipole with substrate (retarded)
- DipoleRetMirror: Dipole with mirror substrate (retarded)
- EELSRet: Electron energy loss spectroscopy (retarded)
- SpectrumRet: Spectral response calculations (retarded)
- ElectronBeam: Electron beam excitation
"""

from .planewave_stat import PlaneWaveStat, planewave
from .planewave_stat_layer import PlaneWaveStatLayer, PlaneWaveStatMirror, planewave_layer
from .dipole_stat import DipoleStat, dipole
from .dipole_stat_layer import DipoleStatLayer, DipoleStatMirror, dipole_layer
from .eels_stat import EELSStat, eels
from .eels_ret import EELSRet, EELSRetLayer, eels_ret
from .spectrum_stat import SpectrumStat
from .planewave_ret import PlaneWaveRet, planewave_ret
from .planewave_ret_layer import PlaneWaveRetLayer, PlaneWaveRetMirror, planewave_ret_layer
from .dipole_ret import DipoleRet, dipole_ret
from .dipole_ret_layer import DipoleRetLayer, DipoleRetMirror, dipole_ret_layer
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
    "PlaneWaveRetLayer",
    "PlaneWaveRetMirror",
    "planewave_ret_layer",
    "DipoleRet",
    "dipole_ret",
    "DipoleRetLayer",
    "DipoleRetMirror",
    "dipole_ret_layer",
    "EELSRet",
    "EELSRetLayer",
    "eels_ret",
    "SpectrumRet",
    "DecayRateSpectrum",
    "spectrum_ret",
    "ElectronBeam",
    "ElectronBeamRet",
    "electronbeam",
]
