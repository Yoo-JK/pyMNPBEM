"""
Electron Energy Loss Spectroscopy (EELS) simulation.

Simulates energy loss of swift electrons passing near or through
metallic nanoparticles.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct, Point
from ..misc.options import BEMOptions
from ..misc.units import SPEED_OF_LIGHT, eV2nm


class EELSStat:
    """
    EELS simulation in quasistatic approximation.

    Computes the energy loss probability for an electron beam
    passing near or through a nanoparticle.

    The electron creates a time-dependent electric field that
    excites the particle's plasmon modes.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions (x, y) in nm.
    velocity : float
        Electron velocity (relative to speed of light, v/c).
        Typical TEM: 0.5-0.7 (100-300 keV).
    width : float, optional
        Beam width for extended beam simulation.

    Examples
    --------
    >>> from mnpbem import EELSStat
    >>> # Single beam position at x=15 nm, y=0
    >>> eels = EELSStat([15, 0], velocity=0.5)
    """

    def __init__(
        self,
        impact: np.ndarray,
        velocity: float = 0.5,
        width: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize EELS simulation.

        Parameters
        ----------
        impact : ndarray
            Impact parameter (x, y) positions in nm.
        velocity : float
            Electron velocity v/c (typically 0.5-0.7).
        width : float, optional
            Beam width.
        """
        self.impact = np.atleast_2d(impact)
        if self.impact.shape[1] == 2:
            # Add z=0 if only (x,y) given
            self.impact = np.column_stack([self.impact, np.zeros(len(self.impact))])

        self.velocity = velocity  # v/c
        self.width = width
        self.options = kwargs

        # Electron properties
        self.v = velocity * SPEED_OF_LIGHT  # nm/fs

    @property
    def n_beams(self) -> int:
        """Number of beam positions."""
        return len(self.impact)

    @property
    def gamma_lorentz(self) -> float:
        """Lorentz factor gamma = 1/sqrt(1 - v^2/c^2)."""
        return 1.0 / np.sqrt(1 - self.velocity ** 2)

    def __call__(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute external potential for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm (or energy if in eV).

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        return self.potential(p, enei)

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential from electron beam.

        The electron creates a time-dependent potential that,
        after Fourier transform, gives the excitation at frequency omega.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        pos = p.pos  # Face centroids (n_faces, 3)
        nvec = p.nvec

        # Angular frequency
        omega = 2 * np.pi * SPEED_OF_LIGHT / enei

        phip = np.zeros((p.n_faces, self.n_beams), dtype=complex)

        for i, r_beam in enumerate(self.impact):
            # Distance from beam to face centroids
            # Beam travels along z-axis through (x0, y0)
            dx = pos[:, 0] - r_beam[0]
            dy = pos[:, 1] - r_beam[1]
            rho = np.sqrt(dx ** 2 + dy ** 2)  # Perpendicular distance
            z = pos[:, 2]  # Distance along beam

            # Avoid division by zero
            rho = np.where(rho < 1e-10, 1e-10, rho)

            # Electric field from electron (in frequency domain)
            # Using modified Bessel functions for retardation
            # In quasistatic limit, field is approximately:
            # E_rho ~ (2*omega/v^2) * K_1(omega*rho/v) * exp(i*omega*z/v)
            # where K_1 is modified Bessel function

            from scipy.special import kv

            arg = omega * rho / self.v / self.gamma_lorentz
            phase = np.exp(1j * omega * z / self.v)

            # Bessel function K_0 for potential
            K0 = kv(0, arg)
            K1 = kv(1, arg)

            # Potential at surfaces
            prefactor = 2 / (self.v * self.gamma_lorentz)
            phi = prefactor * K0 * phase

            # Normal derivative for BEM
            # d(phi)/dn = grad(phi) . n
            rho_hat = np.stack([dx / rho, dy / rho, np.zeros_like(rho)], axis=1)

            grad_phi_rho = -prefactor * omega / (self.v * self.gamma_lorentz) * K1 * phase
            grad_phi_z = 1j * omega / self.v * phi

            grad_phi = grad_phi_rho[:, np.newaxis] * rho_hat
            grad_phi[:, 2] += grad_phi_z

            phip[:, i] = np.sum(grad_phi * nvec, axis=1)

        return CompStruct(p, enei, phip=phip)

    def loss(self, sig: CompStruct) -> np.ndarray:
        """
        Compute energy loss probability.

        The loss probability is proportional to the work done by
        the induced field on the electron.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Energy loss probability for each beam position.
        """
        # Get induced potential at beam positions
        omega = 2 * np.pi * SPEED_OF_LIGHT / sig.enei

        charges = sig.get('sig')
        pos_surf = sig.p.pos
        area = sig.p.area

        loss = np.zeros(self.n_beams)

        for i, r_beam in enumerate(self.impact):
            # Integrate induced field along beam trajectory
            # Loss ~ Im[integral E_z(z) * exp(-i*omega*z/v) dz]

            sig_vals = charges[:, i] if charges.ndim > 1 else charges

            # Sample points along z
            z_min = pos_surf[:, 2].min() - 50
            z_max = pos_surf[:, 2].max() + 50
            z_points = np.linspace(z_min, z_max, 100)

            Ez_integral = 0j
            for z in z_points:
                # Distance from surface elements to point on beam
                dx = r_beam[0] - pos_surf[:, 0]
                dy = r_beam[1] - pos_surf[:, 1]
                dz = z - pos_surf[:, 2]
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                r = np.where(r < 1e-10, 1e-10, r)

                # Z-component of field from surface charges
                Ez = np.sum(sig_vals * dz / (4 * np.pi * r ** 3) * area)

                # Fourier integral weight
                phase = np.exp(-1j * omega * z / self.v)
                Ez_integral += Ez * phase

            dz = (z_max - z_min) / 100
            loss[i] = np.imag(Ez_integral * dz)

        return loss

    def loss_map(
        self,
        p: ComParticle,
        bem,
        enei: float,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute EELS loss probability map.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        bem : BEM solver
            BEM solver (BEMStat or BEMRet).
        enei : float
            Wavelength in nm.
        x_range : tuple
            (x_min, x_max) in nm.
        y_range : tuple
            (y_min, y_max) in nm.
        n_points : int
            Number of points per dimension.

        Returns
        -------
        x : ndarray
            X coordinates.
        y : ndarray
            Y coordinates.
        loss_map : ndarray
            Loss probability map, shape (n_points, n_points).
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        loss_map = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                eels = EELSStat([X[i, j], Y[i, j]], self.velocity)
                exc = eels(p, enei)
                sig = bem.solve(exc)
                loss_map[i, j] = eels.loss(sig)[0]

        return x, y, loss_map

    def bulkloss(self, p: ComParticle, enei: float, phiout: float = 0.01) -> np.ndarray:
        """
        Compute EELS bulk loss probability.

        Based on Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (18).
        This computes the energy loss probability from bulk material
        traversed by the electron beam.

        Parameters
        ----------
        p : ComParticle
            Compound particle (used for dielectric functions and path length).
        enei : float
            Wavelength in nm.
        phiout : float, optional
            Collection angle in radians (default: 0.01 rad ~ 0.57 deg).

        Returns
        -------
        ndarray
            Bulk loss probability for each beam position (1/eV).

        Notes
        -----
        The bulk loss depends on:
        - The dielectric function of the material
        - The electron velocity
        - The path length through the material
        - The collection angle (phiout)
        """
        # Physical constants in atomic units
        bohr = 0.05292  # Bohr radius in nm
        hartree = 27.211  # 2 * Rydberg in eV
        fine = 1 / 137.036  # Fine structure constant
        eV2nm_local = 1239.84193  # eV to nm conversion

        # Photon energy in eV
        ene = eV2nm_local / enei

        # Rest mass of electron in eV
        mass = 0.51e6

        # Get dielectric functions for different media
        if hasattr(p, 'eps'):
            eps_list = []
            for eps_func in p.eps:
                if callable(eps_func):
                    eps_list.append(eps_func(enei))
                else:
                    eps_list.append(eps_func)
            eps = np.array(eps_list)
        else:
            eps = np.array([1.0])

        # Wavenumber of electron beam
        q = 2 * np.pi / (enei * self.velocity)

        # Cutoff wavenumber (depends on collection angle)
        qc = q * np.sqrt((mass / ene) ** 2 * self.velocity ** 2 * phiout ** 2 + 1)

        # Wavenumber of light in each medium
        k = 2 * np.pi / enei * np.sqrt(eps)

        # Compute path length through each medium
        path_length = self._compute_path_length(p)

        # Bulk loss probability [Eq. (18) from Garcia de Abajo]
        # P_bulk = (alpha / (pi * v^2)) * Im[(v^2 - 1/eps) * log((qc^2 - k^2)/(q^2 - k^2))] * L
        # where alpha is fine structure constant

        pbulk = np.zeros(self.n_beams)

        for i in range(self.n_beams):
            path_i = path_length[i] if isinstance(path_length, np.ndarray) else path_length

            # Sum over all media
            for j, eps_j in enumerate(eps):
                if np.abs(eps_j) < 1e-10:
                    continue

                k_j = k[j]

                # Logarithm argument
                num = qc ** 2 - k_j ** 2
                den = q ** 2 - k_j ** 2

                # Avoid numerical issues
                if np.abs(den) < 1e-20:
                    continue

                log_arg = num / den
                if np.real(log_arg) <= 0:
                    log_val = np.log(np.abs(log_arg)) + 1j * np.pi
                else:
                    log_val = np.log(log_arg)

                # Bulk loss contribution
                prefactor = fine ** 2 / (bohr * hartree * np.pi * self.velocity ** 2)
                term = (self.velocity ** 2 - 1.0 / eps_j) * log_val
                pbulk[i] += prefactor * np.imag(term) * path_i

        return pbulk

    def _compute_path_length(self, p: ComParticle) -> np.ndarray:
        """
        Compute path length of electron beam through particle.

        Parameters
        ----------
        p : ComParticle
            Particle geometry.

        Returns
        -------
        ndarray
            Path length for each beam position.
        """
        # Get particle vertices and faces
        if hasattr(p, 'verts') and hasattr(p, 'faces'):
            verts = p.verts
            faces = p.faces
        elif hasattr(p, 'pc'):
            verts = p.pc.verts if hasattr(p.pc, 'verts') else None
            faces = p.pc.faces if hasattr(p.pc, 'faces') else None
        else:
            # Estimate from bounding box
            if hasattr(p, 'pos'):
                z_extent = p.pos[:, 2].max() - p.pos[:, 2].min()
                return np.ones(self.n_beams) * z_extent
            return np.ones(self.n_beams) * 10.0  # Default 10 nm

        if verts is None:
            if hasattr(p, 'pos'):
                z_extent = p.pos[:, 2].max() - p.pos[:, 2].min()
                return np.ones(self.n_beams) * z_extent
            return np.ones(self.n_beams) * 10.0

        path_length = np.zeros(self.n_beams)

        for i, r_beam in enumerate(self.impact):
            # Ray-casting along z-axis to find intersections
            x0, y0 = r_beam[0], r_beam[1]

            # Find z-coordinates where ray intersects particle surface
            z_intersections = []

            # Simple approach: find min/max z of faces near beam
            for face in faces:
                tri_verts = verts[face]

                # Check if beam passes through triangle's xy projection
                if self._point_in_triangle_2d(x0, y0, tri_verts[:, :2]):
                    # Interpolate z at beam position
                    z_int = self._interpolate_z(x0, y0, tri_verts)
                    if z_int is not None:
                        z_intersections.append(z_int)

            if len(z_intersections) >= 2:
                z_intersections.sort()
                # Path length is total traversed distance
                path_length[i] = z_intersections[-1] - z_intersections[0]
            elif len(z_intersections) == 1:
                # Single intersection - estimate
                path_length[i] = 0
            else:
                # No intersection - beam outside particle
                path_length[i] = 0

        return path_length

    def _point_in_triangle_2d(self, px: float, py: float, tri: np.ndarray) -> bool:
        """Check if point (px, py) is inside 2D triangle."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        p = np.array([px, py])
        v1, v2, v3 = tri[0], tri[1], tri[2]

        d1 = sign(p, v1, v2)
        d2 = sign(p, v2, v3)
        d3 = sign(p, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _interpolate_z(self, px: float, py: float, tri: np.ndarray) -> float:
        """Interpolate z-coordinate at point (px, py) on triangle."""
        # Barycentric interpolation
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = np.array([px, py, 0]) - tri[0]

        d00 = v0[0] * v0[0] + v0[1] * v0[1]
        d01 = v0[0] * v1[0] + v0[1] * v1[1]
        d11 = v1[0] * v1[0] + v1[1] * v1[1]
        d20 = v2[0] * v0[0] + v2[1] * v0[1]
        d21 = v2[0] * v1[0] + v2[1] * v1[1]

        denom = d00 * d11 - d01 * d01
        if np.abs(denom) < 1e-10:
            return None

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        if u >= 0 and v >= 0 and w >= 0:
            return u * tri[0, 2] + v * tri[1, 2] + w * tri[2, 2]
        return None

    def field(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute electric field from electron beam.

        Parameters
        ----------
        p : ComParticle
            Particle where field is computed.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Object containing electric field 'e'.
        """
        pos = p.pos
        n_pos = len(pos)

        omega = 2 * np.pi * SPEED_OF_LIGHT / enei

        e = np.zeros((n_pos, self.n_beams, 3), dtype=complex)

        for i, r_beam in enumerate(self.impact):
            dx = pos[:, 0] - r_beam[0]
            dy = pos[:, 1] - r_beam[1]
            rho = np.sqrt(dx ** 2 + dy ** 2)
            z = pos[:, 2]

            rho = np.where(rho < 1e-10, 1e-10, rho)

            from scipy.special import kv

            arg = omega * rho / self.v / self.gamma_lorentz
            phase = np.exp(1j * omega * z / self.v)

            K0 = kv(0, arg)
            K1 = kv(1, arg)

            # Electric field components
            prefactor = 2 * omega / (self.v ** 2 * self.gamma_lorentz ** 2)

            # Radial component (perpendicular to beam)
            rho_hat_x = dx / rho
            rho_hat_y = dy / rho

            E_rho = prefactor * K1 * phase

            e[:, i, 0] = E_rho * rho_hat_x
            e[:, i, 1] = E_rho * rho_hat_y

            # Z component (parallel to beam)
            e[:, i, 2] = 1j * prefactor / self.gamma_lorentz * K0 * phase

        return CompStruct(p, enei, e=e)

    def __repr__(self) -> str:
        return f"EELSStat(n_beams={self.n_beams}, velocity={self.velocity})"


def eels(
    impact: np.ndarray,
    velocity: float = 0.5,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> EELSStat:
    """
    Factory function for EELS excitation.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions.
    velocity : float
        Electron velocity v/c.
    options : BEMOptions or dict, optional
        Options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    EELSStat
        EELS simulation object.
    """
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}

    return EELSStat(impact, velocity, **all_options)
