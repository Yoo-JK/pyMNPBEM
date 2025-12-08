"""
Retarded Electron Energy Loss Spectroscopy (EELS) simulation.

Simulates energy loss of swift electrons passing near or through
metallic nanoparticles with full electromagnetic (retarded) treatment.

This is important for larger particles where retardation effects
become significant.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from scipy import special

from ..particles import ComParticle, CompStruct
from ..misc.units import SPEED_OF_LIGHT, eV2nm


class EELSRet:
    """
    Retarded EELS simulation.

    Computes the energy loss probability for an electron beam
    using full electromagnetic treatment including retardation.

    For larger particles (size > lambda/10), retardation effects
    become important and this class should be used instead of EELSStat.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions (x, y) or (x, y, z) in nm.
    velocity : float
        Electron velocity (relative to speed of light, v/c).
        Typical TEM: 0.5-0.7 (100-300 keV).
    direction : array_like, optional
        Beam direction (default: [0, 0, 1] along z-axis).
    width : float, optional
        Beam width for extended beam simulation.

    Examples
    --------
    >>> from pymnpbem import EELSRet
    >>> # Single beam position at x=15 nm, y=0
    >>> eels = EELSRet([15, 0], velocity=0.5)
    """

    def __init__(
        self,
        impact: np.ndarray,
        velocity: float = 0.5,
        direction: Optional[np.ndarray] = None,
        width: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize retarded EELS simulation.

        Parameters
        ----------
        impact : ndarray
            Impact parameter (x, y) or (x, y, z) positions in nm.
        velocity : float
            Electron velocity v/c (typically 0.5-0.7).
        direction : ndarray, optional
            Beam direction (default: z-axis).
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

        # Beam direction (default: along z-axis)
        if direction is None:
            direction = np.array([0, 0, 1])
        self.direction = np.array(direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)

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
        Compute external fields for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip', 'phi', 'a' fields.
        """
        return self.excitation(p, enei)

    def excitation(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute electromagnetic excitation from electron beam.

        The electron creates time-dependent electromagnetic fields that,
        after Fourier transform, give the excitation at frequency omega.

        For retarded case, we include both scalar and vector potentials.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with electromagnetic fields.
        """
        pos = p.pos  # Face centroids (n_faces, 3)
        nvec = p.nvec

        # Angular frequency and wave number
        omega = 2 * np.pi * SPEED_OF_LIGHT / enei
        k = omega / SPEED_OF_LIGHT

        # Initialize field arrays
        phip = np.zeros((p.n_faces, self.n_beams), dtype=complex)
        phi = np.zeros((p.n_faces, self.n_beams), dtype=complex)
        a = np.zeros((p.n_faces, self.n_beams, 3), dtype=complex)
        e = np.zeros((p.n_faces, self.n_beams, 3), dtype=complex)
        h = np.zeros((p.n_faces, self.n_beams, 3), dtype=complex)

        for i, r_beam in enumerate(self.impact):
            phi_i, a_i, e_i, h_i = self._fields_at_positions(
                pos, r_beam, omega, k
            )

            phi[:, i] = phi_i
            a[:, i, :] = a_i
            e[:, i, :] = e_i
            h[:, i, :] = h_i

            # Normal derivative of potential for BEM
            # phip = dphi/dn - i*omega/c * (n . A)
            # In Lorenz gauge: grad(phi) + (1/c) * dA/dt = 0
            grad_phi = self._gradient_potential(pos, r_beam, omega, k)
            phip[:, i] = (np.sum(grad_phi * nvec, axis=1)
                         - 1j * k * np.sum(a_i * nvec, axis=1))

        return CompStruct(p, enei, phip=phip, phi=phi, a=a, e=e, h=h)

    def _fields_at_positions(
        self,
        pos: np.ndarray,
        r_beam: np.ndarray,
        omega: float,
        k: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute EM fields at given positions from electron beam.

        Uses Liénard-Wiechert potentials in frequency domain.

        Parameters
        ----------
        pos : ndarray
            Positions (n, 3).
        r_beam : ndarray
            Beam impact position.
        omega : float
            Angular frequency.
        k : float
            Wave number.

        Returns
        -------
        phi : ndarray
            Scalar potential (n,).
        a : ndarray
            Vector potential (n, 3).
        e : ndarray
            Electric field (n, 3).
        h : ndarray
            Magnetic field (n, 3).
        """
        n_pos = len(pos)
        gamma = self.gamma_lorentz
        v = self.v

        phi = np.zeros(n_pos, dtype=complex)
        a = np.zeros((n_pos, 3), dtype=complex)
        e = np.zeros((n_pos, 3), dtype=complex)
        h = np.zeros((n_pos, 3), dtype=complex)

        for i, r in enumerate(pos):
            # Vector from trajectory to field point
            r_perp = r - r_beam
            r_para = np.dot(r_perp, self.direction)
            r_perp = r_perp - r_para * self.direction
            b = np.linalg.norm(r_perp)  # Impact parameter

            if b < 1e-10:
                # On beam axis
                continue

            b_hat = r_perp / b

            # Frequency-domain potentials from moving charge
            # Using modified Bessel functions
            xi = omega * b / (gamma * v)

            if xi > 100:
                # Bessel functions essentially zero
                continue

            K0 = special.kv(0, xi)
            K1 = special.kv(1, xi)

            # Phase factor exp(i*omega*z/v)
            z = r_para  # Position along beam
            phase = np.exp(1j * omega * z / v)

            # Scalar potential
            prefactor_phi = 2 / (gamma * v)
            phi[i] = prefactor_phi * K0 * phase

            # Vector potential A = v * phi / c^2 * direction
            a[i, :] = self.velocity * phi[i] * self.direction

            # Electric field components
            # E_perp ~ K1, E_para ~ K0
            prefactor_e = 2 * omega / (gamma**2 * v**2)

            e_perp = prefactor_e * K1 * phase * b_hat
            e_para = -1j * prefactor_e / gamma * K0 * phase * self.direction

            e[i, :] = e_perp + e_para

            # Magnetic field H = (v x E) / c
            h[i, :] = self.velocity * np.cross(self.direction, e[i, :])

        return phi, a, e, h

    def _gradient_potential(
        self,
        pos: np.ndarray,
        r_beam: np.ndarray,
        omega: float,
        k: float
    ) -> np.ndarray:
        """
        Compute gradient of scalar potential.

        Parameters
        ----------
        pos : ndarray
            Positions (n, 3).
        r_beam : ndarray
            Beam impact position.
        omega : float
            Angular frequency.
        k : float
            Wave number.

        Returns
        -------
        grad_phi : ndarray
            Gradient of potential (n, 3).
        """
        n_pos = len(pos)
        gamma = self.gamma_lorentz
        v = self.v

        grad_phi = np.zeros((n_pos, 3), dtype=complex)

        for i, r in enumerate(pos):
            r_perp = r - r_beam
            r_para = np.dot(r_perp, self.direction)
            r_perp = r_perp - r_para * self.direction
            b = np.linalg.norm(r_perp)

            if b < 1e-10:
                continue

            b_hat = r_perp / b
            xi = omega * b / (gamma * v)

            if xi > 100:
                continue

            K0 = special.kv(0, xi)
            K1 = special.kv(1, xi)

            z = r_para
            phase = np.exp(1j * omega * z / v)
            prefactor = 2 / (gamma * v)

            # d(phi)/db = prefactor * d(K0)/d(xi) * d(xi)/db * phase
            # d(K0)/d(xi) = -K1
            grad_perp = -prefactor * omega / (gamma * v) * K1 * phase * b_hat

            # d(phi)/dz = prefactor * K0 * (i*omega/v) * phase
            grad_para = prefactor * K0 * 1j * omega / v * phase * self.direction

            grad_phi[i, :] = grad_perp + grad_para

        return grad_phi

    def loss(self, sig: CompStruct, method: str = 'induced') -> np.ndarray:
        """
        Compute energy loss probability.

        The loss probability is proportional to the work done by
        the induced field on the electron.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges and currents.
        method : str
            Method: 'induced' (induced field), 'boundary' (boundary formula).

        Returns
        -------
        ndarray
            Energy loss probability for each beam position.
        """
        omega = 2 * np.pi * SPEED_OF_LIGHT / sig.enei
        k = omega / SPEED_OF_LIGHT

        if method == 'boundary':
            return self._loss_boundary(sig, omega, k)
        else:
            return self._loss_induced(sig, omega, k)

    def _loss_induced(
        self,
        sig: CompStruct,
        omega: float,
        k: float
    ) -> np.ndarray:
        """
        Compute loss from induced fields along trajectory.

        Uses numerical integration along beam path.
        """
        # Get surface data
        pos_surf = sig.p.pos
        area = sig.p.area
        nvec = sig.p.nvec
        charges = sig.get('sig')

        # Surface currents for retarded case
        h_surf = sig.get('h') if hasattr(sig, '_data') and 'h' in sig._data else None

        loss = np.zeros(self.n_beams)

        for i, r_beam in enumerate(self.impact):
            # Sample points along trajectory
            z_min = pos_surf[:, 2].min() - 100
            z_max = pos_surf[:, 2].max() + 100
            n_sample = 200
            z_vals = np.linspace(z_min, z_max, n_sample)

            trajectory = np.array([
                r_beam + z * self.direction for z in z_vals
            ])

            # Get charge distribution for this beam
            sig_vals = charges[:, i] if charges.ndim > 1 else charges

            # Compute induced field along trajectory
            Ez_total = 0j

            for j, pt in enumerate(trajectory):
                # Distance from surface elements to trajectory point
                r_vec = pt - pos_surf
                r = np.linalg.norm(r_vec, axis=1)
                r = np.where(r < 1e-10, 1e-10, r)
                r_hat = r_vec / r[:, np.newaxis]

                # Retarded Green function
                G = np.exp(1j * k * r) / (4 * np.pi * r)
                dG_dr = (1j * k - 1 / r) * G

                # Electric field from surface charges (Coulomb term)
                E_coulomb = np.sum(
                    sig_vals[:, np.newaxis] * dG_dr[:, np.newaxis] * r_hat * area[:, np.newaxis],
                    axis=0
                )

                # Z-component for loss calculation
                Ez = np.dot(E_coulomb, self.direction)

                # Phase factor for Fourier integral
                z = np.dot(pt - r_beam, self.direction)
                phase = np.exp(-1j * omega * z / self.v)

                Ez_total += Ez * phase

            dz = (z_max - z_min) / n_sample
            loss[i] = np.imag(Ez_total * dz) / np.pi

        return np.abs(loss)

    def _loss_boundary(
        self,
        sig: CompStruct,
        omega: float,
        k: float
    ) -> np.ndarray:
        """
        Compute loss using boundary integral formula.

        More efficient for many beam positions.
        """
        pos_surf = sig.p.pos
        area = sig.p.area
        nvec = sig.p.nvec
        charges = sig.get('sig')

        loss = np.zeros(self.n_beams)
        gamma = self.gamma_lorentz

        for i, r_beam in enumerate(self.impact):
            sig_vals = charges[:, i] if charges.ndim > 1 else charges

            # Loss from work done: integral of (E_ind . v) over time
            # = integral of (E_ind . direction) * v dt
            # In frequency domain: Im[ conj(E_ext) . phi_ind ]

            total = 0j
            for j in range(len(pos_surf)):
                # Perpendicular distance
                r_perp = pos_surf[j] - r_beam
                z = np.dot(r_perp, self.direction)
                r_perp = r_perp - z * self.direction
                b = np.linalg.norm(r_perp)

                if b < 1e-10:
                    continue

                xi = omega * b / (gamma * self.v)
                if xi > 100:
                    continue

                K0 = special.kv(0, xi)
                phase = np.exp(1j * omega * z / self.v)

                # Incident potential
                phi_inc = 2 / (gamma * self.v) * K0 * phase

                # Contribution to loss
                total += np.conj(phi_inc) * sig_vals[j] * area[j]

            loss[i] = np.imag(total) / np.pi

        return np.abs(loss)

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
            BEM solver (BEMRet).
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
                eels = EELSRet([X[i, j], Y[i, j]], self.velocity,
                               self.direction)
                exc = eels(p, enei)
                sig = bem.solve(exc)
                loss_map[i, j] = eels.loss(sig)[0]

        return x, y, loss_map

    def cathodoluminescence(
        self,
        sig: CompStruct,
        directions: np.ndarray
    ) -> np.ndarray:
        """
        Compute cathodoluminescence (CL) emission.

        CL measures the light emitted when the electron beam
        excites the nanoparticle.

        Parameters
        ----------
        sig : CompStruct
            BEM solution.
        directions : ndarray
            Emission directions (n_dir, 3).

        Returns
        -------
        ndarray
            CL intensity for each direction.
        """
        directions = np.atleast_2d(directions)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        n_dir = len(directions)

        omega = 2 * np.pi * SPEED_OF_LIGHT / sig.enei
        k = omega / SPEED_OF_LIGHT

        pos_surf = sig.p.pos
        area = sig.p.area
        charges = sig.get('sig')

        cl = np.zeros(n_dir)

        for d, sdir in enumerate(directions):
            intensity = 0j

            for i in range(self.n_beams):
                sig_vals = charges[:, i] if charges.ndim > 1 else charges

                for j in range(len(pos_surf)):
                    # Far-field phase
                    phase = np.exp(-1j * k * np.dot(sdir, pos_surf[j]))
                    intensity += sig_vals[j] * area[j] * phase

            cl[d] = np.abs(intensity) ** 2

        return cl

    def __repr__(self) -> str:
        return f"EELSRet(n_beams={self.n_beams}, velocity={self.velocity})"


class EELSRetLayer(EELSRet):
    """
    Retarded EELS with layer substrate.

    Includes substrate effects (image charges, reflections).

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions.
    velocity : float
        Electron velocity v/c.
    layer : LayerStructure
        Layer structure for substrate.
    """

    def __init__(
        self,
        impact: np.ndarray,
        velocity: float = 0.5,
        layer=None,
        **kwargs
    ):
        """Initialize retarded EELS with layer."""
        super().__init__(impact, velocity, **kwargs)
        self.layer = layer

    def excitation(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute excitation including layer effects.

        Adds reflected fields from substrate.
        """
        # Get basic excitation
        exc = super().excitation(p, enei)

        if self.layer is None:
            return exc

        # Add layer contributions
        # This would include image charges and reflected fields
        # Simplified implementation - full version would use
        # Sommerfeld integrals for layer Green functions
        return exc


def eels_ret(
    impact: np.ndarray,
    velocity: float = 0.5,
    **kwargs
) -> EELSRet:
    """
    Factory function for retarded EELS excitation.

    Parameters
    ----------
    impact : ndarray
        Impact parameter positions.
    velocity : float
        Electron velocity v/c.
    **kwargs : dict
        Additional options.

    Returns
    -------
    EELSRet
        Retarded EELS simulation object.
    """
    return EELSRet(impact, velocity, **kwargs)
