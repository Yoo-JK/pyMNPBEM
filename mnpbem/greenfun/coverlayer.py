"""
Coverlayer module for coated/core-shell nanoparticles.

This module provides Green functions modified for particles
with thin coating layers (shells).
"""

import numpy as np
from typing import Optional, Union, Tuple


class CoverLayer:
    """
    Coating layer for core-shell particles.

    Represents a thin dielectric layer coating a particle,
    modifying the effective boundary conditions.

    Parameters
    ----------
    eps_layer : callable or complex
        Dielectric function of the layer material
    thickness : float
        Layer thickness in nm
    eps_core : callable or complex, optional
        Dielectric function of the core

    Attributes
    ----------
    eps_layer : callable
        Layer dielectric function
    thickness : float
        Layer thickness
    """

    def __init__(self, eps_layer, thickness, eps_core=None):
        """Initialize cover layer."""
        if callable(eps_layer):
            self.eps_layer = eps_layer
        else:
            self._eps_layer_val = complex(eps_layer)
            self.eps_layer = lambda wl: self._eps_layer_val

        self.thickness = thickness

        if eps_core is not None:
            if callable(eps_core):
                self.eps_core = eps_core
            else:
                self._eps_core_val = complex(eps_core)
                self.eps_core = lambda wl: self._eps_core_val
        else:
            self.eps_core = None

    def effective_eps(self, wavelength, eps_out=1.0):
        """
        Compute effective dielectric function for thin layer.

        For a thin layer, the effective boundary condition can be
        approximated using an effective dielectric constant.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_out : complex
            Outside dielectric function

        Returns
        -------
        eps_eff : complex
            Effective dielectric function
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        # Thin layer approximation
        # For d << wavelength, effective eps is weighted average
        if self.eps_core is not None:
            eps_c = self.eps_core(wavelength)
        else:
            eps_c = eps_l

        # Volume fraction approximation (for spherical core-shell)
        # This is a simplified model; exact solution depends on geometry
        eps_eff = eps_l

        return eps_eff

    def reflection_coefficient(self, wavelength, eps_in, eps_out, angle=0):
        """
        Compute reflection coefficient through layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_in : complex
            Inside dielectric function
        eps_out : complex
            Outside dielectric function
        angle : float
            Incidence angle in radians

        Returns
        -------
        r : complex
            Reflection coefficient
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        n_in = np.sqrt(eps_in)
        n_l = np.sqrt(eps_l)
        n_out = np.sqrt(eps_out)

        # Snell's law
        theta_in = angle
        sin_l = n_in / n_l * np.sin(theta_in)
        if np.abs(sin_l) > 1:
            # Total internal reflection
            cos_l = 1j * np.sqrt(sin_l**2 - 1)
        else:
            cos_l = np.sqrt(1 - sin_l**2)

        sin_out = n_l / n_out * sin_l
        if np.abs(sin_out) > 1:
            cos_out = 1j * np.sqrt(sin_out**2 - 1)
        else:
            cos_out = np.sqrt(1 - sin_out**2)

        # Fresnel coefficients at each interface (s-polarization)
        r_in_l = (n_in * np.cos(theta_in) - n_l * cos_l) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        r_l_out = (n_l * cos_l - n_out * cos_out) / \
                  (n_l * cos_l + n_out * cos_out)

        # Phase factor through layer
        delta = k * n_l * d * cos_l

        # Total reflection (Fabry-Perot)
        r_total = (r_in_l + r_l_out * np.exp(2j * delta)) / \
                  (1 + r_in_l * r_l_out * np.exp(2j * delta))

        return r_total

    def transmission_coefficient(self, wavelength, eps_in, eps_out, angle=0):
        """
        Compute transmission coefficient through layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        eps_in : complex
            Inside dielectric function
        eps_out : complex
            Outside dielectric function
        angle : float
            Incidence angle in radians

        Returns
        -------
        t : complex
            Transmission coefficient
        """
        eps_l = self.eps_layer(wavelength)
        d = self.thickness
        k = 2 * np.pi / wavelength

        n_in = np.sqrt(eps_in)
        n_l = np.sqrt(eps_l)
        n_out = np.sqrt(eps_out)

        theta_in = angle
        sin_l = n_in / n_l * np.sin(theta_in)
        cos_l = np.sqrt(1 - np.clip(sin_l**2, 0, 1) + 0j)

        sin_out = n_l / n_out * sin_l
        cos_out = np.sqrt(1 - np.clip(sin_out**2, 0, 1) + 0j)

        # Fresnel coefficients
        t_in_l = 2 * n_in * np.cos(theta_in) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        t_l_out = 2 * n_l * cos_l / \
                  (n_l * cos_l + n_out * cos_out)

        r_in_l = (n_in * np.cos(theta_in) - n_l * cos_l) / \
                 (n_in * np.cos(theta_in) + n_l * cos_l)
        r_l_out = (n_l * cos_l - n_out * cos_out) / \
                  (n_l * cos_l + n_out * cos_out)

        delta = k * n_l * d * cos_l

        t_total = t_in_l * t_l_out * np.exp(1j * delta) / \
                  (1 + r_in_l * r_l_out * np.exp(2j * delta))

        return t_total


class GreenStatCover:
    """
    Quasistatic Green function with cover layer.

    Modifies the standard Green function to account for
    a thin dielectric layer on the particle surface.

    Parameters
    ----------
    particle : ComParticle
        Composite particle with cover layer
    cover : CoverLayer
        Cover layer object
    options : BEMOptions, optional
        Simulation options
    """

    def __init__(self, particle, cover, options=None):
        """Initialize Green function with cover."""
        self.particle = particle
        self.cover = cover
        self.options = options

        # Get particle geometry
        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

    def G(self, wavelength, inout=None):
        """
        Compute Green function matrix with cover layer correction.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            (inside, outside) medium indices

        Returns
        -------
        G : ndarray
            Green function matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        area = self.pc.area
        n = len(pos)

        # Get dielectric functions
        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_in = self.particle.eps[1](wavelength) if len(self.particle.eps) > 1 else 1.0
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_in = 1.0
            eps_out = 1.0

        # Basic Coulomb Green function
        G = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    G[i, j] = 1.0 / (4 * np.pi * r)

        # Self-term (diagonal)
        for i in range(n):
            # Approximate self-term from area
            a_eff = np.sqrt(area[i] / np.pi)
            G[i, i] = 1.0 / (4 * np.pi * a_eff) * 1.5

        # Apply cover layer correction
        eps_l = self.cover.eps_layer(wavelength)
        d = self.cover.thickness

        # Correction factor for thin layer
        # delta_G ~ d * (eps_l - eps_out) / eps_l
        correction = d * (eps_l - eps_out) / eps_l / wavelength

        # Modify off-diagonal elements
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i, j] *= (1 + correction)

        return G

    def F(self, wavelength, inout=None):
        """
        Compute F matrix (surface derivative of Green function).

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        F : ndarray
            F matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        n = len(pos)

        F = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    r_hat = r_vec / r

                    # F = -dG/dn = (r . n) / (4 * pi * r^3)
                    F[i, j] = np.dot(r_hat, nvec[j]) / (4 * np.pi * r**2)

        return F


class GreenRetCover:
    """
    Retarded Green function with cover layer.

    Parameters
    ----------
    particle : ComParticle
        Composite particle
    cover : CoverLayer
        Cover layer object
    options : BEMOptions, optional
        Simulation options
    """

    def __init__(self, particle, cover, options=None):
        """Initialize retarded Green function with cover."""
        self.particle = particle
        self.cover = cover
        self.options = options

        if hasattr(particle, 'pc'):
            self.pc = particle.pc
        else:
            self.pc = particle

    def G(self, wavelength, inout=None):
        """
        Compute retarded Green function with cover layer.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        G : ndarray
            Retarded Green function matrix
        """
        pos = self.pc.pos
        n = len(pos)
        k = 2 * np.pi / wavelength

        # Get dielectric functions
        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_out = 1.0

        n_med = np.sqrt(eps_out)
        k_med = k * n_med

        G = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    G[i, j] = np.exp(1j * k_med * r) / (4 * np.pi * r)

        # Self-term
        area = self.pc.area
        for i in range(n):
            a_eff = np.sqrt(area[i] / np.pi)
            G[i, i] = np.exp(1j * k_med * a_eff) / (4 * np.pi * a_eff) * 1.5

        # Cover layer correction
        eps_l = self.cover.eps_layer(wavelength)
        d = self.cover.thickness
        n_l = np.sqrt(eps_l)

        # Phase correction through layer
        phase_corr = np.exp(1j * k * n_l * d)

        # Apply correction
        G *= phase_corr

        return G

    def F(self, wavelength, inout=None):
        """
        Compute F matrix for retarded case with cover.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm
        inout : tuple, optional
            Medium indices

        Returns
        -------
        F : ndarray
            F matrix
        """
        pos = self.pc.pos
        nvec = self.pc.nvec
        n = len(pos)
        k = 2 * np.pi / wavelength

        if hasattr(self.particle, 'eps') and len(self.particle.eps) > 0:
            eps_out = self.particle.eps[0](wavelength)
        else:
            eps_out = 1.0

        n_med = np.sqrt(eps_out)
        k_med = k * n_med

        F = np.zeros((n, n), dtype=complex)

        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[i] - pos[j]
                    r = np.linalg.norm(r_vec)
                    r_hat = r_vec / r

                    # dG/dr for retarded Green function
                    G_val = np.exp(1j * k_med * r) / (4 * np.pi * r)
                    dG_dr = G_val * (1j * k_med - 1.0 / r)

                    F[i, j] = np.dot(r_hat, nvec[j]) * dG_dr

        return F


def coverlayer(eps_layer, thickness, eps_core=None):
    """
    Factory function for cover layer.

    Parameters
    ----------
    eps_layer : complex or callable
        Layer dielectric function
    thickness : float
        Layer thickness in nm
    eps_core : complex or callable, optional
        Core dielectric function

    Returns
    -------
    CoverLayer
        Cover layer object
    """
    return CoverLayer(eps_layer, thickness, eps_core)


def refine(p, ind):
    """
    Create refinement function for Green function initialization.

    Green function elements for neighbor cover layer elements are refined
    through polar integration.

    Parameters
    ----------
    p : ComParticle
        Composite particle object.
    ind : ndarray
        Particle indices for refinement, shape (n, 2).

    Returns
    -------
    callable
        Refinement function for Green function initialization.
    """
    # Symmetrize indices
    ind = np.unique(np.vstack([ind, ind[:, ::-1]]), axis=0)

    def refine_func(obj, g, f):
        """Refinement function that dispatches to static or retarded."""
        from .green_stat import GreenStat
        from .green_ret import GreenRet

        if isinstance(obj, GreenStat):
            return refinestat(obj, g, f, p, ind)
        elif isinstance(obj, GreenRet):
            return refineret(obj, g, f, p, ind)
        else:
            raise ValueError(f"Unknown Green function class: {type(obj)}")

    return refine_func


def refinestat(obj, g, f, p, ind):
    """
    Refine quasistatic Green function for cover layer elements.

    Refines Green function elements for neighbor cover layer elements
    through polar integration.

    Parameters
    ----------
    obj : GreenStat
        Green function object.
    g : ndarray
        Green function elements for refinement.
    f : ndarray
        Surface derivative elements for refinement.
    p : ComParticle
        Composite particle object.
    ind : ndarray
        Particle indices for refinement.

    Returns
    -------
    g : ndarray
        Refined Green function.
    f : ndarray
        Refined surface derivative.
    """
    n = p.n_faces if hasattr(p, 'n_faces') else len(p.pos)

    # Get indices
    i1 = ind[:, 0]
    i2 = ind[:, 1]

    # Compute linear indices
    lin_ind = i1 * n + i2

    # Quadrature for polar integration
    n_phi = 12
    n_r = 6
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    w_r = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

    deriv_mode = getattr(obj, 'deriv', 'norm')

    for k, (i, j) in enumerate(zip(i1, i2)):
        pos_i = p.pos[i]
        pos_j = p.pos[j]
        nvec_i = p.nvec[i] if hasattr(p, 'nvec') else np.array([0, 0, 1])
        area_j = p.area[j] if hasattr(p, 'area') else 1.0
        face_radius = np.sqrt(area_j / np.pi)

        g_sum = 0.0
        fx_sum = 0.0
        fy_sum = 0.0
        fz_sum = 0.0

        for ri, rj in enumerate(r_vals):
            r_actual = rj * face_radius
            for phi_k in phi_vals:
                # Integration point on face j
                dx = r_actual * np.cos(phi_k)
                dy = r_actual * np.sin(phi_k)
                pos_q = pos_j + np.array([dx, dy, 0])

                # Distance vector
                x = pos_i[0] - pos_q[0]
                y = pos_i[1] - pos_q[1]
                z = pos_i[2] - pos_q[2]
                r = np.sqrt(x**2 + y**2 + z**2)

                if r < 1e-10:
                    continue

                weight = w_r[ri] * r_actual * (2 * np.pi / n_phi)

                # Green function
                g_sum += weight / r

                # Derivatives
                fx_sum -= weight * x / r**3
                fy_sum -= weight * y / r**3
                fz_sum -= weight * z / r**3

        g.flat[lin_ind[k]] = g_sum

        if deriv_mode == 'cart':
            f[lin_ind[k], 0] = fx_sum
            f[lin_ind[k], 1] = fy_sum
            f[lin_ind[k], 2] = fz_sum
        else:
            f.flat[lin_ind[k]] = (fx_sum * nvec_i[0] +
                                  fy_sum * nvec_i[1] +
                                  fz_sum * nvec_i[2])

    return g, f


def refineret(obj, g, f, p, ind):
    """
    Refine retarded Green function for cover layer elements.

    Refines Green function elements for neighbor cover layer elements
    through polar integration.

    Parameters
    ----------
    obj : GreenRet
        Green function object.
    g : ndarray
        Green function elements for refinement.
    f : ndarray
        Surface derivative elements for refinement.
    p : ComParticle
        Composite particle object.
    ind : ndarray
        Particle indices for refinement.

    Returns
    -------
    g : ndarray
        Refined Green function.
    f : ndarray
        Refined surface derivative.
    """
    n = p.n_faces if hasattr(p, 'n_faces') else len(p.pos)
    order = getattr(obj, 'order', 0)

    i1 = ind[:, 0]
    i2 = ind[:, 1]
    lin_ind = i1 * n + i2

    n_phi = 12
    n_r = 6
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    w_r = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

    deriv_mode = getattr(obj, 'deriv', 'norm')

    for k, (i, j) in enumerate(zip(i1, i2)):
        pos_i = p.pos[i]
        pos_j = p.pos[j]
        nvec_i = p.nvec[i] if hasattr(p, 'nvec') else np.array([0, 0, 1])
        area_j = p.area[j] if hasattr(p, 'area') else 1.0
        face_radius = np.sqrt(area_j / np.pi)

        # Distance between face centroids
        vec0 = pos_j - pos_i
        r0 = np.linalg.norm(vec0)

        # Arrays for different orders
        g_orders = np.zeros(order + 1, dtype=complex)
        f_orders = np.zeros(order + 1, dtype=complex) if deriv_mode == 'norm' else np.zeros((order + 1, 3), dtype=complex)

        for ri, rj in enumerate(r_vals):
            r_actual = rj * face_radius
            for phi_k in phi_vals:
                dx = r_actual * np.cos(phi_k)
                dy = r_actual * np.sin(phi_k)
                pos_q = pos_j + np.array([dx, dy, 0])

                x = pos_i[0] - pos_q[0]
                y = pos_i[1] - pos_q[1]
                z = pos_i[2] - pos_q[2]
                r = np.sqrt(x**2 + y**2 + z**2)

                if r < 1e-10:
                    continue

                weight = w_r[ri] * r_actual * (2 * np.pi / n_phi)

                # Taylor expansion terms
                for ord_i in range(order + 1):
                    factorial_ord = np.math.factorial(ord_i)
                    g_orders[ord_i] += weight * (r - r0)**ord_i / (r * factorial_ord)

                # Surface derivative
                in_prod = x * nvec_i[0] + y * nvec_i[1] + z * nvec_i[2]

                if deriv_mode == 'norm':
                    f_orders[0] -= weight * in_prod / r**3
                    for ord_i in range(1, order + 1):
                        fact_ord = np.math.factorial(ord_i)
                        fact_ord_m1 = np.math.factorial(ord_i - 1)
                        f_orders[ord_i] += weight * in_prod * (
                            (r - r0)**ord_i / (r**3 * fact_ord) +
                            (r - r0)**(ord_i - 1) / (r**2 * fact_ord_m1)
                        )
                else:
                    f_orders[0, :] -= weight * np.array([x, y, z]) / r**3
                    for ord_i in range(1, order + 1):
                        fact_ord = np.math.factorial(ord_i)
                        fact_ord_m1 = np.math.factorial(ord_i - 1)
                        factor = (-(r - r0)**ord_i / (r**3 * fact_ord) +
                                  (r - r0)**(ord_i - 1) / (r**2 * fact_ord_m1))
                        f_orders[ord_i, :] += weight * np.array([x, y, z]) * factor

        # Store results
        for ord_i in range(order + 1):
            if g.ndim == 2:
                g[lin_ind[k], ord_i] = g_orders[ord_i]
            else:
                g.flat[lin_ind[k]] = g_orders[0]

            if deriv_mode == 'norm':
                if f.ndim == 2:
                    f[lin_ind[k], ord_i] = f_orders[ord_i]
                else:
                    f.flat[lin_ind[k]] = f_orders[0]
            else:
                if f.ndim == 3:
                    f[lin_ind[k], :, ord_i] = f_orders[ord_i, :]
                else:
                    f[lin_ind[k], :] = f_orders[0, :]

    return g, f


def shift(p1, d, nvec=None):
    """
    Shift boundary for creation of cover layer structure.

    Parameters
    ----------
    p1 : Particle
        Original particle.
    d : float or ndarray
        Shift distance. If scalar, uniform shift; if array, per-vertex shift.
    nvec : ndarray, optional
        Direction vectors for shifting. If None, uses interpolated normals.

    Returns
    -------
    Particle
        Shifted particle boundary.
    """
    from ..particles import Particle

    verts = p1.verts.copy()
    faces = p1.faces.copy()

    n_verts = len(verts)

    # Handle scalar vs array distance
    if np.isscalar(d):
        d = np.full(n_verts, d)
    else:
        d = np.asarray(d)
        if len(d) != n_verts:
            raise ValueError(f"d must have length {n_verts}, got {len(d)}")

    # Get normal vectors
    if nvec is None:
        # Interpolate face normals to vertices
        nvec = _interp_normals_to_verts(p1)
    else:
        nvec = np.asarray(nvec)
        if nvec.shape != (n_verts, 3):
            raise ValueError(f"nvec must have shape ({n_verts}, 3)")

    # Normalize
    nvec_norm = nvec / (np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-10)

    # Handle duplicate vertices
    _, i1, i2 = np.unique(np.round(verts, 4), axis=0, return_index=True, return_inverse=True)

    # Shift vertices
    verts_shifted = verts + d[:, np.newaxis] * nvec_norm[i1[i2]]

    return Particle(verts_shifted, faces)


def _interp_normals_to_verts(p):
    """
    Interpolate face normals to vertices.

    Parameters
    ----------
    p : Particle
        Particle with face normals.

    Returns
    -------
    ndarray
        Vertex normals, shape (n_verts, 3).
    """
    nvec_faces = p.nvec
    verts = p.verts
    faces = p.faces

    n_verts = len(verts)
    nvec_verts = np.zeros((n_verts, 3))
    counts = np.zeros(n_verts)

    # Average face normals at each vertex
    for i, face in enumerate(faces):
        for j in range(len(face)):
            if face[j] < 0 or np.isnan(face[j]):
                continue
            v_idx = int(face[j])
            if v_idx < n_verts:
                nvec_verts[v_idx] += nvec_faces[i]
                counts[v_idx] += 1

    # Normalize
    counts[counts == 0] = 1
    nvec_verts /= counts[:, np.newaxis]

    # Renormalize to unit vectors
    norms = np.linalg.norm(nvec_verts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    nvec_verts /= norms

    return nvec_verts
