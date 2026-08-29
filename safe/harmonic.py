"""Phase 4b: circumferential-harmonic (Fourier) SAFE for axisymmetric pipes.

For an axisymmetric pipe the displacement field decomposes exactly as
  u(r, theta, z, t) = U(r) * exp(i(n*theta + k*z - omega*t)),
so the 2D cross-section problem block-diagonalizes by circumferential order n:
only the radial line through the wall is meshed (quadratic 3-node elements),
giving ~3*(2*n_elems+1) DOFs per harmonic instead of ~10^3 for the 2D mesh.
This is the standard axisymmetric-SAFE reduction; here it exists to move the
batched eigensolve into the small-matrix regime that GPU batched solvers
handle natively (Phase 4 showed n~720 dense eigh is latency-bound).

Cylindrical strains with d/dtheta -> i n and d/dz -> i k
(Voigt order [e_rr, e_tt, e_zz, g_rt, g_tz, g_rz] — the isotropic C from
materials.py applies unchanged):
  e_rr = U_r'
  e_tt = (U_r + i n U_t) / r
  e_zz = i k U_z
  g_rt = i n U_r / r + U_t' - U_t / r
  g_tz = i n U_z / r + i k U_t
  g_rz = i k U_r + U_z'

Splitting B = Bn + i k Bk (Bn complex, Bk real) yields the familiar pencil
  A(n, k) = K0(n) + i k K1(n) + k^2 K2,   A Hermitian, omega^2 real,
with K0 Hermitian, K1 anti-Hermitian, K2 real symmetric — the same structure
as the 2D solver, so group velocity and parameter sensitivities carry over.

Initial-stress (Lame pressure) term via the cylindrical displacement-gradient
physical components (diagonal prestress: sigma_rr, sigma_tt, sigma_zz):
  (grad u)_r. = (U_r', U_t', U_z')
  (grad u)_t. = ((i n U_r - U_t)/r, (U_r + i n U_t)/r, i n U_z / r)
  (grad u)_z. = i k (U_r, U_t, U_z)
so Kg = Kg0(n) [sigma_rr, sigma_tt rows] + k^2 * sigma_zz * (mass structure);
Kg1 = 0 for the Lame field (no sigma_rz / sigma_tz), matching the 2D module.
"""

import numpy as np
import scipy.linalg as sla

from .materials import Material, isotropic_C
from .prestress import LamePressure

# 3-point Gauss on [-1, 1]
_GP = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def _quad_shape(xi):
    """Quadratic 1D shape functions at nodes xi = -1, 0, +1."""
    N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
    dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN


class HarmonicSAFE:
    """Axisymmetric-pipe SAFE, one circumferential harmonic at a time.

    follower=True adds the pressure load-stiffness (follower) boundary term of
    Mazzotti et al. (2012), Eqs. (21)-(24): the linearized change of surface
    normal Delta_n = (du/deta x n3)/|dx/deta| gives, per boundary circle of
    radius s under pressure p0, the k-independent nodal 3x3 Hermitian block
        Kp = -p0 * orient * [ [1, i n, 0], [-i n, 1, 0], [0, 0, 0] ]
    acting on (Ur, Utheta, Uz) at the surface node (arc-length and 1/r factors
    cancel exactly as the Jacobians cancel in their Eq. (24)). `orient` encodes
    the boundary traversal direction and `follower_sign` the Hamilton insertion
    sign; both conventions are exposed because the paper's counterclockwise-
    numeration rule is stated for a planar mesh, and are pinned numerically by
    reproducing the published pressurized-pipe case behaviors (see
    validate_follower.py)."""

    def __init__(self, a: float, b: float, n_elems: int, mat: Material,
                 pressure: float = 0.0, ends: str = "capped", Pe: float = 0.0,
                 follower: bool = False, follower_sign: float = -1.0,
                 inner_orient: float = -1.0):
        # Default (follower_sign=-1, inner_orient=-1) equals the conservative
        # pressure potential Pi_2 = -p_i*A2(inner) + p_e*A2(outer), where A2 is
        # the second-order enclosed-area variation (= Mazzotti Eq. 24 with
        # solid-outward orientation). Verified consequences (validate_follower):
        # the follower block vanishes on rigid translation, and for T(0,1) it
        # cancels the initial-stress (geometric) term EXACTLY at leading order
        # -- pressure moves the fundamental torsional branch only at O((p/mu)^2).
        # The opposite sign doubles the geometric shift and destabilizes T(0,1)
        # under external pressure (omega^2 < 0), which is unphysical at MPa
        # levels. NOTE: neither convention reproduces the low-frequency T(0,1)
        # behaviors published in Mazzotti et al. (2012), Figs. 9-10 -- an open
        # discrepancy documented in PLAN.md and the paper draft.
        self.a, self.b = a, b
        self.n_elems = n_elems
        self.mat = mat
        self.C = isotropic_C(mat)
        self.r_nodes = np.linspace(a, b, 2 * n_elems + 1)
        self.n_dof = 3 * self.r_nodes.size
        self.field = (LamePressure(pressure, a, b, ends=ends, Pe=Pe)
                      if (pressure or Pe) else None)
        self.follower = follower
        self.follower_sign = follower_sign
        self.inner_orient = inner_orient

    def matrices(self, n: int):
        """Assembled (K0, K1, K2, M) for harmonic n, prestress folded in.

        K0 complex Hermitian, K1 complex anti-Hermitian, K2 & M real symmetric
        (returned as complex/float arrays ready for A = K0 + ik K1 + k^2 K2)."""
        d = self.n_dof
        K0 = np.zeros((d, d), dtype=complex)
        K1 = np.zeros((d, d), dtype=complex)
        K2 = np.zeros((d, d))
        M = np.zeros((d, d))
        C = self.C
        rho = self.mat.rho

        for e in range(self.n_elems):
            nodes = np.array([2 * e, 2 * e + 1, 2 * e + 2])
            re = self.r_nodes[nodes]
            h = re[2] - re[0]
            J = h / 2.0
            dofs = (3 * nodes[:, None] + np.arange(3)).ravel()

            k0 = np.zeros((9, 9), dtype=complex)
            k1 = np.zeros((9, 9), dtype=complex)
            k2 = np.zeros((9, 9))
            me = np.zeros((9, 9))

            for xi, w in zip(_GP, _GW):
                N, dNxi = _quad_shape(xi)
                dN = dNxi / J
                r = N @ re
                dV = w * J * r

                Bn = np.zeros((6, 9), dtype=complex)
                Bk = np.zeros((6, 9))
                Nm = np.zeros((3, 9))
                for a_ in range(3):
                    c = 3 * a_
                    Na, dNa = N[a_], dN[a_]
                    Bn[0, c] = dNa
                    Bn[1, c] = Na / r
                    Bn[1, c + 1] = 1j * n * Na / r
                    Bn[3, c] = 1j * n * Na / r
                    Bn[3, c + 1] = dNa - Na / r
                    Bn[4, c + 2] = 1j * n * Na / r
                    Bn[5, c + 2] = dNa
                    Bk[2, c + 2] = Na
                    Bk[4, c + 1] = Na
                    Bk[5, c] = Na
                    Nm[0, c] = Nm[1, c + 1] = Nm[2, c + 2] = Na

                CBn = C @ Bn
                CBk = C @ Bk
                k0 += dV * (Bn.conj().T @ CBn)
                k1 += dV * (Bn.conj().T @ CBk - Bk.T @ CBn)
                k2 += dV * (Bk.T @ CBk)
                me += (dV * rho) * (Nm.T @ Nm)

                if self.field is not None:
                    s_rr = self.field.A_L - self.field.B_L / r**2
                    s_tt = self.field.A_L + self.field.B_L / r**2
                    Gr = np.zeros((3, 9))
                    Gt = np.zeros((3, 9), dtype=complex)
                    for a_ in range(3):
                        c = 3 * a_
                        Na, dNa = N[a_], dN[a_]
                        Gr[0, c] = Gr[1, c + 1] = Gr[2, c + 2] = dNa
                        Gt[0, c] = 1j * n * Na / r
                        Gt[0, c + 1] = -Na / r
                        Gt[1, c] = Na / r
                        Gt[1, c + 1] = 1j * n * Na / r
                        Gt[2, c + 2] = 1j * n * Na / r
                    k0 += dV * (s_rr * (Gr.T @ Gr) + s_tt * (Gt.conj().T @ Gt))
                    k2 += dV * (self.field.s_zz / rho) * (rho * (Nm.T @ Nm))

            ix = np.ix_(dofs, dofs)
            K0[ix] += k0
            K1[ix] += k1
            K2[ix] += k2
            M[ix] += me

        if self.field is not None and self.follower:
            # nodal follower blocks: inner surface (node 0, pressure P) and
            # outer surface (last node, pressure Pe), outer traversed ccw
            for node, p0, orient in ((0, self.field.P, self.inner_orient),
                                     (self.r_nodes.size - 1, self.field.Pe, 1.0)):
                if p0 == 0.0:
                    continue
                c = self.follower_sign * (-p0) * orient
                i0 = 3 * node
                K0[i0, i0] += c
                K0[i0 + 1, i0 + 1] += c
                K0[i0, i0 + 1] += c * 1j * n
                K0[i0 + 1, i0] += -c * 1j * n

        return K0, K1, K2, M

    def solve(self, n: int, k: float, mats=None):
        """Eigensolve at (harmonic n, wavenumber k): omega ascending, phi
        M-orthonormal. Pass mats=self.matrices(n) to reuse assemblies."""
        K0, K1, K2, M = mats if mats is not None else self.matrices(n)
        A = K0 + (1j * k) * K1 + (k * k) * K2
        w, V = sla.eigh(A, M)
        return np.sqrt(np.clip(w, 0.0, None)), V

    def group_velocities(self, k, omega, phi, mats):
        _, K1, K2, _ = mats
        dA = 1j * K1 + (2.0 * k) * K2
        quad = np.einsum("im,ij,jm->m", phi.conj(), dA, phi).real
        cg = np.full_like(omega, np.nan)
        ok = omega > 0
        cg[ok] = quad[ok] / (2.0 * omega[ok])
        return cg

    def classify(self, phi):
        """('T'|'L') for n = 0 eigenvectors by component energy."""
        U = phi.reshape(-1, 3)
        e_t = np.sum(np.abs(U[:, 1]) ** 2)
        e_rz = np.sum(np.abs(U[:, 0]) ** 2) + np.sum(np.abs(U[:, 2]) ** 2)
        return "T" if e_t > e_rz else "L"

    def find_mode(self, omega, phi, family: str, index: int = 1):
        """index-th (1-based) n=0 mode of a family, by ascending omega."""
        hits = 0
        for m in range(omega.size):
            if omega[m] <= 0:
                continue
            if self.classify(phi[:, m]) == family:
                hits += 1
                if hits == index:
                    return omega[m], m
        return None, None
