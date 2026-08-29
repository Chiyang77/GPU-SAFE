"""Internal-pressure prestress for the SAFE eigenproblem (Phase 2).

Initial-stress (geometric) stiffness from the linearized incremental weak form
(updated Lagrangian, small prestress, geometry update neglected):

  G(u, du) = int_A sum_m (grad u_m)^H . sigma0 . (grad du_m) dA,  d/dz -> ik

With the scalar gradient split grad = G0 + ik*G1 (G0 = in-plane derivatives,
G1 = axial), this yields

  Kg(k) = Kg0 + ik*Kg1 + k^2*Kg2
  Kg0 = int G0^T s G0,  Kg1 = int (G0^T s G1 - G1^T s G0),  Kg2 = int G1^T s G1

per displacement component — the same symmetry structure as the material
matrices (Kg0, Kg2 symmetric; Kg1 skew-symmetric), so the total pencil stays
Hermitian and omega^2 stays real. sigma0 is linear in the applied pressure,
so d(Kg)/dP = Kg(P=1) — used later for analytical pressure sensitivities.

Scope notes (Phase 2): geometric/initial-stress effect only. Not included:
third-order elastic constants (material acoustoelasticity), follower-pressure
boundary stiffness, and the O(P/E) update of the deformed geometry.

The prestress field is the exact thick-wall Lame solution (internal pressure
P, traction-free outer surface):
  s_rr(r) = A_L - B_L/r^2,  s_tt(r) = A_L + B_L/r^2
  A_L = P a^2/(b^2 - a^2),  B_L = P a^2 b^2/(b^2 - a^2)
  s_zz = A_L (capped ends) or 0 (open ends), uniform.
"""

from dataclasses import dataclass

import numpy as np

from .element import GAUSS_PTS, GAUSS_WTS, shape_functions
from .mesh import AnnularMesh


@dataclass(frozen=True)
class LamePressure:
    """Exact Lame prestress in a thick-walled cylinder under internal pressure
    P and (optionally) external pressure Pe. sigma_rr(a) = -P, sigma_rr(b) = -Pe."""
    P: float                 # internal gauge pressure [Pa]
    a: float                 # inner radius
    b: float                 # outer radius
    ends: str = "capped"     # 'capped' -> s_zz = A_L, 'open'/'plane' -> s_zz = 0
    Pe: float = 0.0          # external gauge pressure [Pa]

    @property
    def A_L(self) -> float:
        return (self.P * self.a**2 - self.Pe * self.b**2) / (self.b**2 - self.a**2)

    @property
    def B_L(self) -> float:
        return (self.P - self.Pe) * self.a**2 * self.b**2 / (self.b**2 - self.a**2)

    @property
    def s_zz(self) -> float:
        return self.A_L if self.ends == "capped" else 0.0

    def stress(self, x: float, y: float) -> np.ndarray:
        """Cartesian 3x3 Cauchy prestress tensor at cross-section point (x, y)."""
        r2 = x * x + y * y
        s_rr = self.A_L - self.B_L / r2
        s_tt = self.A_L + self.B_L / r2
        c2, s2 = x * x / r2, y * y / r2
        cs = x * y / r2
        s = np.zeros((3, 3))
        s[0, 0] = s_rr * c2 + s_tt * s2
        s[1, 1] = s_rr * s2 + s_tt * c2
        s[0, 1] = s[1, 0] = (s_rr - s_tt) * cs
        s[2, 2] = self.s_zz
        return s


def assemble_geometric(mesh: AnnularMesh, stress_fn):
    """Geometric stiffness matrices (Kg0, Kg1, Kg2), dense (n_dof, n_dof).

    stress_fn(x, y) must return the 3x3 Cartesian prestress tensor.
    """
    n = mesh.n_dof
    Kg0 = np.zeros((n, n))
    Kg1 = np.zeros((n, n))
    Kg2 = np.zeros((n, n))
    I3 = np.eye(3)

    for conn in mesh.elems:
        coords = mesh.nodes[conn]
        e0 = np.zeros((12, 12))
        e1 = np.zeros((12, 12))
        e2 = np.zeros((12, 12))
        for (xi, eta), w in zip(GAUSS_PTS, GAUSS_WTS):
            N, dN = shape_functions(xi, eta)
            J = dN.T @ coords
            detJ = np.linalg.det(J)
            dNdx = dN @ np.linalg.inv(J).T          # (4,2)
            xg, yg = N @ coords                     # Gauss-point position
            s = stress_fn(xg, yg)

            D0 = np.zeros((3, 4))                   # scalar gradient, in-plane part
            D0[:2, :] = dNdx.T
            D1 = np.zeros((3, 4))                   # scalar gradient, axial part
            D1[2, :] = N

            s00 = D0.T @ s @ D0                     # (4,4) nodal blocks
            s01 = D0.T @ s @ D1
            s11 = D1.T @ s @ D1

            dA = w * detJ
            e0 += dA * np.kron(s00, I3)             # dof = 3*node + component
            e1 += dA * np.kron(s01 - s01.T, I3)
            e2 += dA * np.kron(s11, I3)

        dofs = (3 * conn[:, None] + np.arange(3)).ravel()
        ix = np.ix_(dofs, dofs)
        Kg0[ix] += e0
        Kg1[ix] += e1
        Kg2[ix] += e2

    return Kg0, Kg1, Kg2


def torsional_shift_analytic(field: LamePressure, mu: float, rho: float,
                             k: np.ndarray):
    """Exact first-order Delta(omega^2) of T(0,1) under Lame prestress.

    The unpressurized T(0,1) eigenvector is the rigid in-plane rotation
    u = (-y, x, 0) exp(ikz). Its geometric Rayleigh quotient evaluates in
    closed form:
      Delta(omega^2) = [ (s_rr + s_tt) * Area + k^2 * s_zz * Ip ] / (rho * Ip)
    with s_rr + s_tt = 2 A_L (constant), Area = pi (b^2 - a^2),
    Ip = pi/2 (b^4 - a^4). The 1/k^2 term in cp makes T(0,1) dispersive under
    pressure — the effect reported by Mazzotti et al. (2012).
    """
    a, b = field.a, field.b
    area = np.pi * (b**2 - a**2)
    Ip = np.pi / 2 * (b**4 - a**4)
    k = np.asarray(k, dtype=float)
    return (2 * field.A_L * area + k**2 * field.s_zz * Ip) / (rho * Ip)
