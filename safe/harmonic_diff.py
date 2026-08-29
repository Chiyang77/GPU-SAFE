"""Phase 5: differentiable harmonic SAFE (PyTorch autograd through assembly).

Gradient strategy (robust at eigensolvers' weak spots):
  1. Solve the eigenproblem OUTSIDE the autograd graph (detached parameters,
     fp64 scipy) -> M-orthonormal eigenvector phi of the mode of interest.
  2. Re-assemble A(k; p) and M(p) with torch ops on leaf parameters
     p = (t, P, E, nu, rho) and form the Rayleigh quotient
        omega^2(p) = Re(phi^H A phi) / Re(phi^H M phi),   phi detached.
  3. Reverse-mode backward gives EXACT d omega/dp for simple eigenvalues
     (generalized Hellmann-Feynman: first-order eigenvalue sensitivities do
     not require eigenvector derivatives). For (near-)degenerate clusters,
     the sum of quotients over an M-orthonormal basis of the cluster is
     basis-invariant, so cluster-trace sensitivities remain well-defined.

This avoids differentiating through `eigh` itself, whose autograd backward
contains 1/(lambda_i - lambda_j) factors that blow up at mode crossings —
the failure mode flagged in the literature review (adjoint eigen-derivative
methods excluding repeated eigenvalues).

Exact scaling identities used for validation: at fixed geometry and nu,
all stiffness entries scale with E and mass with rho, so
  d omega/dE = omega/(2E),   d omega/drho = -omega/(2 rho)   (unpressurized).

Not yet exact here: d(cg)/dp (needs eigenvector sensitivities, e.g. Nelson's
method). At ~10 us/eigensolve, finite differences on the batched solver are
the pragmatic interim for cg sensitivities.
"""

import numpy as np
import torch

from .harmonic import HarmonicSAFE
from .materials import Material

_GP = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
_GW = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]


def _quad_shape(xi):
    N = [0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)]
    dN = [xi - 0.5, -2.0 * xi, xi + 0.5]
    return N, dN


class DifferentiableHarmonicSAFE:
    """Torch-native harmonic-SAFE assembly over leaf parameters.

    Parameters enter as 0-dim float64 tensors (requires_grad=True):
      t (wall thickness), P (internal pressure), E, nu, rho.
    Geometry: outer radius r_out fixed, inner radius a = r_out - t.
    """

    def __init__(self, r_out: float, n_elems: int = 4, ends: str = "capped"):
        self.r_out = r_out
        self.n_elems = n_elems
        self.ends = ends
        self.n_dof = 3 * (2 * n_elems + 1)

    def matrices(self, n: int, t, P, E, nu, rho):
        """Differentiable (K0, K1, K2, M), prestress folded, complex128/float64."""
        d = self.n_dof
        cplx = torch.complex128
        K0 = torch.zeros((d, d), dtype=cplx)
        K1 = torch.zeros((d, d), dtype=cplx)
        K2 = torch.zeros((d, d), dtype=torch.float64)
        M = torch.zeros((d, d), dtype=torch.float64)

        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        # C = lam * (normal-block ones) + mu * diag(2,2,2,1,1,1), built from
        # constant masks so autograd flows through lam and mu only
        ones33 = torch.zeros((6, 6), dtype=torch.float64)
        ones33[:3, :3] = 1.0
        diag_mu = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
                                          dtype=torch.float64))
        C = lam * ones33 + mu * diag_mu

        a = self.r_out - t
        b = torch.tensor(self.r_out, dtype=torch.float64)
        A_L = P * a**2 / (b**2 - a**2)
        B_L = P * a**2 * b**2 / (b**2 - a**2)
        s_zz = A_L if self.ends == "capped" else torch.zeros((), dtype=torch.float64)

        n_nodes = 2 * self.n_elems + 1
        s = torch.linspace(0.0, 1.0, n_nodes, dtype=torch.float64)
        r_nodes = a + t * s

        for e in range(self.n_elems):
            nodes = [2 * e, 2 * e + 1, 2 * e + 2]
            re = r_nodes[nodes]
            J = (re[2] - re[0]) / 2.0
            dofs = torch.tensor([3 * q + m for q in nodes for m in range(3)])
            ix = (dofs[:, None], dofs[None, :])

            for xi, w in zip(_GP, _GW):
                N, dNxi = _quad_shape(xi)
                r = N[0] * re[0] + N[1] * re[1] + N[2] * re[2]
                dV = w * J * r

                Bn = torch.zeros((6, 9), dtype=cplx)
                Bk = torch.zeros((6, 9), dtype=torch.float64)
                Nm = torch.zeros((3, 9), dtype=torch.float64)
                Gr = torch.zeros((3, 9), dtype=torch.float64)
                Gt = torch.zeros((3, 9), dtype=cplx)
                for q in range(3):
                    c = 3 * q
                    Na = N[q]
                    dNa = dNxi[q] / J
                    Nr = Na / r
                    Bn[0, c] = dNa + 0j
                    Bn[1, c] = Nr + 0j
                    Bn[1, c + 1] = 1j * n * Nr
                    Bn[3, c] = 1j * n * Nr
                    Bn[3, c + 1] = dNa - Nr + 0j
                    Bn[4, c + 2] = 1j * n * Nr
                    Bn[5, c + 2] = dNa + 0j
                    Bk[2, c + 2] = Na
                    Bk[4, c + 1] = Na
                    Bk[5, c] = Na
                    Nm[0, c] = Nm[1, c + 1] = Nm[2, c + 2] = Na
                    Gr[0, c] = Gr[1, c + 1] = Gr[2, c + 2] = dNa
                    Gt[0, c] = 1j * n * Nr
                    Gt[0, c + 1] = -Nr + 0j
                    Gt[1, c] = Nr + 0j
                    Gt[1, c + 1] = 1j * n * Nr
                    Gt[2, c + 2] = 1j * n * Nr

                Cc = C.to(cplx)
                CBn = Cc @ Bn
                CBk = C @ Bk
                s_rr = A_L - B_L / r**2
                s_tt = A_L + B_L / r**2

                k0 = dV * (Bn.mH @ CBn
                           + (s_rr * (Gr.T @ Gr)).to(cplx)
                           + s_tt * (Gt.mH @ Gt))
                k1 = dV * (Bn.mH @ CBk.to(cplx) - Bk.T.to(cplx) @ CBn)
                k2 = dV * (Bk.T @ CBk + s_zz * (Nm.T @ Nm))
                me = dV * rho * (Nm.T @ Nm)

                K0 = K0.index_put(ix, K0[ix] + k0)
                K1 = K1.index_put(ix, K1[ix] + k1)
                K2 = K2.index_put(ix, K2[ix] + k2)
                M = M.index_put(ix, M[ix] + me)

        return K0, K1, K2, M

    def omega(self, n: int, k: float, t, P, E, nu, rho, mode_select):
        """Differentiable omega of one mode (or a cluster trace).

        mode_select: ('family', index) for n=0 e.g. ('L', 2); an int mode
        index; or a list of ints -> returns sqrt of the cluster-mean of
        omega^2 (basis-invariant for degenerate clusters).
        """
        fv = lambda x: float(x.detach()) if hasattr(x, "detach") else float(x)
        ref = HarmonicSAFE(self.r_out - fv(t), self.r_out, self.n_elems,
                           Material(fv(E), fv(nu), fv(rho)),
                           pressure=fv(P), ends=self.ends)
        om, phi = ref.solve(n, k, ref.matrices(n))
        if isinstance(mode_select, tuple):
            fam, idx = mode_select
            _, m = ref.find_mode(om, phi, fam, idx)
            modes = [m]
        elif isinstance(mode_select, int):
            modes = [mode_select]
        else:
            modes = list(mode_select)
        Phi = torch.tensor(phi[:, modes])                     # detached

        K0, K1, K2, M = self.matrices(n, t, P, E, nu, rho)
        A = K0 + (1j * k) * K1 + (k * k) * K2.to(torch.complex128)
        MPhi = torch.complex(M @ Phi.real, M @ Phi.imag)
        num = torch.einsum("im,im->m", Phi.conj(), A @ Phi).real
        den = torch.einsum("im,im->m", Phi.conj(), MPhi).real
        return torch.sqrt((num / den).mean())

    def gradients(self, n, k, params, mode_select):
        """omega and d omega/d{t,P,E,nu,rho} via one reverse-mode backward."""
        leafs = {name: torch.tensor(v, dtype=torch.float64, requires_grad=True)
                 for name, v in params.items()}
        w = self.omega(n, k, leafs["t"], leafs["P"], leafs["E"],
                       leafs["nu"], leafs["rho"], mode_select)
        grads = torch.autograd.grad(w, list(leafs.values()), allow_unused=True)
        out = {name: (0.0 if g is None else float(g))
               for name, g in zip(leafs, grads)}
        return float(w), out
