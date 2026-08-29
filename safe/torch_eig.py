"""Phase 4: batched generalized Hermitian eigensolver on the GPU (PyTorch).

Solves  [K0 + ik K1 + k^2 K2] phi = omega^2 M phi  for whole batches of cases
(one per parameter set) and wavenumbers at once.

cuSOLVER exposes batched *standard* Hermitian eigensolvers but no batched
*generalized* one (the infrastructure gap documented in the literature
review). We therefore reduce each pencil with a Cholesky factorization of M:

  M = L L^H  (batched over cases, k-independent, computed once)
  A~ = L^-1 A L^-H          (two batched triangular solves)
  A~ y = omega^2 y          (batched torch.linalg.eigh)
  phi = L^-H y              (one batched triangular solve; phi^H M phi = I)

A(k) is assembled cheaply from the real SAFE blocks:
  Re A = K0 + k^2 K2,  Im A = k K1   (K1 skew-symmetric => A Hermitian).

Matrices can be handed over as numpy arrays, torch tensors, or Warp arrays
(zero-copy via wp.to_torch), so the Warp assembly of Phase 3 feeds this
solver without leaving the GPU.

Precision: consumer GeForce GPUs run fp64 at 1/64 of fp32 throughput, which
makes a pure-fp64 eigh barely faster than the CPU. precision="mixed" runs the
Cholesky reduction and eigh in complex64, then refines with a float64
*subspace Rayleigh-Ritz* step: the lowest (m + buffer) fp32 eigenvectors are
projected onto A and M in fp64 and the small (m+buffer)-dim generalized
problem is re-solved. A plain per-vector Rayleigh quotient is NOT enough here:
the SAFE spectrum spans ~5 decades, so fp32 eigenvectors are contaminated by
*nearby* modes (error ~ eps32 * ||A|| / gap); the subspace solve resolves the
within-window rotations exactly, leaving only the (buffered) window-boundary
coupling ~ sweep-grade 1e-6..1e-7 relative accuracy. Use fp64 for final
gradient verification and reference solves.
"""

import numpy as np
import torch


def _to_torch(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(device)
    try:
        import warp as wp
        if isinstance(arr, wp.array):
            return wp.to_torch(arr)          # zero-copy, stays on its device
    except ImportError:
        pass
    raise TypeError(f"unsupported array type {type(arr)}")


class BatchedDispersion:
    """Batched SAFE dispersion solves over (cases x wavenumbers) on one device."""

    def __init__(self, K0, K1, K2, M, device="cuda", case_chunk=64):
        self.device = device
        self.case_chunk = case_chunk
        self.K0 = _to_torch(K0, device)
        self.K1 = _to_torch(K1, device)
        self.K2 = _to_torch(K2, device)
        self.M = _to_torch(M, device)
        self.n_cases, self.n_dof, _ = self.K0.shape
        # M is real SPD and k-independent: one batched Cholesky for everything
        self.L = torch.linalg.cholesky(self.M).to(torch.complex128)
        self.L32 = self.L.to(torch.complex64)

    @torch.no_grad()
    def solve_k(self, k, n_keep=None, want_vectors=False, want_cg=False,
                cases=None, precision="fp64"):
        """One wavenumber, all cases (or the subset `cases`). Returns dict with
        'omega' (C, m) and optionally 'phi' (C, n, m) M-orthonormal, 'cg' (C, m)."""
        idx = (torch.arange(self.n_cases, device=self.device) if cases is None
               else torch.as_tensor(cases, device=self.device).reshape(-1))
        nc = idx.numel()
        m = n_keep or self.n_dof
        out_w = torch.empty((nc, m), dtype=torch.float64, device=self.device)
        out_phi = (torch.empty((nc, self.n_dof, m),
                               dtype=torch.complex128, device=self.device)
                   if (want_vectors or want_cg) else None)

        for s in range(0, nc, self.case_chunk):
            e = min(s + self.case_chunk, nc)
            sel = idx[s:e]
            K0c, K1c, K2c = self.K0[sel], self.K1[sel], self.K2[sel]
            if precision == "fp64":
                Lc = self.L[sel]
                A = torch.complex(K0c + (k * k) * K2c, k * K1c)
                X = torch.linalg.solve_triangular(Lc, A, upper=False, left=True)
                At = torch.linalg.solve_triangular(Lc, X.mH, upper=False,
                                                   left=True).mH
                At = 0.5 * (At + At.mH)                 # kill round-off asymmetry
                w, Y = torch.linalg.eigh(At)            # ascending eigenvalues
                out_w[s:e] = torch.sqrt(torch.clamp(w[:, :m], min=0.0))
                if out_phi is not None:
                    out_phi[s:e] = torch.linalg.solve_triangular(
                        Lc.mH, Y[:, :, :m], upper=True, left=True)
            elif precision == "mixed":
                # complex64 pipeline, then fp64 subspace Rayleigh-Ritz
                buf = min(m + 30, self.n_dof)
                Lc = self.L32[sel]
                A = torch.complex(K0c + (k * k) * K2c, k * K1c)
                A32 = A.to(torch.complex64)
                X = torch.linalg.solve_triangular(Lc, A32, upper=False, left=True)
                At = torch.linalg.solve_triangular(Lc, X.mH, upper=False,
                                                   left=True).mH
                At = 0.5 * (At + At.mH)
                _, Y = torch.linalg.eigh(At)
                Q = torch.linalg.solve_triangular(Lc.mH, Y[:, :, :buf],
                                                  upper=True, left=True)
                Q = Q.to(torch.complex128)
                Mc = self.M[sel]
                MQ = torch.complex(Mc @ Q.real, Mc @ Q.imag)
                H = Q.mH @ (A @ Q)
                Mh = Q.mH @ MQ
                H = 0.5 * (H + H.mH)
                Mh = 0.5 * (Mh + Mh.mH)
                Lh = torch.linalg.cholesky(Mh)
                Xh = torch.linalg.solve_triangular(Lh, H, upper=False, left=True)
                Ht = torch.linalg.solve_triangular(Lh, Xh.mH, upper=False,
                                                   left=True).mH
                Ht = 0.5 * (Ht + Ht.mH)
                w, Z = torch.linalg.eigh(Ht)
                Z = torch.linalg.solve_triangular(Lh.mH, Z[:, :, :m],
                                                  upper=True, left=True)
                out_w[s:e] = torch.sqrt(torch.clamp(w[:, :m], min=0.0))
                if out_phi is not None:
                    out_phi[s:e] = Q @ Z          # M-orthonormal Ritz vectors
            else:
                raise ValueError(f"unknown precision '{precision}'")

        result = {"omega": out_w}
        if want_vectors:
            result["phi"] = out_phi
        if want_cg:
            result["cg"] = self._group_velocity(k, out_w, out_phi, idx)
        return result

    def _group_velocity(self, k, omega, phi, idx):
        """cg = phi^H (i K1 + 2k K2) phi / (2 omega), batched and chunked."""
        cg = torch.full_like(omega, float("nan"))
        nc = idx.numel()
        for s in range(0, nc, self.case_chunk):
            e = min(s + self.case_chunk, nc)
            sel = idx[s:e]
            dA = torch.complex((2.0 * k) * self.K2[sel], self.K1[sel])
            quad = torch.einsum("bim,bij,bjm->bm",
                                phi[s:e].conj(), dA, phi[s:e]).real
            ok = omega[s:e] > 0
            cg[s:e][ok] = quad[ok] / (2.0 * omega[s:e][ok])
        return cg

    @torch.no_grad()
    def sweep(self, ks, n_keep=30, want_cg=False, precision="fp64"):
        """Dispersion surfaces omega(case, k, mode) for all cases and k-points.

        Returns dict of numpy arrays: 'omega' (C, nk, m), optionally 'cg'."""
        nk = len(ks)
        omega = np.empty((self.n_cases, nk, n_keep))
        cg = np.empty_like(omega) if want_cg else None
        for ik, k in enumerate(ks):
            r = self.solve_k(float(k), n_keep=n_keep, want_cg=want_cg,
                             precision=precision)
            omega[:, ik, :] = r["omega"].cpu().numpy()
            if want_cg:
                cg[:, ik, :] = r["cg"].cpu().numpy()
        out = {"omega": omega}
        if want_cg:
            out["cg"] = cg
        return out
