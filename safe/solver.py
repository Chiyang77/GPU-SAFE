"""Dispersion eigenproblem: prescribe real wavenumber k, solve for real omega.

  A(k) phi = omega^2 M phi,   A(k) = K0 + i*k*K1 + k^2*K2

K0, K2, M are real symmetric and K1 is real skew-symmetric, so A(k) is
Hermitian and the pencil (A, M) has real eigenvalues omega^2 >= 0. scipy's
`eigh` handles the complex-Hermitian generalized problem directly and returns
M-orthonormal eigenvectors (phi^H M phi = I), which the group-velocity
perturbation formula relies on.
"""

import numpy as np
import scipy.linalg as sla


def solve_dispersion(k: float, K0, K1, K2, M, omega_max: float | None = None):
    """Solve the SAFE eigenproblem at one wavenumber.

    Returns (omega, phi): omega (m,) ascending in rad/s, phi (n_dof, m) complex
    M-orthonormal eigenvectors. Modes with omega_max < omega are dropped when
    omega_max is given.
    """
    A = K0 + (1j * k) * K1 + (k * k) * K2
    w, V = sla.eigh(A, M)
    w = np.clip(w, 0.0, None)          # clip tiny negative round-off
    omega = np.sqrt(w)
    if omega_max is not None:
        keep = omega <= omega_max
        omega, V = omega[keep], V[:, keep]
    return omega, V


def group_velocities(k: float, omega, phi, K1, K2):
    """cg = d omega / d k via eigenvalue perturbation (simple eigenvalues).

    For phi M-orthonormal: d(omega^2)/dk = phi^H (i K1 + 2 k K2) phi,
    so cg = phi^H (i K1 + 2 k K2) phi / (2 omega). Exact for simple
    eigenvalues; at crossings the value belongs to the perturbed pair basis.
    """
    dA = (1j) * K1 + (2.0 * k) * K2
    quad = np.einsum("im,ij,jm->m", phi.conj(), dA, phi).real
    cg = np.full_like(omega, np.nan)
    ok = omega > 0
    cg[ok] = quad[ok] / (2.0 * omega[ok])
    return cg
