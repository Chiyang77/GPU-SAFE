"""Mode postprocessing: cylindrical components, circumferential order n,
and family classification into torsional T(0,m) / longitudinal L(0,m) /
flexural F(n,m).

Circumferential order: FFT of the nodal displacement around each radial ring;
the dominant harmonic (power summed over ur, utheta, uz and over rings) is n.
Family for n = 0: torsional if the in-plane motion is circumferential-dominated,
longitudinal if radial/axial-dominated.
"""

import numpy as np

from .mesh import AnnularMesh


def cylindrical_components(mesh: AnnularMesh, phi: np.ndarray):
    """Split one eigenvector into nodal (ur, utheta, uz), each (n_nodes,) complex."""
    u = phi.reshape(-1, 3)
    c, s = np.cos(mesh.theta), np.sin(mesh.theta)
    ur = u[:, 0] * c + u[:, 1] * s
    ut = -u[:, 0] * s + u[:, 1] * c
    uz = u[:, 2]
    return ur, ut, uz


def circumferential_order(mesh: AnnularMesh, phi: np.ndarray) -> int:
    """Dominant circumferential harmonic n in [0, n_circ // 2]."""
    ur, ut, uz = cylindrical_components(mesh, phi)
    power = np.zeros(mesh.n_circ)
    for comp in (ur, ut, uz):
        rings = comp.reshape(mesh.n_r + 1, mesh.n_circ)
        power += (np.abs(np.fft.fft(rings, axis=1)) ** 2).sum(axis=0)
    # fold negative harmonics onto positive n
    n_max = mesh.n_circ // 2
    folded = np.zeros(n_max + 1)
    folded[0] = power[0]
    for n in range(1, n_max + 1):
        folded[n] = power[n] + (power[mesh.n_circ - n] if n < mesh.n_circ - n else 0.0)
    return int(np.argmax(folded))


def classify_mode(mesh: AnnularMesh, phi: np.ndarray):
    """Return (n, family) with family in {'T', 'L', 'F'}."""
    n = circumferential_order(mesh, phi)
    if n > 0:
        return n, "F"
    ur, ut, uz = cylindrical_components(mesh, phi)
    e_t = float(np.sum(np.abs(ut) ** 2))
    e_rl = float(np.sum(np.abs(ur) ** 2) + np.sum(np.abs(uz) ** 2))
    return 0, ("T" if e_t > e_rl else "L")


def find_branch(mesh, omega, phi, n_target: int, family: str, index: int = 1):
    """Pick the index-th (1-based, ascending omega) mode matching (n, family).

    Returns (omega_m, mode_index) or (None, None) if absent.
    """
    hits = 0
    for m in range(omega.size):
        if omega[m] <= 0:
            continue
        n, fam = classify_mode(mesh, phi[:, m])
        if n == n_target and fam == family:
            hits += 1
            if hits == index:
                return omega[m], m
    return None, None
