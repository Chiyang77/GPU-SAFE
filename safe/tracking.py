"""MAC-based mode tracking across a wavenumber (or parameter) sweep.

Eigensolvers return modes sorted by frequency, so branches swap places at
crossings. Tracking greedily matches modes at consecutive steps by the
M-weighted modal assurance criterion  MAC_ij = |phi_i(prev)^H M phi_j(cur)|
(both sets M-orthonormal, so MAC is ~1 on the continued branch and ~0 off it).
"""

import numpy as np


def track_branches(phis, M):
    """Track m branches through a sweep.

    phis: list over steps of (n_dof, m) M-orthonormal eigenvector arrays.
    Returns idx (n_steps, m): idx[s, b] is the mode index (column of step s)
    belonging to branch b, with branches seeded by step 0's frequency order.
    """
    n_steps = len(phis)
    m = phis[0].shape[1]
    idx = np.empty((n_steps, m), dtype=np.int64)
    idx[0] = np.arange(m)
    prev = phis[0]

    for s in range(1, n_steps):
        cur = phis[s]
        mac = np.abs(prev.conj().T @ (M @ cur))         # (m_prev, m_cur)
        assign = np.full(m, -1, dtype=np.int64)
        taken = np.zeros(cur.shape[1], dtype=bool)
        # greedy: resolve strongest correlations first
        order = np.dstack(np.unravel_index(np.argsort(-mac, axis=None), mac.shape))[0]
        n_assigned = 0
        for i, j in order:
            if assign[i] == -1 and not taken[j]:
                assign[i] = j
                taken[j] = True
                n_assigned += 1
                if n_assigned == m:
                    break
        idx[s] = assign
        prev = cur[:, assign]                           # keep branch order
    return idx


def apply_tracking(values, idx):
    """Reorder per-step values (n_steps, m) into branch order (n_steps, m)."""
    return np.take_along_axis(np.asarray(values), idx, axis=1)
