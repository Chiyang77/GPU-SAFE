"""Global assembly of the SAFE system matrices K0, K1, K2, M.

Dense storage: the pipe cross-section stays small (O(10^3) DOFs), and dense
Hermitian eigensolvers are the Phase-1 baseline (and the Phase-4 GPU-batched
target format).
"""

import numpy as np

from .element import element_matrices
from .materials import Material, isotropic_C
from .mesh import AnnularMesh


def assemble_global(mesh: AnnularMesh, mat: Material):
    """Return (K0, K1, K2, M) as dense (n_dof, n_dof) float64 arrays."""
    C = isotropic_C(mat)
    n = mesh.n_dof
    K0 = np.zeros((n, n))
    K1 = np.zeros((n, n))
    K2 = np.zeros((n, n))
    M = np.zeros((n, n))

    for conn in mesh.elems:
        coords = mesh.nodes[conn]
        K0e, K1e, K2e, Me = element_matrices(coords, C, mat.rho)
        dofs = (3 * conn[:, None] + np.arange(3)).ravel()   # (12,)
        ix = np.ix_(dofs, dofs)
        K0[ix] += K0e
        K1[ix] += K1e
        K2[ix] += K2e
        M[ix] += Me

    return K0, K1, K2, M
