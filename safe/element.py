"""Q4 SAFE cross-sectional element.

Each node carries 3 displacement DOFs (ux, uy, uz) even though the mesh is 2D:
axial propagation is analytic via u = U(x,y) exp(i(kz - wt)), so d/dz -> ik and
the strain-displacement operator splits as B(k) = B0 + ik*B1.

With Voigt ordering [exx, eyy, ezz, gxy, gyz, gxz] and DOF order (ux, uy, uz):

  B0_a = [[N_a,x, 0,     0    ],        B1_a = [[0,   0,   0  ],
          [0,     N_a,y, 0    ],                [0,   0,   0  ],
          [0,     0,     0    ],                [0,   0,   N_a],
          [N_a,y, N_a,x, 0    ],                [0,   0,   0  ],
          [0,     0,     N_a,y],                [0,   N_a, 0  ],
          [0,     0,     N_a,x]]                [N_a, 0,   0  ]]

Element matrices (all real; K1e is skew-symmetric):
  K0e = int B0^T C B0 dA
  K1e = int (B0^T C B1 - B1^T C B0) dA
  K2e = int B1^T C B1 dA
  Me  = int rho N^T N dA
"""

import numpy as np

# 2x2 Gauss quadrature on [-1,1]^2
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GAUSS_PTS = np.array([(xi, eta) for xi in _GP for eta in _GP])
GAUSS_WTS = np.ones(4)

# Q4 local node order matching mesh.AnnularMesh element connectivity
_XI_A = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA_A = np.array([-1.0, -1.0, 1.0, 1.0])


def shape_functions(xi: float, eta: float):
    """Return N (4,) and dN/d(xi,eta) (4,2) at a natural-coordinate point."""
    N = 0.25 * (1 + _XI_A * xi) * (1 + _ETA_A * eta)
    dN = np.column_stack([0.25 * _XI_A * (1 + _ETA_A * eta),
                          0.25 * _ETA_A * (1 + _XI_A * xi)])
    return N, dN


def element_matrices(coords: np.ndarray, C: np.ndarray, rho: float):
    """K0e, K1e, K2e, Me (each 12x12) for one Q4 SAFE element.

    coords: (4,2) nodal x,y in the element's local node order.
    """
    K0e = np.zeros((12, 12))
    K1e = np.zeros((12, 12))
    K2e = np.zeros((12, 12))
    Me = np.zeros((12, 12))

    for (xi, eta), w in zip(GAUSS_PTS, GAUSS_WTS):
        N, dN = shape_functions(xi, eta)
        J = dN.T @ coords                       # J[a,b] = d x_b / d xi_a
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("non-positive Jacobian — check element orientation")
        dNdx = dN @ np.linalg.inv(J).T          # (4,2): dN/dx, dN/dy

        B0 = np.zeros((6, 12))
        B1 = np.zeros((6, 12))
        Nmat = np.zeros((3, 12))
        for a in range(4):
            c = 3 * a
            Nx, Ny = dNdx[a]
            B0[0, c] = Nx
            B0[1, c + 1] = Ny
            B0[3, c] = Ny
            B0[3, c + 1] = Nx
            B0[4, c + 2] = Ny
            B0[5, c + 2] = Nx
            B1[2, c + 2] = N[a]
            B1[4, c + 1] = N[a]
            B1[5, c] = N[a]
            Nmat[0, c] = Nmat[1, c + 1] = Nmat[2, c + 2] = N[a]

        dA = w * detJ
        CB0 = C @ B0
        CB1 = C @ B1
        K0e += dA * (B0.T @ CB0)
        K1e += dA * (B0.T @ CB1 - B1.T @ CB0)
        K2e += dA * (B1.T @ CB1)
        Me += (dA * rho) * (Nmat.T @ Nmat)

    return K0e, K1e, K2e, Me
