"""Annular Q4 cross-section mesh for a pipe.

The annulus is one continuous mesh: the seam at theta = 0/2*pi shares nodes
(plain connectivity wrap-around, not a periodic boundary condition).

Node numbering: node(i, j) = i * n_circ + j for radial ring i in [0, n_r]
(i=0 inner surface, i=n_r outer surface) and sector j in [0, n_circ).

Element local node ordering is chosen so the isoparametric Jacobian is
positive: xi runs radially outward, eta runs counterclockwise in theta.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnnularMesh:
    r_in: float
    r_out: float
    n_circ: int   # sectors around the circumference
    n_r: int      # element layers through the wall

    nodes: np.ndarray = field(init=False)   # (n_nodes, 2) x,y
    elems: np.ndarray = field(init=False)   # (n_elems, 4) node ids, Q4 order
    theta: np.ndarray = field(init=False)   # (n_nodes,) nodal angle
    radius: np.ndarray = field(init=False)  # (n_nodes,) nodal radius

    def __post_init__(self):
        if self.r_out <= self.r_in:
            raise ValueError("r_out must exceed r_in")
        if self.n_circ < 8 or self.n_r < 1:
            raise ValueError("mesh too coarse: need n_circ >= 8, n_r >= 1")

        r = np.linspace(self.r_in, self.r_out, self.n_r + 1)
        th = 2 * np.pi * np.arange(self.n_circ) / self.n_circ
        R, TH = np.meshgrid(r, th, indexing="ij")           # (n_r+1, n_circ)
        self.radius = R.ravel()
        self.theta = TH.ravel()
        self.nodes = np.column_stack([self.radius * np.cos(self.theta),
                                      self.radius * np.sin(self.theta)])

        elems = []
        for i in range(self.n_r):
            for j in range(self.n_circ):
                jp = (j + 1) % self.n_circ
                # (xi -> radial, eta -> circumferential) => positive det J
                elems.append([i * self.n_circ + j,
                              (i + 1) * self.n_circ + j,
                              (i + 1) * self.n_circ + jp,
                              i * self.n_circ + jp])
        self.elems = np.array(elems, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def ring(self, i: int) -> np.ndarray:
        """Node ids of radial ring i, ordered by increasing theta."""
        return np.arange(i * self.n_circ, (i + 1) * self.n_circ)
