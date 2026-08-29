"""Isotropic linear-elastic material and its full 3D constitutive matrix.

Voigt strain ordering used throughout: [exx, eyy, ezz, gxy, gyz, gxz].
SAFE needs the full 3D C — a plane-stress/strain reduction would drop the
uz coupling that carries longitudinal and torsional modes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    E: float      # Young's modulus [Pa]
    nu: float     # Poisson's ratio
    rho: float    # density [kg/m^3]

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def c_shear(self) -> float:
        """Bulk shear wave speed sqrt(mu/rho) — exact T(0,1) phase velocity."""
        return float(np.sqrt(self.mu / self.rho))

    @property
    def c_bar(self) -> float:
        """Bar velocity sqrt(E/rho) — low-frequency L(0,1) limit."""
        return float(np.sqrt(self.E / self.rho))


STEEL = Material(E=210e9, nu=0.3, rho=7850.0)


def isotropic_C(mat: Material) -> np.ndarray:
    """6x6 stiffness matrix in Voigt ordering [exx, eyy, ezz, gxy, gyz, gxz]."""
    lam, mu = mat.lam, mat.mu
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C
