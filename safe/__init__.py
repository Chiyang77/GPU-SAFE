"""CPU reference implementation of the SAFE (semi-analytical finite element) method
for guided-wave dispersion in pipes. Phase 1: unpressurized isotropic pipe, Q4 annular
mesh, real wavenumber k -> real frequencies omega."""

from .materials import Material, isotropic_C, STEEL
from .mesh import AnnularMesh
from .assembly import assemble_global
from .solver import solve_dispersion, group_velocities
from .modes import classify_mode, cylindrical_components
from .prestress import LamePressure, assemble_geometric, torsional_shift_analytic
