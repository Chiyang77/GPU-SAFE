# GPU-SAFE

A GPU-batched, differentiable semi-analytical finite element (SAFE) framework for computing
pressure-dependent guided wave dispersion in pipes and for solving the associated inverse
problems.

The framework assembles the SAFE cross-sectional matrices, including the geometric stiffness
of the exact Lame prestress field of internal pressure, in batched GPU kernels for thousands
of parameter combinations simultaneously. The generalized eigenvalue problems are solved in
batch through a Cholesky reduction combined with a mixed-precision scheme. A differentiable
assembly provides exact first-order eigenfrequency sensitivities in a single reverse-mode
pass, which supports Fisher-information identifiability analysis and gradient-based inversion
of internal pressure and wall thickness.

## Contents

The `safe/` package contains the solver library:

| Module | Description |
| --- | --- |
| `materials.py`, `mesh.py`, `element.py`, `assembly.py` | 2-D SAFE discretization (Q4 annulus) |
| `solver.py`, `modes.py`, `tracking.py` | Dispersion solution, mode classification, MAC branch tracking |
| `prestress.py` | Lame prestress field and geometric stiffness |
| `harmonic.py` | Circumferential-harmonic SAFE (1-D radial reduction) |
| `batched.py` | NVIDIA Warp batched GPU assembly |
| `torch_eig.py` | Batched mixed-precision generalized eigensolution (PyTorch) |
| `harmonic_diff.py` | Differentiable assembly and eigenfrequency sensitivities |

## Requirements

Python 3.10+, NumPy, SciPy, [NVIDIA Warp](https://github.com/NVIDIA/warp), and PyTorch with
CUDA for the GPU-batched and differentiable paths. The CPU solvers run without a GPU.

## Status

The validation scripts and usage examples will be released upon publication of the
accompanying paper. A preprint is forthcoming.

## License

To be added upon publication.
