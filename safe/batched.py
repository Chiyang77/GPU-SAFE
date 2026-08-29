"""Phase 3: Warp GPU batched assembly of the SAFE matrices.

Assembles K0, K1, K2, M for a whole batch of pipe cases at once — one case per
(wall thickness, internal pressure) pair — with the Lame geometric prestress
folded directly into K0/K2 (Kg1 = 0 for the Lame field: no sigma_xz/yz).

Design notes:
- The annular mesh topology (n_circ x n_r Q4 grid) is shared by every case;
  only the nodal radii depend on wall thickness. Node coordinates are therefore
  computed inside the kernel from (r_out, t[case]) — no per-case mesh uploads.
- One thread per (case, element); each thread integrates 2x2 Gauss points and
  scatters closed-form 3x3 node-pair blocks with atomic adds. All math float64.
- The 3x3 blocks are the analytical evaluation of B0^T C B0 etc. for isotropic
  C, matching safe/element.py entry-for-entry (validated in benchmark_phase3).

Import of this module requires warp; the CPU package (safe/*) does not.
"""

import numpy as np
import warp as wp

F = wp.float64


@wp.kernel
def _assemble_kernel(
    t: wp.array(dtype=F),            # (cases,) wall thickness
    AL: wp.array(dtype=F),           # (cases,) Lame constant A_L = P a^2/(b^2-a^2)
    BL: wp.array(dtype=F),           # (cases,) Lame constant B_L = P a^2 b^2/(b^2-a^2)
    szz: wp.array(dtype=F),          # (cases,) axial prestress
    conn: wp.array2d(dtype=wp.int32),
    r_out: F,
    n_circ: wp.int32,
    inv_n_r: F,       # 1/n_r, computed in double on the host
    dth: F,           # 2*pi/n_circ, computed in double on the host
    sq3: F,           # Gauss abscissa 1/sqrt(3), double precision
    lam: F,
    mu: F,
    rho: F,
    K0: wp.array3d(dtype=F),
    K1: wp.array3d(dtype=F),
    K2: wp.array3d(dtype=F),
    Mm: wp.array3d(dtype=F),
):
    c, e = wp.tid()

    t_c = t[c]
    r_in = r_out - t_c
    dr = t_c * inv_n_r

    # nodal coordinates of this element, computed from the case geometry
    xs = wp.vec4d()
    ys = wp.vec4d()
    for a in range(4):
        nd = conn[e, a]
        i = nd / n_circ           # ring index (integer division)
        j = nd - i * n_circ       # sector index
        r = r_in + F(i) * dr
        th = F(j) * dth
        xs[a] = r * wp.cos(th)
        ys[a] = r * wp.sin(th)

    # local-node sign patterns and Gauss points (2x2); +-1 literals are exact
    xia = wp.vec4d(F(-1.0), F(1.0), F(1.0), F(-1.0))
    eta = wp.vec4d(F(-1.0), F(-1.0), F(1.0), F(1.0))
    gxi = wp.vec4d(-sq3, -sq3, sq3, sq3)
    get = wp.vec4d(-sq3, sq3, -sq3, sq3)

    lam2 = lam + F(2.0) * mu
    al = AL[c]
    bl = BL[c]
    sz = szz[c]

    for gp in range(4):
        xi = gxi[gp]
        et = get[gp]

        N = wp.vec4d()
        dNxi = wp.vec4d()
        dNet = wp.vec4d()
        for a in range(4):
            N[a] = F(0.25) * (F(1.0) + xia[a] * xi) * (F(1.0) + eta[a] * et)
            dNxi[a] = F(0.25) * xia[a] * (F(1.0) + eta[a] * et)
            dNet[a] = F(0.25) * eta[a] * (F(1.0) + xia[a] * xi)

        j00 = F(0.0); j01 = F(0.0); j10 = F(0.0); j11 = F(0.0)
        for a in range(4):
            j00 += dNxi[a] * xs[a]
            j01 += dNxi[a] * ys[a]
            j10 += dNet[a] * xs[a]
            j11 += dNet[a] * ys[a]
        detJ = j00 * j11 - j01 * j10
        inv_det = F(1.0) / detJ
        # dNdx = dN @ inv(J)^T with J = [[j00, j01], [j10, j11]]
        Nx = wp.vec4d()
        Ny = wp.vec4d()
        for a in range(4):
            Nx[a] = (dNxi[a] * j11 - dNet[a] * j01) * inv_det
            Ny[a] = (-dNxi[a] * j10 + dNet[a] * j00) * inv_det

        dA = detJ  # Gauss weights are 1

        # Lame prestress at the Gauss point (Cartesian components)
        xg = F(0.0); yg = F(0.0)
        for a in range(4):
            xg += N[a] * xs[a]
            yg += N[a] * ys[a]
        r2 = xg * xg + yg * yg
        s_rr = al - bl / r2
        s_tt = al + bl / r2
        sxx = (s_rr * xg * xg + s_tt * yg * yg) / r2
        syy = (s_rr * yg * yg + s_tt * xg * xg) / r2
        sxy = (s_rr - s_tt) * xg * yg / r2

        for a in range(4):
            ra = 3 * conn[e, a]
            ax = Nx[a]; ay = Ny[a]; na = N[a]
            for b in range(4):
                rb = 3 * conn[e, b]
                bx = Nx[b]; by = Ny[b]; nb = N[b]

                # geometric (initial-stress) scalars, same for all 3 components
                g0 = ax * (sxx * bx + sxy * by) + ay * (sxy * bx + syy * by)
                g2 = sz * na * nb

                # K0: material B0^T C B0 + in-plane geometric term
                wp.atomic_add(K0, c, ra + 0, rb + 0, dA * (lam2 * ax * bx + mu * ay * by + g0))
                wp.atomic_add(K0, c, ra + 0, rb + 1, dA * (lam * ax * by + mu * ay * bx))
                wp.atomic_add(K0, c, ra + 1, rb + 0, dA * (lam * ay * bx + mu * ax * by))
                wp.atomic_add(K0, c, ra + 1, rb + 1, dA * (lam2 * ay * by + mu * ax * bx + g0))
                wp.atomic_add(K0, c, ra + 2, rb + 2, dA * (mu * (ax * bx + ay * by) + g0))

                # K1: material B0^T C B1 - B1^T C B0 (skew); Lame gives no Kg1
                wp.atomic_add(K1, c, ra + 0, rb + 2, dA * (lam * ax * nb - mu * na * bx))
                wp.atomic_add(K1, c, ra + 1, rb + 2, dA * (lam * ay * nb - mu * na * by))
                wp.atomic_add(K1, c, ra + 2, rb + 0, dA * (mu * ax * nb - lam * na * bx))
                wp.atomic_add(K1, c, ra + 2, rb + 1, dA * (mu * ay * nb - lam * na * by))

                # K2: material B1^T C B1 + axial geometric term
                wp.atomic_add(K2, c, ra + 0, rb + 0, dA * (mu * na * nb + g2))
                wp.atomic_add(K2, c, ra + 1, rb + 1, dA * (mu * na * nb + g2))
                wp.atomic_add(K2, c, ra + 2, rb + 2, dA * (lam2 * na * nb + g2))

                # M
                mv = dA * rho * na * nb
                wp.atomic_add(Mm, c, ra + 0, rb + 0, mv)
                wp.atomic_add(Mm, c, ra + 1, rb + 1, mv)
                wp.atomic_add(Mm, c, ra + 2, rb + 2, mv)


def assemble_batch(thicknesses, pressures, r_out, n_circ, n_r, mat,
                   ends="capped", device="cuda:0"):
    """Assemble pressurized SAFE matrices for every case on the GPU.

    thicknesses, pressures: (n_cases,) arrays — one (t, P) pair per case.
    Returns (K0, K1, K2, M) as wp.arrays of shape (n_cases, n_dof, n_dof),
    float64, living on `device` (use .numpy() to copy back).
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    pressures = np.asarray(pressures, dtype=np.float64)
    n_cases = thicknesses.size
    if pressures.size != n_cases:
        raise ValueError("thicknesses and pressures must have equal length")

    # shared topology (radii are case-dependent, connectivity is not)
    from .mesh import AnnularMesh
    template = AnnularMesh(r_out - thicknesses[0], r_out, n_circ, n_r)
    conn = wp.array(template.elems.astype(np.int32), dtype=wp.int32, device=device)
    n_dof = template.n_dof

    a = r_out - thicknesses
    b = r_out
    AL = pressures * a**2 / (b**2 - a**2)
    BL = pressures * a**2 * b**2 / (b**2 - a**2)
    szz = AL.copy() if ends == "capped" else np.zeros_like(AL)

    dev_args = dict(dtype=F, device=device)
    t_d = wp.array(thicknesses, **dev_args)
    AL_d = wp.array(AL, **dev_args)
    BL_d = wp.array(BL, **dev_args)
    szz_d = wp.array(szz, **dev_args)

    shape = (n_cases, n_dof, n_dof)
    K0 = wp.zeros(shape, **dev_args)
    K1 = wp.zeros(shape, **dev_args)
    K2 = wp.zeros(shape, **dev_args)
    M = wp.zeros(shape, **dev_args)

    wp.launch(_assemble_kernel,
              dim=(n_cases, template.elems.shape[0]),
              inputs=[t_d, AL_d, BL_d, szz_d, conn,
                      F(r_out), np.int32(n_circ), F(1.0 / n_r),
                      F(2.0 * np.pi / n_circ), F(1.0 / np.sqrt(3.0)),
                      F(mat.lam), F(mat.mu), F(mat.rho)],
              outputs=[K0, K1, K2, M],
              device=device)
    return K0, K1, K2, M
