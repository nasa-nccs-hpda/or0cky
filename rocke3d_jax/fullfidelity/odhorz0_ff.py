"""Full-fidelity port of OCNDYN2.f's ODHORZ0 -- Stage 2 of the DYNSI/ocean port, D40.

ODHORZ0 prepares the per-cell pressure profile and seawater-equation-of-state quantities used by
the (not yet ported) horizontal pressure-gradient force solve: for each layer, from the top down,
it accumulates the hydrostatic pressure `P`/`OPBOT`, then -- using a "Linear Upstream Scheme"
(`USE_OPGFQ=0`, confirmed the only live branch for this rundeck; the Quadratic Upstream Scheme
alternative, `USE_OPGFQ=1`, is dead code here) -- estimates the enthalpy/salinity at the upper and
lower quadrature points of each cell (`GUP`/`GDN`/`SUP`/`SDN`), converts those (plus the
corresponding pressure) to specific volume via the seawater equation of state (`VOLGSP`), and
combines the upper/lower volumes into `dZGdP`, `VBAR` (mean specific volume) and `DH3D` (cell
thickness contribution). At the North Pole, results are copied uniformly across all longitudes,
and `polevel()` (ported in D39, reused here unchanged) reconstructs the pole-row UO/VO.

`VOLGSP` is a genuine external-file dependency: a trilinear interpolation over a 43x41x40-entry
lookup table (`VGSP`, `OCNFUNTAB.f`'s `OCFUNC` module) read from the `OFTAB` binary file at
`init_OCEAN` -- the same file D35's `SHCGS` depends on. Rather than replicating the ~70,520-entry
table, its two outputs per cell (`VUP`, `VDN`) are recorded directly as real inputs, the
established "record what's not yet ported" pattern (D29's `GAIRX`/`GWATX`, D33's `oPREC`, D35's
`SHCGS`/`PCORR`).
"""
import numpy as np
from polerelax_ff import polevel, geomo_pole_arrays
from ostres2_ff import geomo_arrays

IM, JM, LMO = 72, 46, 13
GRAV = 9.80665
Z12EH = 0.28867513  # 1/sqrt(12)


def dxypo_array():
    """DXYPO(J) = DXYS(J)+DXYN(J) = the full grid-cell area (DXYS=DXYN=.5*DXYP, OGEOM.f) --
    analytically derivable, reusing D36's ostres2_ff.geomo_arrays(). Length JM+1, 1-indexed."""
    dxys, dxyn, _, _, _ = geomo_arrays()
    return dxys + dxyn


def odhorz0(lmm, lmv, opress, g0m, gzm, s0m, szm, mo0, uo0, vo0, vup, vdn):
    """Direct port of ODHORZ0's live (USE_OPGFQ=0) branch (OCNDYN2.f:1718-1862).

    All 2D fields (`lmm`,`lmv`,`opress`) shape (IM+1,JM+1); all 3D fields shape
    (IM+1,JM+1,LMO+1); 1-indexed throughout (row/col/layer 0 unused). `vup`,`vdn` are VOLGSP's
    recorded real outputs (see module docstring). `lmv` is NOT dumped by this delta -- polevel()
    (called at the end of each layer here, same as D39) uses LMV, not LMM; reused directly from
    D36-D39's already-validated LMV geometry dump (e.g. ffz_polerelax_geom.bin), since LMU/LMV
    are static and unchanging across every delta. Returns a dict with
    `opbot`,`gup`,`gdn`,`sup`,`sdn`,`dzgdp`,`vbar`,`dh3d` (the routine's real outputs) plus
    `mo`,`uo`,`vo` (fresh copies, modified only at the North Pole row/by `polevel`).
    """
    cosic, sinic, cosu, sinu = geomo_pole_arrays()
    dxypo = dxypo_array()

    mo = mo0.copy()
    uo = uo0.copy()
    vo = vo0.copy()

    opbot = np.zeros_like(opress)
    p = np.zeros_like(g0m)
    gup = np.zeros_like(g0m)
    gdn = np.zeros_like(g0m)
    sup = np.zeros_like(g0m)
    sdn = np.zeros_like(g0m)
    dzgdp = np.zeros_like(g0m)
    vbar = np.zeros_like(g0m)
    dh3d = np.zeros_like(g0m)

    def m_active(i, j, l):
        """Replicates nbyzm's mask exactly (OCNDYN.f:505-539): the ordinary per-cell
        LMM(I,J)>=L test everywhere EXCEPT the North Pole row (J=JM), which is hard-restricted
        to I=1 only (i1yzm=i2yzm=1, regardless of LMM's value at I=2..IM there) -- a real design
        gap caught via dump comparison (OPBOT/GUP/etc at (I>1,JM) must stay exactly 0, not
        whatever LMM(I,JM) would otherwise imply)."""
        if j == JM:
            return i == 1 and lmm[1, JM] >= l
        return lmm[i, j] >= l

    # ---- Pressure integration, top-down (l=1..LMO) ----
    # OPBOT(I,J) is initialized to OPRESS(I,J) ONLY where m_active(i,j,1) -- at the North Pole
    # row this means ONLY (1,JM), matching the Fortran's own l=1-only, nbyzm-gated init loop.
    # Cells never touched (land, or (I>1,JM)) stay exactly 0, matching the real dump.
    for j in range(1, JM + 1):
        for i in range(1, IM + 1):
            if m_active(i, j, 1):
                opbot[i, j] = opress[i, j]
    for l in range(1, LMO + 1):
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if m_active(i, j, l):
                    p[i, j, l] = opbot[i, j] + mo0[i, j, l] * GRAV * 0.5
                    opbot[i, j] = opbot[i, j] + mo0[i, j, l] * GRAV

    # ---- Per-layer EOS prep, bottom-up (l=LMO..1), Linear Upstream Scheme ----
    for l in range(LMO, 0, -1):
        for j in range(1, JM + 1):
            for i in range(1, IM + 1):
                if not m_active(i, j, l):
                    continue
                mmi = mo0[i, j, l] * dxypo[j]
                g0l = g0m[i, j, l]
                gzl = gzm[i, j, l]
                s0l = s0m[i, j, l]
                szl = szm[i, j, l]
                gup[i, j, l] = (g0l - 2 * Z12EH * gzl) / mmi
                gdn[i, j, l] = (g0l + 2 * Z12EH * gzl) / mmi
                sup[i, j, l] = (s0l - 2 * Z12EH * szl) / mmi
                sdn[i, j, l] = (s0l + 2 * Z12EH * szl) / mmi
                smean = s0l / mmi
                sup[i, j, l] = max(sup[i, j, l], 0.5 * smean)
                sdn[i, j, l] = max(sdn[i, j, l], 0.5 * smean)

                vup_l = vup[i, j, l]
                vdn_l = vdn[i, j, l]
                dzgdp[i, j, l] = vup_l * (0.5 - Z12EH) + vdn_l * (0.5 + Z12EH)
                vbar[i, j, l] = (vup_l + vdn_l) * 0.5
                dh3d[i, j, l] = mo0[i, j, l] * vbar[i, j, l]

        # Copy to all longitudes at North Pole
        if lmm[1, JM] >= l:
            dh3d[2:IM + 1, JM, l] = dh3d[1, JM, l]
            vbar[2:IM + 1, JM, l] = vbar[1, JM, l]
            dzgdp[2:IM + 1, JM, l] = dzgdp[1, JM, l]
            mo[2:IM + 1, JM, l] = mo[1, JM, l]

        # Initialize polar velocities
        polevel(uo, vo, l, lmv, cosic, sinic, cosu, sinu)

    return dict(opbot=opbot, gup=gup, gdn=gdn, sup=sup, sdn=sdn, dzgdp=dzgdp, vbar=vbar,
                dh3d=dh3d, mo=mo, uo=uo, vo=vo)
