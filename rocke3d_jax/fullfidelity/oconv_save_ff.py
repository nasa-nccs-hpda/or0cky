"""OCONV post-ITER-loop flux save (D61, piece 2b).

After the HBL iteration exits, OCNKPP.f:2399-2405 saves the fluxes used by the gradient
adjustment (FLG3D/FLS3D) and the boundary terms. With DM = DELTAM*DTBYDZ(1) and
DTBYDZ(1) = DTS/MO(I,J,1) (OCNKPP.f:1738/1814 and the DTBYDZ setup):
  FLG3D(1:LMIJ-1) = FLG(1:LMIJ-1)  (from the last G OVDIFFS call)
  FLG3D(LMIJ)     = 0
  FLG3D(0)        = DTS*DELTAE*DXYPO(J)
  FLS3D(1:LMIJ-1) = FLS(1:LMIJ-1)  (from the last S OVDIFFS call)
  FLS3D(LMIJ)     = 0
  FLS3D(0)        = -DTS*DELTAS*DXYPO(J)*(1-DM) + DM*S0M1(I,J)
Arrays are 0-based (length LMO+1) with entry L equal to the Fortran index L.
"""

DTS = 1800.0
LMO = 13


def convergence_continue(iter_, hblp, hbl, ze, kbl):
    """OCNKPP.f:2337-2338: GO TO 510 while (ITER=1, or HBL moved by more than a quarter of the
    layer thickness at KBL) and ITER<4. ze is 0-based with ze[k] = Fortran ZE(k)."""
    moved = abs(hblp - hbl) > (ze[kbl] - ze[kbl - 1]) * 0.25
    return (iter_ == 1 or moved) and iter_ < 4


def flux_save(lmij, fl_g, fl_s, deltae, deltas, deltam, dxypo_j, mo1_cur, s0m1):
    """fl_g, fl_s: last OVDIFFS fluxes (0-based, index L = Fortran FLG(L)/FLS(L)).
    mo1_cur is MO(I,J,1), the setup record's mo1. Returns (flg3d, fls3d), 0-based length LMO+1."""
    dtbydz1 = DTS / mo1_cur
    dm = deltam * dtbydz1
    flg3d = [0.0] * (LMO + 1)
    fls3d = [0.0] * (LMO + 1)
    for L in range(1, lmij):
        flg3d[L] = fl_g[L]
        fls3d[L] = fl_s[L]
    flg3d[lmij] = 0.0
    fls3d[lmij] = 0.0
    flg3d[0] = DTS * deltae * dxypo_j
    fls3d[0] = -DTS * deltas * dxypo_j * (1 - dm) + dm * s0m1
    return dm, flg3d, fls3d
