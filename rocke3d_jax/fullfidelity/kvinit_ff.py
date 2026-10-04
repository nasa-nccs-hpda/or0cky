"""Full-fidelity port of OCNKPP.f's KVINIT -- Stage 2, D58 (OCNKPP.f:1315-1359).

KVINIT saves the pre-source surface snapshot used by the KPP mixing step: the first-layer
(and first LSRPD layers of G0M) values of G0M, S0M, MO, GXMO, GYMO, SXMO, SYMO, UO, VO, UOD, VOD
are copied into KPP_COM's G0M1 / S0M1 / MO1 / GXM1 / GYM1 / SXM1 / SYM1 / UO1 / VO1 / UOD1 / VOD1.
It is a pure copy with no arithmetic, so the port is a copy and the validation is exact equality.
Arrays are numpy arrays shaped (IM, JM, ...) with 1-based layer index 0 = Fortran layer 1.
"""
import numpy as np


def kvinit(g0m, s0m, mo, gxmo, gymo, sxmo, symo, uo, vo, uod, vod, lsrpd):
    snap = {
        "g0m1": g0m[:, :, :lsrpd].copy(),
        "s0m1": s0m[:, :, 0].copy(),
        "mo1": mo[:, :, 0].copy(),
        "gxm1": gxmo[:, :, 0].copy(),
        "gym1": gymo[:, :, 0].copy(),
        "sxm1": sxmo[:, :, 0].copy(),
        "sym1": symo[:, :, 0].copy(),
        "uo1": uo[:, :, 0].copy(),
        "vo1": vo[:, :, 0].copy(),
        "uod1": uod[:, :, 0].copy(),
        "vod1": vod[:, :, 0].copy(),
    }
    return snap
