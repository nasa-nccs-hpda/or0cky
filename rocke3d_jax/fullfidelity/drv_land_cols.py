"""D177 item 4: the land leftovers of the coupled path, computed instead of read from the ffp/ffg records.

Sources (read-only real code) and measured identities on the real records (all 54 steps of nov26_day + the 6-step dec01 / jan01 windows, both substeps,
99,396 land rows; max difference 0 for every one):
  ffg pres   (0-based col 154)  GHY_DRV.f:1261  `ps - dmCO2cond(i,j)*grav*0.01`; dmCO2cond is the CO2-condensation increment (0 in this Earth build:
                               DO_CO2_CONDENSATION is not exercised, the record identity pres == PBL psurf holds with 0 differences) -> pres = PBL psurf column (ffp col 16),
                               which the coupled path already sets from our PEDN(1) (atm_step.override_pbl).
  ffg vs0    (159)              GHY_DRV.f:1266 `pbl_args%ws0` = PBL output ws0 (ffp output col 98).
  ffg gusti  (160)              GHY_DRV.f:1267 `pbl_args%gusti` after the PBL call = the gusti INPUT of the PBL (land: PBL does not change it; ffp col 24 == col 113, 0 diff).
  ffg vs     (158)              `pbl_args%ws` (already taken from our PBL output in land_chain.land_substep).
  ffg ma1    (165)              GHY_DRV.f:1272 `ma1 = atmlnd%am1(i,j)` = MA(1,i,j) of the atmosphere at SURFACE entry (== the state our dynamics/CONDSE/RADIA produce;
                               record check vs the dump of the 'r'/'d' sites: 0 difference).
  ffp land elhx (19)            GHY_DRV.f:1071-1075: elhx = lhe, and lhs when tg1 < 0 (tg1 = tsns_ij, the soil surface temperature carried from the previous GHY call;
                               record identity on 99,396 rows with tg1 := tg - tf: 0 mismatches).  In the coupled path the substep-2 value follows from the substep-1 GHY `tsns`.
The only deviation that cannot be excluded from these data: a cell whose tsns lies within one rounding step of 0 C (tg - tf versus tsns); none occurs in the records.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LHE = 2.5e6
LHS = 2.834e6
TF = 273.15
# 0-based ffg columns
G_PRES, G_VS, G_VS0, G_GUSTI, G_MA1 = 154, 158, 159, 160, 165
# 0-based ffp columns
P_PSURF, P_GUSTI_IN, P_ELHX, P_TG = 16, 24, 19, 18
P_OUT_WS, P_OUT_WS0, P_OUT_GUSTI = 91, 98, 113


def land_elhx(tg1):
    """GHY_DRV.f:1071-1072. tg1: soil surface temperature [C] (tsns); returns (N,)."""
    return np.where(np.asarray(tg1) < 0.0, LHS, LHE)


def land_elhx_from_tg(tg_kelvin):
    return land_elhx(np.asarray(tg_kelvin) - TF)


def land_leftovers(p4, pbl_out, ma1_ij):
    """p4: PBL records (itype 4, (N,154)) of this substep; pbl_out: dict returned by pbl_compare.run(p4) (needs 'ws0', 'ws');
    ma1_ij: (JM, IM) atmosphere MA(1) at SURFACE entry.  Returns dict(pres, vs0, gusti, ma1) arrays (N,)."""
    i = p4[:, 0].astype(int) - 1
    j = p4[:, 1].astype(int) - 1
    return dict(pres=np.array(p4[:, P_PSURF]), vs0=np.asarray(pbl_out['ws0']), gusti=np.array(p4[:, P_GUSTI_IN]), ma1=np.asarray(ma1_ij)[j, i])


def apply_to_g(g, left):
    """ffg-layout copy with the four leftover columns replaced."""
    g = np.array(g, dtype=np.float64)
    g[:, G_PRES], g[:, G_VS0], g[:, G_GUSTI], g[:, G_MA1] = left['pres'], left['vs0'], left['gusti'], left['ma1']
    return g


def land_substep_v2(p4, g, q1, trup, ma1_ij, dtsurf=900.0, dyn=None, set_elhx=False):
    """land_chain.land_substep with pres / vs0 / gusti / ma1 (and optionally the elhx of the PBL rows) computed here.  Same return contract.
    Not a replacement of the existing function (no existing file is edited); the arithmetic after the leftovers is that function's."""
    import land_chain as LC
    import pbl_compare as PC
    assert np.array_equal(p4[:, :2], g[:, :2])
    p4 = np.array(p4, dtype=np.float64)
    if set_elhx:
        p4[:, P_ELHX] = land_elhx_from_tg(p4[:, P_TG])
    out = PC.run(p4)
    left = land_leftovers(p4, out, ma1_ij)
    g2 = apply_to_g(g, left)
    ps = p4[:, 16]
    tsv, qsrf = out["tsv"], out["qsrf"]
    rho = 100.0 * ps / (LC.RGAS * tsv)
    ddml = p4[:, 23] > 0.5
    ma1 = left['ma1']
    forcing = dict(ts=tsv / (1.0 + qsrf * LC.XDELT), qs=qsrf, rho=rho, ch=out["ch"], vs=out["ws"],
                   tprime=np.where(ddml, p4[:, 25] - p4[:, 7], 0.0), qprime=np.where(ddml, p4[:, 26] - p4[:, 39], 0.0),
                   qm1=q1 * ma1, pres=left['pres'], vs0=left['vs0'], gusti=left['gusti'])
    ghy, _ = LC.run_ghy(g2, forcing, dyn)
    rcdmws = out["cm"] * out["ws"] * rho
    dlw = dtsurf * (trup - LC.STBO * (ghy["tbcs"] + LC.TF) ** 4)
    patch = dict(uflux1=rcdmws * out["us"], vflux1=rcdmws * out["vs"],
                 dth1=-(-ghy["ashg"] + dlw) / (LC.SHA * ma1), dq1=ghy["aevap"] / ma1, tsavg=tsv, qsavg=qsrf)
    return dict(patch=patch, pbl=out, ghy=ghy, rho=rho, dyn_next={k: ghy[k] for k in LC.DYN_KEYS},
                evap_max_ij=ghy["evap_max_ij"], fr_sat_ij=ghy["fr_sat_ij"], leftovers=left,
                elhx=np.array(p4[:, P_ELHX]))
