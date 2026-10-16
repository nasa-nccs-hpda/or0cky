"""OCONV per-column setup block -> KPPMIX inputs (D59, piece 1 of the OCONV port).

Recorded inputs per call: G, S, PO, MO(I,J,1)=mo1, UL(1:LMO,1:KMUV), RAVM, LMUV, OGEOZ, HOCEAN,
DELTAE/S/M/SR, U2rho, and the seawater-EOS values the block computes (VOLGSP/VOLGS/ALPHAGSP/
BETAGSP/SHCGS are external-table lookups, recorded directly per project precedent).
ZE comes from the D54 KPPMIX record. Outputs: zgrid/hwide/byhwide, shsq, dvsq, dbloc, ritop,
ustar, bo, bosol -- the KPPMIX inputs.
"""
import math

import numpy as np

LMO = 13
EPSLN = 1e-20


JM = 46
IM = 72


def setup_inputs(rec, ze, lmij_, j):
    """rec: dict with recorded setup fields; ze: 0-based (LMO+1); returns dict of KPPMIX inputs."""
    grav = rec["grav"]
    g, s, po = rec["g"], rec["s"], rec["po"]          # 1-based length LMO+1
    ul = rec["ul"]                                    # (LMO+1, KMUV+1) 1-based
    ravm = rec["ravm"]
    # OCNKPP.f:1714 (pole, IM+2 velocity points) vs :1807 (non-pole, 4)
    kmuv = IM + 2 if j == JM else 4
    lmij = lmij_
    # --- zgrid/hwide/byhwide (OCONV:1928-1948, ZSCALE from OGEOZ/HOCEAN) ---
    zscale = (rec["ogeoz"] + grav * rec["hocean"]) / (ze[lmij] * grav)
    zgrid = np.zeros(LMO + 2); hwide = np.zeros(LMO + 2); byhwide = np.zeros(LMO + 2)
    zgrid[0] = EPSLN; hwide[0] = EPSLN; byhwide[0] = 0.0
    for L in range(1, LMO):
        zgrid[L] = -0.5 * zscale * (ze[L - 1] + ze[L])
        hwide[L] = zgrid[L - 1] - zgrid[L]
        byhwide[L] = 1.0 / hwide[L]
    zgrid[LMO] = -0.5 * zscale * (ze[LMO - 1] + ze[LMO])
    hwide[LMO] = zgrid[LMO - 1] - zgrid[LMO]
    byhwide[LMO] = 1.0 / hwide[LMO]
    zgrid[LMO + 1] = -ze[LMO] * zscale
    hwide[LMO + 1] = EPSLN
    byhwide[LMO + 1] = 0.0
    # --- velocity shears (OCONV:1992-2003), sequential accumulation over K ---
    shsq = np.zeros(LMO + 1); dvsq = np.zeros(LMO + 1)
    for L in range(1, lmij):
        acc_s = 0.0; acc_v = 0.0
        for K in range(1, kmuv + 1):
            d1 = ul[L, K] - ul[L + 1, K]
            acc_s = acc_s + ravm[K] * (d1 * d1)
            d2 = ul[1, K] - ul[L, K]
            acc_v = acc_v + ravm[K] * (d2 * d2)
        shsq[L] = acc_s; dvsq[L] = acc_v
    acc = dvsq[lmij]
    for K in range(1, kmuv + 1):
        d = ul[1, K] - ul[lmij, K]
        acc = acc + ravm[K] * (d * d)
    dvsq[lmij] = acc
    # --- density gradients (OCONV:2008-2024), EOS values recorded ---
    byrho = rec["byrho"]; rhom = rec["rhom"]; rho1 = rec["rho1"]
    dbsfc = np.zeros(LMO + 1); dbloc = np.zeros(LMO + 1); ritop = np.zeros(LMO + 1)
    for L in range(2, lmij + 1):
        dbsfc[L] = grav * (1.0 - rho1[L] * byrho[L])
        dbloc[L - 1] = grav * (1.0 - rhom[L] * byrho[L])
        ritop[L] = (zgrid[1] - zgrid[L]) * dbsfc[L]
    # --- surface forcing (OCONV: talpha/sbeta/Ustar/Bo/Bosol) ---
    talpha1 = rec["alpha1"]; sbeta1 = rec["beta1"]; byshc = 1.0 / rec["shc1"]
    ustar = math.sqrt(rec["u2rho"] * byrho[1])
    grav_b2 = grav * (byrho[1] * byrho[1])
    bo = -(grav_b2 * (
        sbeta1 * rec["deltas"] + talpha1 * byshc * rec["deltae"]
        - (sbeta1 * s[1] + talpha1 * byshc * g[1]) * rec["deltam"]))
    bosol = -(((grav_b2 * talpha1) * byshc) * rec["deltasr"])
    return dict(zgrid=zgrid, hwide=hwide, byhwide=byhwide, shsq=shsq, dvsq=dvsq,
                dbloc=dbloc, ritop=ritop, ustar=ustar, bo=bo, bosol=bosol)
