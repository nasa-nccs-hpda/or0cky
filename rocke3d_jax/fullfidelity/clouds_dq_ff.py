"""Full-fidelity port of CLOUDS2.F90 `get_dq_cond` / `get_dq_evap` (lines 7259-7336) -- D89.

Stateless 3-iteration saturation adjustment shared by MSTCNV and LSCOND.  Vectorised over a flat
record axis (all arguments broadcastable float64 arrays); the Fortran control flow (`if QM>0`,
`if COND>0`, fixed `do N=1,3`, final clamp) is reproduced elementwise with `np.where` masks.

Pure helpers from shared/Utilities.F90 (QSAT :33-51, DQSATDT :53-67) and the constants from
shared/Constants_mod.F90 are rebuilt here with the same operation order as the Fortran parameters:
    A = 6.108d0*MRAT, B = 1./(RVAP*TF), C = 1./RVAP, BYSHA = 1./SHA
    QSAT   = A*exp(LH*(B-C/max(130.d0,TM)))/PR
    DQSATDT= LH*C/(TM*TM)
Single-precision-literal check (D54 hazard): every un-suffixed literal in these routines and helpers
(`1.`, `0.`) is an exactly representable integer, so promoting a REAL(4) 1. or 0. to double changes
nothing; the only non-trivial literals (`6.108d0`, `130.d0`, `0d0`) carry d0 suffixes.  Constants
that DO contain `1.`-style literals (kapa=(srat-1.)/srat, bysha=1./sha) are exact for the same
reason.  No D54-style correction is needed or applied.

Constants (`mrat`, `rvap`, `tf`, `bysha`) are derived from the Constants_mod.F90 definitions
(non-PLANET_PARAMS branch, gasc=8.314510, mair=28.9655, mwat=18.015, srat=1.401, tf=273.15) and are
checked against the values the instrumented real model wrote (ffc_dq_consts.txt) by
clouds_dq_compare.py.
"""
import numpy as np

GASC = 8.314510
MAIR = 28.9655
MWAT = 18.015
SRAT = 1.401
TF = 273.15
RGAS = 1e3 * GASC / MAIR
RVAP = 1e3 * GASC / MWAT
MRAT = MWAT / MAIR
KAPA = (SRAT - 1.0) / SRAT
SHA = RGAS / KAPA
BYSHA = 1.0 / SHA

_A = 6.108 * MRAT
_B = 1.0 / (RVAP * TF)
_C = 1.0 / RVAP

NITER = 3


def qsat(tm, lh, pr):
    """Utilities.F90 QSAT: saturation vapour mixing ratio."""
    tm = np.asarray(tm, dtype=np.float64)
    return _A * np.exp(lh * (_B - _C / np.maximum(130.0, tm))) / pr


def dqsatdt(tm, lh):
    """Utilities.F90 DQSATDT (factor only; multiply by QSAT)."""
    tm = np.asarray(tm, dtype=np.float64)
    return lh * _C / (tm * tm)


def _adjust(sm, qm, plk, mass, lhx, pl, sign, niter=NITER, bysha=BYSHA):
    sm, qm, plk, mass, lhx, pl = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64)
                                                       for a in (sm, qm, plk, mass, lhx, pl)])
    slh = lhx * bysha
    qmt = qm.copy()
    tp = sm * plk / mass
    dqsum = np.zeros_like(qm)
    for _ in range(niter):
        qst = qsat(tp, lhx, pl)
        dq = (qmt - mass * qst) / (1.0 + slh * qst * dqsatdt(tp, lhx))
        tp = tp + slh * dq / mass
        qmt = qmt - dq
        dqsum = dqsum + sign * dq
    return dqsum


def get_dq_cond(sm, qm, plk, mass, lhx, pl, niter=NITER, bysha=BYSHA, clamp=True, guard=True):
    """CLOUDS2.F90 get_dq_cond -> (dqsum, fcond).  dqsum>0 is condensation.

    `niter`, `clamp`, `guard` exist only so tests can mutate the algorithm (mutation checks)."""
    qm_a = np.asarray(qm, dtype=np.float64)
    with np.errstate(all="ignore"):
        dq = _adjust(sm, qm, plk, mass, lhx, pl, +1.0, niter, bysha)
        qmb = np.broadcast_to(qm_a, dq.shape)
        if clamp:
            dq = np.maximum(0.0, np.minimum(dq, qmb))
        fc = dq / qmb
    act = (qmb > 0) if guard else np.ones(dq.shape, bool)
    return np.where(act, dq, 0.0), np.where(act, fc, 0.0)


def get_dq_evap(sm, qm, plk, mass, lhx, pl, cond, niter=NITER, bysha=BYSHA, clamp=True, guard=True):
    """CLOUDS2.F90 get_dq_evap -> (dqsum, fevp).  dqsum>0 is evaporation of `cond`."""
    cond_a = np.asarray(cond, dtype=np.float64)
    with np.errstate(all="ignore"):
        dq = _adjust(sm, qm, plk, mass, lhx, pl, -1.0, niter, bysha)
        cb = np.broadcast_to(cond_a, dq.shape)
        if clamp:
            dq = np.maximum(0.0, np.minimum(dq, cb))
        fe = dq / cb
    act = (cb > 0) if guard else np.ones(dq.shape, bool)
    return np.where(act, dq, 0.0), np.where(act, fe, 0.0)
