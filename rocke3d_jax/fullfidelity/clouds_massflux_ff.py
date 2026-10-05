"""Full-fidelity port of CLOUDS2.F90 `MASS_FLUX` (lines 5683-5804) -- D93.

Cloud-base closure: a fixed-count (8) bisection-like iteration on the plume mass fraction FPLUME that restores
the cloud-base moist static energy balance DMSE1 to |DMSE1|<=1d-3.  Vectorised over a flat record axis; the
Fortran control flow is reproduced with per-record masks:
  * `do ITER=1,8`: a record stays active until it executes the Fortran `exit` (|DMSE1|<=1.d-3) or reaches 8;
    once inactive its FPLUME/FMP2/DQSUM/TNX/QNX/DMSE1 are frozen (the Fortran leaves them at the exit values);
  * `if(DQSUM.gt.0.)` evaporation block and its nested `if(DQSUM.gt.0.)` (lower layer) are masks;
  * after the 8th iteration (no exit) FPLUME is still updated once more by +-DFP while FMP2/DQSUM stay from
    iteration 8 (the Fortran does the same; returned `iters` = 9 in that case, like the Fortran ITER).
Module state read by the Fortran (host-associated arrays AIRM, BYAM, SM, QM, PLK, PL at LMIN..LMIN+2) is passed
explicitly: airm0=AIRM(LMIN), airm1=AIRM(LMIN+1), byam0/1/2=BYAM(LMIN..LMIN+2), sm2=SM(LMIN+2), qm2=QM(LMIN+2),
plk0/1=PLK(LMIN, LMIN+1), pl0/1=PL(LMIN, LMIN+1).  SLHE = LHE*BYSHA (a module parameter) is used in the DMSE1
residual while the evaporation/plume terms use the dummy argument SLH (= LHX*BYSHA).  TNX/QNX are module scalars
side-effected by MASS_FLUX; their final values are returned (only meaningful if some iteration executed the
DQSUM>0 block).
Single-precision literals: .25, 0.5, .5, 1., 8 are exact in REAL(4); 1.d-3 is double -> no f4() corrections.
QSAT/DQSATDT are imported from clouds_dq_ff (D89, validated); THBAR (shared/Utilities.F90:5-31) is ported here.
"""
import numpy as np
import clouds_dq_ff as dq

DELTX = 1.0 / dq.MRAT - 1.0           # Constants_mod: deltx = bymrat-1., bymrat = 1./mrat
SLHE = 2.5e6 * dq.BYSHA               # CLOUDS2.F90:99  SLHE = LHE*BYSHA
NITER = 8
TOL = 1.0e-3

_A = 113.4977618974100
_B = 438.5012518098521
_C = 88.49964112645850
_D = -11.50111432385882
_E = 30.00033943846368
_F = 299.9975118132485
_G = 299.9994728900967


def thbar(x, y):
    """shared/Utilities.F90 THBAR: mean temperature used in vertical differencing."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    q = x / y
    al = (_A + q * (_B + q * (_C + q * (_D + q)))) / (_E + q * (_F + _G * q))
    return x * al


def mass_flux(lmin, lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg,
              airm0, airm1, byam0, byam1, byam2, sm2, qm2, plk0, plk1, pl0, pl1,
              niter=NITER, tol=TOL, slhe=SLHE, deltx=DELTX, nested=True, evap=True):
    """-> dict(fplume, fmp2, dqsum, iters, dmse1, tnx, qnx).  `niter`, `tol`, `nested`, `evap` exist only for
    mutation tests."""
    args = [np.asarray(a, dtype=np.float64) for a in (lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg,
                                                       airm0, airm1, byam0, byam1, byam2, sm2, qm2, plk0, plk1,
                                                       pl0, pl1)]
    (lhx, qmo1, qmo2, smo1, smo2, slh, wmdn, wmup, wmedg, airm0, airm1, byam0, byam1, byam2, sm2, qm2,
     plk0, plk1, pl0, pl1) = np.broadcast_arrays(*args)
    shp = lhx.shape
    fplume = np.full(shp, 0.25)
    dfp = np.full(shp, 0.25)
    fmp2 = np.zeros(shp)
    dqsum = np.zeros(shp)
    dmse1 = np.zeros(shp)
    tnx = np.full(shp, np.nan)
    qnx = np.full(shp, np.nan)
    iters = np.full(shp, niter + 1)
    active = np.ones(shp, bool)
    with np.errstate(all="ignore"):
        for it in range(1, niter + 1):
            a = active
            dfp_n = dfp * 0.5
            fmp2_n = fplume * airm0
            frat1 = fmp2_n * byam1
            frat2 = fmp2_n * byam2
            smn1 = smo1 * (1.0 - fplume) + frat1 * smo2
            qmn1 = qmo1 * (1.0 - fplume) + frat1 * qmo2
            smn2 = smo2 * (1.0 - frat1) + frat2 * sm2
            qmn2 = qmo2 * (1.0 - frat1) + frat2 * qm2
            smp = smo1 * fplume
            qmp = qmo1 * fplume
            tp = smo1 * plk1 * byam0
            qsatmp = fmp2_n * dq.qsat(tp, lhx, pl1)
            gama = slh * qsatmp * dq.dqsatdt(tp, lhx) / fmp2_n
            dqsum_n = (qmp - qsatmp) / (1.0 + gama)
            blk = (dqsum_n > 0.0) if evap else np.zeros(shp, bool)
            fevap = 0.5 * fplume
            # upper layer (LMIN+1) re-evaporation
            mcloud = fevap * airm1
            tnx1 = smo2 * plk1 * byam1
            qnx1 = qmo2 * byam1
            qsatc = dq.qsat(tnx1, lhx, pl1)
            dqe = mcloud * (qsatc - qnx1) / (1.0 + slh * qsatc * dq.dqsatdt(tnx1, lhx))
            dqe = np.where(dqe > dqsum_n, dqsum_n, dqe)
            smn2 = np.where(blk, smn2 - slh * dqe / plk1, smn2)
            qmn2 = np.where(blk, qmn2 + dqe, qmn2)
            dqsum_b = np.where(blk, dqsum_n - dqe, dqsum_n)
            tnx_n = np.where(blk, tnx1, tnx)
            qnx_n = np.where(blk, qnx1, qnx)
            # lower layer (LMIN) re-evaporation, nested under the first block
            blk2 = blk & (dqsum_b > 0.0) if nested else np.zeros(shp, bool)
            mcloud2 = fevap * airm0
            tnx0 = smo1 * plk0 * byam0
            qnx0 = qmo1 * byam0
            qsatc2 = dq.qsat(tnx0, lhx, pl0)
            dq2 = mcloud2 * (qsatc2 - qnx0) / (1.0 + slh * qsatc2 * dq.dqsatdt(tnx0, lhx))
            dq2 = np.where(dq2 > dqsum_b, dqsum_b, dq2)
            smn1 = np.where(blk2, smn1 - slh * dq2 / plk0, smn1)
            qmn1 = np.where(blk2, qmn1 + dq2, qmn1)
            tnx_n = np.where(blk2, tnx0, tnx_n)
            qnx_n = np.where(blk2, qnx0, qnx_n)
            sdn = smn1 * byam0
            sup = smn2 * byam1
            sedge = thbar(sup, sdn)
            qdn = qmn1 * byam0
            qup = qmn2 * byam1
            svdn = sdn * (1.0 + deltx * qdn - wmdn)
            svup = sup * (1.0 + deltx * qup - wmup)
            qedge = 0.5 * (qup + qdn)
            svedg = sedge * (1.0 + deltx * qedge - wmedg)
            dmse = ((svup - svedg) * plk1 + (svedg - svdn) * plk0
                    + slhe * (dq.qsat(sup * plk1, lhx, pl1) - qdn))
            # commit this iteration for active records
            dfp = np.where(a, dfp_n, dfp)
            fmp2 = np.where(a, fmp2_n, fmp2)
            dqsum = np.where(a, dqsum_b, dqsum)
            tnx = np.where(a, tnx_n, tnx)
            qnx = np.where(a, qnx_n, qnx)
            dmse1 = np.where(a, dmse, dmse1)
            ex = a & (np.abs(dmse) <= tol)
            iters = np.where(ex, it, iters)
            up = a & ~ex
            fplume = np.where(up & (dmse > tol), fplume - dfp, fplume)
            fplume = np.where(up & (dmse < -tol), fplume + dfp, fplume)
            active = a & ~ex
    return dict(fplume=fplume, fmp2=fmp2, dqsum=dqsum, iters=iters, dmse1=dmse1, tnx=tnx, qnx=qnx)
