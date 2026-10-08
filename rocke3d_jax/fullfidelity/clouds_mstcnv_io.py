"""Record layouts and loaders for the D110 MSTCNV dumps (instrumentation/CLOUDS2_mstcnv.f90.patch +
ATM_DRV_clouds_mstcnv.f.patch).  The Fortran array constructors in the patch were GENERATED from the tables below
(field name, Fortran expression, Fortran shape), so the layouts here are the single source of truth.

Files (big-endian f8 streams, ff_data/<date>/, written to units 1200-1202):
  ffc_mc_cols_<itime>.bin   one record per sampled MSTCNV call (CLOUDS2_DRV.F90:896).  Record =
        header(8): itime, i, j, ncall (running call count in the step), conv (1 if LMCMAX>0 at exit), nin, nout, ckflag
        followed by the IN fields then the OUT fields (tables IN_FIELDS / OUT_FIELDS, Fortran column-major flattening;
        (9,40) arrays are index [moment, layer], (4,40) arrays are [k-sample, layer]).
        Convective calls are recorded with stride FFM_CSTRIDE (default 1 = every one), non-convective ones with
        stride FFM_NSTRIDE (default 25).
  ffc_mc_ck_<itime>.bin     internal checkpoints for convective calls with mod(ncall, FFM_CKSTRIDE) == 0.  Block =
        header(8): itime, i, j, ncall, nev, nwords, overflow, 0; then nev events, each
        stage, lmin, ic, nppl, n, payload(n).  Stage payloads: tables CK_FIELDS[stage].
        stages: 1 cloud-base DMSE test; 2 after MASS_FLUX; 3 plume set-up (before the ascent; cycle if absent next);
        4 end of ascent; 5 end of downdraft; 6 end of subsidence; 7 before the precip/evap loop; 8 end of one cloud
        type (after precip loop and PRCPMC); 9 after the cloud-base-loop post-processing (before optical thickness).
  ffc_mc_consts.txt         name value text lines (constants, tunables, strides).
K-samples: UM/VM/U_0/V_0/RA are recorded for the 4 indices KS = (1, 1+(KMAX-1)/3, 1+2*(KMAX-1)/3, KMAX) only (the
Fortran K loops are independent of each other; sampling is by design).  ks = values recorded in the 'kmax' field.
LM = 40, NMOM = 9; arrays index 0..LM-1 = layers 1..LM; (LM+1) arrays index 0..LM = layers 1..LM+1.
"""
import glob
import os

import numpy as np

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
LM = 40
NMOM = 9
NKS = 4
HDR = 8
CKHDR = 8
EVHDR = 5

L1 = (LM + 1,)
LL = (LM,)
M9 = (NMOM, LM)
K4 = (NKS, LM)
KS4 = (NKS,)
MC = lambda s: "merge(1d0,0d0,%s)" % s  # noqa: E731

IN_FIELDS = [
    ("pearth", "PEARTH", ()), ("pland", "PLAND", ()), ("dcl", "dble(DCL)", ()), ("lmcm", "dble(LMCM)", ()),
    ("xmass", "XMASS", ()), ("bydtsrc", "BYDTsrc", ()), ("dtsrc", "DTsrc", ()), ("bybr", "BYBR", ()),
    ("kmax", "dble(KMAX)", ()),
    ("pl", "PL", LL), ("ple", "PLE", L1), ("plk", "PLK", LL), ("airm", "AIRM", LL), ("byam", "BYAM", LL),
    ("etal", "ETAL", LL), ("tl", "TL", LL), ("tvl", "TVL", LL), ("sm", "SM", LL), ("qm", "QM", LL),
    ("qcll", "QCLL", LL), ("qcil", "QCIL", LL), ("sdl", "SDL", LL), ("wturb", "WTURB", LL), ("gzl", "GZL", LL),
    ("smom", "SMOM", M9), ("qmom", "QMOM", M9),
    ("ra", "RA(ffm_ks)", KS4), ("um", "UM(ffm_ks,:)", K4), ("vm", "VM(ffm_ks,:)", K4),
    ("u0", "U_0(ffm_ks,:)", K4), ("v0", "V_0(ffm_ks,:)", K4),
]

OUT_FIELDS = [
    ("ierr", "dble(IERR)", ()), ("lerr", "dble(LERR)", ()), ("lmcmin", "dble(LMCMIN)", ()),
    ("lmcmax", "dble(LMCMAX)", ()), ("prcpmc", "PRCPMC", ()), ("cldslwij", "CLDSLWIJ", ()),
    ("clddepij", "CLDDEPIJ", ()), ("airxl", "AIRXL", ()), ("prheat", "PRHEAT", ()), ("wmsum", "WMSUM", ()),
    ("wmctwp", "WMCTWP", ()), ("wmclwp", "WMCLWP", ()), ("fmc1", "FMC1", ()), ("mccont", "dble(MCCONT)", ()),
] + [(n.lower(), n, LL) for n in (
    "TL SM QM FSSL CLDMCL TAUMCL SVLATL SVLAT1 SVWMXL CSIZEL CONDPT VSUBL TPSAV MCFLX DGDSM DGDEEP DGSHLW DPHASE "
    "DPHADEEP DPHASHLW DTOTW DQCOND DGDQM DQMTOTAL DQMSHLW DQMDEEP DQCTOTAL DQCSHLW DQCDEEP DDMFLX TDNL QDNL "
    "U00L QLmc QImc CNVMMRL CONDMMR").split()] + [
    ("lhp", "LHP", L1), ("precnvl", "PRECNVL", L1),
    ("smom", "SMOM", M9), ("qmom", "QMOM", M9), ("um", "UM(ffm_ks,:)", K4), ("vm", "VM(ffm_ks,:)", K4),
]
# fields whose value is stale (module state left from an earlier call) when LMCMIN==0:
STALE_IF_NONCONV = ("airxl", "prheat")


def _arr(names, shape=LL):
    return [(n.lower(), n, shape) for n in names.split()]


CK_FIELDS = {
    1: [("dmse", "DMSE", ()), ("fmp0", "FMP0", ())],
    2: [("fplume", "FPLUME", ()), ("fmp2", "FMP2", ()), ("dqsum", "DQSUM", ())],
    3: [("fctype", "FCTYPE", ()), ("mplum1", "MPLUM1", ()), ("mplume", "MPLUME", ()), ("fmc1", "FMC1", ()),
        ("mccont", "dble(MCCONT)", ()), ("mc1", MC("MC1"), ())],
    4: ([("lmax", "dble(LMAX)", ()), ("mccont", "dble(MCCONT)", ()), ("mc1", MC("MC1"), ()), ("fmc1", "FMC1", ()),
         ("ldraft", "dble(LDRAFT)", ()), ("lfrz", "dble(LFRZ)", ()), ("etadn", "ETADN", ()),
         ("mplume", "MPLUME", ()), ("fplume", "FPLUME", ()), ("smp", "SMP", ()), ("qmp", "QMP", ()),
         ("mpmax", "merge(MPMAX,0d0,LMAX.gt.LMIN)", ()), ("smpmax", "merge(SMPMAX,0d0,LMAX.gt.LMIN)", ()),
         ("qmpmax", "merge(QMPMAX,0d0,LMAX.gt.LMIN)", ()), ("cdhsum", "CDHSUM", ()), ("cdhsum1", "CDHSUM1", ()),
         ("cdhdrt", "CDHDRT", ())]
        + _arr("DM DMR DDR CCM COND CONDP CONDP1 CONDV TAUMC1 CDHEAT ENT DET BUOY WCU TPSAV SMDNL QMDNL DSM DSMR "
               "DQM DQMR VLAT U00L CONDMMR")
        + _arr("DSMOM DSMOMR DQMOM DQMOMR SMOMDNL QMOMDNL", M9)
        + [("smomp", "SMOMP", (NMOM,)), ("qmomp", "QMOMP", (NMOM,)),
           ("smompmax", "merge(SMOMPMAX,0d0*SMOMP,LMAX.gt.LMIN)", (NMOM,)),
           ("qmompmax", "merge(QMOMPMAX,0d0*QMOMP,LMAX.gt.LMIN)", (NMOM,)),
           ("ump", "UMP(ffm_ks)", KS4), ("vmp", "VMP(ffm_ks)", KS4)]
        + [("dum", "DUM(ffm_ks,:)", K4), ("dvm", "DVM(ffm_ks,:)", K4), ("umdnl", "UMDNL(ffm_ks,:)", K4),
           ("vmdnl", "VMDNL(ffm_ks,:)", K4)]),
    5: ([("ldmin", "dble(LDMIN)", ()), ("llmin", "dble(LLMIN)", ()), ("edraft", "merge(EDRAFT,0d0,ETADN.gt.1d-10)", ()),
         ("ddraft", "merge(DDRAFT,0d0,ETADN.gt.1d-10)", ()), ("smdn", "merge(SMDN,0d0,ETADN.gt.1d-10)", ()),
         ("qmdn", "merge(QMDN,0d0,ETADN.gt.1d-10)", ()), ("evpsum", "EVPSUM", ()), ("cdhsum1", "CDHSUM1", ()),
         ("etadn", "ETADN", ())]
        + _arr("DM DSM DQM DMR DSMR DQMR DDM COND CDHEAT TAUMCL TDNL QDNL")
        + _arr("DSMOM DQMOM DSMOMR DQMOMR", M9)
        + [("dum", "DUM(ffm_ks,:)", K4), ("dvm", "DVM(ffm_ks,:)", K4)]),
    6: ([("ksub", "dble(KSUB)", ()), ("ierr", "dble(IERR)", ()), ("lerr", "dble(LERR)", ()),
         ("ldmin", "dble(LDMIN)", ()), ("lmax", "dble(LMAX)", ()), ("cm", "CM", (LM + 1,)), ]
        + _arr("CMNEG SM QM") + _arr("SMOM QMOM", M9)
        + [("um", "UM(ffm_ks,:)", K4), ("vm", "VM(ffm_ks,:)", K4)]),
    7: ([("prcp", "PRCP", ()), ("prheat", "PRHEAT", ()), ("told", "TOLD", ())]
        + _arr("COND CONDP HEAT1 SVLATL SVWMXL CONDPT MCFLX DGDSM DQMTOTAL SM QM VLAT")
        + [("lhp", "LHP", L1)] + _arr("SMOM", M9)),
    8: ([("prcp", "PRCP", ()), ("prcpmc", "PRCPMC", ()), ("cldslwij", "CLDSLWIJ", ()), ("clddepij", "CLDDEPIJ", ()),
         ("lmcmin", "dble(LMCMIN)", ()), ("lmcmax", "dble(LMCMAX)", ()), ("prheat", "PRHEAT", ()),
         ("fmc1", "FMC1", ())]
        + _arr("CLDMCL")
        + [("precnvl", "PRECNVL", L1), ("lhp", "LHP", L1)]
        + _arr("HEAT1 COND SM QM DPHASE DQCOND DQCTOTAL DGDSM DDMFLX SVLATL SVWMXL CONDPT DQMTOTAL")
        + _arr("SMOM QMOM", M9)),
    9: ([("lmcmin", "dble(LMCMIN)", ()), ("lmcmax", "dble(LMCMAX)", ()), ("fmc1", "FMC1", ()),
         ("airxl", "merge(AIRXL,0d0,LMCMIN.gt.0)", ())]
        + _arr("FSSL DGDSM SM")),
}

CONST_NAMES = ("entrainment_cont1 entrainment_cont2 radiusl_multiplier radiusi_multiplier u00a u00b wmu_multiplier "
               "rwcldox rimax mc_fddrt mc_entr_mass_lim_plume mc_new_ddrft_thetav mc_revp_abv_cldbase ccmul ccmul2 "
               "coetau wmu wmul ti cldmin fddet dtmin1 brcld lmcm lm seconds_per_hour").split()
CONST_FEXPR = ("entrainment_cont1 entrainment_cont2 radiusl_multiplier radiusi_multiplier U00a U00b wmu_multiplier "
               "RWCldOX RIMAX MC_FDDRT dble(MC_ENTR_MASS_LIM_PLUME) dble(MC_NEW_DDRFT_THETAV) "
               "dble(MC_REVP_ABV_CLDBASE) CCMUL CCMUL2 COETAU WMU WMUL TI CLDMIN FDDET DTMIN1 BRCLD dble(LMCM) "
               "dble(LM) SECONDS_PER_HOUR").split()


def size(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def total(fields):
    return sum(size(f[2]) for f in fields)


def fortran_constructor(fields, indent="         "):
    """'(/ expr, expr, ... /)' continuation text for a field table."""
    parts = [f[1] for f in fields]
    lines, cur = [], ""
    for p in parts:
        piece = p + ", "
        if len(cur) + len(piece) > 90:
            lines.append(cur)
            cur = ""
        cur += piece
    lines.append(cur.rstrip().rstrip(","))
    return "(/ " + (" &\n" + indent).join(lines) + " /)"


def split_record(vec, fields):
    """Cut a 1-D (or (nrec, n)) array into a dict of arrays shaped (nrec,) + C-order reversed Fortran shape.
    Fortran (9,40) -> numpy (nrec, 40, 9)?  No: returned as (nrec, 40, 9)^T convention is avoided: result keeps
    numpy shape (nrec, LM, NMOM) transposed back to [moment, layer] = vec reshape((nrec,)+shape[::-1]) -> use
    .swapaxes so that out[name][r] has the SAME index order as the Fortran shape (moment first)."""
    v = np.atleast_2d(vec)
    out, p = {}, 0
    for name, _, shape in fields:
        n = size(shape)
        blk = v[:, p:p + n]
        p += n
        if shape == ():
            out[name] = blk[:, 0]
        else:
            # Fortran column-major (first index fastest) -> reshape with reversed dims, then transpose back
            a = blk.reshape((v.shape[0],) + tuple(shape[::-1]))
            out[name] = np.transpose(a, (0,) + tuple(range(len(shape), 0, -1)))
    assert p == v.shape[1], (p, v.shape)
    return out


def load_cols(date, ff=FF_DEFAULT):
    """-> dict with 'hdr' (nrec, 8) columns by name, 'in' and 'out' dicts (see split_record)."""
    files = sorted(glob.glob(f"{ff}/{date}/ffc_mc_cols_[0-9]*.bin"))
    if not files:
        return None
    nin, nout = total(IN_FIELDS), total(OUT_FIELDS)
    rl = HDR + nin + nout
    parts = []
    for f in files:
        raw = np.fromfile(f, dtype=">f8")
        assert raw.size % rl == 0, (f, raw.size, rl)
        parts.append(raw.reshape(-1, rl).astype(np.float64))
    r = np.concatenate(parts)
    hdr = {k: r[:, i] for i, k in enumerate("itime i j ncall conv nin nout ckflag".split())}
    assert np.all(hdr["nin"] == nin) and np.all(hdr["nout"] == nout)
    return dict(hdr=hdr, inp=split_record(r[:, HDR:HDR + nin], IN_FIELDS),
                out=split_record(r[:, HDR + nin:], OUT_FIELDS))


def load_ck(date, ff=FF_DEFAULT):
    """-> list of dict(itime, i, j, ncall, overflow, events=[dict(stage, lmin, ic, nppl, f=dict)])."""
    blocks = []
    for f in sorted(glob.glob(f"{ff}/{date}/ffc_mc_ck_[0-9]*.bin")):
        raw = np.fromfile(f, dtype=">f8").astype(np.float64)
        p = 0
        while p < raw.size:
            h = raw[p:p + CKHDR]
            nev, nw = int(h[4]), int(h[5])
            body = raw[p + CKHDR:p + CKHDR + nw]
            assert body.size == nw, (f, p)
            evs, q = [], 0
            for _ in range(nev):
                st, lmin, ic, nppl, n = (int(x) for x in body[q:q + EVHDR])
                pay = body[q + EVHDR:q + EVHDR + n]
                assert pay.size == n and n == total(CK_FIELDS[st]), (f, st, n, total(CK_FIELDS[st]))
                evs.append(dict(stage=st, lmin=lmin, ic=ic, nppl=nppl, f=split_record(pay, CK_FIELDS[st])))
                q += EVHDR + n
            assert q == nw
            blocks.append(dict(itime=int(h[0]), i=int(h[1]), j=int(h[2]), ncall=int(h[3]), overflow=int(h[6]),
                               events=evs))
            p += CKHDR + nw
    return blocks


def load_consts(date, ff=FF_DEFAULT):
    out = {}
    p = f"{ff}/{date}/ffc_mc_consts.txt"
    if not os.path.exists(p):
        return None
    for ln in open(p):
        t = ln.split()
        if len(t) == 2:
            out[t[0]] = float(t[1])
    return out
