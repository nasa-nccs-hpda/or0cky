"""D186 (stage S2): CONDSE of phase 1 as a DEVICE-RESIDENT program (translation of clouds_condse_batch.condse_step_batch to jnp, statement by statement).

What is on the device (jit units, all arrays stay on the device):
  cse_setup   column set-up (gathers of the 3,168 non-polar columns, CLOUDS2_DRV.F90:680-842) + the SOUTH POLE callback (see below)
  MSTCNV      clouds_mstcnv_jax kernels; the cloud-base loop LMIN=1..LMCM-1 is a HOST loop (see "declared host stages"), the kernels it calls are jitted
  cse_post    convective post-processing, DDML search, LSCOND set-up, LSCOND core (clouds_lscond_jax._core, mode xla), precipitation/energy bookkeeping,
              snow-age exp, stores, hand-off arrays (the 'hand_off_b' arrays of the radiation packet), final T/Q/moment/UKM merge, the NORTH POLE callback,
              momentum back-transfer (avg_replicated_duv_to_vgrid) and recalc_agrid_uv
DECLARED host stages (named in every result):
  pole_south, pole_north   the two pole columns (KMAX=72) run the per-column NumPy port clouds_condse_ff.condse_column through jax.pure_callback
                           (inputs: the pole slices of the entry arrays and the 7 LSCOND module-array vectors; outputs: the pole slices of all X arrays and the
                           module-array vectors).  Not ported to JAX.
  mstcnv_lmin_loop         clouds_mstcnv_jax's host loop over LMIN=1..LMCM-1 (22 iterations at LMCM=23): per iteration one device->host read of a 3,168-element
                           bool mask (compaction to a bucket), then bucket-sized jit calls.  The loop and compaction are the D146 design, not changed.
  mstcnv_qus_subsidence    clouds_mstcnv_batch._adv_groups (NumPy QUS ADV1D of the subsidence), a pure_callback inside the jitted event block.
Stop-model conditions (SUBSID ERROR, cloud-property errors, negative cloud cover, VMP) are returned as device flags, not raised.
Libm mode (jnp.exp / jnp.power, 'xla' mode of LSCOND); the Intel libimf is not used.  Needs the XLA flags of clouds_jax_env_fast.
"""
import clouds_jax_env_fast  # noqa: F401  (XLA flags BEFORE jax)
import functools
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import clouds_condse_ff as cf
import clouds_lscond_batch as lb
import clouds_lscond_jax as lj
import clouds_mstcnv_jax as mj
import jax_p1_glue as G

IM, JM, LM = 72, 46, 40
F = np.float64
GRAV, RGAS, TF, LHE, LHS, LHM = cf.GRAV, cf.RGAS, cf.TF, cf.LHE, cf.LHS, cf.LHM
BYGRAV, DELTX, ENTCON, TINY = cf.BYGRAV, cf.DELTX, cf.ENTCON, cf.TINY
f4 = cf.f4

KEYS3 = ("TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV CLDSAV1 FSS TAUSS TAUSSIP TAUMC CLDSS CLDMC CSIZMC CSIZSS CSIZSSIP QLSS QISS QLMC "
         "QIMC W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE "
         "DIM_ST_WATER DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV LMC SNOAGE TMOM QMOM UKM VKM UKMSP VKMSP UKMNP "
         "VKMNP").split()
KEYS2 = "P_ACC PM_ACC PREC EPREC PRECSS DDM1 DDMS TDN1 QDN1 DDML AIRX".split()
KEYS_OLD = "T Q QCL QCI U V".split()
X_KEYS = KEYS3 + KEYS2 + KEYS_OLD
# entry arrays the pole columns read in addition to the X keys (and the X keys themselves, which carry the initial values of the output slices)
POLE_KEYS = [k for k in X_KEYS if k not in ("UKM", "VKM")] + ["CLDSAV", "DCLEV", "EGCM", "FEARTH", "FLAND", "GZ", "MWS", "PDSIG", "PEDN", "PK", "PMID",
                                                              "PMIDOLD", "QSAVG", "RNDSS", "TSAVG"]
POLE_KEYS = list(dict.fromkeys(POLE_KEYS))
MS_KEYS = ("tausslip", "csizelip", "cldsal", "cldsv1", "dqlsc", "lhp", "prebar1")


def ms_zero():
    return {k: jnp.zeros(LM + 1 if k in ("lhp", "prebar1") else LM) for k in MS_KEYS}


# ============================================================================================================== pole columns (declared host stage)
def pole_slice(x, j):
    """Pole column (i=0, row j) of an entry/output array, any of the layouts of cse_in."""
    if x.shape == (IM, LM):
        return x
    if x.ndim == 2:
        return x[0, j]
    if x.ndim == 3:
        return x[0, j, :] if x.shape[:2] == (IM, JM) else x[:, 0, j]
    if x.ndim == 4:
        return x[:, 0, j, :] if x.shape[1:3] == (IM, JM) else x[:, :, 0, j]
    raise ValueError(x.shape)


def _pole_put(full, sl, j):
    if full.shape == (IM, LM):
        full[...] = sl
    elif full.ndim == 2:
        full[0, j] = sl
    elif full.ndim == 3:
        if full.shape[:2] == (IM, JM):
            full[0, j, :] = sl
        else:
            full[:, 0, j] = sl
    else:
        if full.shape[1:3] == (IM, JM):
            full[:, 0, j, :] = sl
        else:
            full[:, :, 0, j] = sl


class PoleHost:
    """Host side of the pole callbacks: builds a sparse entry dict from the pole slices, runs the per-column NumPy port, returns the pole slices of all
    output arrays and the module-array vectors.  `calls`/`seconds`/`bytes` are the callback accounting."""

    def __init__(self, ctx, shapes):
        self.cfg, self.G = ctx.cfg, ctx.cfg['geom']
        self.shapes = shapes
        self.calls = 0
        self.seconds = 0.0
        self.bytes = 0
        self.log = []

    def run(self, j, a_slices, ms_in):
        import time
        t0 = time.perf_counter()
        cf.set_backend('numpy')
        A = {}
        for k, shp in self.shapes.items():
            arr = np.zeros(shp)
            if k in a_slices:
                _pole_put(arr, np.asarray(a_slices[k]), j)
            A[k] = arr
        X = cf._fresh_outputs(A)
        for k in ("AIRX", "DDM1", "DDMS", "DDML", "TDN1", "QDN1"):
            X[k][:, :] = 0.0
        ms = dict(S={k: (list(np.asarray(ms_in[k], float)) if k in ("dqlsc", "lhp", "prebar1") else np.array(ms_in[k], float)) for k in MS_KEYS})
        cnt = cf.new_counts()
        cf.condse_column(X, A, self.G, self.cfg, 0, j, ms, cnt)
        out = {k: np.array(pole_slice(X[k], j), dtype=np.float64, copy=True) for k in a_slices if k in X}
        msout = {k: np.asarray(ms['S'][k], dtype=np.float64) for k in MS_KEYS}
        dt = time.perf_counter() - t0
        self.calls += 1
        self.seconds += dt
        nb = sum(np.asarray(v).nbytes for v in a_slices.values()) + sum(v.nbytes for v in out.values()) + sum(np.asarray(v).nbytes for v in ms_in.values()) \
            + sum(v.nbytes for v in msout.values())
        self.bytes += nb
        self.log.append(dict(pole=j, seconds=dt, bytes=nb, columns=cnt.get('columns', 0), ierr=(cnt.get('mc_ierr', 0), cnt.get('ls_ierr', 0))))
        return out, msout


def pole_call(host, j, A, ms):
    """Inside jit: pure_callback to the pole column.  A: dict of device entry arrays (the POLE_KEYS); returns (dict of X pole slices, ms vectors)."""
    sl = {k: pole_slice(A[k], j) for k in POLE_KEYS if k in A}
    out_spec = ({k: jax.ShapeDtypeStruct(v.shape, jnp.float64) for k, v in sl.items() if k in host.out_keys},
                {k: jax.ShapeDtypeStruct(v.shape, jnp.float64) for k, v in ms.items()})

    def cb(a, m):
        return host.run(j, a, m)
    return jax.pure_callback(cb, out_spec, sl, ms)


# ============================================================================================================== device CONDSE
class CondseDevice:
    def __init__(self, ctx):
        self.ctx = ctx
        cfg = ctx.cfg
        Gm = cfg['geom']
        self.cfg = cfg
        I = np.array([i for j in range(1, JM - 1) for i in range(int(Gm['IMAXJ'][j]))])
        J = np.array([j for j in range(1, JM - 1) for i in range(int(Gm['IMAXJ'][j]))])
        assert (np.asarray(Gm['KMAXJ'])[1:JM - 1] == 4).all()
        self.I, self.J, self.N = I, J, I.size
        self.axyp = jnp.asarray(np.asarray(Gm['AXYP'], dtype=np.float64)[I, J])
        self.byaxyp = jnp.asarray(np.asarray(Gm['BYAXYP'], dtype=np.float64)[I, J])
        self.ra = jnp.asarray(np.asarray(Gm['RAVJ'], dtype=np.float64)[:4, J])
        self.agrid = G.make_agrid_consts(cfg['glue_geom'])
        self.Kls = {n: jnp.asarray(v) for n, v in lj.make_K().items()}
        self.host = None
        self.out_keys = None
        self._setup = jax.jit(self._setup_fn)
        from types import SimpleNamespace as _NS
        self._mc_setup = jax.jit(lambda K, R, lmcm: mj._setup(_NS(**K), R, lmcm), static_argnums=2)
        self._post = jax.jit(self._post_fn)

    # ---------------------------------------------------------------- host side bring-up
    def bind(self, A):
        """Create the pole host from the shapes of the entry arrays (call once with the device dict of entry arrays)."""
        shapes = {k: tuple(A[k].shape) for k in X_KEYS + ["CLDSAV", "DCLEV", "EGCM", "FEARTH", "FLAND", "GZ", "MWS", "PDSIG", "PEDN", "PK", "PMID", "PMIDOLD",
                                                      "QSAVG", "RNDSS", "TSAVG"] if k in A}
        self.host = PoleHost(self.ctx, shapes)
        self.host.out_keys = set(k for k in X_KEYS if k in A and k not in ("UKM", "VKM"))
        return self.host

    # ---------------------------------------------------------------- unit 1: set-up
    def _setup_fn(self, A, ms):
        cfg, I, J, N = self.cfg, self.I, self.J, self.N
        dtsrc, bydtsrc, xmass, bybr = cfg['dtsrc'], cfg['bydtsrc'], cfg['xmass'], cfg['bybr']
        g = lambda a: jnp.transpose(a[I, J, :])            # (IM,JM,LM) -> (LM,N)   # noqa: E731
        g2 = lambda a: a[:, I, J]                          # (L,IM,JM)  -> (L,N)    # noqa: E731
        pearth, pland = A['FEARTH'][I, J], A['FLAND'][I, J]
        ts, qs = A['TSAVG'][I, J], A['QSAVG'][I, J]
        tsv = ts * (1 + qs * DELTX)
        dcl = (A['DCLEV'][I, J] + 0.5).astype(jnp.int64)
        pl, ple, plk, airm = g2(A['PMID']), g2(A['PEDN']), g2(A['PK']), g2(A['PDSIG'])
        byam = 1.0 / airm
        wturb = jnp.sqrt(F(f4(0.6666667)) * g2(A['EGCM']))
        dpdt = (pl - g2(A['PMIDOLD'])) * bydtsrc
        t, q = g(A['T']), g(A['Q'])
        sm = t * airm
        smom = jnp.transpose(A['TMOM'][:, I, J, :], (0, 2, 1)) * airm[None]
        tl = t * plk
        qm = q * airm
        qmom = jnp.transpose(A['QMOM'][:, I, J, :], (0, 2, 1)) * airm[None]
        qcll, qcil = g(A['QCL']), g(A['QCI'])
        sdl = g(A['MWS']) / dtsrc * self.byaxyp[None, :]
        tvl = tl * (1.0 + DELTX * q)
        gz = g(A['GZ'])
        etal = jnp.zeros((LM, N)).at[1:LM - 1].set(0.5 * ENTCON * (gz[2:LM] - gz[0:LM - 2]) * 1.0e-3 * BYGRAV)
        gzl = jnp.zeros((LM, N)).at[1:LM - 1].set(etal[1:LM - 1] / ENTCON)
        etal = etal.at[LM - 1].set(etal[LM - 2])
        gzl = gzl.at[LM - 1].set(gzl[LM - 2])
        u0, v0 = A['UKM'][:, :, I, J], A['VKM'][:, :, I, J]
        um, vm = u0 * airm[None], v0 * airm[None]
        tprcp = t[0] * plk[0] - TF
        R = dict(pearth=pearth, pland=pland,
                 dcl=dcl, pl=pl, plk=plk, airm=airm, byam=byam, etal=etal, tl=tl, tvl=tvl, sm=sm, qm=qm, qcll=qcll,
                 qcil=qcil, sdl=sdl, wturb=wturb, gzl=gzl, ple=ple, smom=smom, qmom=qmom, um=um, vm=vm, u0=u0, v0=v0, ra=self.ra)
        W0 = dict(pearth=pearth, tsv=tsv, dcl=dcl, pl=pl, ple=ple, plk=plk, airm=airm, byam=byam, dpdt=dpdt, t=t, q=q, sm=sm, qm=qm, smom=smom,
                  qmom=qmom, qcll=qcll, qcil=qcil, sdl=sdl, um=um, vm=vm, u0=u0, v0=v0, tprcp=tprcp)
        # south pole (Fortran order: first)
        Xs, ms_s = pole_call(self.host, 0, A, ms)
        return R, W0, Xs, ms_s

    def setup(self, A, ms):
        return self._setup(A, ms)

    # ---------------------------------------------------------------- MSTCNV host loop (declared)
    def mstcnv(self, R, stats=None):
        """clouds_mstcnv_jax.mstcnv_jax with the cloud-base loop unchanged and the 'negative cloud cover' error accumulated on the device (returned as a flag)
        instead of a per-iteration host read.  The per-iteration host read of the column mask (compaction) remains."""
        c = self.cfg['tune']
        N = self.N
        lmcm = int(self.cfg['lmcm'])
        K = mj.make_K(c)
        for n_ in ("xmass", "bydtsrc", "dtsrc", "bybr"):
            K[n_] = np.float64(self.cfg[n_])
        K['fmpscale'] = np.float64(min(1.0, F(self.cfg['dtsrc']) / (F(1.0) * mj.mc.SECONDS_PER_HOUR)))
        Kj = {n_: jnp.asarray(v) for n_, v in K.items()}
        from types import SimpleNamespace
        k = SimpleNamespace(**Kj)
        S, Sc, I_, aux = self._mc_setup(Kj, R, lmcm)
        err_all = jnp.zeros((), bool)
        nev = 0
        syncs = 0
        for lmin in range(1, lmcm):
            mask, fmp0, fmp2 = mj._base(Kj, lmin, S, I_)
            idx = np.flatnonzero(jax.device_get(mask))          # the declared per-iteration device->host read (counted)
            syncs += 1
            if not idx.size:
                continue
            nev += idx.size
            nb = next((b for b in mj.BUCKETS if b >= idx.size), N)
            idxg = np.zeros(nb, np.int64)
            idxg[:idx.size] = idx
            idxs = np.full(nb, N, np.int64)
            idxs[:idx.size] = idx
            valid = np.arange(nb) < idx.size
            S, Sc, err = mj._event_call(Kj, lmin, fmp0, fmp2, jnp.asarray(idxg), jnp.asarray(idxs), jnp.asarray(valid), S, Sc, I_, aux)
            err_all = err_all | err
        if stats is not None:
            stats['events'] = nev
            stats['lmin_host_syncs'] = syncs
        o = mj._post(Kj, S, Sc, I_, aux)
        return o, err_all

    # ---------------------------------------------------------------- unit 3: post-processing
    def _post_fn(self, A, W0, o, ms_s, Xs, Kls):
        cfg, I, J, N = self.cfg, self.I, self.J, self.N
        dtsrc, bydtsrc, bybr, lmcld = cfg['dtsrc'], cfg['bydtsrc'], cfg['bybr'], cfg['lmcld']
        pearth, tsv, dcl, pl, ple, plk, airm, byam = (W0[k] for k in ('pearth', 'tsv', 'dcl', 'pl', 'ple', 'plk', 'airm', 'byam'))
        t, q, sm, qm, smom, qmom, qcll, qcil, sdl, um, vm, u0, v0, tprcp = (W0[k] for k in ('t', 'q', 'sm', 'qm', 'smom', 'qmom', 'qcll', 'qcil', 'sdl', 'um', 'vm', 'u0', 'v0', 'tprcp'))
        axyp = self.axyp
        flags = {}
        # ---- fresh outputs (clouds_condse_ff._fresh_outputs) and the zeroing before the south pole
        X = {k: A[k] for k in X_KEYS}
        X['FSS'] = jnp.ones_like(A['FSS'])
        X['TLS'], X['QLS'], X['TMC'], X['QMC'] = A['T'], A['Q'], A['T'], A['Q']
        for k in ("AIRX", "DDM1", "DDMS", "DDML", "TDN1", "QDN1"):
            X[k] = jnp.zeros_like(A[k])
        flags['subsid_error'] = (o['ierr'] == 2).any().astype(jnp.int32)
        lmcmin, lmcmax = o['lmcmin'].astype(jnp.int64), o['lmcmax'].astype(jnp.int64)
        conv = lmcmin > 0
        Lr = jnp.arange(LM)[:, None]
        inmc = (Lr < lmcmax[None, :])
        sm_m, qm_m = o['sm'].T, o['qm'].T
        tmc = jnp.where(inmc, sm_m * byam, t)
        qmc = jnp.where(inmc, qm_m * byam, q)
        smom_mc = jnp.where(inmc[None], jnp.transpose(o['smom'], (1, 2, 0)), smom)
        qmom_mc = jnp.where(inmc[None], jnp.transpose(o['qmom'], (1, 2, 0)), qmom)
        um1 = jnp.where(inmc[None], jnp.transpose(o['um'], (1, 2, 0)), um)
        vm1 = jnp.where(inmc[None], jnp.transpose(o['vm'], (1, 2, 0)), vm)
        prcpmc = o['prcpmc']
        prcp = jnp.where(conv, prcpmc * 100.0 * BYGRAV, 0.0)
        enrgp = jnp.zeros(N)
        neg = conv & ~(tprcp > 0)
        enrgp = jnp.where(neg, (enrgp + 0.0) - prcp * LHM, enrgp)
        X['CSIZMC'] = X['CSIZMC'].at[:, I, J].set(jnp.where(inmc, o['csizel'].T, X['CSIZMC'][:, I, J]))
        fssl = jnp.where(conv[None, :], o['fssl'].T, 1.0)
        X['FSS'] = X['FSS'].at[:, I, J].set(jnp.where(conv[None, :], o['fssl'].T, X['FSS'][:, I, J]))
        X['AIRX'] = X['AIRX'].at[I, J].set(jnp.where(conv, o['airxl'] * axyp, 0.0))
        ddm = o['ddmflx']
        posd = (ddm > 0.0) & ((jnp.arange(1, LM + 1)[None, :]) <= dcl[:, None])
        ddml = jnp.where(posd.any(axis=1), posd.argmax(axis=1) + 1, dcl)
        cols = jnp.arange(N)
        X['DDML'] = X['DDML'].at[I, J].set(jnp.where(conv, ddml, 0.0))
        X['TDN1'] = X['TDN1'].at[I, J].set(jnp.where(conv, o['tdnl'][cols, ddml - 1], 0.0))
        X['QDN1'] = X['QDN1'].at[I, J].set(jnp.where(conv, o['qdnl'][cols, ddml - 1], 0.0))
        X['DDMS'] = X['DDMS'].at[I, J].set(jnp.where(conv, -100.0 * ddm[cols, ddml - 1] / (GRAV * dtsrc), 0.0))
        X['DDM1'] = X['DDM1'].at[I, J].set(jnp.where(conv, ddm[:, 0] * RGAS * tsv / (GRAV * ple[0] * dtsrc), 0.0))
        X['LMC'] = X['LMC'].at[0, I, J].set(lmcmin.astype(jnp.float64)).at[1, I, J].set((lmcmax + 1).astype(jnp.float64))
        # ---- LSCOND set-up (1259-1284)
        svlatl, svwmxl = o['svlatl'].T, o['svwmxl'].T
        qclx = jnp.where(inmc & (svlatl == LHE), qcll + svwmxl, qcll)
        qcix = jnp.where(inmc & (svlatl == LHS), qcil + svwmxl, qcil)
        ql_ls = q
        aq = (ql_ls - A['QTOLD'][:, I, J]) * bydtsrc
        S = dict(qcll=qcll, qcil=qcil, svlatl=svlatl, svlat1=o['svlat1'].T, svwmxl=svwmxl, sdl=sdl, vsubl=o['vsubl'].T, fssl=fssl,
                 ttoldl=A['TTOLD'][:, I, J], aq=aq, dpdt=W0['dpdt'], pl=pl, plk=plk, airm=airm, byam=byam, u00l=o['u00l'].T, taumcl=o['taumcl'].T,
                 precnvl=o['precnvl'].T, tl=t * plk, ql=ql_ls, th=t, rh=A['RHSAV'][:, I, J], qclx=qclx, qcix=qcix, svlhxl=A['SVLHX'][:, I, J],
                 cldsavl=A['CLDSAV'][:, I, J], csizel=o['csizel'].T, sm=sm_m, qm=qm_m, qmom=jnp.transpose(qmom, (1, 0, 2)),
                 smom=jnp.transpose(smom, (1, 0, 2)), um=jnp.transpose(um, (1, 0, 2)), vm=jnp.transpose(vm, (1, 0, 2)))
        S['pdsigl00'] = jnp.asarray(cf.PDSIGL00)
        S['dqlsc'] = jnp.zeros((LM, N))
        S['tausslip_init'], S['csizelip_init'], S['cldsal_init'], S['cldsv1_init'] = ms_s['tausslip'], ms_s['csizelip'], ms_s['cldsal'], ms_s['cldsv1']
        S['tausslip'] = jnp.tile(ms_s['tausslip'][:, None], (1, N))
        S['cldsv1'] = jnp.tile(ms_s['cldsv1'][:, None], (1, N))
        c = cf.LS_CONST
        sndo = 59.68 / (c['rwcldox'] ** 3)
        P = {k: c[k] for k in ("cmx", "u00a", "rimax", "rwmax", "rwcldox", "rcldix")}
        P.update(rcldlx=c['rcldlx'], wmui=cf.WMUIX * float(f4(0.001)), scdnci=0.06417127, bybr=bybr, bydtsrc=bydtsrc, dtsrc=dtsrc, lmcld=lmcld,
                 pearth=pearth, wconst=lb.L0.WMU * (1.0 - pearth) + lb.L0.WMUL * pearth, scdncw=sndo * (1.0 - pearth) + 174.0 * pearth, dcl=dcl, ra=self.ra)
        Sj, Pj, lm_ = lj.prepare(S, P)
        Sout, Wout = lj._core(Sj, Pj, Kls, lm_, "xla")
        flags['lscond_vmp'] = Wout['vmp_err'].astype(jnp.int32)
        Sx = Sout
        # ---- bookkeeping (1442-1464)
        prcpss = Wout['prcpss']
        lhp = Sx['lhp']
        prcp = prcp + prcpss * 100.0 * BYGRAV
        ice0 = lhp[0] == LHS
        enrgp = jnp.where(ice0, (enrgp + 0.0) - prcpss * 100.0 * BYGRAV * LHM, enrgp + 0.0)
        sa = enrgp < 0.0
        e = jnp.exp(-prcp)
        sn0 = X['SNOAGE'][:, I, J]
        X['SNOAGE'] = X['SNOAGE'].at[:, I, J].set(jnp.where(sa[None, :], sn0 * e[None, :], sn0))
        # ---- stores (1932-1945)
        cldmcl = o['cldmcl'].T
        st = lambda key, val: X.__setitem__(key, X[key].at[:, I, J].set(val))     # noqa: E731
        st('TAUMC', o['taumcl'].T)
        st('CLDMC', cldmcl)
        st('SVLAT', svlatl)
        cldssl = Sx['cldssl']
        st('TAUSS', Sx['taussl'])
        st('CLDSS', cldssl)
        st('CLDSAV', Sx['cldsavl'])
        st('CLDSAV1', Sx['cldsv1'])
        st('SVLHX', Sx['svlhxl'])
        st('CSIZSS', Sx['csizel'])
        st('QLSS', Sx['qlss'])
        st('QISS', Sx['qiss'])
        st('QLMC', o['qlmc'].T)
        st('QIMC', o['qimc'].T)
        flags['handoff_error'] = self._hand_off(X, I, J, cldmcl, cldssl, svlatl, Sx['svlhxl'], Sx['qclx'], Sx['qcix'], o['cnvmmrl'].T, Sx['cldsal'])
        st('TAUSSIP', Sx['tausslip'])
        st('CSIZSSIP', Sx['csizelip'])
        st('RHSAV', Sx['rh'])
        st('TTOLD', Sx['th'])
        st('QTOLD', Sx['ql'])
        X['PREC'] = X['PREC'].at[I, J].set(prcp)
        X['EPREC'] = X['EPREC'].at[I, J].set(enrgp)
        precss = prcpss * 100.0 * BYGRAV
        X['PRECSS'] = X['PRECSS'].at[I, J].set(precss)
        X['P_ACC'] = X['P_ACC'].at[I, J].set(X['P_ACC'][I, J] + prcp)
        X['PM_ACC'] = X['PM_ACC'].at[I, J].set(X['PM_ACC'][I, J] + prcp - precss)
        # ---- final merge (2152-2185)
        fs = fssl
        X['T'] = X['T'].at[I, J, :].set(jnp.transpose(Sx['th'] * fs + tmc * (1.0 - fs)))
        X['Q'] = X['Q'].at[I, J, :].set(jnp.transpose(Sx['ql'] * fs + qmc * (1.0 - fs)))
        smom_f = jnp.transpose(Sx['smom'], (1, 0, 2)) * fs[None] + smom_mc * (1.0 - fs[None])
        qmom_f = jnp.transpose(Sx['qmom'], (1, 0, 2)) * fs[None] + qmom_mc * (1.0 - fs[None])
        X['TMOM'] = X['TMOM'].at[:, I, J, :].set(jnp.transpose(smom_f * byam[None], (0, 2, 1)))
        X['QMOM'] = X['QMOM'].at[:, I, J, :].set(jnp.transpose(qmom_f * byam[None], (0, 2, 1)))
        X['QCI'] = X['QCI'].at[I, J, :].set(jnp.transpose(Sx['qcix']))
        X['QCL'] = X['QCL'].at[I, J, :].set(jnp.transpose(Sx['qclx']))
        X['TMC'] = X['TMC'].at[I, J, :].set(jnp.transpose(tmc))
        X['QMC'] = X['QMC'].at[I, J, :].set(jnp.transpose(qmc))
        ums = jnp.transpose(Sx['um'], (1, 0, 2))
        vms = jnp.transpose(Sx['vm'], (1, 0, 2))
        du = (ums * fs[None] + um1 * (1.0 - fs[None])) * byam[None] - u0
        dv = (vms * fs[None] + vm1 * (1.0 - fs[None])) * byam[None] - v0
        X['UKM'] = X['UKM'].at[:, :, I, J].set(du)
        X['VKM'] = X['VKM'].at[:, :, I, J].set(dv)
        # carry for the north pole: the last column's LSCOND arrays
        ms_b = dict(tausslip=Sx['tausslip'][:, -1], csizelip=Sx['csizelip'][:, -1], cldsal=Sx['cldsal'][:, -1], cldsv1=Sx['cldsv1'][:, -1],
                    dqlsc=jnp.zeros(LM), lhp=lhp[:, -1], prebar1=Sx['prebar1'][:, -1])
        # ---- poles: south result (already computed in the set-up unit) and north (callback after the batch, with the batch's carry)
        Xn, ms_n = pole_call(self.host, JM - 1, A, ms_b)
        for k, v in Xs.items():
            if k not in ('UKMNP', 'VKMNP'):
                X[k] = self._put_pole(X[k], v, 0)
        for k, v in Xn.items():
            if k not in ('UKMSP', 'VKMSP'):
                X[k] = self._put_pole(X[k], v, JM - 1)
        # ---- momentum back-transfer and the A-grid winds
        X['U'], X['V'], X['UKM'], X['VKM'] = G.avg_replicated_duv_to_vgrid(X['U'], X['V'], X['UKM'], X['VKM'], X['UKMSP'], X['VKMSP'], X['UKMNP'], X['VKMNP'])
        X['UALIJ'], X['VALIJ'] = G.recalc_agrid_uv(X['U'], X['V'], self.agrid)
        # the cloud arrays the radiation packet and the next step's entry read: kept in X
        return X, ms_n, flags, dict(prcp=prcp, enrgp=enrgp)

    @staticmethod
    def _hand_off(X, I, J, cldmcl, cldssl, svlatl, svlhxl, qclx, qcix, cnvmmrl, cldsal):
        """clouds_condse_batch.hand_off_b (CONDSE:1949-2080) for all columns; arrays (LM,N).  Returns the error flag (stop_model 255 conditions)."""
        st = lambda key, val: X.__setitem__(key, X[key].at[:, I, J].set(val))     # noqa: E731
        csizss = X['CSIZSS'][:, I, J]
        csizmc = X['CSIZMC'][:, I, J]
        cs = cldssl > 0.0
        den = cldmcl + cldssl + TINY
        w = cs & (svlhxl == LHE)
        ic_ = cs & (svlhxl == LHS)
        err1 = (cs & ~w & ~ic_).any()
        z = jnp.zeros_like(cldssl)
        fsw = jnp.where(w, cldssl / den, z)
        fsi = jnp.where(ic_, cldssl / den, z)
        st('MIX_ST_WATER', jnp.where(w, qclx, z))
        st('MIX_ST_ICE', jnp.where(ic_, qcix, z))
        st('DIM_ST_WATER', jnp.where(w, csizss * 1.0e-06, z))
        st('DIM_ST_ICE', jnp.where(ic_, csizss * 1.0e-06, z))
        st('FRAC_AREA_ST', jnp.where(cs, cldsal, z))
        cm = cldmcl > 0.0
        cw = cm & (svlatl == LHE)
        ci = cm & (svlatl == LHS)
        err2 = (cm & ~cw & ~ci).any()
        fcw = jnp.where(cw, cldmcl / den, z)
        fci = jnp.where(ci, cldmcl / den, z)
        mcw = jnp.where(cw, cnvmmrl, z)
        mci = jnp.where(ci, cnvmmrl, z)
        dcw = jnp.where(cw, csizmc * 1.0e-06, z)
        dci = jnp.where(ci, csizmc * 1.0e-06, z)
        fac_old = X['FRAC_AREA_CNV'][:, I, J]
        fac = jnp.where(cm, cldmcl, fac_old)
        zm = cm & (cnvmmrl == 0.0)
        fcw, fci, mcw, mci, dcw, dci, fac = (jnp.where(zm, z, v) for v in (fcw, fci, mcw, mci, dcw, dci, fac))
        wc = cldmcl + cldssl - cldmcl * cldssl
        wc = jnp.where(zm, cldssl, wc)
        renorm = zm & cs
        fw_n = fsw / (fsw + fsi)
        fi_n = fsi / (fw_n + fsi)
        st('FRAC_ST_WATER', jnp.where(renorm, fw_n, fsw))
        st('FRAC_ST_ICE', jnp.where(renorm, fi_n, fsi))
        st('W_CLOUD', wc)
        st('FRAC_CNV_WATER', fcw)
        st('FRAC_CNV_ICE', fci)
        st('MIX_CNV_WATER', mcw)
        st('MIX_CNV_ICE', mci)
        st('DIM_CNV_WATER', dcw)
        st('DIM_CNV_ICE', dci)
        st('FRAC_AREA_CNV', fac)
        return (err1 | err2).astype(jnp.int32)

    @staticmethod
    def _put_pole(arr, sl, j):
        if arr.shape == (IM, LM):
            return sl
        if arr.ndim == 2:
            return arr.at[0, j].set(sl)
        if arr.ndim == 3:
            return arr.at[0, j, :].set(sl) if arr.shape[:2] == (IM, JM) else arr.at[:, 0, j].set(sl)
        return arr.at[:, 0, j, :].set(sl) if arr.shape[1:3] == (IM, JM) else arr.at[:, :, 0, j].set(sl)

    def post(self, A, W0, o, ms_s, Xs):
        return self._post(A, W0, o, ms_s, Xs, self.Kls)

    # ---------------------------------------------------------------- whole CONDSE
    def run(self, A, ms, stats=None):
        """A: dict of device entry arrays (all of cse_in plus the chained state); ms: module-array vectors.  Returns (X dict, ms', flags dict, extra)."""
        R, W0, Xs, ms_s = self.setup(A, ms)
        o, err = self.mstcnv(R, stats)
        X, ms_n, flags, extra = self.post(A, W0, o, ms_s, Xs)
        flags['mstcnv_negative_cloud'] = err.astype(jnp.int32)
        return X, ms_n, flags, extra
