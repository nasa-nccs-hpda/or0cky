"""D128: one 30-minute source step of P2SAoM40, chained from the validated full-fidelity ports in the real call order
(see scoping/ATM_STEP_PLAN.md, D127, for the line-by-line order, which port implements each call, what is recorded
and what is not ported).

Live order (MODELE.f:316-339, ATM_DRV.f atm_phase1 / atm_phase2, SURFACE.f):
    dyn        atm_phase1 88-235  DYNAM + QDYNAM + energy fix + CALC_TROP + PGRAD_PBL + calc_kea_3d   (dyn_step.py, D121-D123)
    condse     ATM_DRV.f:264 CONDSE                                                                    (clouds_condse_ff.py, D124-D126)
    radia      ATM_DRV.f:274 RADIA: recorded SOCRATES output (SRHR, TRHR, COSZ1) + the T update RAD_DRV.f:5474-5478
    surface    SURFACE.f DO NS=1,2: PBL, tile fluxes, land-ice, GHY (Ent exports recorded), aggregation, first-layer
               TMOM/QMOM update (SURFACE.f:1068-1089), ATM_DIFFUS(1,1) = ATURB + velocity diffusion  (chain_two_substeps.py,
               land_chain.py, aturb_ff.py, aturb_uv_ff.py ...), twice
    dissip     ATM_DRV.f:466 DISSIP                                                                    (dyn_glue_ff.dissip, D114-D117)
    filter     ATM_DRV.f:482 FILTER (SLP filter, MAtoPMB, energy fix)                                  (dyn_filter_ff.filter_slp, D101-D102)
The second ATM_DIFFUS call of atm_phase2 (ATM_DRV.f:458, lbase_min=2) returns at ATURB.f:116: nothing to chain; DRYCNV is not
linked into this executable (ledger D1).

Recorded inputs (explicit, none computed here): SOCRATES output SRHR/TRHR/COSZ1; all non-dynamic CONDSE entry fields (ffc_cse_in);
the ATURB/PBL hidden state at the step start (ffa_step_<it>_a); the SURFACE tile/PBL/land-ice/GHY/aggregation records of the real step
(ffp/ffs/ffl/ffg/fft) for every column that is NOT overwritten from the chained state (see `override_pbl`), including Ent exports and
land forcing; the pole-row conventions.

A step can start at any stage boundary from the REAL boundary state (start=...) so that a stage's own error is isolated from the
propagation of earlier ones: start in {'dyn','condse','radia','surface','dissip','filter'}; stop likewise.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_step as ds
import dyn_glue_ff as gf
import dyn_filter_ff as ffl
import clouds_condse_ff as cf
import clouds_condse_io as cio
import ffdump_reader as fr
import intel_libm_ff

IM, JM, LM = 72, 46, 40
FF = cio.FF_DEFAULT
DATES = cio.DATES
STAGES = ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter')
NIS = 2

HIDDEN = ('EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'USTARPBL', 'LMONINPBL', 'DDM1',
          'TSAVG', 'QSAVG', 'USAVG', 'VSAVG', 'TGVAVG', 'QGAVG', 'UALIJ', 'VALIJ', 'SRHR', 'TRHR', 'COSZ1')
DYN_OUT = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'MASUM', 'TMOM', 'QMOM', 'MUS', 'MVS', 'MWS', 'GZ',
           'KEA', 'PTROPO', 'LTROPO', 'WSAVE', 'DPDX', 'DPDY', 'DPDX0', 'DPDY0', 'PHI', 'PMIDOLD', 'PDSIG')
CONDSE_OUT = ('T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'U', 'V', 'UALIJ', 'VALIJ', 'PREC', 'EPREC', 'PRECSS', 'DDM1', 'DDMS', 'TDN1',
              'QDN1', 'DDML')
END_FIELDS = ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'PEK', 'P', 'TMOM', 'QMOM', 'UALIJ', 'VALIJ',
              'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'USTARPBL', 'LMONINPBL', 'DDM1', 'TSAVG', 'QSAVG')


# ------------------------------------------------------------------------------------------------ real boundary data
class Real:
    """Lazy loaders for the real boundary dumps of one step (all in ff_data/<date>/)."""

    def __init__(self, date, itime, ff=FF):
        self.date, self.itime, self.ff = date, itime, ff
        self._c = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def site(self, s):
        return self._get('site' + s, lambda: cio.read_cse(f"{self.ff}/{self.date}/ffa_step_{self.itime}_{s}.bin"))

    a = property(lambda self: self.site('a'))
    r = property(lambda self: self.site('r'))
    d = property(lambda self: self.site('d'))
    e = property(lambda self: self.site('e'))
    s1 = property(lambda self: self._get('s1', lambda: ds.load_state(ds.state_path(self.date, self.itime, 1, self.ff))))
    s3 = property(lambda self: self._get('s3', lambda: ds.load_state(ds.state_path(self.date, self.itime, 3, self.ff))))
    s4 = property(lambda self: self._get('s4', lambda: ds.load_exports(ds.state_path(self.date, self.itime, 4, self.ff))))
    cse_in = property(lambda self: self._get('ci', lambda: cio.read_cse(f"{self.ff}/{self.date}/ffc_cse_in_{self.itime}.bin")))
    cse_out = property(lambda self: self._get('co', lambda: cio.read_cse(f"{self.ff}/{self.date}/ffc_cse_out_{self.itime}.bin")))

    def post_surface(self):
        return self._get('ps', lambda: cio.old_state(self.date, self.itime, 'post_surface', self.ff))

    def filt_in(self):
        import dyn_filter_compare as dfc
        return self._get('fi', lambda: dfc.load_in(f"{self.ff}/{self.date}/ffd_filt_{self.itime}_in.bin"))

    def filt_out(self):
        import dyn_filter_compare as dfc
        return self._get('fo', lambda: dfc.load_out(f"{self.ff}/{self.date}/ffd_filt_{self.itime}_out.bin"))

    def available(self):
        need = [f"{self.ff}/{self.date}/ffa_step_{self.itime}_{s}.bin" for s in 'arde']
        need += [f"{self.ff}/{self.date}/ffc_cse_in_{self.itime}.bin", f"{self.ff}/{self.date}/ffc_cse_out_{self.itime}.bin"]
        need += [f"{self.ff}/{self.date}/{n}_{self.itime}.bin" for n in ('ffp', 'ffs', 'ffl', 'ffg', 'fft')]
        need += [f"{self.ff}/{self.date}/ffa_{self.itime}_c{k}_{io}.bin" for k in (1, 2) for io in ('in', 'out')]
        return all(os.path.exists(p) for p in need) and ds.available(self.date, self.ff)


def have_dumps(date, itime, ff=FF):
    return Real(date, itime, ff).available()


# ------------------------------------------------------------------------------------------------ comparison utilities
def _ij_axes(shape):
    """axes of the (IM, JM) horizontal indices in an array shape (None if not a horizontal field)."""
    ai = [k for k, n in enumerate(shape) if n == IM]
    aj = [k for k, n in enumerate(shape) if n == JM]
    if not ai or not aj:
        return None
    return ai[0], aj[0]


def field_stats(a, b, tag=None, bound=1e-12):
    """Difference statistics of a against the reference b: max abs, scale (max |ref|), max/scale, rms, fraction of bitwise-equal
    cells, first differing cell (1-based, array order), worst cell, number of cells and of horizontal columns above
    bound*scale, and the category A (bitwise) / B (<= 1e-12 scale) / C (<= 1e-6 scale) / D (worse)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = np.abs(a - b)
    nn = np.isnan(a) & np.isnan(b)
    d = np.where(nn, 0.0, d)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    mx = float(d.max()) if d.size else 0.0
    rel = mx / scale if scale > 0 else (0.0 if mx == 0 else np.inf)
    neq = int(((a != b) & ~nn).sum())
    first = worst = None
    n_over = cols_over = cols_over6 = 0
    if mx > 0:
        idx = np.argwhere((a != b) & ~nn)
        first = tuple(int(x) + 1 for x in idx[0]) if len(idx) else None
        worst = tuple(int(x) + 1 for x in np.unravel_index(np.argmax(d), d.shape))
        over = d > bound * scale
        n_over = int(over.sum())
        ax = _ij_axes(d.shape)
        if ax is not None and n_over:
            other = tuple(k for k in range(d.ndim) if k not in ax)
            cols_over = int(over.any(axis=other).sum()) if other else n_over
            over6 = d > 1e-6 * scale
            cols_over6 = int(over6.any(axis=other).sum()) if other else int(over6.sum())
    cat = 'A' if mx == 0 else ('B' if rel <= bound else ('C' if rel <= 1e-6 else 'D'))
    return dict(max_abs=mx, scale=scale, rel=rel, rms=float(np.sqrt((d ** 2).mean())) if d.size else 0.0, n=int(a.size), n_diff=neq,
                first=first, worst=worst, n_over=n_over, cols_over=cols_over, cols_over6=cols_over6, cat=cat)


def compare_state(S, ref, fields):
    out = {}
    for f in fields:
        if f in S and f in ref:
            out[f] = field_stats(S[f], ref[f])
    return out


def fmt_stats(st):
    return " ".join(f"{k}:{v['cat']}({v['max_abs']:.1e}/{v['rel']:.0e})" for k, v in st.items())


# ------------------------------------------------------------------------------------------------ context
class Ctx:
    pass


def make_ctx(date, imf=False, ff=FF, surface=True):
    """Everything one date needs: dynamics context (dyn_step.load_ctx), CONDSE cfg (clouds_condse_ff.make_cfg), filter geometry,
    IMAXJ and constants.  imf=True routes pow/exp through the Intel libimf (as the real ifort build) where the ports support it."""
    c = Ctx()
    c.date, c.imf, c.ff = date, bool(imf), ff
    if imf and not intel_libm_ff.available():
        raise RuntimeError("Intel libimf not available on this host")
    c.dyn = ds.load_ctx(date, ff, imf_pow=bool(imf))
    cf.set_backend('imf' if imf else 'numpy')
    c.cfg = cf.make_cfg(date, ff, glue_geom=c.dyn.gg)
    c.gg = c.dyn.gg
    import dyn_filter_ff as _f
    c.fg = _f.load_consts(f"{ff}/{date}/ffd_filt_consts.bin")
    c.imaxj = np.array(c.cfg['geom']['IMAXJ'], int)
    c.dtsrc = float(c.cfg['dtsrc'])
    c.sha = float(c.gg['sha'])
    c.kapa = float(c.gg['kapa'])
    return c


def _pow(ctx, x, y):
    return intel_libm_ff.pow_imf(x, y) if ctx.imf else np.power(x, y)


def imaxj_mask(ctx):
    m = np.zeros((IM, JM), bool)
    for j in range(JM):
        m[:ctx.imaxj[j], j] = True
    return m


# ------------------------------------------------------------------------------------------------ state
def init_state(R):
    """Real state at the start of the step (ffd_state s1 + hidden state of ffa_step_a)."""
    S = {k.upper(): np.array(v, copy=True) for k, v in R.s1.items() if k in ds.STATE_KEYS}
    for k in HIDDEN:
        S[k] = np.array(R.a[k], copy=True)
    S['PDSIG'] = np.array(R.a['PDSIG'], copy=True)
    S['PEK'] = np.array(R.a['PEK'], copy=True)
    return S


# ------------------------------------------------------------------------------------------------ stage: dyn
def stage_dyn(S, R, ctx, tm=None):
    w = ds.dyn_step({k: S[k.upper()] for k in ds.STATE_KEYS}, ctx.dyn, itime=R.itime, timing=tm)
    for k in DYN_OUT:
        if k in w:
            S[k] = np.array(w[k], copy=True)
    S['PEK'] = _pow(ctx, S['PEDN'], ctx.kapa)            # MAtoPMB: PEK = PEDN**KAPA
    return S


# ------------------------------------------------------------------------------------------------ stage: condse
def condse_inputs(S, R):
    """CONDSE entry arrays: the real recorded entry set with every field that the dynamics block (or an earlier stage) produced
    replaced by the chained value; UKM/VKM rebuilt from the chained U,V (replicate_uv_to_agrid)."""
    inp = dict(R.cse_in)
    for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'PK', 'PMID', 'PEDN', 'PDSIG', 'PMIDOLD', 'GZ', 'MWS'):
        inp[k] = np.array(S[k], copy=True)
    inp['PEK'] = np.array(S['PEK'], copy=True)
    for k in ('EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'TSAVG', 'QSAVG'):      # hidden step-start state (equal to the record at step 0)
        inp[k] = np.array(S[k], copy=True)
    if '_carry' in S:                                  # free-running chain: cloud/precipitation carry state of OUR previous step
        inp.update({k: np.array(v, copy=True) for k, v in S['_carry'].items()})
    ukm, vkm, usp, vsp, unp, vnp = cf.replicate_uv_to_agrid(S['U'], S['V'])
    inp.update(UKM=ukm, VKM=vkm, UKMSP=usp, VKMSP=vsp, UKMNP=unp, VKMNP=vnp)
    return inp


def stage_condse(S, R, ctx, ms=None, cols=None, tm=None):
    inp = condse_inputs(S, R)
    X, cnt = cf.condse_step(inp, ctx.cfg, cols=cols, ms=ms if ms is not None else {})
    for k in CONDSE_OUT:
        if k in X:
            S[k] = np.array(X[k], copy=True)
    S['_condse_counts'] = cnt
    S['_condse_X'] = X
    return S


def condse_from_real(S, R):
    """Replay: the real CONDSE exit state replaces the stage output."""
    o = R.cse_out
    for k in CONDSE_OUT:
        if k in o:
            S[k] = np.array(o[k], copy=True)
    return S


# ------------------------------------------------------------------------------------------------ stage: radia
def radia_apply(T, srhr, trhr, cosz1, ma, pk, ctx):
    """RAD_DRV.f:5474-5478: T(I,J,L) += (SRHR(L)*COSZ1+TRHR(L))*DTsrc*bysha*byMA/PK for I<=IMAXJ(J).
    srhr,trhr (LM+1,IM,JM) with index 0 the surface term (not used here), ma/pk (LM,IM,JM), T (IM,JM,LM)."""
    bysha = 1.0 / ctx.sha
    byma = 1.0 / ma
    tn = np.array(T, copy=True)
    h = (np.transpose(srhr[1:], (1, 2, 0)) * cosz1[:, :, None] + np.transpose(trhr[1:], (1, 2, 0)))
    inc = h * ctx.dtsrc * bysha * np.transpose(byma, (1, 2, 0)) / np.transpose(pk, (1, 2, 0))
    m = imaxj_mask(ctx)
    tn[m] = T[m] + inc[m]
    return tn


TAULIM = 1.0e-3          # RAD_DRV.f:2464 taulim = min(tauwc0, tauic0) = .001
NRAD = 5                 # P2SAoM40.R:253


def is_radiation_step(itime):
    return (itime - ds.ITIMEI) % NRAD == 0


def radia_cloud_masking(cldss, cldmc, tauss, taumc):
    """RAD_DRV.f:2611-2615 (radiation steps only): CLDSS=0 where TAUSS<=taulim, CLDMC=0 where TAUMC<=taulim.  These two arrays are
    carried to the next step's CONDSE entry (the only cloud arrays that change between CONDSE exit and the next entry)."""
    return np.where(tauss <= TAULIM, 0.0, cldss), np.where(taumc <= TAULIM, 0.0, cldmc)


def stage_radia(S, R, ctx):
    r = R.r
    S['SRHR'], S['TRHR'], S['COSZ1'] = np.array(r['SRHR']), np.array(r['TRHR']), np.array(r['COSZ1'])
    S['T'] = radia_apply(S['T'], S['SRHR'], S['TRHR'], S['COSZ1'], S['MA'], S['PK'], ctx)
    if is_radiation_step(R.itime):
        X = S.get('_condse_X') or R.cse_out
        S['_cloud_rad'] = radia_cloud_masking(X['CLDSS'], X['CLDMC'], X['TAUSS'], X['TAUMC'])
    return S


# ------------------------------------------------------------------------------------------------ stage: surface
_SURF = {}


def _surf_mods():
    if not _SURF:
        import jax.numpy as jnp
        import tile_aggregate_ff as TA
        import surface_tile_ff as ST
        import surface_chain_ff as CH
        import landice_tile_ff as LI
        import pbl_compare as PC
        import pbl_ff as P
        import aturb_compare as AC
        import substep_chain as SC
        import chain_aggregate_aturb as CA
        import land_chain as LC
        import ghy_compare as GC
        import chain_two_substeps as C2
        _SURF.update(jnp=jnp, TA=TA, ST=ST, CH=CH, LI=LI, PC=PC, P=P, AC=AC, SC=SC, CA=CA, LC=LC, GC=GC, C2=C2)
    return _SURF


PBL_COLS = dict(zs1=5, tkv=7, ps=16, ddml=23, gusti=24, tdns=25, qdns=26, dbl=29, ug=31, vg=32, utop=37, vtop=38, qtop=39, ztop=40,
                mdf=41, dpdxr=42, dpdyr=43, dpdxr0=44, dpdyr0=45, dtdt=46, u1aa=47, v1aa=48)
PBL_GUSTI_OUT = 113
FLOAT4_4375 = float(np.float32(4375.0))


def atm_layout(S, ctx):
    """Chained state -> the ATURB (J,I,L) layout used by chain_two_substeps."""
    t3 = lambda x: np.transpose(x, (1, 0, 2))
    l3 = lambda x: np.transpose(x, (2, 1, 0))
    return dict(T=t3(S['T']), Q=t3(S['Q']), U=t3(S['U']), V=t3(S['V']), UA=l3(S['UALIJ']), VA=l3(S['VALIJ']), E=l3(S['EGCM']),
                PMID=l3(S['PMID']), PEDN=l3(S['PEDN']), PK=l3(S['PK']), PEK1=S['PEK'][0].T, PDSIG=l3(S['PDSIG']), MA1=S['MA'][0].T)


def cell_inputs(S, atm, ctx, ustar, lmonin, pblht, dclev, coriol_ji, t1aa_ji, u1aa_ji, v1aa_ji):
    """Atmosphere-dependent PBL inputs per cell, arrays (JM,IM): get_atm_layer1 scalars, get_dbl (PBL_DRV.f:1294), the PGRAD_PBL
    exports, the CONDSE downdraft exports (mdf, gusti, tdns, qdns, ddml flag), surface pressure and dtdt_gcm."""
    M = _surf_mods()
    SC = M['SC']
    jnp = M['jnp']
    e = SC.layer1_exports(atm['T'], atm['Q'], atm['UA'], atm['VA'], atm['MA1'], atm['PEK1'], atm['PMID'][..., 0])
    ug, vg, dbl = SC.get_dbl(jnp.asarray(ustar), jnp.asarray(lmonin), jnp.asarray(coriol_ji), jnp.asarray(pblht), jnp.asarray(dclev),
                             jnp.asarray(atm['T']), jnp.asarray(atm['Q']), jnp.asarray(atm['UA']), jnp.asarray(atm['VA']),
                             jnp.asarray(atm['PMID']), jnp.asarray(atm['PK']), jnp.asarray(e['ztop']))
    ddms, tdn1, qdn1, ddml = S['DDMS'].T, S['TDN1'].T, S['QDN1'].T, S['DDML'].T
    flag = (ddml == 1).astype(float)
    mdn = np.maximum(ddms, -0.07)
    gusti = np.where(flag > 0, np.log(1.0 - 600.4 * mdn - FLOAT4_4375 * mdn * mdn), 0.0)
    pek1 = atm['PEK1']
    pk1 = atm['PK'][..., 0]
    c = dict(zs1=e['zs1'], tkv=e['tkv'], ps=atm['PEDN'][..., 0], ddml=flag, gusti=gusti,
             tdns=np.where(flag > 0, tdn1 * pek1 / pk1, 0.0), qdns=np.where(flag > 0, qdn1, 0.0),
             dbl=dbl, ug=ug, vg=vg, utop=e['utop'], vtop=e['vtop'], qtop=e['qtop'], ztop=e['ztop'],
             mdf=S['DDM1'].T, dpdxr=S['DPDX'].T, dpdyr=S['DPDY'].T, dpdxr0=S['DPDX0'].T, dpdyr0=S['DPDY0'].T,
             u1aa=u1aa_ji, v1aa=v1aa_ji)
    c['dtdt'] = (np.asarray(c['tkv']) - t1aa_ji * pek1) / 900.0
    return {k: np.asarray(v) for k, v in c.items()}


def override_pbl(rec, cell, skip=()):
    """Overwrite the atmosphere-dependent columns of PBL records `rec` (N,154) by the per-cell chained values. Every column not
    listed in PBL_COLS (ground state, profiles, radiation inputs, constants, ...) stays as recorded."""
    rec = np.array(rec)
    j = rec[:, 1].astype(int) - 1
    i = rec[:, 0].astype(int) - 1
    for nm, c in PBL_COLS.items():
        if nm in skip:
            continue
        rec[:, c] = cell[nm][j, i]
    rec[:, PBL_GUSTI_OUT] = cell['gusti'][j, i]
    return rec


def override_tiles(tile, li, atm):
    """Tile (ffs) / land-ice (ffl) record columns that depend on the atmosphere at substep entry: surface pressure, MA(1), q1, thv1."""
    M = _surf_mods()
    S_, LI = M['ST'], M['LI']
    t = np.array(tile)
    j = t[:, 1].astype(int) - 1
    i = t[:, 0].astype(int) - 1
    q1 = atm['Q'][..., 0]
    t[:, S_.IN['ps']] = atm['PEDN'][..., 0][j, i]
    t[:, S_.IN['ma1']] = atm['MA1'][j, i]
    t[:, S_.IN['byma1']] = 1.0 / atm['MA1'][j, i]
    t[:, S_.IN['q1']] = q1[j, i]
    t[:, S_.IN['thv1']] = (atm['T'][..., 0] * (1.0 + q1 * M['SC'].XDELT))[j, i]
    l = np.array(li)
    j = l[:, 1].astype(int) - 1
    i = l[:, 0].astype(int) - 1
    l[:, LI.IN['ps']] = atm['PEDN'][..., 0][j, i]
    l[:, LI.IN['ma1']] = atm['MA1'][j, i]
    l[:, LI.IN['q1']] = q1[j, i]
    return t, l


def first_layer_update(tmom, qmom, dth1, dq1, t1, q1, pk1):
    """SURFACE.f:1068-1089 'UPDATE FIRST LAYER QUANTITIES' (the TMOM/QMOM part): TMOM(:,I,J,1) *= (1-FTEVAP),
    QMOM(:,I,J,1) *= (1-FQEVAP) and QMOM(:,I,J,1)=0 when Q+DQ1 < qmin (1e-12).  All arrays (IM,JM); tmom/qmom (9,IM,JM,LM)."""
    ft = np.where(dth1 * t1 < 0, -dth1 / (t1 * pk1), 0.0)
    fq = np.where((dq1 < 0) & (q1 > 0), -dq1 / q1, 0.0)
    tm = np.array(tmom, copy=True)
    qm = np.array(qmom, copy=True)
    tm[:, :, :, 0] = tmom[:, :, :, 0] * (1.0 - ft)[None]
    qm[:, :, :, 0] = qmom[:, :, :, 0] * (1.0 - fq)[None]
    small = (q1 + dq1) < 1e-12
    qm[:, small, 0] = 0.0
    return tm, qm


def _aturb(atm, fl, blk, dt):
    """chain_two_substeps.run_aturb, returning also W2GCM and PBLPTOP."""
    M = _surf_mods()
    C2, AC, jnp = M['C2'], M['AC'], M['jnp']
    i = blk[:, 0].astype(int) - 1
    j = blk[:, 1].astype(int) - 1
    shape = atm['T'].shape[:2]
    m = AC.valid_mask(shape)
    full = {}
    for k in ('UFLUX1', 'VFLUX1', 'TFLUX1', 'QFLUX1', 'TSAVG', 'QSAVG'):
        a = np.full(shape, {'TSAVG': 280.0, 'QSAVG': 0.0}.get(k, 1e-3))
        a[j, i] = np.asarray(fl[k])
        full[k] = a
    args = {k: jnp.asarray(atm[k]) for k in ('T', 'Q', 'UA', 'VA', 'E', 'PMID', 'PEDN', 'PK', 'PEK1', 'PDSIG')}
    args.update({k: jnp.asarray(v) for k, v in full.items()})
    res, Un, Vn, ua, va = C2._aturb_uv_jit(args, jnp.asarray(atm['U']), jnp.asarray(atm['V']), dt)
    return dict(T=np.asarray(res['t']), Q=np.asarray(res['q']), E=np.asarray(res['e']), W2=np.asarray(res['w2']),
                pblht=np.asarray(res['pblht']), dclev=np.asarray(res['dclev']), pblptop=np.asarray(res['pblptop']),
                U=np.asarray(Un), V=np.asarray(Vn), UA=np.asarray(ua), VA=np.asarray(va), m=m)


def _substep(pbl12, tile, pbl3, li, blk, atm, dt, land):
    """chain_two_substeps.substep with the extended ATURB return."""
    M = _surf_mods()
    TA, CH, C, LC, jnp = M['TA'], M['CH'], M['CA'], M['LC'], M['jnp']
    C2 = M['C2']
    ftype, patch, _ = TA.unpack(blk)
    patch = {k: np.array(v) for k, v in patch.items()}
    got, pout = CH.run_chain(pbl12, tile, return_pbl=True)
    lut = C._cell_lookup(blk)
    idx = lut[tile[:, 0].astype(int), tile[:, 1].astype(int)]
    k = tile[:, 2].astype(int) - 1
    patch['uflux1'][idx, k] = np.asarray(got['dmua']); patch['vflux1'][idx, k] = np.asarray(got['dmva'])
    patch['dth1'][idx, k] = np.asarray(got['dth1']); patch['dq1'][idx, k] = np.asarray(got['dq1'])
    patch['tsavg'][idx, k] = np.asarray(pout['tsv']); patch['qsavg'][idx, k] = np.asarray(pout['qsrf'])
    gli, pli = C.landice_chain(pbl3, li)
    idl = lut[li[:, 0].astype(int), li[:, 1].astype(int)]
    patch['uflux1'][idl, 2] = np.asarray(gli['uflux1']); patch['vflux1'][idl, 2] = np.asarray(gli['vflux1'])
    patch['dth1'][idl, 2] = np.asarray(gli['dth1']); patch['dq1'][idl, 2] = np.asarray(gli['dq1'])
    patch['tsavg'][idl, 2] = np.asarray(pli['tsv']); patch['qsavg'][idl, 2] = np.asarray(pli['qsrf'])
    lr = None
    if land is not None:
        gi = land['g'][:, 0].astype(int) - 1
        gj = land['g'][:, 1].astype(int) - 1
        lr = LC.land_substep(land['p4'], land['g'], atm['Q'][gj, gi, 0], land['trup'], dt, land.get('dyn'))
        idg = lut[gi + 1, gj + 1]
        for k_ in ('uflux1', 'vflux1', 'dth1', 'dq1', 'tsavg', 'qsavg'):
            patch[k_][idg, 3] = lr['patch'][k_]
    comp = {k_: np.asarray(v) for k_, v in TA.aggregate(jnp.asarray(ftype), {k_: jnp.asarray(v) for k_, v in patch.items()}).items()}
    i = blk[:, 0].astype(int) - 1
    j = blk[:, 1].astype(int) - 1
    return dict(tile=got, pbl=pout, li=gli, pbl_li=pli, comp=comp, ftype=ftype, land=lr, blk=blk, ci=i, cj=j)


def _cell_to_ij(arr_ji):
    return np.asarray(arr_ji).T


def surface_records(R):
    M = _surf_mods()
    PC, ST, LI, TA, GC = M['PC'], M['ST'], M['LI'], M['TA'], M['GC']
    d, it = f"{R.ff}/{R.date}", R.itime
    p = PC.load(f"{d}/ffp_{it}.bin"); t = ST.load(f"{d}/ffs_{it}.bin"); l = LI.load(f"{d}/ffl_{it}.bin")
    fft = TA.load(f"{d}/fft_{it}.bin"); g = GC.load(f"{d}/ffg_{it}.bin")
    n, nt, nl, B, ng = len(p) // 2, len(t) // 2, len(l) // 2, len(fft) // 2, len(g) // 2
    rec = dict(pa=p[:n], pb=p[n:], ta=t[:nt], tb=t[nt:], la=l[:nl], lb=l[nl:], blk1=fft[:B], blk2=fft[B:], g1=g[:ng], g2=g[ng:])
    return rec


def stage_surface(S, R, ctx, rec=None, tm=None, land_mode='ghy'):
    """Both NIsurf substeps (SURFACE.f:385-1178) from the chained atmosphere.  Fills S with the ATURB exit state of substep 2 and
    the composites needed later (TSAVG, QSAVG, USTARPBL, LMONINPBL, T1AA, U1AA, V1AA ...).
    land_mode 'ghy': the land patch is computed by our PBL + GHY port (Ent exports and land forcing recorded);
              'recorded': the land patch (GHY outputs) is the recorded one from the fft records, as in D19-D21."""
    M = _surf_mods()
    C2, LC, TA, jnp, SC = M['C2'], M['LC'], M['TA'], M['jnp'], M['SC']
    rec = rec or surface_records(R)
    dt = 900.0
    atm1 = atm_layout(S, ctx)
    pa, pb = rec['pa'], rec['pb']
    a12, a3, a4 = pa[pa[:, 2] <= 2], pa[pa[:, 2] == 3], pa[pa[:, 2] == 4]
    b12, b3, b4 = pb[pb[:, 2] <= 2], pb[pb[:, 2] == 3], pb[pb[:, 2] == 4]
    coriol = np.zeros(atm1['T'].shape[:2])
    coriol[pb[:, 1].astype(int) - 1, pb[:, 0].astype(int) - 1] = pb[:, 36]
    cell1 = cell_inputs(S, atm1, ctx, S['USTARPBL'].T, S['LMONINPBL'].T, S['PBLHT'].T, S['DCLEV'].T, coriol,
                        S['T1AA'].T, S['U1AA'].T, S['V1AA'].T)
    a12c, a3c, a4c = override_pbl(a12, cell1), override_pbl(a3, cell1), override_pbl(a4, cell1)
    ta, la = override_tiles(rec['ta'], rec['la'], atm1)
    g1, g2, blk1, blk2 = rec['g1'], rec['g2'], rec['blk1'], rec['blk2']
    _, patch1, _ = TA.unpack(blk1)
    lut1 = {(int(a), int(b)): k for k, (a, b) in enumerate(blk1[:, :2])}
    idx1 = np.array([lut1[(int(a), int(b))] for a, b in g1[:, :2]])
    trup = LC.infer_trup(g1, patch1['dth1'][idx1, 3], dt)
    t0 = time.perf_counter()
    r1 = _substep(a12c, ta, a3c, la, blk1, atm1, dt, dict(p4=a4c, g=g1, trup=trup, dyn=None) if land_mode == 'ghy' else None)
    # SURFACE.f:1068-1089 first-layer TMOM/QMOM update (before this substep's ATURB), composite DTH1/DQ1
    dth1, dq1 = _grid(r1['comp']['dth1'], r1, 0.0), _grid(r1['comp']['dq1'], r1, 0.0)
    S['TMOM'], S['QMOM'] = first_layer_update(S['TMOM'], S['QMOM'], dth1, dq1, S['T'][:, :, 0], S['Q'][:, :, 0], S['PK'][0])
    fl = M['CA'].aturb_flux_arrays(r1['comp'], atm1['MA1'][r1['cj'], r1['ci']], dt)
    ex1 = _aturb(atm1, fl, blk1, dt)
    atm2 = dict(atm1)
    for k in ('T', 'Q', 'U', 'V', 'UA', 'VA'):
        atm2[k] = _merge_valid(ex1[k], atm1[k], ex1['m'], k)
    atm2['E'] = _merge_valid(ex1['E'], atm1['E'], ex1['m'], 'E')
    atm2['pblht'], atm2['dclev'] = ex1['pblht'], ex1['dclev']
    # substep-2 inputs: the existing predictor (D19) on the recorded substep-2 rows, then the CONDSE/dynamics-dependent columns
    p12, p3, tnew, lnew, p4 = C2.predict_ns2(a12c, a3c, a4c, b12, b3, ta, rec['tb'], la, rec['lb'], r1, blk1, atm2, r1['ftype'], pb, b4)
    cell2 = dict(cell1)
    t1aa = atm2['T'][..., 0]
    cell2['dtdt'] = (_tkv2(atm2) - t1aa * atm1['PEK1']) / dt
    cell2['u1aa'], cell2['v1aa'] = atm2['UA'][..., 0], atm2['VA'][..., 0]
    skip = ('zs1', 'tkv', 'dbl', 'ug', 'vg', 'utop', 'vtop', 'qtop', 'ztop')       # predicted from our ATURB exit by predict_ns2
    p12, p3 = (override_pbl(x, cell2, skip) for x in (p12, p3))
    p4 = override_pbl(p4, cell2, skip) if p4 is not None else None
    tnew2, lnew2 = override_tiles(tnew, lnew, atm2)
    r2 = _substep(p12, tnew2, p3, lnew2, blk2, atm2, dt,
                  dict(p4=p4, g=g2, trup=trup, dyn=r1['land']['dyn_next']) if land_mode == 'ghy' else None)
    dth1, dq1 = _grid(r2['comp']['dth1'], r2, 0.0), _grid(r2['comp']['dq1'], r2, 0.0)
    S['TMOM'], S['QMOM'] = first_layer_update(S['TMOM'], S['QMOM'], dth1, dq1, np.transpose(atm2['T'], (1, 0, 2))[:, :, 0],
                                              np.transpose(atm2['Q'], (1, 0, 2))[:, :, 0], S['PK'][0])
    fl2 = M['CA'].aturb_flux_arrays(r2['comp'], atm2['MA1'][r2['cj'], r2['ci']], dt)
    ex2 = _aturb(atm2, fl2, blk2, dt)
    if tm is not None:
        tm['surface'] = tm.get('surface', 0.0) + time.perf_counter() - t0
    m = ex2['m']
    to_ijl = lambda x: np.transpose(np.asarray(x), (1, 0, 2))
    to_lij = lambda x: np.transpose(np.asarray(x), (2, 1, 0))
    S['T'] = to_ijl(_merge_valid(ex2['T'], atm2['T'], m, 'T'))
    S['Q'] = to_ijl(_merge_valid(ex2['Q'], atm2['Q'], m, 'Q'))
    S['U'] = to_ijl(_merge_valid(ex2['U'], atm2['U'], m, 'U'))
    S['V'] = to_ijl(_merge_valid(ex2['V'], atm2['V'], m, 'V'))
    S['UALIJ'] = to_lij(_merge_valid(ex2['UA'], atm2['UA'], m, 'UA'))
    S['VALIJ'] = to_lij(_merge_valid(ex2['VA'], atm2['VA'], m, 'VA'))
    S['EGCM'] = to_lij(_merge_valid(ex2['E'], atm2['E'], m, 'E'))
    S['W2GCM'] = to_lij(_merge_valid(ex2['W2'], np.transpose(S['W2GCM'], (2, 1, 0)), m, 'W2'))
    for k, kk in (('PBLHT', 'pblht'), ('DCLEV', 'dclev'), ('PBLPTOP', 'pblptop')):
        S[k] = np.where(m.T, np.asarray(ex2[kk]).T, S[k])
    S['T1AA'] = np.where(m.T, S['T'][:, :, 0], S['T1AA'])
    S['U1AA'], S['V1AA'] = np.where(m.T, S['UALIJ'][0], S['U1AA']), np.where(m.T, S['VALIJ'][0], S['V1AA'])
    S['TSAVG'] = _grid_ij(r2['comp']['tsavg'], r2, S['TSAVG'])
    S['QSAVG'] = _grid_ij(r2['comp']['qsavg'], r2, S['QSAVG'])
    ust, lmo = composite_ustar_lmonin(r2, p12, p3, p4 if p4 is not None else b4, pb)
    S['USTARPBL'], S['LMONINPBL'] = ust.T, lmo.T
    S['_surface'] = dict(r1=r1, r2=r2, ex1=ex1, ex2=ex2)
    return S


def _tkv2(atm):
    M = _surf_mods()
    e = M['SC'].layer1_exports(atm['T'], atm['Q'], atm['UA'], atm['VA'], atm['MA1'], atm['PEK1'], atm['PMID'][..., 0])
    return np.asarray(e['tkv'])


def _grid(vals, r, fill):
    """block-cell vector -> (IM,JM) array."""
    a = np.full((IM, JM), fill, float)
    a[r['ci'], r['cj']] = np.asarray(vals)
    return a


def _grid_ij(vals, r, prev):
    a = np.array(prev, copy=True)
    a[r['ci'], r['cj']] = np.asarray(vals)
    return a


def _merge_valid(new, old, m, name):
    """ATURB computes only the valid columns (i=1 on the pole rows); U,V (B grid) are not defined on row J=1."""
    new = np.asarray(new)
    old = np.asarray(old)
    if name in ('U', 'V'):                       # B grid: all longitudes valid on rows J=2..JM, row J=1 undefined
        mm = np.ones(new.shape, bool)
        mm[0] = False
    else:
        mm = m[..., None] if new.ndim == 3 else m
    return np.where(mm, new, old)


def composite_ustar_lmonin(r2, p12, p3, p4, pb):
    """atmsrf%ustar_pbl/lmonin_pbl: area-fraction-weighted sum of the PBL outputs of substep 2 over the tile types (same
    composition predict_ns2 uses for the next substep's get_dbl)."""
    M = _surf_mods()
    C = M['CA']
    blk = r2['blk']
    lut = C._cell_lookup(blk)
    ft = r2['ftype']
    ust = np.zeros((JM, IM)); lmo = np.zeros((JM, IM))

    def acc(rows, ustar, lmonin, itp):
        idx = lut[rows[:, 0].astype(int), rows[:, 1].astype(int)]
        j = rows[:, 1].astype(int) - 1
        i = rows[:, 0].astype(int) - 1
        np.add.at(ust, (j, i), ft[idx, itp - 1] * ustar)
        np.add.at(lmo, (j, i), ft[idx, itp - 1] * lmonin)
    for itp in (1, 2):
        m = p12[:, 2] == itp
        acc(p12[m], np.asarray(r2['pbl']['ustar'])[m], np.asarray(r2['pbl']['lmonin'])[m], itp)
    acc(p3, np.asarray(r2['pbl_li']['ustar']), np.asarray(r2['pbl_li']['lmonin']), 3)
    if r2['land'] is not None:
        acc(p4, np.asarray(r2['land']['pbl']['ustar']), np.asarray(r2['land']['pbl']['lmonin']), 4)
    else:                                       # recorded land PBL outputs (ustar, lmonin = rec(100), rec(101))
        acc(p4, p4[:, 99], p4[:, 100], 4)
    return ust, lmo


# ------------------------------------------------------------------------------------------------ stages: dissip, filter
def stage_dissip(S, ctx):
    dke, tn, ke = gf.dissip(S['U'], S['V'], S['KEA'], S['T'], S['PK'], ctx.gg)
    S['T'] = tn
    S['KEA_NEW'] = ke
    return S


def stage_filter(S, ctx, tsavg=None, masum0=None):
    o = ffl.filter_slp(S['PEDN'][0], S['TSAVG'] if tsavg is None else tsavg, S['MA'], S['PK'], S['T'], S['Q'], S['QCL'], S['QCI'],
                       S['QMOM'], S['U'], S['V'], ctx.fg, masum0=masum0, imf_pow=ctx.imf)
    for k_o, k_s in (('pedn', 'PEDN'), ('pmid', 'PMID'), ('pk', 'PK'), ('ma', 'MA'), ('masum', 'MASUM'), ('t', 'T'), ('q', 'Q'),
                     ('qcl', 'QCL'), ('qci', 'QCI'), ('qmom', 'QMOM')):
        S[k_s] = np.array(o[k_o], copy=True)
    mp = ffl.matopmb(S['MA'], ctx.fg, imf_pow=ctx.imf)
    S['P'] = np.array(mp['p'], copy=True)
    S['PDSIG'] = np.array(mp['pdsig'], copy=True)
    S['PEK'] = _pow(ctx, S['PEDN'], ctx.kapa)
    return S


def pdsig_from_pedn(pedn):
    return pedn[:-1] - pedn[1:]


# ------------------------------------------------------------------------------------------------ real boundary states for start=...
def real_state_at(R, stage, ctx):
    """The real state at the ENTRY of `stage` (so that a run can start there)."""
    S = init_state(R)
    if stage == 'dyn':
        return S
    s3 = R.s3
    s4 = R.s4
    if stage == 'condse':
        for k, v in s3.items():
            if k in ds.STATE_KEYS:
                S[k.upper()] = np.array(v, copy=True)
        S['PMIDOLD'] = np.array(R.cse_in['PMIDOLD'])
        for k in ('KEA', 'PHI', 'WSAVE', 'PTROPO', 'LTROPO'):
            S[k] = np.array(s4[k.lower()], copy=True)
        S['DPDX'], S['DPDY'], S['DPDX0'], S['DPDY0'] = (np.array(s4[k]) for k in ('dpdx', 'dpdy', 'dpdx0', 'dpdy0'))
        S['PDSIG'] = np.array(R.cse_in['PDSIG'])
        S['PEK'] = np.array(R.cse_in['PEK'])
        return S
    ci, co = R.cse_in, R.cse_out
    for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'PEK', 'GZ', 'MWS', 'PMIDOLD'):
        if k in ci:
            S[k] = np.array(ci[k], copy=True)
    S['MA'] = np.array(R.r['MA']); S['P'] = np.array(R.r['P'])
    for k in ('KEA',):
        S[k] = np.array(R.r[k])
    S['DPDX'], S['DPDY'], S['DPDX0'], S['DPDY0'] = (np.array(R.s4[k]) for k in ('dpdx', 'dpdy', 'dpdx0', 'dpdy0'))
    condse_from_real(S, R)
    if stage == 'radia':
        return S
    for k in ('T', 'Q', 'QCL', 'QCI', 'U', 'V'):
        S[k] = np.array(R.r[k], copy=True)
    S['SRHR'], S['TRHR'], S['COSZ1'] = np.array(R.r['SRHR']), np.array(R.r['TRHR']), np.array(R.r['COSZ1'])
    if stage == 'surface':
        return S
    ps = R.post_surface()
    for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI'):
        S[k] = np.array(ps[k], copy=True)
    d = R.d
    S['TMOM'] = np.array(R.e['TMOM'])
    for k in ('UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'USTARPBL', 'LMONINPBL', 'TSAVG', 'QSAVG'):
        S[k] = np.array(R.e[k], copy=True)
    fi = R.filt_in()
    S['QMOM'] = np.array(fi['qmom'], copy=True)
    S['TSAVG'] = np.array(fi['tsavg'], copy=True)
    if stage == 'dissip':
        return S
    for k_f, k_s in (('t', 'T'), ('q', 'Q'), ('qcl', 'QCL'), ('qci', 'QCI')):
        S[k_s] = np.array(fi[k_f], copy=True)
    S['U'], S['V'] = np.array(d['U']), np.array(d['V'])
    return S


# ------------------------------------------------------------------------------------------------ driver
def run_step(date, itime, ctx, start='dyn', stop='filter', ms=None, ff=FF, condse='run', condse_cols=None, R=None, S=None,
             rec=None, timing=None, hook=None, land_mode='ghy'):
    """Run stages start..stop from the real state at the entry of `start`.  condse='run' runs our CONDSE port, 'real' replays the
    recorded exit state.  Returns (S, snapshots) where snapshots[stage] = copy of the state after that stage."""
    R = R or Real(date, itime, ff)
    tm = timing if timing is not None else {}
    S = S or real_state_at(R, start, ctx)
    _native(S)
    i0, i1 = STAGES.index(start), STAGES.index(stop)
    snaps = {}
    for st in STAGES[i0:i1 + 1]:
        t0 = time.perf_counter()
        if st == 'dyn':
            stage_dyn(S, R, ctx, tm)
        elif st == 'condse':
            if condse == 'run':
                stage_condse(S, R, ctx, ms=ms, cols=condse_cols)
            else:
                condse_from_real(S, R)
        elif st == 'radia':
            stage_radia(S, R, ctx)
        elif st == 'surface':
            stage_surface(S, R, ctx, rec=rec, tm=tm, land_mode=land_mode)
        elif st == 'dissip':
            stage_dissip(S, ctx)
        elif st == 'filter':
            stage_filter(S, ctx)
        tm['stage_' + st] = tm.get('stage_' + st, 0.0) + time.perf_counter() - t0
        _native(S)
        snaps[st] = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in S.items() if not k.startswith('_')}
        if hook is not None:
            hook(st, S, R)
    return S, snaps


def _native(S):
    """Big-endian arrays read from the dumps (dtype '>f8') are converted to native float64 (JAX rejects them)."""
    for k, v in list(S.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f' and v.dtype.byteorder not in ('=', '|'):
            S[k] = v.astype(np.float64)
    return S


def end_reference(R):
    """Real end-of-step state (ffa_step_<it>_e) in the same names/layouts."""
    return {k: R.e[k] for k in R.e if k != 'HDR'}


def stage_references(R):
    """Real boundary references per stage exit, in the state naming."""
    ci, co = R.cse_in, R.cse_out
    refs = {}
    s3 = {k.upper(): v for k, v in R.s3.items()}
    s3.update({k.upper(): v for k, v in R.s4.items()})
    refs['dyn'] = s3
    refs['condse'] = {k: co[k] for k in co}
    r = {k: R.r[k] for k in R.r}
    refs['radia'] = r
    ps = R.post_surface()
    refs['surface'] = {k: ps[k] for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI')}
    refs['dissip'] = {k: R.d[k] for k in R.d if k != 'HDR'}
    refs['filter'] = end_reference(R)
    return refs


# ------------------------------------------------------------------------------------------------ free-running chain (start of F2)
CARRY_KEYS = ("TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV CLDSAV1 FSS TAUSS TAUSSIP TAUMC CLDSS CLDMC CSIZMC CSIZSS CSIZSSIP QLSS QISS QLMC "
              "QIMC W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE "
              "DIM_ST_WATER DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV LMC SNOAGE P_ACC PM_ACC AIRX").split()


def run_free(date, it0, nsteps, ctx, land_mode='recorded', condse='run', ff=FF, log=None):
    """Chain `nsteps` consecutive steps from the REAL state at it0, each step starting from OUR previous end state (atmosphere and
    ATURB/PBL hidden state, CONDSE cloud/precipitation carry state, CONDSE module arrays).  Everything that is not atmosphere
    (sea-ice, lake, land, ocean surface state, Ent exports, radiation SRHR/TRHR/COSZ1, the SURFACE tile records of step k) is the
    RECORDED one of step k.  Returns a list of per-step end-state comparisons against the real end state."""
    S, ms, out = None, {}, []
    for k in range(nsteps):
        R = Real(date, it0 + k, ff)
        S, sn = run_step(date, it0 + k, ctx, R=R, S=S, ms=ms, land_mode=land_mode, condse=condse)
        X = S.get('_condse_X')
        carry = {key: np.array(X[key], copy=True) for key in CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:                                   # RADIA's CLDSS/CLDMC masking on radiation steps
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        st = compare_state(sn['filter'], end_reference(R), END_FIELDS)
        out.append(dict(itime=it0 + k, stats=st))
        if log:
            log(it0 + k, st)
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
    return out
