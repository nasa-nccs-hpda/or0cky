"""D122: the whole dynamics block of one 30-minute physics step (atm_phase1 from the energy bookkeeping before
CALL DYNAM through calc_kea_3d), chained from the individual full-fidelity ports, in the real call order.

Fortran (model/ of modelE2_planet_2.0; see scoping/ATM_DYNAMICS_CHAIN_PLAN.md for the line-by-line plan):
  ATM_DRV.f atm_phase1 88-95 (MAOLD, SEINIT, KEINIT), 105 DYNAM, 134 COMPUTE_WSAVE, 136-142 QCL/QCI rescale,
  143 QDYNAM, 176-205 energy fix, 212 CALC_TROP, 216 PGRAD_PBL, 235 calc_kea_3d;
  ATMDYN.f DYNAM 186-390 (five leapfrog passes: forward, backward, even, odd, even).

Design.  The step is a PLAN: a list of Stage objects (kind + argument bindings by Fortran variable name) executed by
`run_plan` on a workspace dict `w` that mirrors the Fortran variables (U,V,T,TMOM,MA,MASUM,UX,VX,UT,VT,MODD1,MODD3,
MEVEN,MSUMODD, and the DYNAMICS/ATM_COM module arrays MU,MV,MW,CONV,SPA,PEDN,PMID,PK,P,MUS,MVS,MWS,GZ ...).  Every
call is a data binding (e.g. aflux reads u='UX', v='VX', ma='MODD3', ...), exactly the actual-argument list of the
Fortran CALL, so the plan is the call sequence and can be mutated (drop / swap a stage) by tests.  The numerical
work is done only by the existing ports (dyn_aflux_ff, dyn_advecv_ff, dyn_pgf_ff, dyn_aadvt_ff, dyn_isotropuv_ff,
dyn_sdrag_ff, dyn_fltruv_ff, dyn_aadvq_ff, dyn_filter_ff, dyn_glue_ff); the glue in DYNAM that no port covers is
implemented here: MUs/MVs/MWs zeroing and accumulation, MASUM re-initialisation, UX/UT/VX/VT/TZ copies, MEVEN/MODD1
copies, PU/PV/SD scaling, MMA = MEVEN*AXYP, TT/TZT averaging, the flux scaling by DTLF, the leapfrog control flow
(NS counter), the SDRAG full-field wrapper and the MAtoP/MAtoPMB module-state bookkeeping.

Not ported (diagnostics only, no state feedback, inferred from names and call patterns): DIAGA0, DIAGA, DIAGB, EPFLUX
(empty entry in STRAT_DUM), COMPUTE_MASS_FLUX_DIAGS, COMPUTE_DYNAM_AIJ_DIAGNOSTICS, AIJ/AJL accumulations, the
mid-step MAtoPMB that DYNAM calls when MODDA<2 (its results are overwritten by the next ADVECM MAtoP / the final
MAtoPMB, and the MASUM it writes equals ADVECM's MSUM bit for bit because both sum L=LM..1 sequentially).

Array conventions as in the ports: U,V,T,Q,QCL,QCI,MU,MV,MUS,MVS,MWS,GZ (IM,JM,LM); MW (IM,JM,LM-1) inside the
leapfrog; MA,PMID,PK (LM,IM,JM); PEDN (LM+1,IM,JM); TMOM,QMOM (9,IM,JM,LM); MASUM,P (IM,JM).
"""
import time

import numpy as np

import dyn_aflux_ff as fa
import dyn_aadvq_ff as aq
import dyn_aadvt_ff as ft
import dyn_advecv_ff as fv
import dyn_avrx_compare as ac
import dyn_avrx_ff as favrx
import dyn_filter_ff as ffl
import dyn_fltruv_ff as flt
import dyn_glue_ff as gf
import dyn_glue_io as gio
import dyn_isotropuv_ff as fi
import dyn_pgf_ff as fp
import dyn_qdynam_io as qio
import dyn_sdrag_compare as sc
import dyn_sdrag_ff as fs

IM, JM, LM = fa.IM, fa.JM, fa.LM
FF_DEFAULT = ac.FF_DEFAULT
DATES = ac.DATES
NSTEP = 6
DT = 450.0
MZ = 2                                   # TMOM index of the z moment (0-based; Fortran MZ=3)
NDAA = 13                                # rundeck P2SAoM40.R:265 ndaa=13 (DIAG_COM default 7)
ITIMEI = 16032                           # 'itimei' of the three restarts fort1_<date>_itime<N>.nc (netCDF variable)
STATE_KEYS = ('u', 'v', 't', 'q', 'qcl', 'qci', 'ma', 'pedn', 'pmid', 'pk', 'p', 'masum', 'tmom', 'qmom',
              'mus', 'mvs', 'mws', 'gz')


# ----------------------------------------------------------------------------- context (one-time constants)
class Ctx:
    """Geometry / constants / tables of one run, all from the recorded one-time dumps (ffd_aflux_geom, ffd_avrx_consts,
    ffd_sdrag_consts, ffd_glue_consts, ffd_qdyn_geom)."""


def load_ctx(date, ff=FF_DEFAULT, imf_pow=False):
    c = Ctx()
    c.date = date
    c.imf_pow = imf_pow
    c.g = g = fa.load_geom(f"{ff}/{date}/ffd_aflux_geom.bin")
    c.tab = fa.avrx_tables(g)
    k = ac.load_consts(date, ff)
    c.C, c.S = k['C'], k['S']
    c.sdp, c.sdgeo = sc.load_consts(date, ff)
    c.gg = gio.load_g(date, ff)
    c.qg = qio.load_geom(qio.gname(date, ff))
    c.iso_geo = dict(cosv=g['cosv'], dxv=g['dxv'], cosiv=g['cosiv'], siniv=g['siniv'], fjeq=.5 * (1 + JM))
    c.flt_geo = dict(dxyn=g['dxyn'], dxys=g['dxys'], cosv=g['cosv'], radius=g['radius'])
    c.omega = g['omega']
    c.axyp = c.gg['axyp']
    return c


# ----------------------------------------------------------------------------- state I/O (ffd_state_<itime>_s<k>)
def state_path(date, itime, site, ff=FF_DEFAULT):
    return f"{ff}/{date}/ffd_state_{itime}_s{site}.bin"


def _r(raw, o, shape):
    n = int(np.prod(shape))
    return raw[o:o + n].reshape(shape, order='F').copy(), o + n


def load_state(path):
    """ffd_state_<itime>_s1|2|3.bin (written by ATM_DRV_dynG.f.patch ffds_state): header [itime,site,0,0], then U V T Q QCL
    QCI (IM,JM,LM), MA (LM,IM,JM), PEDN (LM+1,IM,JM), PMID PK (LM,IM,JM), P MASUM (IM,JM), TMOM QMOM (9,IM,JM,LM),
    MUs MVs MWs GZ (IM,JM,LM)."""
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), site=int(raw[1]))
    o = 4
    for k, sh in (('u', (IM, JM, LM)), ('v', (IM, JM, LM)), ('t', (IM, JM, LM)), ('q', (IM, JM, LM)),
                  ('qcl', (IM, JM, LM)), ('qci', (IM, JM, LM)), ('ma', (LM, IM, JM)), ('pedn', (LM + 1, IM, JM)),
                  ('pmid', (LM, IM, JM)), ('pk', (LM, IM, JM)), ('p', (IM, JM)), ('masum', (IM, JM)),
                  ('tmom', (9, IM, JM, LM)), ('qmom', (9, IM, JM, LM)), ('mus', (IM, JM, LM)), ('mvs', (IM, JM, LM)),
                  ('mws', (IM, JM, LM)), ('gz', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size, (o, raw.size)
    return d


def load_exports(path):
    """ffd_state_<itime>_s4.bin: [itime,4,0,0], PTROPO LTROPO (IM,JM), WSAVE (IM,JM,LM-1), KEA (IM,JM,LM), DPDX_BY_RHO
    DPDY_BY_RHO DPDX_BY_RHO_0 DPDY_BY_RHO_0 (IM,JM), PHI (IM,JM,LM)."""
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]))
    o = 4
    for k, sh in (('ptropo', (IM, JM)), ('ltropo', (IM, JM)), ('wsave', (IM, JM, LM - 1)), ('kea', (IM, JM, LM)),
                  ('dpdx', (IM, JM)), ('dpdy', (IM, JM)), ('dpdx0', (IM, JM)), ('dpdy0', (IM, JM)),
                  ('phi', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    d['ltropo'] = d['ltropo'].astype(int)
    return d


def available(date, ff=FF_DEFAULT):
    import os
    it0 = dict(DATES)[date]
    need = [f"{ff}/{date}/ffd_aflux_geom.bin", f"{ff}/{date}/ffd_qdyn_geom.bin", f"{ff}/{date}/ffd_glue_consts.bin",
            f"{ff}/{date}/ffd_avrx_consts.bin", f"{ff}/{date}/ffd_sdrag_consts.bin"]
    need += [state_path(date, it0 + k, s, ff) for k in range(NSTEP) for s in (1, 2, 3, 4)]
    need += [f"{ff}/{date}/ffd_aflux_{it0 + k}_p5_out.bin" for k in range(NSTEP)]
    return all(os.path.exists(p) for p in need)


# ----------------------------------------------------------------------------- the plan
class Stage:
    """kind + argument bindings (names of workspace variables, i.e. the actual arguments of the Fortran CALL)."""

    def __init__(self, kind, pas=0, **a):
        self.kind = kind
        self.pas = pas
        self.a = a

    def __repr__(self):
        return f"Stage({self.kind}, p{self.pas}, {self.a})"

    @property
    def name(self):
        return f"p{self.pas}.{self.kind}" if self.pas else self.kind


def dynam_plan(nidyn=4, nstep=None, ndaa=NDAA):
    """The call sequence of DYNAM (ATMDYN.f:186-390) for NIdyn=nidyn, as a list of Stage.  The leapfrog control flow
    (labels 300/340/360 and the NS counter) is simulated here; `pas` numbers the AFLUX calls 1..5 as in the dumps.
    nstep = (Itime-ItimeI)*NIdyn (ATM_DRV.f:80): with it the diagnostic branch `If (MODDA < 2) {MAtoPMB, DIAGA, DIAGB,
    EPFLUX}` of the even pass (ATMDYN.f:352-357, MODDA = Mod(NSTEP+4-NS+NDAA*NIDYN, NDAA*NIDYN+2)) is added to the
    plan: DIAGA fills Q at the poles (DIAG.f:221-233), the only state effect of the diagnostic calls (stage 'diaga').
    nstep=None omits it."""
    dtfs = DT * 2. / 3.
    dtlf = 2. * DT
    P = []
    add = lambda kind, pas=0, **a: P.append(Stage(kind, pas, **a))
    state = dict(pas=0)

    def fwd_bwd(ns):
        p = state['pas'] + 1
        # initial forward step (MRCH=0)
        add('reinit', p)
        add('aflux', p, ns=ns, u='U', v='V', ma='MA', masum='MASUM', me='MA', mesum='MASUM')
        add('advecm', p, dt=dtfs, mold='MA', mnew='MODD3', msum='MSUMODD')
        add('advecv', p, dt=dtfs, u='U', v='V', mmean='MA', mbefor='MA', ut='UX', vt='VX', mafter='MODD3')
        add('pgf', p, dt=dtfs, mam='MA', ut='UX', vt='VX', mafter='MODD3', s0='T', sz='TZ')
        add('pscale', p)
        add('iso', p, a='UX', b='VX')
        # initial backward step (MRCH=-1)
        p += 1
        add('aflux', p, ns=ns, u='UX', v='VX', ma='MODD3', masum='MSUMODD', me='MA', mesum='MASUM')
        add('advecm', p, dt=DT, mold='MA', mnew='MODD1', msum='MSUMODD')
        add('advecv', p, dt=DT, u='UX', v='VX', mmean='MODD3', mbefor='MA', ut='UT', vt='VT', mafter='MODD1')
        add('pgf', p, dt=DT, mam='MODD3', ut='UT', vt='VT', mafter='MODD1', s0='T', sz='TZ')
        add('pscale', p)
        add('iso', p, a='UT', b='VT')
        state['pas'] = p

    def odd(ns):
        p = state['pas'] + 1
        add('aflux', p, ns=ns, u='U', v='V', ma='MA', masum='MASUM', me='MODD1', mesum='MSUMODD')
        add('advecm', p, dt=dtlf, mold='MODD1', mnew='MODD3', msum='MSUMODD')
        add('advecv', p, dt=dtlf, u='U', v='V', mmean='MA', mbefor='MODD1', ut='UT', vt='VT', mafter='MODD3')
        add('pgf', p, dt=dtlf, mam='MA', ut='UT', vt='VT', mafter='MODD3', s0='T', sz='TZ')
        add('pscale', p)
        add('iso', p, a='UT', b='VT')
        add('copy', p, dst='MODD1', src='MODD3')
        state['pas'] = p

    def even(ns, call):
        p = state['pas'] + 1
        add('copy', p, dst='MEVEN', src='MA')
        add('aflux', p, ns=ns, u='UT', v='VT', ma='MODD1', masum='MSUMODD', me='MEVEN', mesum='MASUM')
        add('advecm', p, dt=dtlf, mold='MEVEN', mnew='MA', msum='MASUM')
        add('advecv', p, dt=dtlf, u='UT', v='VT', mmean='MODD1', mbefor='MEVEN', ut='U', vt='V', mafter='MA')
        add('pscale', p)
        add('accum', p)
        add('copy', p, dst='TT', src='T')
        add('copy', p, dst='TZT', src='TZ')
        add('mma', p)
        add('aadvt', p, dt=dtlf, call=call)
        add('tz', p)
        add('avg', p)
        add('pgf', p, dt=dtlf, mam='MODD1', ut='U', vt='V', mafter='MA', s0='TT', sz='TZT')
        add('iso', p, a='U', b='V')
        add('sdrag', p, dt=dtlf)
        if nstep is not None and (nstep + 4 - ns + ndaa * nidyn) % (ndaa * nidyn + 2) < 2:
            add('matopmb', p)                         # If (MODDA < 2): Call MAtoPMB, DIAGA, DIAGB, EPFLUX
            add('diaga', p)
        state['pas'] = p

    add('flux_zero')
    ns = nidyn
    nsold = nidyn
    fwd_bwd(ns)                                       # label 300 and the two initial steps, then GO TO 360
    call = 0
    phase = 'even'
    while True:
        if phase == 'odd':
            odd(ns)                                   # label 340
            ns -= 1
        call += 1
        even(ns, call)                                # label 360
        ns -= 1
        if nsold - ns < 8 and ns > 1:
            phase = 'odd'
            continue
        nsold = ns
        if ns > 1:
            fwd_bwd(ns)
            phase = 'even'
            continue
        break
    add('matopmb')                                    # If (MODDA >= 2) Call MAtoPMB (identical result either way)
    add('flux_scale')
    add('filter_chain')                               # FLTRUV, conserv_amb_ext, fltry2 x2, conserv_amb_ext, glue
    return P


def step_plan(nidyn=4, nstep=None):
    """atm_phase1 from the pre-dynamics bookkeeping to calc_kea_3d (ATM_DRV.f:88-235)."""
    P = [Stage('save_old'), Stage('se_init'), Stage('ke_init')]
    P += dynam_plan(nidyn, nstep)
    P += [Stage('wsave'), Stage('qscale'), Stage('qdynam'), Stage('se_final'), Stage('ke_final'),
          Stage('efix'), Stage('trop'), Stage('pgrad'), Stage('kea')]
    return P


# ----------------------------------------------------------------------------- stage execution
def workspace(state):
    """Workspace (Fortran variable names) from a state dict (keys STATE_KEYS)."""
    w = {k.upper(): np.array(state[k], copy=True) for k in STATE_KEYS}
    w['DUT'] = np.zeros((IM, JM, LM))
    w['DVT'] = np.zeros((IM, JM, LM))
    return w


def sdrag_field(u, v, t, pk, pedn, ma, dt1, ctx):
    """SDRAG (ATMDYN.f:1993-2141) on the whole field: the column port applied to every B-grid column (I=1..IM,
    J=2..JM) with its four neighbouring MA columns (Ip1, cyclic).  pedn is PEDN(LM+1,IM,JM)."""
    jj = np.arange(1, JM)
    ii = np.arange(IM)
    I, J = np.meshgrid(ii, jj, indexing='ij')
    I = I.ravel(); J = J.ravel()
    ip1 = (I + 1) % IM
    un, vn, ex = fs.sdrag_columns(u[I, J, :], v[I, J, :], t[I, J, :], pk[:, I, J].T, pedn[1:, I, J].T,
                                  ma[:, ip1, J - 1].T, ma[:, I, J - 1].T, ma[:, ip1, J].T, ma[:, I, J].T,
                                  J + 1, dt1, ctx.sdp, ctx.sdgeo)
    u = u.copy(); v = v.copy()
    u[I, J, :] = un
    v[I, J, :] = vn
    return u, v, ex


def _set_matop(w, r):
    """Module-state side effect of MAtoP (called by ADVECM): PEDN(1:LM), PMID, PDSIG, PK, P rewritten."""
    pedn = w['PEDN'].copy()
    pedn[:LM] = r['pedn']
    w['PEDN'] = pedn
    w['PMID'] = r['pmid']; w['PDSIG'] = r['pdsig']; w['PK'] = r['pk']; w['P'] = r['p']


def exec_stage(st, w, ctx):
    k, a = st.kind, st.a
    g = ctx.g
    kg2mb = g['kg2mb']
    if k == 'save_old':
        w['MAOLD'] = w['MA'].copy()
        w['PMIDOLD'] = w['PMID'].copy()
    elif k == 'se_init':
        w['SEINIT'] = gf.conserv_se(w['MA'], w['MASUM'], w['PK'], w['T'], w['Q'], w['QCI'], ctx.gg)
    elif k == 'ke_init':
        w['KEINIT'] = ffl.conserv_ke(w['MA'], w['U'], w['V'], ctx.gg)[0]
    elif k == 'flux_zero':
        for n in ('MUS', 'MVS', 'MWS'):
            w[n] = np.zeros((IM, JM, LM))
    elif k == 'reinit':                                  # label 300
        w['MASUM'] = fa.seqsum(w['MA'], 0)               # Do I,J: MASUM = Sum(MA(:,I,J))
        w['UX'] = w['U'].copy(); w['UT'] = w['U'].copy()
        w['VX'] = w['V'].copy(); w['VT'] = w['V'].copy()
        w['TZ'] = w['TMOM'][MZ].copy()                   # TZ(:,:,:) = TMOM(MZ,:,:,:)
    elif k == 'aflux':
        r = fa.aflux(a['ns'], w[a['u']], w[a['v']], w[a['ma']], w[a['masum']], w[a['me']], w[a['mesum']], g,
                     tab=ctx.tab)
        w['MU'] = r['mu']; w['MV'] = r['mv']; w['MW'] = r['mw']; w['CONV'] = r['conv']; w['SPA'] = r['spa']
    elif k == 'advecm':
        r = fa.advecm(a['dt'], w[a['mold']], w['CONV'], w['MW'], g, imf_pow=ctx.imf_pow)
        if r['n_exception'] == 2:
            raise RuntimeError('ADVECM: Mass diagnostic error (stop_model 11)')
        w[a['mnew']] = r['mnew']; w[a['msum']] = r['msum']
        _set_matop(w, r)
    elif k == 'advecv':
        ut, vt = fv.advecv(a['dt'], w[a['u']], w[a['v']], w[a['mmean']], w[a['mbefor']], w[a['ut']], w[a['vt']],
                           w[a['mafter']], w['MU'], w['MV'], w['MW'], w['SPA'], g)
        w[a['ut']] = ut; w[a['vt']] = vt
        w['DUT'] = np.zeros((IM, JM, LM)); w['DVT'] = np.zeros((IM, JM, LM))   # ADVECV leaves DUT,DVT zero
    elif k == 'pgf':
        r = fp.pgf(a['dt'], w[a['mam']], w[a['ut']], w[a['vt']], w[a['mafter']], w[a['s0']], w[a['sz']],
                   w['DUT'], w['DVT'], g, tab=ctx.tab, imf_pow=ctx.imf_pow)
        w[a['ut']] = r['ut']; w[a['vt']] = r['vt']; w['DUT'] = r['dut']; w['DVT'] = r['dvt']
        w['GZ'] = r['gz']; w['PHI'] = r['phi']; w['SPA'] = r['adm']
    elif k == 'pscale':
        w['PU'] = w['MU'] * kg2mb; w['PV'] = w['MV'] * kg2mb; w['SD'] = w['MW'] * kg2mb
    elif k == 'iso':
        w[a['a']], w[a['b']] = fi.isotropuv(w[a['a']], w[a['b']], ctx.iso_geo, ctx.C, ctx.S)
    elif k == 'copy':
        w[a['dst']] = w[a['src']].copy()
    elif k == 'accum':                                   # ACCUMULATE MASS FLUXES FOR TRACERS and Q
        w['MUS'] = w['MUS'] + w['PU']
        w['MVS'] = w['MVS'] + w['PV']
        mws = w['MWS'].copy()
        mws[:, :, :LM - 1] = mws[:, :, :LM - 1] + w['SD']
        w['MWS'] = mws
    elif k == 'mma':                                     # MMA(:,:,L) = MEVEN(L,:,:)*AXYP(:,:)
        mma = np.empty((IM, JM, LM))
        for l in range(LM):
            mma[:, :, l] = w['MEVEN'][l] * ctx.axyp
        w['MMA'] = mma
    elif k == 'aadvt':
        r = ft.aadvt(a['dt'], w['MMA'], w['T'], w['TMOM'], w['MU'], w['MV'], w['MW'], False)
        w['T'] = r['rm']; w['TMOM'] = r['rmom']; w['MMA'] = r['mm']; w['FPEU'] = r['fqu']; w['FPEV'] = r['fqv']
    elif k == 'tz':
        w['TZ'] = w['TMOM'][MZ].copy()
    elif k == 'avg':
        w['TT'] = .5 * (w['T'] + w['TT'])
        w['TZT'] = .5 * (w['TZ'] + w['TZT'])
    elif k == 'sdrag':
        w['U'], w['V'], _ = sdrag_field(w['U'], w['V'], w['T'], w['PK'], w['PEDN'], w['MA'], a['dt'], ctx)
    elif k == 'matopmb':
        r = ffl.matopmb(w['MA'], g, imf_pow=ctx.imf_pow)
        w['MASUM'] = r['masum']; w['PEDN'] = r['pedn']; w['PMID'] = r['pmid']; w['PK'] = r['pk']
        w['PDSIG'] = r['pdsig']; w['P'] = r['p']
    elif k == 'flux_scale':                              # (mb*m^2/s) -> (mb*m^2)
        dtlf = 2. * DT
        w['MUS'] = w['MUS'] * dtlf
        w['MVS'] = w['MVS'] * dtlf
        mws = w['MWS'].copy()
        mws[:, :, :LM - 1] = mws[:, :, :LM - 1] * dtlf
        w['MWS'] = mws
    elif k == 'diaga':                                   # DIAGA: Q at the poles := Q(1,pole,:) (DIAG.f:221-233)
        q = w['Q'].copy()
        q[1:, 0, :] = q[0, 0, :][None, :]
        q[1:, JM - 1, :] = q[0, JM - 1, :][None, :]
        w['Q'] = q
    elif k == 'filter_chain':
        u, v, dam = flt.filter_chain(w['U'], w['V'], w['MA'], w['MASUM'], ctx.flt_geo, ctx.omega)
        w['U'] = u; w['V'] = v; w['DAMSUM'] = dam
    elif k == 'wsave':
        w['WSAVE'] = gf.compute_wsave(w['MWS'], w['T'], w['PK'], w['PEDN'], ctx.gg)
    elif k == 'qscale':                                  # QCL,QCI scaled by MAOLD/MA
        for n in ('QCL', 'QCI'):
            x = w[n].copy()
            for l in range(LM):
                x[:, :, l] = x[:, :, l] * (w['MAOLD'][l] / w['MA'][l])
            w[n] = x
    elif k == 'qdynam':
        qg = ctx.qg
        r = aq.qdynam(w['Q'], w['QMOM'], w['MAOLD'], w['MUS'], w['MVS'], w['MWS'], qg['axyp'], qg['imaxj'],
                      qg['kg2mb'], qg['byim_geom'], qg['byim_qus'])
        w['Q'] = r['q']; w['QMOM'] = r['qmom']
        w['MUS'] = r['q0']['mu']; w['MVS'] = r['q0']['mv']; w['MWS'] = r['q0']['mw']   # AADVQ0 rewrites MUs,MVs,MWs
    elif k == 'se_final':
        w['SEFINAL'] = gf.conserv_se(w['MA'], w['MASUM'], w['PK'], w['T'], w['Q'], w['QCI'], ctx.gg)
    elif k == 'ke_final':
        w['KEFINAL'] = ffl.conserv_ke(w['MA'], w['U'], w['V'], ctx.gg)[0]
    elif k == 'efix':
        r = gf.energy_fix(w['SEINIT'], w['KEINIT'], w['SEFINAL'], w['KEFINAL'], w['MASUM'], w['T'], w['PK'], ctx.gg)
        w['T'] = r['t']; w['DSEPKE'] = r['dsepke']; w['MMGLOB'] = r['mmglob']
    elif k == 'trop':
        pt, lt, ierr, tt = gf.calc_trop(w['T'], w['PK'], w['PMID'], ctx.gg, imf_pow=ctx.imf_pow)
        w['PTROPO'] = pt; w['LTROPO'] = lt
    elif k == 'pgrad':
        o = gf.pgrad_pbl(w['T'][:, :, 0], w['PK'][0], w['PMID'][0], w['PEDN'][0], w['PHI'][:, :, 0],
                         ctx.gg['zatmo'], ctx.gg)
        w['DPDX'], w['DPDY'], w['DPDX0'], w['DPDY0'] = o
    elif k == 'kea':
        w['KEA'] = gf.calc_kea_3d(w['U'], w['V'], ctx.gg)
    else:
        raise ValueError(k)


def run_plan(plan, w, ctx, hook=None, timing=None):
    for st in plan:
        if hook is not None:
            hook('pre', st, w, ctx)
        t0 = time.perf_counter()
        exec_stage(st, w, ctx)
        if timing is not None:
            timing[st.kind] = timing.get(st.kind, 0.0) + time.perf_counter() - t0
        if hook is not None:
            hook('post', st, w, ctx)
    return w


def dyn_step(state, ctx, itime=None, plan=None, hook=None, timing=None):
    """Run the dynamics block of one physics step from `state` (dict with STATE_KEYS, the real state at the start of
    the step); `itime` (the model Itime of the step) selects the steps on which DIAGA fires (see dynam_plan).  Returns the workspace `w` (uppercase Fortran names): final state U,V,T,Q,QCL,QCI,MA,PEDN,PMID,PK,P,
    MASUM,TMOM,QMOM,MUS,MVS,MWS,GZ plus the physics exports WSAVE,PTROPO,LTROPO,DPDX..,KEA,PHI."""
    w = workspace(state)
    if plan is None:
        plan = step_plan(nstep=None if itime is None else (itime - ITIMEI) * 4)
    return run_plan(plan, w, ctx, hook, timing)
