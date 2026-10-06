"""D123: validate dyn_step.py (the chained dynamics block of one physics step) against the real-Fortran dumps.

Real data (per date nov26 33312, dec01 33552, jan01 17520; 6 steps each):
  ffd_state_<itime>_s1|s2|s3|s4.bin   written by instrumentation/ATM_DRV_dynG.f.patch (D122): full atmosphere state at the
      step start (s1), at DYNAM exit (s2), at the end of the dynamics block after QDYNAM and the energy fix (s3) and the
      physics exports after CALC_TROP / PGRAD_PBL / COMPUTE_WSAVE / calc_kea_3d (s4).  Loaders: dyn_step.load_state/load_exports.
  ffd_<itime>_pre_condse.bin          existing whole-state dump at the same point as s3 (an independent second reference)
  per-call dumps of the stage routines (ffd_aflux_*, ffd_aflux_advecm_*, ffd_advecv_*, ffd_pgf_*, ffd_aadvt_*): 5 leapfrog
      passes per step (p1 fwd, p2 bwd, p3 even, p4 odd, p5 even), 2 AADVT calls (c1, c2 = passes 3 and 5).

Modes
  chain     the plan runs from the real step-start state only; every stage's INPUTS (as produced by the earlier port
            stages) and OUTPUTS are compared with the stage's real per-call dump.  Final state compared with s2/s3/s4.
  boundary  stage-boundary replay: before every dumped stage all inputs except those produced by the port stages since
            the previous dumped stage are replaced by the real recorded inputs, so each routine is tested with the port's
            own output of the preceding stage(s) and real everything else.
Usage (from fullfidelity/): python dyn_step_compare.py [--numpy-pow] [--both-pow] [--boundary] [--date nov26] [--steps N]
  default: libimf pow (bitwise mode; needs the Intel runtime), chain mode, 3 dates x 6 steps.
"""
import os
import sys
import time

import numpy as np

import dyn_aadvt_compare as tc
import dyn_advecv_compare as vc
import dyn_aflux_compare as cm
import dyn_pgf_compare as pc
import dyn_step as ds
import ffdump_reader
import intel_libm_ff

IM, JM, LM = ds.IM, ds.JM, ds.LM


def stats(a, b):
    """max |a-b|, scale-relative (max|a-b|/max|b|), elementwise relative on |b|>1e-6 max|b|, #unequal, N."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    sc = float(np.max(np.abs(b))) if b.size else 0.0
    mx = float(d.max()) if d.size else 0.0
    m = np.abs(b) > 1e-6 * sc
    er = float(np.max(d[m] / np.abs(b[m]))) if m.any() else 0.0
    return dict(abs=mx, rel=(mx / sc if sc > 0 else 0.0), erel=er, nne=int(np.sum(a != b)), n=int(a.size))


def merge(acc, key, s):
    o = acc.get(key)
    if o is None:
        acc[key] = dict(s)
    else:
        for k in ('abs', 'rel', 'erel'):
            o[k] = max(o[k], s[k])
        o['nne'] += s['nne']
        o['n'] += s['n']


# ----------------------------------------------------------------------------- per-call dump access
def _fn(kind, date, itime, pas, ff):
    return cm.fname(date, itime, pas, kind, ff)


DUMP_KINDS = ('aflux', 'advecm', 'advecv', 'pgf', 'aadvt')


def written(st):
    """Workspace variables written by a stage."""
    k, a = st.kind, st.a
    return {
        'aflux': {'MU', 'MV', 'MW', 'CONV', 'SPA'},
        'advecm': {a.get('mnew'), a.get('msum'), 'PEDN', 'PMID', 'PDSIG', 'PK', 'P'},
        'advecv': {a.get('ut'), a.get('vt'), 'DUT', 'DVT'},
        'pgf': {a.get('ut'), a.get('vt'), 'DUT', 'DVT', 'GZ', 'PHI', 'SPA'},
        'iso': {a.get('a'), a.get('b')},
        'copy': {a.get('dst')},
        'pscale': {'PU', 'PV', 'SD'},
        'accum': {'MUS', 'MVS', 'MWS'},
        'mma': {'MMA'},
        'aadvt': {'T', 'TMOM', 'MMA', 'FPEU', 'FPEV'},
        'tz': {'TZ'},
        'avg': {'TT', 'TZT'},
        'sdrag': {'U', 'V'},
        'reinit': {'MASUM', 'UX', 'UT', 'VX', 'VT', 'TZ'},
        'diaga': {'Q'},
        'matopmb': {'MASUM', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P'},
    }.get(k, set())


def stage_inputs(st, date, itime, ff):
    """-> list of (workspace var, real array, row slice used for the comparison)."""
    a, p = st.a, st.pas
    full = slice(None)
    r1 = (slice(None), slice(1, None))              # J=2..JM of an (IM,JM,..) array
    if st.kind == 'aflux':
        i = cm.load_aflux_in(_fn('aflux', date, itime, p, ff) + "_in.bin")
        return [(a['u'], i['u'], full), (a['v'], i['v'], full), (a['ma'], i['ma'], full), (a['masum'], i['masum'], full),
                (a['me'], i['me'], full), (a['mesum'], i['mesum'], full)]
    if st.kind == 'advecm':
        i = cm.load_advecm_in(_fn('aflux_advecm', date, itime, p, ff) + "_in.bin")
        return [(a['mold'], i['mold'], full), ('CONV', i['conv'], full), ('MW', i['mw'], full)]
    if st.kind == 'advecv':
        i = vc.load_advecv_in(_fn('advecv', date, itime, p, ff) + "_in.bin")
        return [(a['u'], i['u'], full), (a['v'], i['v'], full), (a['mmean'], i['mmean'], full),
                (a['mbefor'], i['mbefor'], full), (a['mafter'], i['mafter'], full), (a['ut'], i['ut'], full),
                (a['vt'], i['vt'], full), ('MU', i['mu'], full), ('MV', i['mv'], r1), ('MW', i['mw'], full),
                ('SPA', i['spa'], full)]
    if st.kind == 'pgf':
        i = pc.load_pgf_in(_fn('pgf', date, itime, p, ff) + "_in.bin")
        return [(a['mam'], i['mam'], full), (a['mafter'], i['mafter'], full), (a['s0'], i['s0'], full),
                (a['sz'], i['sz'], full), (a['ut'], i['ut'], full), (a['vt'], i['vt'], full),
                ('DUT', i['dut'], r1), ('DVT', i['dvt'], r1)]
    if st.kind == 'aadvt':
        i = tc.load_in(tc.fname(date, itime, a['call'], 'in', ff))
        return [('MMA', i['mma'], full), ('T', i['t'], full), ('TMOM', i['tmom'], full), ('MU', i['mu'], full),
                ('MV', i['mv'], r1), ('MW', i['mw'], full)]
    return []


def _sl(x, sl):
    return x if sl == slice(None) else x[sl]


def stage_outputs(st, w, date, itime, ff):
    """-> list of (label, port array, real array)."""
    a, p = st.a, st.pas
    if st.kind == 'aflux':
        o = cm.load_aflux_out(_fn('aflux', date, itime, p, ff) + "_out.bin")
        return [('MU', w['MU'], o['mu']), ('MV', w['MV'][:, 1:], o['mv'][:, 1:]), ('MW', w['MW'], o['mw']),
                ('CONV', w['CONV'], o['conv']), ('SPA', w['SPA'], o['spa'])]
    if st.kind == 'advecm':
        o = cm.load_advecm_out(_fn('aflux_advecm', date, itime, p, ff) + "_out.bin")
        return [('MNEW', w[a['mnew']], o['mnew']), ('MSUM', w[a['msum']], o['msum']), ('PEDN', w['PEDN'][:LM], o['pedn']),
                ('PMID', w['PMID'], o['pmid']), ('PDSIG', w['PDSIG'], o['pdsig']), ('PK', w['PK'], o['pk']),
                ('P', w['P'], o['p'])]
    if st.kind == 'advecv':
        o = vc.load_advecv_out(_fn('advecv', date, itime, p, ff) + "_out.bin")
        return [('UT', w[a['ut']], o['ut']), ('VT', w[a['vt']], o['vt'])]
    if st.kind == 'pgf':
        o = pc.load_pgf_out(_fn('pgf', date, itime, p, ff) + "_out.bin")
        return [('UT', w[a['ut']], o['ut']), ('VT', w[a['vt']], o['vt']), ('DUT', w['DUT'][:, 1:], o['dut'][:, 1:]),
                ('DVT', w['DVT'][:, 1:], o['dvt'][:, 1:]), ('GZ', w['GZ'], o['gz']), ('PHI', w['PHI'], o['phi']),
                ('ADM', w['SPA'], o['adm'])]
    if st.kind == 'aadvt':
        o = tc.load_out(tc.fname(date, itime, a['call'], 'out', ff))
        return [('MMA', w['MMA'], o['mma']), ('T', w['T'], o['t']), ('TMOM', w['TMOM'], o['tmom']),
                ('FPEU', w['FPEU'], o['fpeu']), ('FPEV', w['FPEV'], o['fpev'])]
    return []


class Recorder:
    """Hook for dyn_step.run_plan: stage input/output comparison (chain) or boundary resync + output comparison."""

    def __init__(self, date, itime, ff, boundary=False):
        self.date, self.itime, self.ff, self.boundary = date, itime, ff, boundary
        self.res = {}                  # key -> stats, key = 'p3.aflux.in.U' etc (per pass, per step)
        self.fresh = set()

    def __call__(self, when, st, w, ctx):
        if st.kind in DUMP_KINDS:
            if when == 'pre':
                for var, real, sl in stage_inputs(st, self.date, self.itime, self.ff):
                    if self.boundary:
                        if var not in self.fresh:
                            w[var] = np.array(real, copy=True)
                    else:
                        merge(self.res, f"{st.name}.in.{var}", stats(_sl(w[var], sl), _sl(real, sl)))
            else:
                for lab, a, b in stage_outputs(st, w, self.date, self.itime, self.ff):
                    merge(self.res, f"{st.name}.out.{lab}", stats(a, b))
                self.fresh = set(written(st))
        elif when == 'post':
            self.fresh |= written(st)


FIELDS_S2 = ('u', 'v', 't', 'q', 'qcl', 'qci', 'ma', 'pedn', 'pmid', 'pk', 'p', 'masum', 'tmom', 'qmom', 'mus', 'mvs',
             'mws', 'gz')


def compare_final(w, ref, keys=FIELDS_S2, rows=None):
    out = {}
    for k in keys:
        out[k] = stats(w[k.upper()], ref[k])
    return out


class SiteHook:
    """Chains Recorder with a snapshot of the workspace right after 'filter_chain' (= DYNAM exit)."""

    def __init__(self, rec):
        self.rec = rec
        self.snap = None

    def __call__(self, when, st, w, ctx):
        self.rec(when, st, w, ctx)
        if when == 'post' and st.kind == 'filter_chain':
            self.snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}


def run_step_compare(date, itime, ctx, ff=ds.FF_DEFAULT, boundary=False, per_stage=True, plan=None):
    """One step: returns dict(s2=..., s3=..., s4=..., pre_condse=..., stages=..., timing, w)."""
    s1 = ds.load_state(ds.state_path(date, itime, 1, ff))
    s2 = ds.load_state(ds.state_path(date, itime, 2, ff))
    s3 = ds.load_state(ds.state_path(date, itime, 3, ff))
    s4 = ds.load_exports(ds.state_path(date, itime, 4, ff))
    rec = Recorder(date, itime, ff, boundary) if per_stage else None
    hook = SiteHook(rec) if per_stage else None
    tm = {}
    t0 = time.perf_counter()
    w = ds.dyn_step(s1, ctx, itime=itime, plan=plan, hook=hook, timing=tm)
    wall = time.perf_counter() - t0
    out = dict(timing=tm, wall=wall, w=w, s1=s1, s2=s2, s3=s3, s4=s4)
    if per_stage:
        out['stages'] = rec.res
        snap = hook.snap
        out['s2cmp'] = compare_final(snap, s2) if snap is not None else None
    out['s3cmp'] = compare_final(w, s3)
    ex = {}
    for k, wk in (('ptropo', 'PTROPO'), ('ltropo', 'LTROPO'), ('wsave', 'WSAVE'), ('kea', 'KEA'), ('dpdx', 'DPDX'),
                  ('dpdy', 'DPDY'), ('dpdx0', 'DPDX0'), ('dpdy0', 'DPDY0'), ('phi', 'PHI')):
        a, b = w[wk], s4[k]
        if k.startswith('dpd'):                      # only the cells the Fortran sets (J=2..JM-1, plus the pole cells)
            mask = np.zeros((IM, JM), bool); mask[:, 1:JM - 1] = True; mask[0, 0] = mask[0, JM - 1] = True
            a, b = a[mask], b[mask]
        ex[k] = stats(a, b)
    out['s4cmp'] = ex
    pcp = f"{ff}/{date}/ffd_{itime}_pre_condse.bin"          # only the first 2 steps of each date were dumped
    if os.path.exists(pcp):
        pc_ = ffdump_reader.read_dump(pcp)
        out['precond'] = {k.lower(): stats(w[k], pc_[k]) for k in ('MA', 'U', 'V', 'T', 'Q', 'QCL', 'QCI', 'PK', 'PMID',
                                                                    'PDSIG', 'P')}
        out['precond']['pedn'] = stats(w['PEDN'], pc_['PEDN'])
    return out


def fmt(s):
    return f"abs={s['abs']:.2e} rel={s['rel']:.1e} erel={s['erel']:.1e} neq={s['nne']}/{s['n']}"


def time_step(date, itime, ctx, n=3, ff=ds.FF_DEFAULT):
    """Warm CPU timing of the chained step without any comparison hooks: one cold call (imports, first-touch), then n
    timed calls.  Returns (cold_s, [warm_s...], per-stage-kind seconds of the last call)."""
    s1 = ds.load_state(ds.state_path(date, itime, 1, ff))
    t0 = time.perf_counter()
    ds.dyn_step(s1, ctx, itime=itime)
    cold = time.perf_counter() - t0
    warm = []
    tm = {}
    for _ in range(n):
        tm = {}
        t0 = time.perf_counter()
        ds.dyn_step(s1, ctx, itime=itime, timing=tm)
        warm.append(time.perf_counter() - t0)
    return cold, warm, tm


def main(argv):
    if '--timing' in argv:
        import platform
        for imf in ([True, False] if intel_libm_ff.available() else [False]):
            ctx = ds.load_ctx('nov26', imf_pow=imf)
            cold, warm, tm = time_step('nov26', 33312, ctx)
            print(f"imf_pow={imf}: cold {cold:.2f}s, warm {', '.join(f'{x:.2f}' for x in warm)}s (min {min(warm):.2f}s, "
                  f"mean {np.mean(warm):.2f}s) on {platform.node()}, numpy {np.__version__}")
            print("  per stage kind (last warm call, s):", {k: round(v, 3) for k, v in sorted(tm.items(), key=lambda x: -x[1])[:8]})
        return
    both = '--both-pow' in argv
    modes = [True, False] if both else [not ('--numpy-pow' in argv)]
    boundary = '--boundary' in argv
    dates = [d for d in ds.DATES if '--date' not in argv or d[0] == argv[argv.index('--date') + 1]]
    nsteps = int(argv[argv.index('--steps') + 1]) if '--steps' in argv else ds.NSTEP
    for imf in modes:
        if imf and not intel_libm_ff.available():
            print("libimf not available: skipping imf-pow mode")
            continue
        print(f"=== imf_pow={imf} boundary={boundary}")
        agg = {'s2': {}, 's3': {}, 's4': {}, 'pre': {}, 'stage': {}}
        wall = []
        for date, it0 in dates:
            ctx = ds.load_ctx(date, imf_pow=imf)
            for k in range(nsteps):
                r = run_step_compare(date, it0 + k, ctx, boundary=boundary)
                wall.append(r['wall'])
                for key, tag in (('s2cmp', 's2'), ('s3cmp', 's3'), ('s4cmp', 's4'), ('precond', 'pre')):
                    if r.get(key):
                        for f, s in r[key].items():
                            merge(agg[tag], f, s)
                for f, s in r['stages'].items():
                    merge(agg['stage'], f, s)
                bad = {f: s['abs'] for f, s in r['s3cmp'].items() if s['abs'] != 0}
                print(f"{date} {it0 + k}: step {r['wall']:.1f}s  s3 non-exact fields: {bad if bad else 'none'}", flush=True)
        for tag, title in (('s2', 'DYNAM exit (s2)'), ('s3', 'end of dynamics block (s3)'), ('s4', 'physics exports (s4)'),
                           ('pre', 'existing pre_condse dump')):
            print(f"--- {title}: worst over all steps")
            for f, s in agg[tag].items():
                print(f"  {f:8s} {fmt(s)}")
        nz = {f: s for f, s in agg['stage'].items() if s['nne'] != 0}
        print(f"--- stage comparisons: {len(agg['stage'])} (stage,field) series, {len(nz)} with any non-exact element")
        for f, s in sorted(nz.items()):
            print(f"  {f:28s} {fmt(s)}")
        print(f"wall per step: mean {np.mean(wall):.2f}s min {np.min(wall):.2f}s (includes dump I/O of the hooks)")


if __name__ == "__main__":
    main(sys.argv[1:])
