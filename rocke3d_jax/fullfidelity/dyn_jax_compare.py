"""D140/D141: validate and time the JAX dynamics stages (dyn_jax_*.py, dyn_step_jax.py) against the numpy chain
(dyn_step.py, numpy-pow mode) and against the real-Fortran dumps.

per-stage   `StageCheck` hook: the numpy plan runs in boundary mode (dyn_step_compare.Recorder: before every dumped stage
            all inputs not produced by earlier port stages are replaced by the real recorded inputs); before every JAX-converted stage the
            workspace is copied, the JAX stage is run on the copy and its written variables are compared with the numpy
            stage's output (JAX-vs-numpy), and for advecv/pgf also with the real per-call dump (JAX-vs-real, numpy-vs-real).
end-state   whole chained step, numpy (numpy-pow) vs JAX vs the real states s2/s3 (and s4 exports).
timing      warm per-stage / whole-step CPU times and compile (cold-call) times.
Usage (from fullfidelity/): python dyn_jax_compare.py [--steps N] [--dates nov26,dec01,jan01] [--timing]
"""
import sys
import time

import numpy as np

import dyn_step as ds
import dyn_step_compare as dc
import dyn_step_jax as dj

IM, JM, LM = ds.IM, ds.JM, ds.LM
EXTRA = {'filter_chain': ('U', 'V'), 'kea': ('KEA',), 'wsave': ('WSAVE',)}


def stage_vars(st):
    return tuple(v for v in (EXTRA.get(st.kind) or tuple(dc.written(st))) if v)


class StageCheck:
    def __init__(self, date, itime, kit, kinds=dj.JAX_KINDS, ff=ds.FF_DEFAULT, boundary=True):
        self.date, self.itime, self.kit, self.kinds, self.ff = date, itime, kit, kinds, ff
        self.rec = dc.Recorder(date, itime, ff, boundary=boundary)
        self.jn, self.jr, self.nr = {}, {}, {}

    def __call__(self, when, st, w, ctx):
        self.rec(when, st, w, ctx)
        if st.kind not in self.kinds:
            return
        if when == 'pre':
            self.w2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            dj.exec_stage_jax(st, self.w2, ctx, self.kit, self.kinds)
        else:
            for v in stage_vars(st):
                dc.merge(self.jn, f"{st.kind}.{v}", dc.stats(self.w2[v], w[v]))
            if st.kind in ('advecv', 'pgf'):
                for (lab, pa, real), (_, ja, _) in zip(dc.stage_outputs(st, w, self.date, self.itime, self.ff),
                                                      dc.stage_outputs(st, self.w2, self.date, self.itime, self.ff)):
                    dc.merge(self.nr, f"{st.kind}.{lab}", dc.stats(pa, real))
                    dc.merge(self.jr, f"{st.kind}.{lab}", dc.stats(ja, real))


def per_stage(dates, nsteps, ff=ds.FF_DEFAULT, kinds=dj.JAX_KINDS):
    agg = dict(jn={}, jr={}, nr={})
    for date, it0 in dates:
        ctx = ds.load_ctx(date, imf_pow=False)
        kit = dj.Kit(ctx)
        for k in range(nsteps):
            it = it0 + k
            s1 = ds.load_state(ds.state_path(date, it, 1, ff))
            ck = StageCheck(date, it, kit, kinds, ff)
            ds.dyn_step(s1, ctx, itime=it, hook=ck)
            for n in agg:
                for key, s in getattr(ck, n).items():
                    dc.merge(agg[n], key, s)
    return agg


def end_state(date, itime, ctx, kit, ff=ds.FF_DEFAULT, kinds=dj.JAX_KINDS):
    s1 = ds.load_state(ds.state_path(date, itime, 1, ff))
    s3 = ds.load_state(ds.state_path(date, itime, 3, ff))
    s4 = ds.load_exports(ds.state_path(date, itime, 4, ff))
    wn = ds.dyn_step(s1, ctx, itime=itime)
    wj = dj.dyn_step_jax(s1, ctx, kit, itime=itime, kinds=kinds)
    out = dict(jn={}, jr={}, nr={})
    for k in dc.FIELDS_S2:
        out['jn'][k] = dc.stats(wj[k.upper()], wn[k.upper()])
        out['jr'][k] = dc.stats(wj[k.upper()], s3[k])
        out['nr'][k] = dc.stats(wn[k.upper()], s3[k])
    for k, wk in (('wsave', 'WSAVE'), ('kea', 'KEA'), ('ptropo', 'PTROPO'), ('phi', 'PHI')):
        out['jn'][k] = dc.stats(wj[wk], wn[wk])
        out['jr'][k] = dc.stats(wj[wk], s4[k])
        out['nr'][k] = dc.stats(wn[wk], s4[k])
    return out


def timing(date='nov26', itime=33312, n=3, kinds=dj.JAX_KINDS, ff=ds.FF_DEFAULT):
    ctx = ds.load_ctx(date, imf_pow=False)
    kit = dj.Kit(ctx)
    s1 = ds.load_state(ds.state_path(date, itime, 1, ff))
    tn_cold, tj_cold = {}, {}
    t0 = time.perf_counter(); ds.dyn_step(s1, ctx, itime=itime, timing=tn_cold); n_cold = time.perf_counter() - t0
    t0 = time.perf_counter(); dj.dyn_step_jax(s1, ctx, kit, itime=itime, kinds=kinds, timing=tj_cold)
    j_cold = time.perf_counter() - t0
    wn, wj, tns, tjs = [], [], [], []
    for _ in range(n):
        tm = {}; t0 = time.perf_counter(); ds.dyn_step(s1, ctx, itime=itime, timing=tm)
        wn.append(time.perf_counter() - t0); tns.append(tm)
        tm = {}; t0 = time.perf_counter(); dj.dyn_step_jax(s1, ctx, kit, itime=itime, kinds=kinds, timing=tm)
        wj.append(time.perf_counter() - t0); tjs.append(tm)
    best = lambda L: {k: min(t[k] for t in L) for k in L[0]}
    return dict(numpy_cold=n_cold, jax_cold=j_cold, numpy_warm=wn, jax_warm=wj, np_stage=best(tns), jax_stage=best(tjs),
                jax_cold_stage=tj_cold)


def main(argv):
    dates = [d for d in ds.DATES if '--dates' not in argv or d[0] in argv[argv.index('--dates') + 1].split(',')]
    nsteps = int(argv[argv.index('--steps') + 1]) if '--steps' in argv else 1
    if '--timing' in argv:
        r = timing()
        print("numpy whole step: cold %.2fs warm %s" % (r['numpy_cold'], ', '.join('%.3f' % x for x in r['numpy_warm'])))
        print("jax   whole step: cold %.2fs warm %s" % (r['jax_cold'], ', '.join('%.3f' % x for x in r['jax_warm'])))
        print("stage        numpy_warm  jax_warm  jax_cold(compile incl.)")
        for k in dj.JAX_KINDS:
            print(f"{k:13s}{r['np_stage'][k]:9.3f}  {r['jax_stage'][k]:9.3f}  {r['jax_cold_stage'][k]:9.2f}")
        rest = [k for k in r['np_stage'] if k not in dj.JAX_KINDS]
        print("numpy-only stages total: numpy %.3f jax-path %.3f" % (sum(r['np_stage'][k] for k in rest),
                                                                   sum(r['jax_stage'][k] for k in rest)))
        return
    print(f"=== per-stage (boundary mode, real inputs for dumped stages), {nsteps} step(s)/date, dates {[d[0] for d in dates]}")
    agg = per_stage(dates, nsteps)
    for tag, title in (('jn', 'JAX vs numpy'), ('jr', 'JAX vs real dump'), ('nr', 'numpy vs real dump')):
        print(f"--- {title}")
        for k, s in agg[tag].items():
            print(f"  {k:18s} {dc.fmt(s)}")
    nend = int(argv[argv.index('--endsteps') + 1]) if '--endsteps' in argv else 1
    print(f"=== end state of the chained step, steps 0..{nend - 1} of each date: worst over steps, per date")
    for date, it0 in dates:
        ctx = ds.load_ctx(date, imf_pow=False)
        kit = dj.Kit(ctx)
        agg = dict(jn={}, jr={}, nr={})
        for k in range(nend):
            r = end_state(date, it0 + k, ctx, kit)
            for n in agg:
                for key, s in r[n].items():
                    dc.merge(agg[n], key, s)
        for tag, title in (('jn', 'JAX vs numpy'), ('jr', 'JAX vs real'), ('nr', 'numpy vs real')):
            print(f"--- {date} {title}")
            for k, s in agg[tag].items():
                print(f"  {k:8s} {dc.fmt(s)}")


if __name__ == "__main__":
    main(sys.argv[1:])
