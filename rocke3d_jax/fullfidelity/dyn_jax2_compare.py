"""D145: validate / time dyn_step_jax2 (AFLUX, ADVECM, AADVT, QDYNAM added to the JAX stages) against the numpy chain
(dyn_step.py, numpy-pow) and the real dumps.  Usage (from fullfidelity/): python dyn_jax2_compare.py [--stages] [--end N] [--timing]
--stages: per-stage JAX-vs-numpy in boundary mode (real inputs for dumped stages), 1 step per date
--end N : end state of the chained step, steps 0..N-1 of each date: JAX vs numpy, JAX vs real, numpy vs real (worst per field)
--timing: warm/cold whole-step and per-stage times."""
import sys
import time

import numpy as np

import dyn_step as ds
import dyn_step_compare as dc
import dyn_step_jax2 as dj2
import dyn_jax_compare as jc

NEWV = {'qdynam': ('Q', 'QMOM', 'MUS', 'MVS', 'MWS')}


class StageCheck2(jc.StageCheck):
    def __init__(self, date, itime, kit, kinds=dj2.JAX_KINDS):
        super().__init__(date, itime, kit, kinds)

    def __call__(self, when, st, w, ctx):
        self.rec(when, st, w, ctx)
        if st.kind not in self.kinds:
            return
        if when == 'pre':
            self.w2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            dj2.exec_stage_jax(st, self.w2, ctx, self.kit, self.kinds)
        else:
            for v in (NEWV.get(st.kind) or jc.stage_vars(st)):
                dc.merge(self.jn, f"{st.kind}.{v}", dc.stats(self.w2[v], w[v]))


def stages(dates, nsteps=1):
    agg = {}
    for date, it0 in dates:
        ctx = ds.load_ctx(date, imf_pow=False)
        kit = dj2.Kit(ctx)
        for k in range(nsteps):
            s1 = ds.load_state(ds.state_path(date, it0 + k, 1))
            ck = StageCheck2(date, it0 + k, kit)
            ds.dyn_step(s1, ctx, itime=it0 + k, hook=ck)
            for key, s in ck.jn.items():
                dc.merge(agg, key, s)
    return agg


def end_state(date, itime, ctx, kit):
    s1 = ds.load_state(ds.state_path(date, itime, 1))
    s3 = ds.load_state(ds.state_path(date, itime, 3))
    s4 = ds.load_exports(ds.state_path(date, itime, 4))
    wn = ds.dyn_step(s1, ctx, itime=itime)
    wj = dj2.dyn_step_jax(s1, ctx, kit, itime=itime)
    out = dict(jn={}, jr={}, nr={})
    for k in dc.FIELDS_S2:
        out['jn'][k] = dc.stats(wj[k.upper()], wn[k.upper()]); out['jr'][k] = dc.stats(wj[k.upper()], s3[k])
        out['nr'][k] = dc.stats(wn[k.upper()], s3[k])
    for k, wk in (('wsave', 'WSAVE'), ('kea', 'KEA'), ('ptropo', 'PTROPO'), ('phi', 'PHI')):
        out['jn'][k] = dc.stats(wj[wk], wn[wk]); out['jr'][k] = dc.stats(wj[wk], s4[k]); out['nr'][k] = dc.stats(wn[wk], s4[k])
    return out


def timing(date='nov26', itime=33312, n=3):
    ctx = ds.load_ctx(date, imf_pow=False)
    kit = dj2.Kit(ctx)
    s1 = ds.load_state(ds.state_path(date, itime, 1))
    tjc = {}
    t0 = time.perf_counter(); dj2.dyn_step_jax(s1, ctx, kit, itime=itime, timing=tjc); cold = time.perf_counter() - t0
    wn, wj, tn, tj = [], [], [], []
    for _ in range(n):
        tm = {}; t0 = time.perf_counter(); ds.dyn_step(s1, ctx, itime=itime, timing=tm); wn.append(time.perf_counter() - t0); tn.append(tm)
        tm = {}; t0 = time.perf_counter(); dj2.dyn_step_jax(s1, ctx, kit, itime=itime, timing=tm); wj.append(time.perf_counter() - t0); tj.append(tm)
    best = lambda L: {k: min(t[k] for t in L) for k in L[0]}
    return dict(jax_cold=cold, numpy_warm=wn, jax_warm=wj, np_stage=best(tn), jax_stage=best(tj), cold_stage=tjc)


def main(argv):
    if '--timing' in argv:
        r = timing()
        print("jax cold %.1fs; numpy warm %s; jax warm %s" % (r['jax_cold'], ['%.3f' % x for x in r['numpy_warm']], ['%.3f' % x for x in r['jax_warm']]))
        for k in r['np_stage']:
            print(f"{k:13s} numpy {r['np_stage'][k]:7.3f}  jax {r['jax_stage'][k]:7.3f}  cold {r['cold_stage'][k]:7.2f}")
    if '--stages' in argv:
        for k, s in stages(ds.DATES).items():
            print(f"  {k:18s} {dc.fmt(s)}")
    if '--end' in argv:
        nend = int(argv[argv.index('--end') + 1])
        for date, it0 in ds.DATES:
            ctx = ds.load_ctx(date, imf_pow=False); kit = dj2.Kit(ctx)
            agg = dict(jn={}, jr={}, nr={})
            for k in range(nend):
                r = end_state(date, it0 + k, ctx, kit)
                for n in agg:
                    for key, s in r[n].items():
                        dc.merge(agg[n], key, s)
            for tag in ('jn', 'jr', 'nr'):
                print(f"--- {date} {tag}: worst abs / scale-rel / #unequal of N per field")
                print("  " + "; ".join(f"{k} {s['abs']:.1e}/{s['rel']:.1e}/{s['nne']}" for k, s in agg[tag].items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
