"""Speed profile wrapper (speed/step-time, 2026-10-12): gpu_day_run with jax.profiler traces around one held step and one radiation step.
    ROCKE_PROF_DIR=DIR GPU_DAY_DRIVER=d212_day python -u gpu_profile_run.py OUTDIR <d212_day flags>
Patches nothing in the model: wraps Coupled.step (trace start/stop, block_until_ready of the returned state, per-step wall).  Writes DIR/trace_<held|rad>_k<k>/ and DIR/step_walls.json.
Steps traced: the first held step with k >= PROF_K0 whose predecessor was held, and the first radiation step with k >= PROF_K0 (PROF_K0 default 6)."""
import json
import os
import time

import gpu_day_run as G   # noqa: F401  (GPU preamble first)
import jax
import atm_step as A
import jax_coupled as JC

D = os.environ['ROCKE_PROF_DIR']
os.makedirs(D, exist_ok=True)
K0 = int(os.environ.get('PROF_K0', '6'))
WALLS, DONE = [], set()
_orig = JC.Coupled.step


def step(self, k, state, rec, dev, sr, *a, **kw):
    it = self.it0 + k
    rad = bool(A.is_radiation_step(it))
    kind = 'rad' if rad else 'held'
    prev_held = (k > 0) and not A.is_radiation_step(it - 1)
    tr = os.environ.get('PROF_TRACE', '1') == '1' and k >= K0 and kind not in DONE and (rad or prev_held)
    srt0 = {n: v['seconds'] for n, v in sr._st.items()}
    if tr:
        jax.block_until_ready(state)
        jax.profiler.start_trace(os.path.join(D, f'trace_{kind}_k{k}'))
    t = time.perf_counter()
    out = _orig(self, k, state, rec, dev, sr, *a, **kw)
    jax.block_until_ready(out[0])
    w = time.perf_counter() - t
    if tr:
        jax.profiler.stop_trace()
        DONE.add(kind)
    stg = {n: round(v['seconds'] - srt0[n], 4) for n, v in sr._st.items() if v['seconds'] - srt0[n] > 0}
    WALLS.append(dict(k=k, it=it, rad=rad, wall=w, traced=tr, stages=stg, p1_sub=out[1].get('phase1_stage_seconds'), p1_s=out[1].get('phase1_seconds')))
    json.dump(WALLS, open(os.path.join(D, 'step_walls.json'), 'w'), indent=1)
    return out


JC.Coupled.step = step
if __name__ == '__main__':
    import runpy
    runpy.run_module(os.environ.get('GPU_DAY_DRIVER', 'd212_day'), run_name='__main__')
