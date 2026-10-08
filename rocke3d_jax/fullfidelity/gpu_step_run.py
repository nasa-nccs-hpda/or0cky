"""D198: GPU-capable runner of the ASSEMBLED coupled step (jax_coupled.Coupled) in LIBM mode with REPLAYED radiation.

    python -u gpu_step_run.py [DATE] OUTDIR [--steps N] [--ref REF.npz | REF_{k}.npz] [--save all|first-last|last|none] [--steady-repeat R] [--timed]
                              [--cpu-flags auto|on|off] [--io-audit]

Defaults: DATE nov26, N = 1 (up to 6 with the coupled6 data subset), --save first-last, --steady-repeat 1 when N == 1 (step 0 is run once more from the same
start state for a steady-state time), --cpu-flags auto.

What it is (labels printed at the start, written into every result file):
  * LIBM MODE: every exp / pow is the XLA backend's (CPU: glibc/XLA CPU, GPU: CUDA device math); NO Intel libimf, NO libimf host callback.  This file never imports
    d187_imf_p1.install: jax_atm_phase1.Phase1 is built directly with its libm context (ctx imf=False), libimf_fused / libimf_ops are set to 'libm'.
  * RADIATION REPLAYED from the recorded record (not computed, no Fortran server, no callback).
  * Ent exports, land forcing, tile radiation columns, the CONDSE entry set and the SURFACE templates are RECORDED inputs, uploaded every step (as D191).
  * NOT BITWISE with the CPU / Fortran result on a GPU: device exp / pow differ from the CPU libm in the last bit for 6-12 % of values (gpu/DISCOVER_RUN.md, probe).
    Judge a GPU result at rounding level (categories B/C) for one step and statistically beyond; this runner never claims category A for a GPU result.
  * Host callbacks that remain (and are kept): MSTCNV QUS subsidence (NumPy), the two CONDSE pole columns (NumPy), the OADVT2 pre-pass (NumPy).  They need >= 2 CPU cores.
  * The CPU-only XLA flags of clouds_jax_env (--xla_cpu_max_isa=AVX, --xla_disable_hlo_passes=algsimp,reshape-mover) are applied ONLY when the backend is the CPU
    (--cpu-flags auto); on a GPU they are not set (jax is imported before clouds_jax_env, whose flags are then a no-op).  GPU_KEEP_ALGSIMP / CLOUDS_JAX_KEEP_ALGSIMP are
    irrelevant on the GPU (no algsimp flag is set).
Needs: numpy, jax (+jaxlib CUDA for the GPU), netCDF4, scipy, mpmath (as the model code).  Does NOT need pytest, hydra / omegaconf, the NumPy reference, the Intel runtime.
SOCRATES / RADIA are never ported or modified.  Nothing existing is edited: the libm wiring is done here by rebinding two names of d187_imf_p1 and one of libimf_fused
BEFORE jax_coupled is imported (jax_coupled.Coupled otherwise forces libimf)."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('date_or_out', nargs='*', help='[DATE] OUTDIR  (DATE default nov26)')
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--ref', default=None, help='reference end-state npz (keys as written by this runner / d191_run); a {k} in the name is replaced by the step number')
    ap.add_argument('--save', default='first-last', choices=['all', 'first-last', 'last', 'none'])
    ap.add_argument('--steady-repeat', type=int, default=None, help='extra repeats of step 0 for a steady time (default 1 when --steps 1, else 0)')
    ap.add_argument('--timed', action='store_true', help='device block after every stage (per-stage times; slower, only for profiling)')
    ap.add_argument('--cpu-flags', default='auto', choices=['auto', 'on', 'off'])
    ap.add_argument('--compare', nargs=2, metavar=('CAND.npz', 'REF.npz'), help='stand-alone: compare two saved end-state npz files (categories A/B/C/D, max relative differences) and exit; no model, no jax')
    ap.add_argument('--io-audit', action='store_true', help='list the data files opened (Python level) and write io_audit.json')
    return ap.parse_args()


import numpy as np  # noqa: E402
import jax_harness as H  # noqa: E402  (numpy-only at import)


def compare_to_ref(cand, ref):
    """Harness categories (jax_harness.field_category; A bitwise, B rel <= 1e-12 of the field scale, C <= 1e-6, D worse) per group and in total, with the largest
    relative differences (rel = max|diff| / max|ref|; rel_pointwise = max |diff| / |ref| over elements with ref != 0).  No tolerance is applied here."""
    groups = {}
    common = sorted(set(cand) & set(ref))
    for k in common:
        a, b = cand[k], ref[k]
        if a.dtype.kind not in 'fiub' or b.dtype.kind not in 'fiub':
            continue
        r = H.field_category(a, b)
        if a.shape == b.shape and a.size:
            af, bf = np.asarray(a, float), np.asarray(b, float)
            m = (bf != 0) & np.isfinite(af) & np.isfinite(bf)
            pw = float(np.max(np.abs(af - bf)[m] / np.abs(bf[m]))) if m.any() else 0.0
        else:
            pw = float('inf')
        g = k.split('/')[0]
        groups.setdefault(g, {})[k] = dict(cat=r['cat'], rel=float(r['rel']), max_abs=float(r['max_abs']), rel_pointwise=pw, n_diff=r.get('n_diff'), n=r.get('n'),
                                           n_cols_over=r.get('n_cols_over'))
    out = dict(n_common=sum(len(v) for v in groups.values()), only_in_candidate=sorted(set(cand) - set(ref))[:20], only_in_reference=sorted(set(ref) - set(cand))[:20],
               n_only_in_candidate=len(set(cand) - set(ref)), n_only_in_reference=len(set(ref) - set(cand)), groups={})
    tot = {c: 0 for c in 'ABCD'}
    allrows = {}
    for g, rows in groups.items():
        cats = {c: sum(1 for v in rows.values() if v['cat'] == c) for c in 'ABCD'}
        for c in 'ABCD':
            tot[c] += cats[c]
        allrows.update(rows)
        worst = sorted(((v['rel'], k) for k, v in rows.items() if v['cat'] != 'A'), reverse=True)[:8]
        out['groups'][g] = dict(n_fields=len(rows), categories=cats, max_rel=max([v['rel'] for v in rows.values()] or [0.0]),
                                max_rel_pointwise=max([v['rel_pointwise'] for v in rows.values()] or [0.0]), worst_fields=[(k, r) for r, k in worst])
    out['categories'] = tot
    out['max_rel'] = max([v['rel'] for v in allrows.values()] or [0.0])
    out['max_rel_by_category'] = {c: max([v['rel'] for v in allrows.values() if v['cat'] == c] or [0.0]) for c in 'ABCD'}
    out['worst_fields'] = [(k, r, allrows[k]['cat']) for r, k in sorted(((v['rel'], k) for k, v in allrows.items() if v['cat'] != 'A'), reverse=True)[:25]]
    out['fields'] = allrows
    out['statement'] = ('categories are those of ACCEPTANCE section 3 (fixed, not tunable). This is a LIBM-mode result compared with a reference; category A is not '
                        'expected unless the reference is the same backend, XLA version and mode.')
    return out


A_ = _args()
if A_.compare:
    cand_p, ref_p = A_.compare
    res = compare_to_ref(H.load_npz(cand_p), H.load_npz(ref_p))
    print(f'candidate {cand_p}\nreference {ref_p}\ncategories {res["categories"]}  max_rel {res["max_rel"]:.3e}  by category {res["max_rel_by_category"]}  '
          f'(common {res["n_common"]}, only candidate {res["n_only_in_candidate"]}, only reference {res["n_only_in_reference"]})')
    for g, v in res['groups'].items():
        print(f'  {g:9s} n={v["n_fields"]:3d} {v["categories"]} max_rel {v["max_rel"]:.2e}  max_rel_pointwise {v["max_rel_pointwise"]:.2e}')
    print('  worst fields (not A):', [(k, f'{r:.2e}', c) for k, r, c in res['worst_fields'][:15]])
    print('  categories (ACCEPTANCE section 3): A bitwise, B rel<=1e-12 of the field scale, C <=1e-6, D worse; rel = max|diff|/max|ref|')
    if A_.date_or_out:
        os.makedirs(A_.date_or_out[-1], exist_ok=True)
        json.dump({k: v for k, v in res.items() if k != 'fields'}, open(os.path.join(A_.date_or_out[-1], 'compare.json'), 'w'), indent=1, default=str)
    sys.exit(0)
if not A_.date_or_out:
    sys.exit('usage: gpu_step_run.py [DATE] OUTDIR [options]   or   gpu_step_run.py --compare CAND.npz REF.npz [OUTDIR]')
if len(A_.date_or_out) == 1:
    DATE, OUTDIR = 'nov26', A_.date_or_out[0]
else:
    DATE, OUTDIR = A_.date_or_out[0], A_.date_or_out[1]
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------------- backend decision BEFORE any model import
# clouds_jax_env* apply their XLA flags only `if 'jax' not in sys.modules`.  Importing jax first therefore turns them into no-ops (jax does not read XLA_FLAGS before
# the backend is created).  The CPU flags are wanted only on the CPU backend: decided from the environment, then verified after the backend is up.
_plat = os.environ.get('JAX_PLATFORMS', '').lower()
_gpu_visible = (os.path.exists('/dev/nvidiactl') or bool(os.environ.get('NVIDIA_VISIBLE_DEVICES')) or os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1')) and 'cpu' not in _plat.split(',')[:1]
_want_flags = {'on': True, 'off': False, 'auto': not _gpu_visible}[A_.cpu_flags]
if _want_flags:
    import clouds_jax_env  # noqa: F401  (flags BEFORE jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update('jax_enable_x64', True)
BACKEND = jax.default_backend()
DEVICES = [str(d) for d in jax.devices()]
XLA_FLAGS = os.environ.get('XLA_FLAGS', '')
if BACKEND == 'cpu' and _gpu_visible and A_.cpu_flags == 'auto':
    print('NOTE: a GPU device node is visible but JAX found only the CPU backend (no CUDA jaxlib?); the CPU XLA flags were NOT applied (decided before the backend was known). '
          'Use --cpu-flags on for a CPU-bitwise-style run.', flush=True)
if BACKEND != 'cpu' and ('xla_cpu' in XLA_FLAGS):
    print('WARNING: CPU XLA flags are set in XLA_FLAGS on a non-CPU backend:', XLA_FLAGS, flush=True)
assert not os.environ.get('JAX_COMPILATION_CACHE_DIR') and not os.environ.get('CLOUDS_JAX_CACHE'), 'no JAX compile cache for validated runs (D175)'
assert len(os.sched_getaffinity(0)) >= 2, 'host callbacks inside jit need >= 2 CPU cores'

# ---------------------------------------------------------------------------------------------------- libm wiring (nothing existing is edited)
import intel_libm_ff  # noqa: E402
import d187_imf_p1 as W  # noqa: E402  (imports jax_atm_phase1 first: execution counters)
import jax_atm_phase1 as P1  # noqa: E402
import libimf_fused as FX  # noqa: E402
import libimf_ops as LI  # noqa: E402

LI.set_mode('libm')
FX.set_mode('libm')
_set_mode_orig = FX.set_mode
W.install = lambda: {'libm_mode': 'no libimf wiring: jax_atm_phase1 libm context, libimf_ops / libimf_fused in libm mode (gpu_step_run.py)'}
W.make_phase1 = lambda date, **kw: P1.Phase1(date, **kw)          # Phase1 builds ctx imf=False itself (D186 libm); no KitImf
FX.set_mode = lambda m: _set_mode_orig('libm')                    # Coupled.__init__ asks for 'libimf': refused here, libm is kept

import jax_coupled as C  # noqa: E402
import jax_harness as H  # noqa: E402
import jax_p1_count as CNT  # noqa: E402
import atm_step as A  # noqa: E402
import clouds_mstcnv_dev as md  # noqa: E402
import jax_ocean as JO  # noqa: E402

assert LI.mode() == 'libm' and FX.mode() == 'libm'

LABELS = [
    'LIBM MODE: exp/pow from the XLA backend (' + BACKEND + '); no Intel libimf, no libimf host callback',
    'RADIATION REPLAYED from the real record (not computed; no Fortran callback in this result)',
    'RECORDED inputs: Ent exports, land forcing, tile radiation columns, CONDSE entry set, SURFACE templates (uploaded every step)',
    ('GPU result: NOT BITWISE with the CPU / Fortran (device exp/pow differ by 1 ulp in 6-12 % of values); judge at rounding level and statistically'
     if BACKEND != 'cpu' else 'CPU result in libm mode: not bitwise with the libimf (Fortran-faithful) result by construction; compare at rounding level'),
    'HYBRID step, not end-to-end JAX: host callbacks for QUS, the CONDSE pole columns and the OADVT2 pre-pass; host template build',
]


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True, timeout=10).stdout.strip() or 'unknown'
    except Exception:  # noqa: BLE001
        return 'unknown'


def header():
    import importlib.metadata as md_
    ver = {}
    for p in ('jax', 'jaxlib', 'numpy', 'netCDF4', 'scipy', 'mpmath'):
        try:
            ver[p] = md_.version(p)
        except Exception:  # noqa: BLE001
            ver[p] = 'NOT INSTALLED'
    import jaxlib  # versions of the modules actually imported (importlib.metadata can report a stale dist-info)
    ver.update(jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__)
    return dict(task='D198 assembled coupled step, libm mode, replayed radiation', date=DATE, steps=A_.steps, backend=BACKEND, devices=DEVICES, x64=bool(jax.config.jax_enable_x64),
                xla_flags=XLA_FLAGS, cpu_flags_applied=bool(_want_flags), python=platform.python_version(), versions=ver, host=platform.node(),
                cores=sorted(os.sched_getaffinity(0)), omp_num_threads=os.environ.get('OMP_NUM_THREADS'), git_head=git_head(),
                libimf_available_on_host=bool(intel_libm_ff.available()), libimf_used=False, libimf_ops=LI.status(), libimf_fused=FX.status(),
                ff_data=os.environ.get('FF_DATA', 'unset (module default: ' + str(H.FF_DEFAULT) + ')'), modelE_prod_input=os.environ.get('MODELE_PROD_INPUT', 'unset (module default)'), labels=LABELS,
                time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


# ---------------------------------------------------------------------------------------------------- helpers
def build_registry():
    sr = H.StageRegistry()
    for n, (k, d) in P1.STAGE_TABLE.items():
        if n == 'melt_si':
            continue
        if n == 'radia':
            k = 'JJ/REC'
        if n == 'condse_mstcnv':
            d = 'MSTCNV (D189): cloud-base loop and event block as ONE device program, libm exp/pow; QUS ADV1D subsidence = NumPy host callback'
        sr.register(n, k, d)
    for n, k, d in C.STAGES2:
        sr.register(n, k, d)
    return sr


def flatten(tree, prefix='', out=None):
    out = {} if out is None else out
    if isinstance(tree, dict):
        for k in tree:
            flatten(tree[k], f'{prefix}{k}/', out)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            flatten(v, f'{prefix}{i}/', out)
    elif hasattr(tree, 'dtype') or isinstance(tree, (int, float, bool, np.generic)):
        a = np.asarray(tree)
        if a.dtype.kind in 'fiub':
            out[prefix.rstrip('/')] = a
    return out


def snap_np(d):
    return {k: np.asarray(v) for k, v in d.items() if hasattr(v, 'shape') and np.asarray(v).dtype.kind in 'fiub'}


def end_arrays(arrays):
    """The same arrays and key names as d191_run.validate (group/field), pulled to the host (the only device_get of a saved step)."""
    allarr = {}
    for stg in ('dyn', 'condse', 'radia', 'surface', 'dissip', 'filter', 'X'):
        for k, v in snap_np(arrays[stg]).items():
            allarr[f'{stg}/{k}'] = v
    SS2 = C.tree_np(arrays['SS2'])
    V2n = C.tree_np(arrays['V2'])
    for grp in ('ocean', 'ice', 'lake', 'li', 'atm'):
        allarr.update(flatten(SS2[grp], f'surf/{grp}/'))
    allarr.update(flatten(dict(rsix=V2n['rsix'], rsiy=V2n['rsiy'], usi=V2n['usi'], vsi=V2n['vsi']), 'surf/v2/'))
    allarr.update(flatten(C.tree_np(arrays['aux_land']), 'land/'))
    return allarr


def run_step(cp, k, state, rec, dev, sr, timed):
    ct = H.Counters()
    ct.listen_compiles()
    LI.reset_counters()
    FX.reset_counters()
    md.reset_qus()
    for kk in JO.XPRE_STATS:
        JO.XPRE_STATS[kk] = 0 if kk != 'seconds' else 0.0
    CNT.reset()
    hp0 = cp.ph.cs.host
    p0 = (hp0.calls, hp0.seconds, hp0.bytes) if hp0 is not None else (0, 0.0, 0)
    try:
        c0 = CNT.snapshot()
        t0 = time.perf_counter()
        with ct.stage('step'):
            new, info, arrays = cp.step(k, state, rec, dev, sr, timed=timed, keep=True)
            jax.block_until_ready(new)
        wall = time.perf_counter() - t0
        c1 = CNT.snapshot()
        tot = ct.totals()
        hp = cp.ph.cs.host
        out = dict(k=k, itime=info['itime'], step_seconds=wall, jit_executions=c1['jit_calls'] - c0['jit_calls'],
                   eager_primitive_dispatches=(c1['eager_primitive_calls'] or 0) - (c0['eager_primitive_calls'] or 0),
                   xla_compilations=tot['compiles'], xla_compile_seconds=tot['compile_seconds'], jit_by_stage=info['jit_by_stage'], flags=info['flags'],
                   stage_rebuilt=bool(info.get('stage_rebuilt', False)), host_template=info['host'], nit_rebuild=info.get('nit_rebuild'),
                   libimf_callbacks=dict(libimf_ops=LI.counters()['total'], libimf_fused=FX.counters()['total']),
                   qus_callbacks=dict(md.QUS), oadvt2_prepass_callbacks=dict(JO.XPRE_STATS),
                   pole_callbacks=dict(calls=hp.calls - p0[0], seconds=hp.seconds - p0[1], bytes=hp.bytes - p0[2]),
                   phase1_seconds=info.get('phase1_seconds'))
    finally:
        ct.stop_listening()
    return out, new, arrays, sr


def main():
    hdr = header()
    print('=' * 100, flush=True)
    for L in LABELS:
        print('LABEL:', L, flush=True)
    print(json.dumps({k: v for k, v in hdr.items() if k != 'labels'}, indent=1, default=str), flush=True)
    print('=' * 100, flush=True)
    json.dump(hdr, open(os.path.join(OUTDIR, 'header.json'), 'w'), indent=1, default=str)
    io = None
    if A_.io_audit:
        import d187_common as K
        io = K.IOAudit(roots=(os.environ.get('FF_DATA') or str(H.FF_DEFAULT), os.path.dirname(os.environ.get('MODELE_PROD_INPUT') or '') or '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0'))

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    on = (lambda tag: io.on(tag)) if io is not None else (lambda tag: _Null())
    t_all = time.perf_counter()
    with on('setup'):
        cp = C.Coupled(DATE)
    print(f'setup {cp.setup_seconds:.1f} s; libimf_ops {LI.mode()} fused {FX.mode()} ctx.imf={cp.ctx.imf}', flush=True)
    assert not cp.ctx.imf, 'libm runner: the context must not be an imf context'
    sr = build_registry()
    res = dict(header=hdr, labels=LABELS, date=DATE, steps=[], setup_seconds=cp.setup_seconds)
    state0 = None
    with on('step0-load'):
        state, rec, dev = cp.initial_state()
        jax.block_until_ready((state, dev))
    state0 = (state, rec, dev)
    first_end = None
    for k in range(A_.steps):
        if k > 0:
            with on(f'step{k}-load'):
                rec = cp.ph.load_records(cp.it0 + k, first=False)
                dev = cp.ph.to_device(rec, first=False)
        with on(f'step{k}'):
            out, state, arrays, sr = run_step(cp, k, state, rec, dev, sr, A_.timed)
        out['label'] = 'cold (includes compilation)' if k == 0 else 'steady'
        save = A_.save == 'all' or (A_.save == 'last' and k == A_.steps - 1) or (A_.save == 'first-last' and k in (0, A_.steps - 1))
        refp = None
        if A_.ref:
            refp = A_.ref.format(k=k) if '{k}' in A_.ref else (A_.ref if k == 0 else None)
        if save or refp:
            t0 = time.perf_counter()
            allarr = end_arrays(arrays)
            out['seconds_device_to_host_snapshot'] = time.perf_counter() - t0
            if save:
                fn = os.path.join(OUTDIR, f'{DATE}_libm_step{k}.npz')
                np.savez(fn, **allarr)
                out['npz'] = fn
                out['n_arrays'] = len(allarr)
            if refp and os.path.exists(refp):
                ref = H.load_npz(refp)
                cmpd = compare_to_ref(allarr, ref)
                out['reference'] = refp
                out['vs_reference'] = cmpd
                print(f'[step {k}] vs {os.path.basename(refp)}: categories {cmpd["categories"]} max_rel {cmpd["max_rel"]:.3e} by category {cmpd["max_rel_by_category"]} '
                      f'(common {cmpd["n_common"]}, only cand {cmpd["n_only_in_candidate"]}, only ref {cmpd["n_only_in_reference"]})', flush=True)
                for g, v in cmpd['groups'].items():
                    print(f'    {g:9s} n={v["n_fields"]:3d} {v["categories"]} max_rel {v["max_rel"]:.2e}', flush=True)
                print('    worst:', [(k_, f'{r:.2e}', c) for k_, r, c in cmpd['worst_fields'][:8]], flush=True)
            elif refp:
                print(f'[step {k}] reference file {refp} not found: comparison skipped', flush=True)
            if k == 0 and A_.steps == 1:
                first_end = {kk: allarr[kk] for kk in ('filter/T', 'filter/Q', 'filter/U', 'filter/V') if kk in allarr}
        res['steps'].append(out)
        print(f'[step {k}] it {out["itime"]} {out["label"]}: {out["step_seconds"]:.2f} s  jit executions {out["jit_executions"]}  eager {out["eager_primitive_dispatches"]}  '
              f'XLA compilations {out["xla_compilations"]} ({out["xla_compile_seconds"]:.1f} s)  flags {sum(out["flags"].values())}  stage_rebuilt {out["stage_rebuilt"]}', flush=True)
        json.dump(res, open(os.path.join(OUTDIR, 'result.json'), 'w'), indent=1, default=str)
    # ---- steady repeats of step 0 (from the same start state)
    rep = A_.steady_repeat if A_.steady_repeat is not None else (1 if A_.steps == 1 else 0)
    res['repeats'] = []
    for r_ in range(rep):
        state, rec, dev = state0
        # step 0 consumes/overwrites parts of `dev` (cse_in RSI): reload the records exactly as the first run did
        with on('repeat-load'):
            state, rec, dev = cp.initial_state()
            jax.block_until_ready((state, dev))
        with on('repeat'):
            out, new, arrays, sr = run_step(cp, 0, state, rec, dev, sr, A_.timed)
        out['label'] = f'steady repeat {r_ + 1} of step 0'
        if first_end is not None:
            cur = {kk: np.asarray(v) for kk, v in snap_np(arrays['filter']).items()}
            out['bitwise_equal_to_first_run'] = {kk: bool(np.array_equal(cur[kk.split('/')[1]], first_end[kk])) for kk in first_end}
        res['repeats'].append(out)
        print(f'[repeat {r_ + 1}] step 0 again: {out["step_seconds"]:.2f} s  jit {out["jit_executions"]}  XLA compilations {out["xla_compilations"]}  same as first run: {out.get("bitwise_equal_to_first_run")}', flush=True)
    st = [s['step_seconds'] for s in res['steps']]
    steady = [s['step_seconds'] for s in res['steps'][1:]] + [s['step_seconds'] for s in res['repeats']]
    res['summary'] = dict(backend=BACKEND, devices=DEVICES, steps=A_.steps, first_step_seconds_with_compile=st[0] if st else None,
                          steady_step_seconds=steady, steady_mean_seconds=float(np.mean(steady)) if steady else None,
                          total_process_seconds=time.perf_counter() - t_all, libimf_callbacks_total=sum(s['libimf_callbacks']['libimf_ops']['calls'] + s['libimf_callbacks']['libimf_fused']['calls'] for s in res['steps']),
                          labels=LABELS)
    if io is not None:
        res['io_audit'] = io.report()
        json.dump(res['io_audit'], open(os.path.join(OUTDIR, 'io_audit.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(OUTDIR, 'result.json'), 'w'), indent=1, default=str)
    print('=' * 100, flush=True)
    for L in LABELS:
        print('LABEL:', L)
    print('SUMMARY', json.dumps(res['summary'], indent=1, default=str), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
