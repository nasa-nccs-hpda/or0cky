"""Where inside OADVT2 (ocean tracer advection, stage 'dynamics') does the GPU first produce non-finite values? (D214 follow-up to gpu_ocean_stage_probe)

usage: python -u gpu_oadvt_probe.py [DATE]      (one step; ~12 min cold on the A100; run on the CPU as a control: all counts must be zero)

Wraps jax_ocean.oadvt2_dev / oadvtx2_dev so each call also reports, for the heat (G0M, qlimit False) and salt (S0M, qlimit True) advection:
  in      non-finite values of the incoming rm, rx, ry, rz and of the mass mmi
  xpre    the host-callback outputs of both X pre-passes: non-finite count of snap, min/max of ncour, count of lane
  x1 y z x2   non-finite counts of (rm, rx, ry, rz, mass) after each of the four sweeps
The first non-zero entry in call order is the culprit. Diagnostic only (libm mode, radiation replayed, hybrid).
"""
import sys
import gpu_step_run as G
import jax
import jax.numpy as jnp
import jax_ocean as JO
import oadvt_jax as OJ

DATE = sys.argv[1] if len(sys.argv) > 1 else 'nov26'
C = G.C
cp = C.Coupled(DATE)
PROBE = []                      # tracers of the CURRENT trace only; read back inside the same trace by the stage wrapper


def nf(x):
    return jnp.sum(~jnp.isfinite(x)).astype(jnp.int64)


def oadvtx2_probe(Ko, rm, rx, ry, rz, mm, mu, dt, qlimit):
    ls, js = Ko['x_ls'], Ko['x_js']
    shp = (jax.ShapeDtypeStruct((JO.LMO + 1, JO.JM + 1, JO.IM + 1), jnp.float64), jax.ShapeDtypeStruct((JO.LMO + 1, JO.JM + 1), jnp.int64),
           jax.ShapeDtypeStruct((JO.LMO + 1, JO.JM + 1), jnp.bool_))
    lmu1, lmm1 = Ko['x_lmu1'], Ko['x_lmm1']
    snap, ncour, lane = jax.pure_callback(lambda a, b: JO._xpre_np(a, b, dt, lmu1, lmm1), shp, mm, mu)
    PROBE.append(('xpre snap_nonfinite', nf(snap)))
    PROBE.append(('xpre ncour_min', jnp.min(ncour)))
    PROBE.append(('xpre ncour_max', jnp.max(ncour)))
    PROBE.append(('xpre lane_true', jnp.sum(lane).astype(jnp.int64)))
    PROBE.append(('xpre mass_in_nonfinite', nf(mm)))
    PROBE.append(('xpre mu_in_nonfinite', nf(mu)))
    return OJ._x_sweep(rm, rx, ry, rz, mm, snap[ls, js], ncour[ls, js], lane[ls, js], jnp.asarray(Ko['x_lmu_l']), jnp.asarray(Ko['x_lmm_l']),
                       1.0 if qlimit else 0.0)


def oadvt2_probe(Ko, mmi, rm, rx, ry, rz, dt, qlimit, smu, smv, smw):
    tag = 'S0M' if qlimit else 'G0M'
    lmu1, lmv1, lmm1 = Ko['lmu1'], Ko['lmv1'], Ko['lmm1']
    PROBE.append((tag + ' in rm,rx,ry,rz,mmi,smu,smv,smw nonfinite', jnp.stack([nf(a) for a in (rm, rx, ry, rz, mmi, smu, smv, smw)])))
    out = oadvtx2_probe(Ko, rm, rx, ry, rz, mmi, smu, 0.5 * dt, qlimit)
    PROBE.append((tag + ' after x1 (rm,rx,ry,rz,ma)', jnp.stack([nf(a) for a in out])))
    out = OJ.oadvty2_jax(*out[:4], out[4], smv, dt, qlimit, lmm1, lmv1)
    PROBE.append((tag + ' after y  (rm,rx,ry,rz,ma)', jnp.stack([nf(a) for a in out])))
    out = OJ.oadvtz2_jax(*out[:4], out[4], smw, dt, qlimit, lmm1)
    PROBE.append((tag + ' after z  (rm,rx,ry,rz,ma)', jnp.stack([nf(a) for a in out])))
    rm, rx, ry, rz, ma = oadvtx2_probe(Ko, *out[:4], out[4], smu, 0.5 * dt, qlimit)
    PROBE.append((tag + ' after x2 (rm,rx,ry,rz,ma)', jnp.stack([nf(a) for a in (rm, rx, ry, rz, ma)])))
    rm = rm.at[1:JO.IM + 1, JO.JM, 1:].set(rm[1, JO.JM, 1:][None, :])
    rz = rz.at[1:JO.IM + 1, JO.JM, 1:].set(rz[1, JO.JM, 1:][None, :])
    return ma, rm, rx, ry, rz


_orig_stage_dynamics = JO.stage_dynamics


def stage_dynamics_probe(K, Kb, s, fx):
    del PROBE[:]
    out = _orig_stage_dynamics(K, Kb, s, fx)
    out = dict(out)
    for i, (name, v) in enumerate(PROBE):
        out['__p%02d|%s' % (i, name)] = v
    del PROBE[:]
    return out


JO.oadvt2_dev = oadvt2_probe
JO.stage_dynamics = stage_dynamics_probe
JO.STAGES[:] = [(n, (stage_dynamics_probe if n == 'dynamics' else f)) for n, f in JO.STAGES]
captured = []


def make(which):
    K = cp.K

    def f(oc, fx, Kb):
        return JO.ocean_stages(K, Kb, oc, fx, which=which)
    return jax.jit(f)


oa2, ob2 = make('a'), make('b')


def wrap_b(oc, fx, Kb):
    out = ob2(oc, fx, Kb)
    captured.append(jax.device_get({k: v for k, v in out.items() if k.startswith('__p')}))
    return {k: v for k, v in out.items() if not k.startswith('__p')}


cp.oa, cp.ob = oa2, wrap_b
sr = G.build_registry()
state, rec, dev = cp.initial_state()
out, state, arrays, sr = G.run_step(cp, 0, state, rec, dev, sr, False)
print(f'[step 0] done; backend {jax.default_backend()}', flush=True)
print('\ncall-order table (the first non-zero entry is the culprit):')
first = None
for c in captured[:1]:
    for k in sorted(c):
        v = c[k]
        arr = [int(x) for x in jnp.atleast_1d(jnp.asarray(v))]
        print(f'  {k[0:5]}  {k[6:]:62s} {arr}')
        if first is None and 'ncour' not in k and 'lane_true' not in k and any(arr):
            first = k[6:]
print('\nFIRST non-zero entry:', first)
