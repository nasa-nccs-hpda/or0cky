"""D219: GHY with the Ent exports computed INSIDE the sub-iteration loop (ent_jax via ghy_jax.advnc_gdtm(ent=...)) against the same GHY call with the RECORDED Ent exports,
on the real nov26 ffg records of one step (all land cells).  python ent_live_check.py [IT]"""
import os
import sys
import time

import clouds_jax_env  # noqa: F401
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import atm_step as A
import ent_ff as E
import ent_jax as EJ
import ghy_advnc_test as AT
import ghy_jax as J
import land_chain as LC

IT = int(sys.argv[1]) if len(sys.argv) > 1 else 33312
E.set_use_imf(False)
R = A.Real('nov26_day', IT)
g = A.surface_records(R)['g1']
cells = E.load_restart_cells(f'{A.FF}/_pristine_restarts/fort1_nov26_itime33312.nc')
keys = [(int(r[0]), int(r[1])) for r in g]
const, state = EJ.pack_cells(cells, keys)
(s0, d0, f, edts, ecnc, ebet, elai, ns, dt, snowm, wsc, shc, refs) = AT.build_batch_recorded(g)
f = dict(f)
pr = np.maximum(f["pr"], 0.0)
prs = np.minimum(np.maximum(f["prs"], 0.0), pr)
f["pr"], f["prs"] = pr, prs
f["htprs"] = np.where(pr <= 0.0, 0.0, f["htpr"] / np.where(pr <= 0.0, 1.0, pr) * prs)
fv = np.asarray(f["fv"]); pos = fv > 0.0
f["irrig"] = np.where(pos, g[:, 147] / np.where(pos, fv, 1.0), 0.0)
f["htirrig"] = np.where(pos, g[:, 148] / np.where(pos, fv, 1.0), 0.0)
st = dict(J.init_static(jnp.asarray(s0["dz"]), jnp.asarray(s0["q"]), jnp.asarray(s0["qk"]), jnp.asarray(f["fb"]), jnp.asarray(f["fv"])))
st["ws"] = st["ws"].at[:, 0, 1].set(jnp.asarray(wsc))
st["shc"] = st["shc"].at[:, 0, 1].set(jnp.asarray(shc))
st = J.init_xklh_static(st)
st["sl"] = jnp.asarray(s0["sl"])
args = (st, {k: jnp.asarray(v) for k, v in d0.items()}, {k: jnp.asarray(v) for k, v in f.items()}, jnp.asarray(edts), jnp.asarray(ecnc), jnp.asarray(ebet),
        jnp.asarray(elai), jnp.asarray(ns), jnp.asarray(dt), jnp.asarray(snowm))
ms = edts.shape[1]
ref = jax.jit(J.advnc_gdtm, static_argnames=('max_substeps',))(*args, max_substeps=ms)
EOD = 163
ent = dict(const={k: jnp.asarray(v) for k, v in const.items() if isinstance(v, np.ndarray)}, state={k: jnp.asarray(v) for k, v in state.items()},
           qf0=jnp.asarray(g[:, 7]), entf=dict(Ca=jnp.asarray(g[:, 3]), cosz1=jnp.asarray(g[:, 4]), vis=jnp.asarray(g[:, 5]), dvis=jnp.asarray(g[:, 6]),
                                               eod=jnp.asarray(g[:, EOD] == 1.0)))
live_fn = jax.jit(lambda a, e: J.advnc_gdtm(*a, max_substeps=ms, ent=e))
t0 = time.perf_counter()
live = live_fn(args, ent)
jax.block_until_ready(live)
print(f'live Ent GHY call: compile+run {time.perf_counter() - t0:.1f} s; cells {len(keys)}, max sub-iterations in the reference {int(np.max(np.asarray(ref["nit_gdtm"])))}', flush=True)
t0 = time.perf_counter()
live = live_fn(args, ent)
jax.block_until_ready(live)
print(f'steady {1e3 * (time.perf_counter() - t0):.1f} ms', flush=True)
worst = []
for k in ('w', 'ht', 'tp', 'tbcs', 'tsns', 'ashg', 'alhg', 'aevap', 'aruns', 'arunu', 'aeruns', 'aerunu', 'ae0', 'abetad', 'evap_max_ij', 'fr_sat_ij', 'fice', 'wsn', 'hsn', 'fr_snow', 'dzsn'):
    a, b = np.asarray(ref[k]), np.asarray(live[k])
    sc = max(float(np.max(np.abs(a))), 1e-30)
    worst.append((float(np.max(np.abs(a - b))) / sc, k))
print('nit equal:', bool(np.array_equal(np.asarray(ref['nit_gdtm']), np.asarray(live['nit_gdtm']))), ' n cells with different nit', int((np.asarray(ref['nit_gdtm']) != np.asarray(live['nit_gdtm'])).sum()))
for w, k in sorted(worst, reverse=True)[:8]:
    print(f'  {k}: max |live - recorded-Ent| / scale = {w:.2e}')
