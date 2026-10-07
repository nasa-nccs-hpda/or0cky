"""D158: ghy_jax.advnc on the day-long files that carry ffnit >= 12 cells, built with ghy_advnc_test_nit.build_batch_nit, vs the real record
(tolerances as in test_ghy_jax.py, which marks these four files xfail with the old build_batch)."""
import glob, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
pytestmark = pytest.mark.skipif(not os.path.exists(f"{FF}/nov26_day/ffg_33337.bin"), reason="dumps not available")

import ghy_jax as J                 # noqa: E402
import ghy_compare as GC            # noqa: E402
import ghy_advnc_test as AT         # noqa: E402
import ghy_advnc_test_nit as ATN    # noqa: E402

TOL = dict(tbcs=1e-3, tsns=1e-3, ashg=1e-2, alhg=5e-2, aevap=5e-2, arunu=1e-2, aerunu=1e-2, ae0=5e-2, abetad=1e-10,
           w_out=0.15, ht_out=0.05, tp_out=0.15)
STRICT = dict(ashg=1e-5, tbcs=1e-5, alhg=1e-4, aevap=1e-4)    # stiff cells specifically


def _run(builder, rec):
    (s0, d0, f, edts, ecnc, ebet, elai, ns, dtt, snowm, wsc, shc, refs) = builder(rec)
    st = J.init_static(jnp.asarray(s0["dz"]), jnp.asarray(s0["q"]), jnp.asarray(s0["qk"]), jnp.asarray(f["fb"]), jnp.asarray(f["fv"]))
    st = dict(st)
    st["ws"] = st["ws"].at[:, 0, 1].set(jnp.asarray(wsc)); st["shc"] = st["shc"].at[:, 0, 1].set(jnp.asarray(shc))
    st = J.init_xklh_static(st); st["sl"] = jnp.asarray(s0["sl"])
    out = J.advnc(st, {k: jnp.asarray(v) for k, v in d0.items()}, {k: jnp.asarray(v) for k, v in f.items()}, jnp.asarray(edts),
                  jnp.asarray(ecnc), jnp.asarray(ebet), jnp.asarray(elai), jnp.asarray(ns), jnp.asarray(dtt), jnp.asarray(snowm), max_substeps=edts.shape[1])
    return {k: np.asarray(v) for k, v in out.items()}, refs


def _relerr(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-6)


@pytest.mark.parametrize("itime", [33321, 33329, 33336, 33337])
def test_jax_matches_real_incl_stiff_cells(itime):
    rec = GC.load(f"{FF}/nov26_day/ffg_{itime}.bin")
    nit = np.round(rec[:, 289]).astype(int)
    stiff = np.where(nit >= 12)[0]
    assert len(stiff) >= 1
    sel = np.unique(np.concatenate([np.arange(40), stiff]))
    out, refs = _run(ATN.build_batch_nit, rec[sel])
    is_stiff = nit[sel] >= 12
    for k, tol in STRICT.items():
        assert _relerr(out[k][is_stiff], refs[k][is_stiff]).max() < tol, k
    for k, tol in TOL.items():
        key = {"w_out": "w", "ht_out": "ht", "tp_out": "tp"}.get(k, k)
        a = out[key][:, :7, :] if k in ("w_out", "ht_out", "tp_out") else out[key]
        e = _relerr(a, refs[k])
        if k == "abetad":
            # abetad (mean root-beta) of a stiff cell uses the Ent betadl of iterations > 11, which the record does not hold (iteration-11 values reused)
            assert e[~is_stiff].max() < tol and e[is_stiff].max() < 1e-5, k
        else:
            assert e.max() < tol, k


def test_old_build_batch_fails_on_stiff_cell():
    rec = GC.load(f"{FF}/nov26_day/ffg_33337.bin")
    sel = np.where(np.round(rec[:, 289]) >= 12)[0]
    out, refs = _run(AT.build_batch_recorded, rec[sel])
    assert _relerr(out["ashg"], refs["ashg"]).max() > 1e-2
