"""Tests (D28): landice_precip_jax.precip_li (batched PRECIP_LI/PRECLI, Stage 1 of the DYNSI/ocean port) vs
real Fortran, the plain-Python reference, and mutation checks. The real 3-date record is entirely cold
(ENRGP<0, snow accumulation) -- the "rain" branch (ENRGP>=0: melt, possible ice-layer transfer) is never
exercised, so it is cross-checked against the plain-Python reference on synthetic inputs instead (same
honest-scoping pattern as D14's ADDICE synthetic-branch test)."""
import os, sys, glob
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffx_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran PRECIP_LI dumps not available")

import landice_precip_jax as J        # noqa: E402
import landice_precip_ff as L         # noqa: E402
import precli_compare as C            # noqa: E402
import precli_jax_compare as JC       # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return JC.load_all(FILES)


def test_matches_fortran(rec):
    out = JC.batched_precip_li(rec)
    for k, c in (("snow", 9), ("tg1", 10), ("tg2", 11), ("runo", 12), ("e1", 13), ("implm", 14), ("implh", 15)):
        d = np.abs(np.asarray(out[k]) - rec[:, c])
        assert d.max() == 0.0, k
    assert len(rec) > 3000


def test_non_vacuous(rec):
    assert (rec[:, 4] > 0).mean() > 0.99      # precip active on essentially every real cell here
    assert (rec[:, 14] != 0).sum() > 0        # real snow-compaction (DIFS) cells


def test_matches_plain_python_reference_row_for_row(rec):
    out = JC.batched_precip_li(rec)
    idx = np.linspace(0, len(rec) - 1, min(300, len(rec))).astype(int)
    worst = 0.0
    for i in idx:
        r = rec[i]
        py = L.precip_li(r[3], r[4], r[5], r[6], r[7], r[8])
        for k in ("snow", "tg1", "tg2", "runo"):
            got = float(np.asarray(out[k])[i])
            worst = max(worst, abs(got - py[k]) / max(abs(py[k]), 1e-6))
    assert worst < 1e-9


def test_jit_compiles_and_matches_eager(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    args = (a(3), a(4), a(5), a(6), a(7), a(8))
    eager = J.precip_li(*args)
    jitted = jax.jit(J.precip_li)(*args)
    for k in ("snow", "tg1", "tg2"):
        assert float(jnp.max(jnp.abs(eager[k] - jitted[k]))) < 1e-9, k


def test_no_nan_or_inf(rec):
    out = JC.batched_precip_li(rec)
    for k, v in out.items():
        v = np.asarray(v)
        assert not np.isnan(v).any(), k
        assert not np.isinf(v).any(), k


def test_mutations_are_detected(rec):
    out = JC.batched_precip_li(rec)
    old = J.LHM
    try:
        J.LHM = 3.0e5
        out2 = JC.batched_precip_li(rec)
        d = np.max(np.abs(np.asarray(out2["tg1"]) - np.asarray(out["tg1"])))
        assert d > 1e-4
    finally:
        J.LHM = old


def test_rain_branches_match_plain_python_on_synthetic_inputs():
    """ENRGP>=0 (rain) is never exercised by the real 3-date record: cross-check against
    landice_precip_ff on synthetic inputs covering both rain sub-branches (partial melt, and
    melt-through-to-layer-2)."""
    rng = np.random.default_rng(0)
    n = 200
    ftype = np.ones(n)
    tg1 = -rng.uniform(0.5, 15.0, n)
    tg2 = -rng.uniform(0.5, 15.0, n)
    snow = rng.uniform(0.0, 200.0, n)
    hc1 = L.HC1LI + snow * L.SHI
    # partial melt: enrgp in [0, -tg1*hc1) roughly -> dwater < snow enforced by scaling
    enrgp_partial = rng.uniform(0.0, 1.0, n) * (-tg1 * hc1) * 0.5
    prcp = rng.uniform(0.0, 5.0, n)
    got = J.precip_li(jnp.asarray(ftype), jnp.asarray(prcp), jnp.asarray(enrgp_partial), jnp.asarray(snow),
                      jnp.asarray(tg1), jnp.asarray(tg2))
    worst = 0.0
    for i in range(n):
        py = L.precip_li(ftype[i], prcp[i], enrgp_partial[i], snow[i], tg1[i], tg2[i])
        for k in ("snow", "tg1", "tg2", "runo"):
            worst = max(worst, abs(float(got[k][i]) - py[k]) / max(abs(py[k]), 1e-6))
    assert worst < 1e-9
    assert bool(np.all(enrgp_partial >= 0))       # confirms the rain branch is what's being exercised

    # melt-through: large enrgp forces dwater > snow (melts all snow, moves into ice)
    enrgp_big = snow * L.LHM - tg1 * hc1 + rng.uniform(1000.0, 1e5, n)   # forces dwater = snow*LHM.../LHM > snow
    got2 = J.precip_li(jnp.asarray(ftype), jnp.asarray(prcp), jnp.asarray(enrgp_big), jnp.asarray(snow),
                       jnp.asarray(tg1), jnp.asarray(tg2))
    worst2 = 0.0
    n_transfer = 0
    for i in range(n):
        py = L.precip_li(ftype[i], prcp[i], enrgp_big[i], snow[i], tg1[i], tg2[i])
        if py["implm"] != 0:
            n_transfer += 1
        for k in ("snow", "tg1", "tg2", "runo", "implm", "implh", "e1"):
            worst2 = max(worst2, abs(float(got2[k][i]) - py[k]) / max(abs(py[k]), 1e-6))
    assert worst2 < 1e-9
    assert n_transfer > 100   # confirms the melt-through-to-layer-2 branch is genuinely exercised
