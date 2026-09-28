"""Tests (D26): seaice_core_jax.prec_si (batched PRECIP_SI/PREC_SI, Stage 1 of the DYNSI/ocean port) vs real
Fortran, the plain-Python reference, and mutation checks. Precip fell over sea ice on ~91% of real cells here
(non-vacuous); the snow-compression (CMPRS) and salt-in-runoff (SRUN0) branches never trigger in this record --
documented, not hidden (same pattern as D14's ADDICE)."""
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
FILES = sorted(glob.glob(f"{FF}/*/ffw_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran PRECIP_SI dumps not available")

import seaice_core_jax as J        # noqa: E402
import seaice_core_ff as S         # noqa: E402
import precsi_compare as C         # noqa: E402
import precsi_jax_compare as JC    # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return JC.load_all(FILES)


def relerr(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    return np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-6))


def test_matches_fortran(rec):
    out = JC.batched_prec_si(rec)
    ref = dict(snow=rec[:, 15], msi2=rec[:, 16], hsil=rec[:, 17:21], ssil=rec[:, 21:25], tsil=rec[:, 25:29],
               run0=rec[:, 29], srun0=rec[:, 30], erun0=rec[:, 31], cmprs=rec[:, 33])
    for k in ("snow", "msi2", "cmprs"):
        assert relerr(out[k], ref[k]) < 1e-9, k
    for k in ("run0", "srun0", "erun0"):    # near-zero floor: relerr's 1e-6 denominator floor makes tiny
        assert relerr(out[k], ref[k]) < 1e-8, k   # absolute differences look relatively large; still ~1e-11 abs
    assert relerr(out["hsil"], ref["hsil"]) < 1e-9
    assert relerr(out["ssil"], ref["ssil"]) < 1e-9
    assert float(np.max(np.abs(np.asarray(out["tsil"]) - ref["tsil"]))) < 1e-6   # absolute (deg C, near-zero floor issue for relerr)
    assert bool(np.all(np.asarray(out["wetsnow"]) == (rec[:, 32] > 0.5)))
    assert len(rec) > 10000


def test_non_vacuous(rec):
    prcp = rec[:, 13]
    assert (prcp > 0).mean() > 0.5
    assert (rec[:, 29] != 0).sum() > 10          # real runoff cells
    assert (rec[:, 32] > 0.5).sum() > 10         # real wetsnow cells
    assert (rec[:, 3] == 0).sum() > 5            # real no-existing-snow cells


def test_matches_plain_python_reference_row_for_row(rec):
    out = JC.batched_prec_si(rec)
    idx = np.linspace(0, len(rec) - 1, 300).astype(int)
    worst = 0.0
    for i in idx:
        py = S.prec_si(rec[i, 3], rec[i, 4], list(rec[i, 5:9]), list(rec[i, 9:13]), rec[i, 13], rec[i, 14])
        for k in ("snow", "msi2", "run0", "srun0", "erun0"):
            got = float(np.asarray(out[k])[i])
            worst = max(worst, abs(got - py[k]) / max(abs(py[k]), 1e-6))
    assert worst < 1e-9


def test_jit_compiles_and_matches_eager(rec):
    a = lambda c: jnp.asarray(rec[:, c])
    args = (a(3), a(4), jnp.asarray(rec[:, 5:9]), jnp.asarray(rec[:, 9:13]), a(13), a(14))
    eager = J.prec_si(*args)
    jitted = jax.jit(J.prec_si)(*args)
    for k in ("snow", "msi2", "run0"):
        assert float(jnp.max(jnp.abs(eager[k] - jitted[k]))) < 1e-9, k


def test_no_nan_or_inf(rec):
    out = JC.batched_prec_si(rec)
    for k, v in out.items():
        v = np.asarray(v)
        assert not np.isnan(v.astype(float)).any(), k
        assert not np.isinf(v.astype(float)).any(), k


def test_mutations_are_detected(rec):
    out = JC.batched_prec_si(rec)
    old = J.LHM
    try:
        J.LHM = 3.0e5
        out2 = JC.batched_prec_si(rec)
        assert relerr(out2["hsil"], out["hsil"]) > 1e-4
    finally:
        J.LHM = old

    old = J.SNOMAX
    try:
        J.SNOMAX = old * 0.01   # force the "too much snow" compression branch to fire far more often
        out3 = JC.batched_prec_si(rec)
        assert relerr(out3["snow"], out["snow"]) > 1e-6
    finally:
        J.SNOMAX = old
