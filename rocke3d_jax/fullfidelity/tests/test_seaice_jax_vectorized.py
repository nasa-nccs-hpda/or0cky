"""Tests: seaice_core_jax (batched/vectorized SEA_ICE+SSIDEC+snowice+SIMELT) vs the SAME real-Fortran
dumps used for seaice_core_ff (plain Python, already validated -- D10/D12), and cross-checked against
that plain-Python reference row-for-row. ADDICE is out of scope here (see seaice_core_jax.py module
docstring for why)."""
import os, sys, glob

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FFI_FILES = sorted(glob.glob(f"{FF}/*/ffi_*.bin"))
FFM_FILES = sorted(glob.glob(f"{FF}/*/ffm_*.bin"))
pytestmark = pytest.mark.skipif(len(FFI_FILES) < 6 or len(FFM_FILES) < 6,
                                reason="real-Fortran GROUND_SI/SIMELT dumps not available")

import seaice_core_jax as J       # noqa: E402
import seaice_core_ff as S        # noqa: E402
import seaice_compare as C        # noqa: E402
import seaice_jax_compare as JC   # noqa: E402


def relerr(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    return np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-6))


@pytest.fixture(scope="module")
def ffi_rec():
    return JC.load_all_ffi(FFI_FILES)


@pytest.fixture(scope="module")
def ffm_rec():
    return JC.load_all_ffm(FFM_FILES)


def test_ground_si_matches_fortran_at_same_tolerance_as_plain_python(ffi_rec):
    """Same tolerances as D10 in FULL_FIDELITY_DELTAS.md -- the batched port should be exactly as
    accurate as the plain-Python reference, not looser."""
    out = JC.batched_ground_si(ffi_rec)
    ref = dict(snow=ffi_rec[:, 42], hsil=ffi_rec[:, 43:47], ssil=ffi_rec[:, 47:51], msi2=ffi_rec[:, 51],
               runosi=ffi_rec[:, 52], erunosi=ffi_rec[:, 53], srunosi=ffi_rec[:, 54])
    assert relerr(out["snow"], ref["snow"]) < 1e-10
    assert relerr(out["hsil"], ref["hsil"]) < 1e-6
    assert relerr(out["ssil"], ref["ssil"]) < 1e-5
    assert relerr(out["msi2"], ref["msi2"]) < 1e-6
    assert relerr(out["runosi"], ref["runosi"]) < 1e-3
    assert relerr(out["erunosi"], ref["erunosi"]) < 1e-3
    assert relerr(out["srunosi"], ref["srunosi"]) < 1e-3
    assert len(ffi_rec) > 4000


def test_simelt_matches_fortran_bitwise(ffm_rec):
    out = JC.batched_simelt(ffm_rec)
    ref = dict(roice=ffm_rec[:, 18], snow=ffm_rec[:, 19], msi2=ffm_rec[:, 20],
               hsil=ffm_rec[:, 21:25], ssil=ffm_rec[:, 25:29], enrgused=ffm_rec[:, 29])
    for k in ("roice", "snow", "msi2", "hsil", "ssil"):
        assert relerr(out[k], ref[k]) == 0.0, k
    assert relerr(out["enrgused"], ref["enrgused"]) < 1e-10
    melted_out = int(np.sum(np.asarray(out["melted_out"])))
    assert melted_out > 50   # matches the plain-Python D12 count (376 on this record set)


def test_no_nan_or_inf_over_full_record(ffi_rec, ffm_rec):
    out = JC.batched_ground_si(ffi_rec)
    for k, v in out.items():
        v = np.asarray(v)
        assert not np.isnan(v).any(), k
        assert not np.isinf(v).any(), k
    out_m = JC.batched_simelt(ffm_rec)
    for k, v in out_m.items():
        v = np.asarray(v)
        if k == "tsil":
            continue   # NaN is the documented sentinel for the undefined-TSIL branch
        assert not np.isnan(v).any(), k
        assert not np.isinf(v).any(), k


def test_matches_plain_python_reference_row_for_row(ffi_rec):
    """Cross-check against seaice_core_ff (already validated bitwise/near-bitwise vs Fortran) on a
    stratified sample, independent of the direct-vs-Fortran comparison above."""
    out = JC.batched_ground_si(ffi_rec)
    idx = np.linspace(0, len(ffi_rec) - 1, 300).astype(int)
    worst = 0.0
    for i in idx:
        py = C.compare_full(ffi_rec[i])
        for k in ("snow", "msi2", "runosi", "erunosi", "srunosi"):
            got = float(np.asarray(out[k])[i])
            worst = max(worst, abs(got - py[k]["got"]) / max(abs(py[k]["got"]), 1e-6))
    assert worst < 1e-9


def test_jit_compiles_and_matches_eager(ffi_rec):
    dtsrc = jnp.asarray(ffi_rec[:, 3]); snow = jnp.asarray(ffi_rec[:, 4])
    hsil = jnp.asarray(ffi_rec[:, 6:10]); ssil = jnp.asarray(ffi_rec[:, 10:14])
    msi2 = jnp.asarray(ffi_rec[:, 14]); f0dt = jnp.asarray(ffi_rec[:, 15]); f1dt = jnp.asarray(ffi_rec[:, 16])
    evap = jnp.asarray(ffi_rec[:, 17]); srox0 = jnp.asarray(ffi_rec[:, 18])
    fmoc = jnp.asarray(ffi_rec[:, 19]); fhoc = jnp.asarray(ffi_rec[:, 20]); fsoc = jnp.asarray(ffi_rec[:, 21])
    wetsnow = jnp.asarray(ffi_rec[:, 22]) > 0.5

    eager = J.sea_ice(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    jitted = jax.jit(J.sea_ice)(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    for k in ("snow", "msi2", "run", "erun", "srun"):
        assert float(jnp.max(jnp.abs(eager[k] - jitted[k]))) < 1e-9, k


def test_mutations_are_detected(ffi_rec, ffm_rec):
    out = JC.batched_ground_si(ffi_rec)
    active_freeze = np.asarray(np.abs(np.asarray(out["snow"]) - ffi_rec[:, 4])) > 1e-6
    assert active_freeze.sum() > 50

    old = J.LHM
    try:
        J.LHM = 3.0e5
        out2 = JC.batched_ground_si(ffi_rec)
        d = relerr(out2["hsil"], out["hsil"])
        assert d > 1e-4
    finally:
        J.LHM = old

    out_m = JC.batched_simelt(ffm_rec)
    old = J.SILMFAC
    try:
        J.SILMFAC = old * 5
        out_m2 = JC.batched_simelt(ffm_rec)
        assert relerr(out_m2["roice"], out_m["roice"]) > 1e-6
    finally:
        J.SILMFAC = old
