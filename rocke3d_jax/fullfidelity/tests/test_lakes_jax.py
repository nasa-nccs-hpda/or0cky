"""Tests: lakes_core_jax (batched/vectorized LKSOURC/LKMIX) vs the SAME real-Fortran dumps used for
lakes_ff.py (plain Python, already validated -- D13), plus a synthetic check of lkmix's tke>0 branch
(dead code in this rundeck -- GROUND_LK always calls LKMIX with TKE=0, see D13 -- so real data never
exercises it)."""
import glob
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
FILES = sorted(glob.glob(f"{FF}/*/ffl2_*.bin"))
pytestmark = pytest.mark.skipif(len(FILES) < 6, reason="real-Fortran lake (GROUND_LK) dumps not available")

import lakes_ff as L           # noqa: E402
import lakes_core_jax as LJ    # noqa: E402
import lakes_compare as LC     # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return np.concatenate([LC.load(f) for f in FILES], axis=0)


def _run(rec):
    roice = jnp.asarray(rec[:, 2]); mlake0 = jnp.asarray(rec[:, 3]); mlake1 = jnp.asarray(rec[:, 4])
    elake0 = jnp.asarray(rec[:, 5]); elake1 = jnp.asarray(rec[:, 6]); run0 = jnp.asarray(rec[:, 7])
    fodt = jnp.asarray(rec[:, 8]); fidt = jnp.asarray(rec[:, 9]); srox0 = jnp.asarray(rec[:, 10])
    srox1 = jnp.asarray(rec[:, 11]); fsr2 = jnp.asarray(rec[:, 12]); evapo = jnp.asarray(rec[:, 13])
    hlake = jnp.asarray(rec[:, 14]); dtsrc = jnp.asarray(rec[:, 15]); tke = jnp.zeros_like(roice)
    src = LJ.lksourc_full(roice, mlake0, mlake1, elake0, elake1, run0, fodt, fidt, srox0, srox1, fsr2, evapo)
    mix = LJ.lkmix(src["mlake0"], src["mlake1"], src["elake0"], src["elake1"], hlake, tke, roice, dtsrc)
    return src, mix


def test_matches_real_fortran_bitwise(rec):
    src, mix = _run(rec)
    ref_mlake_src = rec[:, 16:18]; ref_elake_src = rec[:, 18:20]
    ref_enrgfo, ref_acefo, ref_acefi, ref_enrgfi = rec[:, 20], rec[:, 21], rec[:, 22], rec[:, 23]
    ref_mlake_mix = rec[:, 24:26]; ref_elake_mix = rec[:, 26:28]

    def relerr(a, b):
        return np.max(np.abs(np.asarray(a) - b) / np.maximum(np.abs(b), 1e-6))

    for got, ref in ((src["mlake0"], ref_mlake_src[:, 0]), (src["mlake1"], ref_mlake_src[:, 1]),
                    (src["elake0"], ref_elake_src[:, 0]), (src["elake1"], ref_elake_src[:, 1]),
                    (src["enrgfo"], ref_enrgfo), (src["acefo"], ref_acefo), (src["acefi"], ref_acefi),
                    (src["enrgfi"], ref_enrgfi), (mix["mlake0"], ref_mlake_mix[:, 0]),
                    (mix["mlake1"], ref_mlake_mix[:, 1]), (mix["elake0"], ref_elake_mix[:, 0]),
                    (mix["elake1"], ref_elake_mix[:, 1])):
        assert relerr(got, ref) == 0.0

    freeze_active = int(np.sum((np.abs(rec[:, 21]) > 1e-12) | (np.abs(rec[:, 22]) > 1e-12)))
    assert freeze_active > 100   # non-vacuous: real freezing genuinely exercised


def test_no_nan_or_inf(rec):
    src, mix = _run(rec)
    for d in (src, mix):
        for k, v in d.items():
            a = np.asarray(v)
            assert not np.isnan(a).any(), k
            assert not np.isinf(a).any(), k


def test_lkmix_tke_branch_matches_plain_python_synthetic():
    """The real record always calls LKMIX with TKE=0 (dead code in this rundeck, D13) -- cross-check
    the tke>0 branch against lakes_ff.py on synthetic inputs instead."""
    rng = np.random.default_rng(3)
    N = 500
    mlake0 = rng.uniform(50, 2000, N); mlake1 = rng.uniform(0, 2000, N)
    elake0 = mlake0 * rng.uniform(-2, 6, N) * 4185.0
    elake1 = mlake1 * rng.uniform(-2, 6, N) * 4185.0
    hlake = rng.uniform(0.5, 20, N)
    tke = np.abs(rng.normal(0, 50, N))
    roice = rng.uniform(0, 1, N)
    dtsrc = rng.uniform(500, 1000, N)

    mine = LJ.lkmix(jnp.asarray(mlake0), jnp.asarray(mlake1), jnp.asarray(elake0), jnp.asarray(elake1),
                    jnp.asarray(hlake), jnp.asarray(tke), jnp.asarray(roice), jnp.asarray(dtsrc))
    worst = 0.0
    n_active = 0
    for i in range(N):
        ref = L.lkmix([mlake0[i], mlake1[i]], [elake0[i], elake1[i]], hlake[i], tke[i], roice[i], dtsrc[i])
        if mlake1[i] > 0 and tke[i] > 0:
            n_active += 1
        worst = max(worst, abs(float(mine["mlake0"][i]) - ref["mlake"][0]),
                   abs(float(mine["elake1"][i]) - ref["elake"][1]))
    assert worst < 1e-9
    assert n_active > 400   # tke>0 branch genuinely exercised in this synthetic sample


def test_jit_compiles_and_matches_eager(rec):
    args = tuple(jnp.asarray(rec[:, c]) for c in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

    def full(roice, mlake0, mlake1, elake0, elake1, run0, fodt, fidt, srox0, srox1, fsr2, evapo, hlake, dtsrc):
        src = LJ.lksourc_full(roice, mlake0, mlake1, elake0, elake1, run0, fodt, fidt, srox0, srox1, fsr2, evapo)
        return LJ.lkmix(src["mlake0"], src["mlake1"], src["elake0"], src["elake1"], hlake,
                        jnp.zeros_like(roice), roice, dtsrc)

    eager = full(*args)
    jitted = jax.jit(full)(*args)
    for k in eager:
        assert float(jnp.max(jnp.abs(eager[k] - jitted[k]))) < 1e-9, k


def test_mutations_are_detected(rec):
    src, mix = _run(rec)
    old = LJ.LHM
    try:
        LJ.LHM = old * 1.2
        src2, _ = _run(rec)
        d = float(jnp.max(jnp.abs(src2["mlake0"] - src["mlake0"])))
        assert d > 1e-6
    finally:
        LJ.LHM = old
