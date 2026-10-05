"""Tests for clouds_massflux_ff.py (CLOUDS2.F90 MASS_FLUX) -- D93.

Real-Fortran validation uses ff_data/<date>/ffc_mf_*.bin (skipped if absent).  Hand-derived and mutation tests at
the bottom use only the port (the mutation tests still compare against real dumps and need them)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_massflux_ff as mf  # noqa: E402
import clouds_massflux_compare as cmp  # noqa: E402
import clouds_dq_ff as dq  # noqa: E402

DATES = [d for d, _ in cmp.DATES]
HAVE = {d: os.path.exists(f"{cmp.FF_DEFAULT}/{d}/ffc_h_consts.txt") for d in DATES}
needs_dumps = pytest.mark.skipif(not all(HAVE.values()), reason="ff_data ffc_mf_* dumps not present on this host")
_cache = {}


def _data(date):
    if date not in _cache:
        _cache[date] = cmp.load_date(date)
    return _cache[date]


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_decision_flow_matches_real(date):
    """Iteration count at exit, the final DQSUM>0 branch and the caller's FPLUME<=.001 test agree on every record
    (the only places a last-bit difference could flip a branch)."""
    d = _data(date)
    r = cmp.run_port(d)
    assert np.array_equal(r["iters"], d["iters"])
    assert np.array_equal(r["dqsum"] > 0, d["dqsum"] > 0)
    assert np.array_equal(r["fplume"] <= 0.001, d["fplume"] <= 0.001)


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_outputs_match_real(date):
    d = _data(date)
    st = cmp.compare(d)
    assert st["fplume"]["bitwise"] == st["fplume"]["n"]        # FPLUME bitwise on every record
    assert st["fmp2"]["bitwise"] == st["fmp2"]["n"]
    assert st["dqsum"]["max_rel"] < 1e-12 and st["dqsum"]["bitwise"] / st["dqsum"]["n"] > 0.97
    assert st["dmse1"]["max_abs"] < 1e-12
    assert st["tnx"]["bitwise"] == st["tnx"]["n"] and st["qnx"]["bitwise"] == st["qnx"]["n"]


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_non_vacuous(date):
    d = _data(date)
    it = d["iters"].astype(int)
    for k in range(1, 10):                                      # every exit iteration 1..8 and the no-exit case (9)
        assert (it == k).sum() > 5, k
    assert (d["dqsum"] > 0).mean() > 0.8                        # evaporation blocks exercised
    assert (d["fplume"] <= 0.001).sum() > 10                    # caller's `cycle` threshold exercised
    assert 0.0009 < d["fplume"].min() < 0.01 and d["fplume"].max() < 0.5
    r = cmp.run_port(d)
    # both inner (nested) evaporation block states occur: TNX set for the lower layer vs only the upper layer
    assert np.isfinite(r["tnx"]).mean() > 0.9 and np.isnan(r["tnx"]).sum() > 100   # stale-TNX records exist


@needs_dumps
@pytest.mark.parametrize("mut", [dict(niter=7), dict(niter=9), dict(tol=1e-2), dict(tol=1e-4), dict(nested=False),
                                 dict(evap=False), dict(slhe=1.01 * mf.SLHE), dict(deltx=1.001 * mf.DELTX)])
def test_mutations_detected_on_real_data(mut):
    d = _data("nov26")
    good = cmp.run_port(d)
    bad = cmp.run_port(d, **mut)
    changed = (~np.isclose(bad["fplume"], d["fplume"], rtol=1e-12, atol=0)) | (bad["iters"] != d["iters"]) \
        | (~np.isclose(bad["dqsum"], d["dqsum"], rtol=1e-9, atol=1e-18))
    assert changed.sum() > 20, mut
    assert np.array_equal(good["fplume"], d["fplume"])


# ------------------------- hand-derived cases (NOT validated against real Fortran)
def _state(**kw):
    s = dict(lmin=3., lhx=2.5e6, qmo1=5e-3 * 100, qmo2=4e-3 * 100, smo1=290. * 100 / 1.0, smo2=288. * 100 / 1.0,
             slh=2.5e6 * dq.BYSHA, wmdn=0., wmup=0., wmedg=0., airm0=100., airm1=100., byam0=0.01, byam1=0.01,
             byam2=0.01, sm2=2.8e4, qm2=0.3, plk0=1.0, plk1=1.0, pl0=900., pl1=800.)
    s.update(kw)
    return s


def test_thbar_equal_temperatures_is_identity_to_1e6():
    # THBAR(T,T) = T*g(1), g(1)=1 within the rational-function accuracy (<1e-6 per the source comment)
    t = np.array([200., 250., 300.])
    assert np.allclose(mf.thbar(t, t), t, rtol=1e-6)


def test_always_returns_in_range_and_flags_no_exit():
    s = {k: np.full(4, v) for k, v in _state().items()}
    r = mf.mass_flux(**s)
    assert ((r["fplume"] > 0) & (r["fplume"] < 0.5)).all()
    assert ((r["iters"] >= 1) & (r["iters"] <= 9)).all()
    # convergence implies a small residual: any record that exited has |DMSE1|<=1e-3
    ex = r["iters"] <= 8
    assert (np.abs(r["dmse1"][ex]) <= 1e-3).all()


def test_dry_plume_no_evaporation_block():
    # QMO1=0: plume has no vapour, DQSUM=(0-QSATMP)/(1+GAMA)<0 so the evaporation block is skipped, TNX stays unset
    s = {k: np.full(2, v) for k, v in _state(qmo1=0.0).items()}
    r = mf.mass_flux(**s)
    assert (r["dqsum"] < 0).all() and np.isnan(r["tnx"]).all()
