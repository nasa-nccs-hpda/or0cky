"""Tests for clouds_dq_ff.py (CLOUDS2.F90 get_dq_cond / get_dq_evap) -- D89.

Real-Fortran validation uses ff_data/<date>/ffc_dq_*.bin (skipped if absent).  The hand-derived /
mutation tests at the bottom do not need dumps and are NOT validation against real Fortran.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_dq_ff as dq  # noqa: E402
import clouds_dq_compare as cmp  # noqa: E402

DATES = [d for d, _ in cmp.DATES]
HAVE = {d: os.path.exists(f"{cmp.FF_DEFAULT}/{d}/ffc_dq_consts.txt") for d in DATES}
needs_dumps = pytest.mark.skipif(not all(HAVE.values()), reason="ff_data ffc_dq_* dumps not present on this host")

TOL_SCALE = 1e-12       # |port - real| / (qm or cond), see FULL_FIDELITY_DELTAS.md D89
_cache = {}


def _data(date):
    if date not in _cache:
        _cache[date] = cmp.load_date(date)
    return _cache[date]


# ---------------------------------------------------------------- real-Fortran validation
@needs_dumps
def test_constants_match_real_model():
    c = cmp.read_consts(f"{cmp.FF_DEFAULT}/nov26/ffc_dq_consts.txt")
    assert dq.BYSHA == c["bysha"]
    assert dq.MRAT == c["mrat"]
    assert dq.RVAP == c["rvap"]
    assert dq.TF == c["tf"]


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_all_sites_match_real_fortran(date):
    d = _data(date)
    st = cmp.stats(d)
    assert set(st) == {1, 2, 3, 4, 5, 6}
    for s, r in st.items():
        assert r["n_branch_mismatch"] == 0, (date, s)
        assert r["max_rel_scale"] < TOL_SCALE, (date, s, r)
        assert r["max_f_abs"] < 1e-12, (date, s, r)


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_mostly_bitwise(date):
    d = _data(date)
    mdq, mf = cmp.run_port(d)
    assert (mdq == d["dqsum"]).mean() > 0.98
    assert (mf == d["f"]).mean() > 0.98


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_non_vacuous(date):
    """Real data actually exercises condensation/evaporation, the QM/COND guards and both clamps."""
    d = _data(date)
    assert (d["dqsum"] > 0).sum() > 10000
    assert ((d["kind"] == 2) & (d["cond"] <= 0)).sum() > 100          # inactive evap records (guard)
    scale = np.where(d["kind"] == 1, d["qm"], d["cond"])
    assert ((d["dqsum"] == scale) & (scale > 0)).sum() > 100          # upper clamp reached
    assert ((d["dqsum"] == 0) & (scale > 0)).sum() > 100              # lower clamp reached


@needs_dumps
@pytest.mark.parametrize("mutation", [dict(niter=2), dict(niter=4), dict(clamp=False), dict(bysha=1.01 * dq.BYSHA)])
def test_mutations_detected_on_real_data(mutation):
    """The comparison must fail if the algorithm is perturbed (non-vacuity of the validation)."""
    d = _data("nov26")
    c = d["kind"] == 1
    e = d["kind"] == 2
    a1, _ = dq.get_dq_cond(d["sm"][c], d["qm"][c], d["plk"][c], d["mass"][c], d["lhx"][c], d["pl"][c], **mutation)
    a2, _ = dq.get_dq_evap(d["sm"][e], d["qm"][e], d["plk"][e], d["mass"][e], d["lhx"][e], d["pl"][e],
                           d["cond"][e], **mutation)
    r1 = np.abs(a1 - d["dqsum"][c]) / d["qm"][c]
    r2 = np.abs(a2 - d["dqsum"][e]) / np.maximum(d["cond"][e], 1e-300)
    assert max(r1.max(), np.nanmax(r2[d["cond"][e] > 0])) > 1e-9


@needs_dumps
def test_evap_guard_mutation_detected():
    d = _data("nov26")
    e = (d["kind"] == 2) & (d["cond"] <= 0)
    _, f = dq.get_dq_evap(d["sm"][e], d["qm"][e], d["plk"][e], d["mass"][e], d["lhx"][e], d["pl"][e],
                          d["cond"][e], guard=False)
    assert not np.array_equal(f, d["f"][e])      # unguarded divides by zero


# ------------------------- hand-derived cases (NOT validated against real Fortran)
def test_guards_return_zero():
    a, f = dq.get_dq_cond(300.0, 0.0, 1.0, 1.0, 2.5e6, 1000.0)
    assert (a, f) == (0.0, 0.0)
    a, f = dq.get_dq_evap(300.0, 0.01, 1.0, 1.0, 2.5e6, 1000.0, 0.0)
    assert (a, f) == (0.0, 0.0)


def test_dry_air_no_condensation():
    a, f = dq.get_dq_cond(300.0, 1e-8, 1.0, 1.0, 2.5e6, 1000.0)
    assert a == 0.0 and f == 0.0


def test_supersaturated_condenses_and_clamps():
    tm, pr, lh = 280.0, 900.0, 2.5e6
    qs = float(dq.qsat(tm, lh, pr))
    q = 3.0 * qs
    a, f = dq.get_dq_cond(tm, q, 1.0, 1.0, lh, pr)
    assert 0.0 < a < q and abs(f - a / q) < 1e-15
    # latent heating warms the parcel, so less than (q - qs) condenses, and after the 3 Newton
    # steps the remaining vapour is within 1% of saturation at the warmed temperature
    assert a < q - qs
    qs_warm = float(dq.qsat(tm + lh * dq.BYSHA * a, lh, pr))
    assert abs((q - a) - qs_warm) < 0.01 * qs_warm
    # tiny vapour amount cannot condense more than it holds
    a2, f2 = dq.get_dq_cond(tm, 1e-12, 1.0, 1.0, lh, pr)
    assert a2 == 0.0 or a2 <= 1e-12


def test_unsaturated_with_condensate_evaporates_up_to_cond():
    tm, pr, lh = 285.0, 900.0, 2.5e6
    qs = float(dq.qsat(tm, lh, pr))
    a, f = dq.get_dq_evap(tm, 0.2 * qs, 1.0, 1.0, lh, pr, 5e-3)
    assert 0 < a < 5e-3 and abs(f - a / 5e-3) < 1e-15
    a, f = dq.get_dq_evap(tm, 0.2 * qs, 1.0, 1.0, lh, pr, 1e-9)     # little condensate: fully evaporates
    assert a == 1e-9 and f == 1.0


def test_qsat_closed_form():
    tm, pr, lh = 273.15, 1000.0, 2.5e6
    expect = 6.108 * dq.MRAT * np.exp(lh / (dq.RVAP * dq.TF) - lh / (dq.RVAP * tm)) / pr
    assert abs(float(dq.qsat(tm, lh, pr)) - expect) / expect < 1e-13
    assert abs(6.108 * dq.MRAT / pr - float(dq.qsat(dq.TF, lh, pr))) < 1e-9   # 6.108 mb * mrat / P at 0 C
    assert abs(float(dq.qsat(100.0, lh, pr)) - float(dq.qsat(130.0, lh, pr))) == 0.0   # max(130, T) floor
