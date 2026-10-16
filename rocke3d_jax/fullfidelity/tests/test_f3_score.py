"""D172 tests: the F3 scoring tool (f3_score.py) validated on the real JAN1950 ensemble by leave-one-out.  Skipped when the data are absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f3_score as S  # noqa: E402
import f3_diagnostics as f3  # noqa: E402

HAVE = all(os.path.exists(f"{S.ENS_DIR}/{m}/JAN1950.accP2SAoM40.nc") for m in S.MEMBERS) and os.path.exists(S.STORED)
pytestmark = pytest.mark.skipif(not HAVE, reason="JAN1950 ensemble / stored acc not present")
COLS = sorted(set(f3.AIJ_COLS.values()))


@pytest.fixture(scope="module")
def ref():
    meta = S.column_meta()
    X, ax, mem = S.load_reference(COLS, meta=meta)
    loo = S.leave_one_out(COLS, X=X, axyp=ax, members=mem, meta=meta)
    return dict(meta=meta, X=X, ax=ax, mem=mem, loo=loo, ls=S.loo_summaries(loo))


def test_ctrl_is_the_stored_month(ref):
    st = S.read_acc(S.STORED)
    f = S.monthly_fields(st["aij"], st["idacc"], ref["meta"], COLS)
    assert np.array_equal(np.nan_to_num(f, nan=-9e99), np.nan_to_num(ref["X"][0], nan=-9e99))


def test_loo_members_look_within_noise(ref):
    """Each member against the other 7: global-mean exceedance fraction near the t-expectation, zonal bins within 2 sigma near P(|t6|<2)=0.908,
    pooled rms ratios near 1.  Bounds are generous statistical checks of the method (8 members), not acceptance thresholds."""
    for m, s in ref["ls"].items():
        assert s["frac_gm_beyond_tcrit"] < 0.15, m
        assert 0.80 < s["zon_frac_within2_mean"] < 0.99, m
        assert 0.7 < s["median_R_grid"] < 1.4 and 0.5 < s["median_R_zon"] < 1.4, m
    mean_frac = np.mean([s["frac_gm_beyond_tcrit"] for s in ref["ls"].values()])
    assert 0.01 < mean_frac < 0.10
    mean_in2 = np.mean([s["zon_frac_within2_mean"] for s in ref["ls"].values()])
    assert abs(mean_in2 - ref["ls"]["ctrl"]["p_expected_within2"]) < 0.04


def test_stored_month_vs_perturbed_members_inside_loo_range(ref):
    r = S.score_month(S.ModelMonth.from_file(S.STORED, "stored"), COLS, X=ref["X"][1:], axyp=ref["ax"], meta=ref["meta"])
    c = S.compare_to_loo(S.summarize(r["rows"], 7), ref["ls"])
    assert all(v["inside"] for v in c.values())


def test_zero_spread_field_is_exact(ref):
    """incsw_toa is prescribed: zero ensemble spread, member difference exactly zero (or float noise) -> t finite or 0, never flagged."""
    rows = {r["name"]: r for r in ref["loo"]["per_member"]["p3"]}
    assert abs(rows["incsw_toa"]["gm_diff"]) < 1e-9


def test_detects_offset_and_wrong_month(ref):
    meta = ref["meta"]
    n2c = S.name_to_col(meta)
    mm = S.ModelMonth.from_file(S.STORED, "biased")
    a = mm.aij.copy()
    c = n2c["t_500"]
    a[c - 1] += 0.2 * mm.idacc[meta[c]["ia"] - 1]            # +0.2 K, ~4.4 sigma of a new member
    mm.aij = a
    r = S.score_month(mm, COLS, X=ref["X"][1:], axyp=ref["ax"], meta=meta)
    fl = S.flagged_fields(r["rows"])
    assert "t_500" in [x[0] for x in fl["flagged"]]
    feb = os.path.join(S.PROD_DIR, "FEB1950.accP2SAoM40.nc")
    if os.path.exists(feb):
        r = S.score_month(S.ModelMonth.from_file(feb, "feb"), COLS, X=ref["X"], axyp=ref["ax"], meta=meta)
        s = S.summarize(r["rows"], 8)
        assert s["frac_gm_beyond_tcrit"] > 0.4 and s["median_R_grid"] > 2.0


def test_model_month_from_f3acc_roundtrip():
    acc = f3.F3Acc()
    acc._a(f3.AIJ_COLS["prec"])[...] += 1.0
    acc.idacc[1] = 4
    mm = S.ModelMonth.from_f3acc(acc)
    assert mm.available([315, 316]) == [315]
    meta = S.column_meta() if os.path.exists(S.STORED) else None
    if meta:
        f = S.monthly_fields(mm.aij, mm.idacc, meta, [315])
        assert np.allclose(f[0], meta[315]["scale"] / 4.0)
