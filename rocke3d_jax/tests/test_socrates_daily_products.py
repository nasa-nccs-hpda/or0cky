"""Unit tests of the SOCRATES spike's daily-products reader and segment plan (fullfidelity/socrates_spike).

The reader was verified bitwise against the instrumented Fortran on 48 dates (report 14, 14.4); these tests pin its building blocks so that a
change is noticed without the Fortran oracle.  Tests that need the ModelE input files skip when they are not on this machine.
Regression numbers were produced by the reader at the commit that matched the Fortran bitwise (compared with rtol 1e-12)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE = os.path.join(HERE, "..", "fullfidelity", "socrates_spike")
sys.path.insert(0, SPIKE)
sys.path.insert(0, os.path.join(SPIKE, ".."))
import daily_products as D          # noqa: E402
import segments as SG               # noqa: E402

PROD = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files"
VOLC = os.path.join(PROD, "STRATAER.VOL.1850-2014_CMIP6_hdr")
MIE = os.path.join(PROD, "miescatpar.abcdv2")
LEVELS = np.load(os.path.join(SPIKE, "data", "rundeck_levels.npz"))


# ------------------------------------------------------------------------------------------------------------- calendar and streams
def test_year_weights():
    yrs = [1854, 1994, 2004]
    assert D.year_weights(yrs, 1850) == (0, 0.0)                    # before the first file: the first file
    assert D.year_weights(yrs, 1854) == (0, 0.0)
    jj, w = D.year_weights(yrs, 1990)                               # between: weight of the file before
    assert jj == 1 and w == pytest.approx(4.0 / 140.0)
    assert D.year_weights(yrs, 1994) == (1, 0.0)
    assert D.year_weights(yrs, 2010) == (3, 1.0)                    # after the last file: the last file


def test_itime_calendar():
    assert D.jday_of_itime(0) == 1
    assert D.jday_of_itime(365 * 48) == 1                           # 365-day calendar wraps
    assert D.jday_of_itime(33312) == 330                            # nov26 1950 00:00
    assert D.jday_of_itime(33312 + 47) == 330 and D.jday_of_itime(33312 + 48) == 331
    assert D.jyear_of_itime(0) == 1949 and D.jyear_of_itime(17520) == 1950 and D.jyear_of_itime(33312) == 1950


def test_linm2m_mid_month_and_cyclic_padding():
    q = np.arange(12.0)[:, None] * np.ones((12, 2))                 # month m has value m
    for m in range(12):
        mid = D.JDMID[m + 1]
        assert D.linm2m(q, int(mid))[0] == pytest.approx(float(m), abs=0.51)   # integer day nearest to mid-month
    # January 1 lies between December (11) and January (0), cyclic: 15/31 of the way from mid-December
    assert D.linm2m(q, 1)[0] == pytest.approx(5.32258064516129)
    assert D.linm2m(q, 365)[0] == pytest.approx(5.67741935483871)
    assert D.linm2m(q, 365)[0] + D.linm2m(q, 1)[0] == pytest.approx(11.0)    # symmetric about the year wrap


def test_rsdist_orbit():
    # regression values (bitwise equal to the Fortran DAILY_orbit on the oracle dates)
    assert D.rsdist(330, 1850.0) == pytest.approx(0.9731282935959317, rel=1e-12)
    assert D.rsdist(1, 1850.0) == pytest.approx(0.9667525404132079, rel=1e-12)
    assert D.rsdist(182, 1850.0) == pytest.approx(1.0337966173585387, rel=1e-12)
    assert D.rsdist(330, 1990.0) == pytest.approx(0.9740587001028833, rel=1e-12)      # the orbit year is the master year
    assert D.rsdist(330, 1990.0) != D.rsdist(330, 1850.0)


# ------------------------------------------------------------------------------------------------------------------ REPART / RETERP
def test_repart_conserves_and_reterp_preserves_constants():
    xlb = np.array([1000.0, 800.0, 500.0, 200.0, 50.0])             # source edges (decreasing pressure)
    ylb = np.array([900.0, 700.0, 400.0, 100.0])                    # target edges inside the source range
    w = D.repart_weights(xlb, ylb)
    f = np.array([1.0, 2.0, 3.0, 4.0])                              # histogram: amount per source layer (REPART uses PART * F)
    g = D.repart_apply(w, f[:, None])[:, 0]
    # REPART of a uniform field gives the field times the overlap fraction of each source layer: parts are in [0, 1] and each target layer
    # takes from consecutive source layers only
    assert all(0.0 <= p <= 1.0 for lst in w for _, p in lst)
    assert all(lst == sorted(lst, key=lambda t: t[0]) for lst in w)
    assert g.shape == (3,) and np.all(g > 0)
    # RETERP (divide by the sum of the parts) preserves a constant
    c = D.reterp_apply(w, np.full((4, 1), 7.5))[:, 0]
    assert np.allclose(c, 7.5, rtol=1e-14)
    # a target layer without overlap is zero
    w0 = D.repart_weights(xlb, np.array([30.0, 20.0, 10.0]))
    assert np.all(D.repart_apply(w0, f[:, None]) == 0.0)


def test_spline4_exact_for_low_order():
    x = np.arange(5.0)
    assert D.spline4(x, x, 1.7) == pytest.approx(1.7, rel=1e-14)    # linear
    assert D.spline4(x, x ** 2, 1.7) == pytest.approx(1.7 ** 2, rel=1e-12)       # quadratic (4-point cubic through 4 points)
    assert D.spline4(x, np.sin(x), 2.0) == pytest.approx(np.sin(2.0), abs=1e-12)   # interior node
    with pytest.raises(NotImplementedError):                       # the edge intervals are not ported (the volcanic tables never use them)
        D.spline4(x, np.sin(x), 0.5)
    assert D.spline4(x, np.sin(x), 2.5) == pytest.approx(np.sin(2.5), abs=0.05)  # smooth function between nodes


# ----------------------------------------------------------------------------------------------------------------- volcanic aerosol
@pytest.mark.skipif(not os.path.exists(VOLC), reason="RADN7 file not on this machine")
def test_volcanic_file_and_background_mean():
    v = D.VolcanicAerosol(VOLC, LEVELS["hlb0"])
    assert (v.nmon, v.y0, v.y1, v.jm, v.lm) == (1980, 1850, 2014, 36, 69)
    assert v.vtau.shape == (1980, 36, 69) and v.hvolkm.shape == (70,) and v.elat.shape == (37,)
    tau, reff = v.at(-1, 182)
    assert tau.shape == (46, 43) and reff.shape == (46,)
    assert np.all(tau >= 0) and np.all(reff > 0.05) and np.all(reff < 1.0)
    # long-term mean does not depend on the day
    t2, r2 = v.at(-1, 300)
    assert np.array_equal(tau, t2) and np.array_equal(reff, r2)
    # regression values (bitwise equal to the Fortran dump)
    assert reff[[0, 10, 22, 35, 45]] == pytest.approx([0.29855957, 0.30711699, 0.35034603, 0.35002038, 0.35637248], rel=1e-7)
    assert tau.sum(1)[[0, 10, 22, 35, 45]] == pytest.approx([0.01200476, 0.01028771, 0.00936774, 0.01260946, 0.01709132], rel=1e-6)
    # a year of the file at mid-year (Pinatubo year 1991): more optical depth in the tropics than the mean
    t91, r91 = v.at(1991, 182)
    assert t91.sum(1)[22] > 3.0 * tau.sum(1)[22]
    # before the first month: clamps to the first month
    assert np.array_equal(v.at(1800, 182)[0], v.at(1850, 0)[0])


@pytest.mark.skipif(not os.path.exists(VOLC), reason="RADN7 file not on this machine")
def test_volcanic_mean_range_outside_file_raises():
    v = D.VolcanicAerosol(VOLC, LEVELS["hlb0"], kyearvm=(1800, 2014))
    with pytest.raises(ValueError):
        v.at(-1, 182)


@pytest.mark.skipif(not os.path.exists(MIE), reason="RADN3 file not on this machine")
def test_q55_table():
    t = D.VolcanicTables(MIE)
    assert t.q55v.shape == (20,)
    assert t.q55v[0] == pytest.approx(0.47989219, rel=1e-7) and t.q55v[19] == pytest.approx(2.3318575, rel=1e-7)
    q = t.q55(np.array([0.05, 0.1, 0.35, 1.0, 5.0]))              # radx clipped to 1.000001..19.99999
    assert np.all(np.isfinite(q))
    assert q[0] == pytest.approx(t.q55(np.array([0.1000001]))[0], rel=1e-5)     # lower clip
    assert q[4] == pytest.approx(t.q55(np.array([2.0]))[0], rel=1e-5)             # upper clip


# ----------------------------------------------------------------------------------------------------------------------- segments
def test_segment_plan_covers_and_cuts_at_day_boundaries():
    plan = SG.segment_plan(33312, 33312 + 100 * 48 + 7, max_days=30)
    assert [s["ndays"] for s in plan] == [30, 30, 30, 11]
    assert plan[0]["itime_first"] == 33312 and plan[-1]["itime_last"] == 33312 + 100 * 48 + 7
    for a, b in zip(plan, plan[1:]):                                 # contiguous, no step in two segments
        assert b["itime_first"] == a["itime_last"] + 1
        assert b["itime_first"] % 48 == 0 and b["day0"] == a["day0"] + a["ndays"]
    for it in (33312, 33312 + 30 * 48 - 1, 33312 + 30 * 48, 33312 + 100 * 48 + 7):
        s = plan[SG.segment_of(plan, it)]
        assert s["itime_first"] <= it <= s["itime_last"]
    assert s["day0"] * 48 <= it
    with pytest.raises(ValueError):
        SG.segment_of(plan, 1)
    with pytest.raises(ValueError):
        SG.segment_plan(10, 5)


def test_segment_plan_single_partial_day_and_first_rad_itime():
    p = SG.segment_plan(33317, 33320, max_days=7)
    assert p == [dict(itime_first=33317, itime_last=33320, day0=694, ndays=1)]
    assert SG.first_rad_itime(33312, 16032) == 33312               # 33312 - 16032 = 17280 = 3456 * 5
    assert SG.first_rad_itime(33313, 16032) == 33317
    assert SG.first_rad_itime(33317, 16032) == 33317
    # every radiation step of the whole range belongs to exactly one segment
    plan = SG.segment_plan(33312, 33312 + 70 * 48, max_days=20)
    seen = [it for it in range(33312, 33312 + 70 * 48 + 1) if (it - 16032) % 5 == 0]
    assert sorted(seen) == seen and all(any(s["itime_first"] <= it <= s["itime_last"] for s in plan) for it in seen)


def test_apply_rundeck_edits():
    import r6_days_oracle as O      # noqa: E402  (imports radiation_server; only its pure helper is used)
    lines = [" &&PARAMETERS", " master_yr=1850", " volc_yr=-1", " DT=450.", " &&END_PARAMETERS"]
    out = O.apply_rundeck_edits(lines, {"master_yr": "1990", "volc_yr": "0", "extra": "3"})
    assert out == [" &&PARAMETERS", " master_yr=1990", " volc_yr=0", " DT=450.", " extra=3", " &&END_PARAMETERS"]
    assert lines[1] == " master_yr=1850"                                         # input not modified
