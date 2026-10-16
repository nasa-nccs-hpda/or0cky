"""Tests for clouds_lscond_ff.py (LSCOND main layer loop D108, CTEI + whole call D109; the tail block is D107).

Real-Fortran validation uses ff_data/<date>/ffc_ls_{bnd,mid,tail}_*.bin (skipped if absent).  To keep the suite fast the
tests use the first NCOL (=1000) column calls of each date; clouds_lscond_compare.py runs all 1902 per date.
Tests marked SYNTHETIC exercise branches that never occur in the six-step windows; they are NOT validated against Fortran."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_dq_ff as dq  # noqa: E402
import clouds_lscond_compare as cc  # noqa: E402
import clouds_lscond_ff as ls  # noqa: E402
import clouds_lscond_io as io  # noqa: E402
import clouds_lscond_size_ff as sz  # noqa: E402

DATES = [d for d, _ in io.DATES]
HAVE = all(os.path.exists(f"{io.FF_DEFAULT}/{d}/ffc_ls_consts.txt") for d in DATES)
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffc_ls_* dumps not present on this host")
needs_imf = pytest.mark.skipif(not sz.imf_available(), reason="Intel libimf not available")
NCOL = 1000
_data = {}
_runs = {}


@pytest.fixture(autouse=True)
def _libm_mode():
    sz.use_imf(False)
    yield
    sz.use_imf(False)


def data(date):
    if date not in _data:
        _data[date] = (io.load_bnd(date), io.load_mid(date), io.load_tail(date))
    return _data[date]


def run(kind, date, imf=False):
    key = (kind, date, imf)
    if key not in _runs:
        sz.use_imf(imf)
        b, mid, tail = data(date)
        n = min(NCOL, b["i"].size)
        if kind == "main":
            _runs[key] = cc.run_main(b, mid, n)
        elif kind == "full":
            _runs[key] = cc.run_full(b, n)
        elif kind == "ctei":
            _runs[key] = cc.run_ctei_from_mid(b, mid, n, tail=False)
        elif kind == "chain":
            _runs[key] = cc.run_ctei_from_mid(b, mid, n, tail=True)
    return _runs[key]


def _ok(st, tol=1e-6):
    return {k: r for k, r in st.items() if not (r["max_rel"] <= tol or (k in cc.SQRT_AMP and r["max_abs"] <= 5e-7))}


# ------------------------------------------------------------------------------------------------ layout
@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_dump_layout_and_scalars(date):
    b, mid, tail = data(date)
    assert b["i"].size == mid["i"].size == 1902 and tail["i"].size == 4754
    for k in ("itime", "i", "j"):
        assert np.array_equal(b[k], mid[k])
    assert set(np.unique(b["kmax"])) == {4.0, 72.0} and np.all(b["lmcld"] == 29) and np.all(b["use_vmp"] == 1)
    assert np.all(b["do_blu00"] == 0) and np.all(b["u00a"] == 0.695) and np.all(b["rcldlx"] == 1.01) and np.all(b["dtsrc"] == 1800)
    assert np.array_equal(b["kmax"] == 72, (b["j"] == 1) | (b["j"] == 46))   # pole rows have KMAX=IM
    assert np.all(b["ierr"] == 0) and np.all(b["rtemp"] == 22.0) and np.all(b["cmx"] == 1.0)
    assert len(np.unique(b["itime"])) == 6


# ------------------------------------------------------------------------------------------------ D108: main loop
@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_main_loop_matches_checkpoint_libm(date):
    b, mid, _ = data(date)
    Ss, Ws, _c = run("main", date)
    st = cc.compare_main(b, mid, Ss, Ws)
    assert not _ok(st), _ok(st)
    nb, ne = cc.cleara_sqrt_explained(mid, Ss, Ws)
    assert nb == ne
    tb = sum(r["bitwise"] for r in st.values())
    tn = sum(r["n"] for r in st.values())
    assert tb / tn > 0.999


@needs_dumps
@needs_imf
@pytest.mark.parametrize("date", DATES)
def test_main_loop_bitwise_with_libimf(date):
    b, mid, _ = data(date)
    Ss, Ws, _c = run("main", date, imf=True)
    st = cc.compare_main(b, mid, Ss, Ws)
    assert all(r["bitwise"] == r["n"] for r in st.values()), {k: r for k, r in st.items() if r["bitwise"] != r["n"]}


# ------------------------------------------------------------------------------------------------ D109: CTEI + whole call
@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_whole_call_matches_exit_libm(date):
    b, _, _ = data(date)
    Ss, Ws, _c = run("full", date)
    st = cc.compare_exit(b, Ss, Ws)
    assert not _ok(st), _ok(st)
    assert all(v == 0 for v in cc.flips(b, Ss, Ws).values())          # no branch flips from last-bit differences


@needs_dumps
@needs_imf
@pytest.mark.parametrize("date", DATES)
def test_whole_call_bitwise_with_libimf(date):
    b, _, _ = data(date)
    Ss, Ws, _c = run("full", date, imf=True)
    st = cc.compare_exit(b, Ss, Ws)
    assert all(r["bitwise"] == r["n"] for r in st.values()), {k: r for k, r in st.items() if r["bitwise"] != r["n"]}
    assert all(v == 0 for v in cc.flips(b, Ss, Ws).values())


@needs_dumps
@needs_imf
@pytest.mark.parametrize("date", DATES)
def test_ctei_matches_tail_entry_and_chain_matches_exit(date):
    b, mid, tail = data(date)
    Ss, Ws, _c = run("ctei", date, imf=True)
    pairs = [p for p in cc.align_mid_tail(mid, tail) if p[0] < NCOL]
    assert len(pairs) > 200
    res = cc.compare_ctei_vs_tail(b, mid, tail, pairs, Ss, Ws)
    res = {k: v for k, v in res.items() if k in ("cldssl", "qclx", "qcix", "svlhxl", "tl", "cleara", "ckij", "hcndss")}
    assert all(r["bitwise"] == r["n"] for r in res.values()), res
    Ss, Ws, _c = run("chain", date, imf=True)
    st = cc.compare_exit(b, Ss, Ws)
    assert all(r["bitwise"] == r["n"] for r in st.values())


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_ctei_libm_matches_tail_entry(date):
    b, mid, tail = data(date)
    Ss, Ws, _c = run("ctei", date)
    pairs = [p for p in cc.align_mid_tail(mid, tail) if p[0] < NCOL]
    res = cc.compare_ctei_vs_tail(b, mid, tail, pairs, Ss, Ws)
    res = {k: v for k, v in res.items() if k in ("cldssl", "qclx", "qcix", "svlhxl", "tl", "cleara", "ckij", "hcndss")}
    assert not _ok(res, 1e-9), _ok(res, 1e-9)


# ------------------------------------------------------------------------------------------------ non-vacuity
@needs_dumps
def test_non_vacuous_branch_coverage():
    tot = {}
    for date in DATES:
        for k, v in run("full", date)[2].items():
            tot[k] = tot.get(k, 0) + v
    must = dict(lhx_ice_forced=1000, lhx_water=1000, lhp_ice_tl_lt_tf=1000, phase_le_to_ls=1, phase_ls_to_le=1,
                oldlat_ls_le=100, qcx_pos=1000, wtliq_1=100, wtliq_0=100, wtliq_interp=100, wtliq_det_ice=10,
                vdef_cm0=100, tem_cap10=100, cm_cap=10, form_clouds=1000, no_form=1000, er_qcl=100, er_ice_precip=5,
                er_rh=10, cleara_pos=100, wtem_wconst=10, drhdt_zero=5, qcl_neg=10, fqtow_pos=100, ls_branch_clear=1000,
                rhw_branch=1000, rh1_lt238=1000, cond_rh1=50, cond_dqsum_pos=50, rainout_liq=5, rainout_ice=1,
                rainout_melt=5, hphase_melt_icep=10, hphase_freeze_liq=1, hphase_lhp_ne_lhx=100, lhp_zero=1000,
                preice_set=100, no_form_evap_liq=10, no_form_evap_ice=1, ctei_visits=1000, ctei_cycle_qcl=100,
                ctei_cycle_clear=1000, ctei_cycle_ck_lt_ckr=10, ctei_cycle_dse=1, ctei_mixed=10, ctei_iter_exit=10,
                ctei_iter_exhaust=1, ctei_fsslrat_ne1=1, ctei_ckij_set=1, ctei_lhx_ice=10)
    for k, n in must.items():
        assert tot[k] >= n, (k, tot[k])
    # never exercised by the real 6-step windows (all dates, all 5706 sampled columns) -- see the SYNTHETIC tests / docs
    never = ("oldlat_le_ls", "er_clip_max", "er_clip_zero", "qnew_neg", "ierr", "no_form_evap_zero_cond",
             "ctei_cycle_ckr_gt_ckm", "ctei_cycle_fpmax", "ctei_fmass_clip")
    assert all(tot[k] == 0 for k in never), {k: tot[k] for k in never}


# ------------------------------------------------------------------------------------------------ mutation checks
def _bad_main(mutate_fn, n=600):
    b, mid, _ = data("nov26")
    sz.use_imf(True)
    Ss, Ws, _c = [], [], ls.new_counters()
    for r in range(n):
        S, P = cc.column_state(b, r)
        Ws.append(ls.lscond_main(S, P, _c))
        Ss.append(S)
    st = cc.compare_main(b, mid, Ss, Ws)
    return sum(r["n"] - r["bitwise"] for r in st.values())


def _bad_full(n=600):
    b, _, _ = data("nov26")
    sz.use_imf(True)
    Ss, Ws, _c = [], [], ls.new_counters()
    for r in range(n):
        S, P = cc.column_state(b, r)
        W = ls.lscond_main(S, P, _c)
        ls.lscond_ctei(S, W, P, _c)
        ls.lscond_tail(S, W, P, _c)
        Ss.append(S)
        Ws.append(W)
    st = cc.compare_exit(b, Ss, Ws)
    return sum(r["n"] - r["bitwise"] for r in st.values())


@needs_dumps
@needs_imf
def test_baseline_exact_in_imf_mode_for_mutation_tests():
    assert _bad_main(None) == 0 and _bad_full() == 0


@needs_dumps
@needs_imf
@pytest.mark.parametrize("name,val", [
    ("_F02", 0.2),                                    # the REAL(4) literal -0.2 in cm0*10.**(-0.2*vdef)
    ("_F20783", 207.83),                              # REAL(4) 207.83 in the Karcher-Lohmann RH1 formula
    ("GBYAIRM0", ls.GBYAIRM0 * (1 + 1e-15)),
    ("COESIG", 2e-3),
    ("SLHE", ls.SLHE * (1 + 1e-15)),
    ("DELTX", ls.DELTX * (1 + 1e-15)),
    ("TMAX_ICE", ls.TF - 5.0001),
    ("TMIN_WATER", ls.TF - 35.0001),
    ("CM00LIQ", 1.1e-4),
    ("CM00ICE", 3.1e-4),
])
def test_constant_mutations_are_detected(monkeypatch, name, val):
    monkeypatch.setattr(ls, name, val)
    assert _bad_full() > 100, name


@needs_dumps
@needs_imf
def test_coeec_mutation_detected_in_main_loop_locals(monkeypatch):
    monkeypatch.setattr(ls, "COEEC", 1001.0)                   # EC is a local: visible at the checkpoint, not at the exit
    assert _bad_main(None) > 10


@needs_dumps
@needs_imf
def test_pow5_as_multiplication_is_detected(monkeypatch):
    """The integer power of the CTEI SIGK formula is a libm pow call in the real build, not a multiplication chain."""
    monkeypatch.setattr(ls, "_pow5", lambda x: (x * x * x * x) * x)
    assert _bad_full(1902) > 20


@needs_dumps
@needs_imf
def test_single_precision_literals_not_observable_by_data(monkeypatch):
    """238.16 and .999999 (REAL(4) literals) cannot be distinguished from their double values on this data (no real
    temperature/RH lies in the 1e-6-wide gap): replicated from the source reading only."""
    monkeypatch.setattr(ls, "_F23816", 238.16)
    monkeypatch.setattr(ls, "_F999999", 0.999999)
    assert _bad_main(None) == 0


@needs_dumps
@needs_imf
def test_ctei_iteration_cap_mutation_is_detected(monkeypatch):
    import inspect
    src = inspect.getsource(ls.lscond_ctei).replace("range(1, 10)", "range(1, 4)")
    ns = dict(ls.__dict__)
    exec(compile("def lscond_ctei_m" + src[src.index("("):], "mut", "exec"), ns)
    monkeypatch.setattr(ls, "lscond_ctei", ns["lscond_ctei_m"])
    assert _bad_full(1902) > 20


# ------------------------------------------------------------------------------------------------ unit / synthetic tests
def test_f4_literals():
    assert ls._F02 == float(np.float32(0.2)) != 0.2
    assert ls._F20783 == float(np.float32(207.83)) and ls._F23816 == float(np.float32(238.16))
    assert ls._F999999 == float(np.float32(0.999999))


def test_clear_from_rh_hand_derived():
    assert ls._clear_from_rh(0.9, 0.2, 0.3) == pytest.approx(((1 - 0.9) / ((1 - 0.2) + 1e-30)) ** 0.5, rel=1e-15)
    assert ls._clear_from_rh(0.5, 1.0, 0.3) == 1.0                  # RH00==1 -> clear
    assert ls._clear_from_rh(1.5, 0.7, 0.3) == 0.0                  # RH>1 -> overcast
    assert ls._clear_from_rh(0.0, 0.01, 0.3) == 1.0                 # sqrt > 1 clamps to 1


def test_ctmix_conserves_mean_and_scales_vertical_moments():
    rm = [2.0, 5.0]
    rmom = [[float(k + 1) for k in range(9)], [float(10 + k) for k in range(9)]]
    tot = rm[0] + rm[1]
    ls.ctmix(rm, rmom, 0.25, 0.1, 0.2)
    assert rm[0] + rm[1] == pytest.approx(tot, rel=1e-15)
    assert rm[0] == pytest.approx(2.0 * 0.9 + 0.2 * 5.0, rel=1e-15)
    for m in (0, 1, 3, 4, 6):                                       # X, Y, XX, YY, XY mixed like the mean
        assert rmom[0][m] == pytest.approx((m + 1) * 0.9 + 0.2 * (10 + m), rel=1e-15)
    for m in (2, 5, 7, 8):                                          # moments with a vertical component scaled by (1-FMAIR)
        assert rmom[0][m] == pytest.approx((m + 1) * 0.75, rel=1e-15) and rmom[1][m] == pytest.approx((10 + m) * 0.75, rel=1e-15)


def test_scalar_dq_agrees_with_imported_helpers():
    """The scalar copy used in libimf mode equals the imported D89 helpers (both use the platform exp here, last bit aside)."""
    for args in [(280.0, 4e-3, 1.0, 1.0, 2.5e6, 900.0), (240.0, 1e-3, 0.9, 40.0, 2.834e6, 300.0)]:
        sm, qm, plk, mass, lhx, pl = args
        a, fa = ls.get_dq_cond(*args)
        sz.use_imf(False)
        d = ls._dq_adjust(sm, qm, plk, mass, lhx, pl, 1.0)
        d = max(0.0, min(d, qm))
        assert d == pytest.approx(a, rel=1e-12) and d / qm == pytest.approx(fa, rel=1e-12)
        e, fe = ls.get_dq_evap(sm, qm, plk, mass, lhx, pl, 1e-3)
        d = max(0.0, min(ls._dq_adjust(sm, qm, plk, mass, lhx, pl, -1.0), 1e-3))
        assert d == pytest.approx(e, rel=1e-12)
    assert ls.get_dq_cond(280.0, 0.0, 1.0, 1.0, 2.5e6, 900.0) == (0.0, 0.0)


def test_non_vmp_arm_not_ported():
    with pytest.raises(NotImplementedError):
        ls.lscond_main({"tl": [0.0] * 40}, {"use_vmp": False})


def _syn(row, mod, keys):
    b, _, _ = data("nov26")
    S, P = cc.column_state(b, row)
    mod(S, P)
    cnt = ls.new_counters()
    S, W, cnt = ls.lscond(S, P, cnt)
    assert all(cnt[k] > 0 for k in keys), {k: cnt[k] for k in keys}
    return S, W, cnt


@needs_dumps
def test_synthetic_negative_water_vapour_branch():
    """SYNTHETIC (QNEW<0 branch, never reached in real data): QL is clamped to 0, QHEAT reset, no NaN, QL stays >= 0."""
    S, W, cnt = _syn(0, lambda S, P: S.__setitem__("aq", [x * 1e4 + 1e-6 for x in S["aq"]]), ["qnew_neg"])
    assert min(S["ql"]) >= 0.0 and all(np.isfinite(S[k]).all() for k in ("tl", "ql", "qclx", "qcix", "cldssl"))
    assert W["ierr"] == 0                                           # IERR needs the second (QCLNEW<0) failure: never reached


@needs_dumps
def test_synthetic_convective_ice_to_water_transition():
    """SYNTHETIC (OLDLAT=LHE with LHX=LHS, never reached in real data): detrained convective water switches to ice."""
    def mod(S, P):
        S["svlatl"] = [2.5e6] * 40
        S["svwmxl"] = [1e-5] * 40
    S, W, cnt = _syn(0, mod, ["oldlat_le_ls"])
    assert all(np.isfinite(S[k]).all() for k in ("tl", "ql", "qclx", "qcix"))


@needs_dumps
def test_synthetic_evaporation_clip_to_ermax():
    """SYNTHETIC (ER>ERMAX clip, never reached in real data: needs layers thicker than 100/(1-RH)^2 mb)."""
    def mod(S, P):
        S["airm"] = [x * 30 for x in S["airm"]]
        S["byam"] = [1.0 / x for x in S["airm"]]
    S, W, cnt = _syn(0, mod, ["er_clip_max"])
    assert min(W["prebar"]) >= 0.0
