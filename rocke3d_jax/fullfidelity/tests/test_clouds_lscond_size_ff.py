"""Tests for clouds_lscond_size_ff.py (LSCOND particle size / optical thickness tail block) -- D107.

Real-Fortran validation uses ff_data/<date>/ffc_ls_tail_*.bin (skipped if absent).  Synthetic tests are labelled
SYNTHETIC: they exercise branches that never occur in the real 6-step windows and are NOT validated against the Fortran."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_lscond_io as io  # noqa: E402
import clouds_lscond_size_compare as cmp  # noqa: E402
import clouds_lscond_size_ff as sz  # noqa: E402

DATES = [d for d, _ in io.DATES]
HAVE = {d: os.path.exists(f"{io.FF_DEFAULT}/{d}/ffc_ls_consts.txt") for d in DATES}
needs_dumps = pytest.mark.skipif(not all(HAVE.values()), reason="ff_data ffc_ls_* dumps not present on this host")
needs_imf = pytest.mark.skipif(not sz.imf_available(), reason="Intel libimf not available")
_cache = {}


@pytest.fixture(autouse=True)
def _libm_mode():
    sz.use_imf(False)
    yield
    sz.use_imf(False)


def _data(date):
    if date not in _cache:
        _cache[date] = io.load_tail(date)
    return _cache[date]


def _run(date, imf=False, **kw):
    key = (date, imf, tuple(sorted((k, id(v)) for k, v in kw.items())))
    sz.use_imf(imf)
    d = _data(date)
    return d, cmp.run_port(d, **kw)


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_constants_match_real(date):
    assert all(cmp.const_check(date).values())


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_outputs_match_real_libm(date):
    d, res = _run(date)
    st = cmp.compare(d, res)
    for k in ("svlhxl", "qclx", "qcix", "cldsv1", "qlss", "qiss", "wmpr", "wmsum"):
        assert st[k]["bitwise"] == st[k]["n"], k                 # inputs passed through / sums: exact
    for k, r in st.items():
        assert r["max_rel"] < 1e-12, (k, r)                      # pow/exp last-bit differences only
        assert r["bitwise"] / r["n"] > 0.999, (k, r)


@needs_dumps
@needs_imf
@pytest.mark.parametrize("date", DATES)
def test_outputs_bitwise_with_libimf(date):
    d, res = _run(date, imf=True)
    st = cmp.compare(d, res)
    assert all(r["bitwise"] == r["n"] for r in st.values()), {k: r for k, r in st.items() if r["bitwise"] != r["n"]}
    assert not cmp.mismatch_records(d, res).any()


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_no_branch_flips(date):
    d, res = _run(date)
    assert all(v == 0 for v in cmp.branch_flips(d, res).values())


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_non_vacuous(date):
    """Every live branch of the block is exercised by real data (counts are layer visits over all records)."""
    d, res = _run(date)
    c = res[2]
    for k, n in dict(liq=10000, ice=10000, vmp_ip_liq=1000, tau_lhp_eq=1000, tau_ip=10000, tau_ip_cap=100,
                     tau_cap=20, csize_cap=50, rcld_rwmax=10, fcld_teeny=10000, ip_fcld_zero=10000, wtem_floor=10000,
                     rescale_bl=1000, rescale_free=10000, bmax_095=500, skip_taumcl=1000, svlhx_reset=10000).items():
        assert c[k] >= n, (k, c[k])
    # branches that never occur in the six-step windows (covered by the SYNTHETIC tests below only)
    assert c["neg_tau"] == 0 and c["lhp_lhe_stop"] == 0
    assert (d["out_taussl"] > 0).sum() > 1000 and (d["out_tausslip"] > 0).sum() > 1000
    assert (d["out_wmsum"] > 0).sum() > 1000


# ------------------------------------------------------------------------------------------------ mutation checks
def _bad_records(date, mutate=None, imf=False, nmax=1500):
    sz.use_imf(imf)
    d = _data(date)
    res = cmp.run_port(d, mutate=mutate, nmax=nmax)
    n = res[1].size
    d2 = {k: (v[:n] if hasattr(v, "__len__") else v) for k, v in d.items()}
    return int(cmp.mismatch_records(d2, res).sum())


@needs_dumps
@needs_imf
def test_unmutated_baseline_is_exact_in_imf_mode():
    assert _bad_records("nov26", imf=True) == 0


@needs_dumps
@needs_imf
@pytest.mark.parametrize("name,mut", [
    ("rcldlx", lambda a, p: p.__setitem__("rcldlx", p["rcldlx"] * (1 + 1e-9))),
    ("rcldix", lambda a, p: p.__setitem__("rcldix", p["rcldix"] * (1 + 1e-9))),
    ("bybr", lambda a, p: p.__setitem__("bybr", p["bybr"] * (1 + 1e-9))),
    ("rimax", lambda a, p: p.__setitem__("rimax", p["rimax"] * 0.5)),
    ("rwmax", lambda a, p: p.__setitem__("rwmax", p["rwmax"] * 0.5)),
    ("ckij", lambda a, p: p.__setitem__("ckij", 0.5)),
    ("dcl", lambda a, p: p.__setitem__("dcl", p["dcl"] + 1)),
    ("lmcld", lambda a, p: p.__setitem__("lmcld", p["lmcld"] - 1)),
    ("pearth", lambda a, p: p.__setitem__("pearth", p["pearth"] + 0.5)),
    ("lhp", lambda a, p: a.__setitem__("lhp", [sz.LHS] * len(a["lhp"]))),
])
def test_mutations_are_detected(name, mut):
    assert _bad_records("nov26", mut, imf=True) > 20, name


@needs_dumps
@needs_imf
def test_mutated_algorithm_is_detected(monkeypatch):
    monkeypatch.setattr(sz, "BYGRAV", sz.BYGRAV * (1 + 1e-12))
    assert _bad_records("nov26", imf=True) > 20
    monkeypatch.undo()
    # wrong exponent (BY3 -> 0.34) in the radius law
    monkeypatch.setattr(sz, "BY3", 0.34)
    assert _bad_records("nov26", imf=True) > 20


@needs_dumps
def test_non_vmp_arm_differs_from_real():
    """use_vmp=False (the dead arm: no TAUSSLIP, precip optical thickness always added) must not reproduce the real VMP run."""
    d = _data("nov26")
    res = cmp.run_port(d, use_vmp=False, nmax=1000)
    n = res[1].size
    assert (res[0]["tausslip"] != d["out_tausslip"][:n]).any(axis=1).sum() > 50


# ------------------------------------------------------------------------------------------------ synthetic tests
def _col(**over):
    LM = 40
    a = {k: [0.0] * LM for k in io.TAIL_IN_LM}
    a["pl"] = [800.0] * LM
    a["tl"] = [270.0] * LM
    a["airm"] = [50.0] * LM
    a["fssl"] = [1.0] * LM
    a["cleara"] = [0.5] * LM
    a["cldssl"] = [0.5] * LM
    a["svlhxl"] = [sz.LHE] * LM
    a["qclx"] = [1e-4] * LM
    a["lhp"] = [sz.LHE] * (LM + 1)
    a.update(over)
    par = dict(pearth=0.0, lmcld=29, dcl=5, ckij=1.0, bybr=0.7829735282337728, rimax=100.0, rwmax=20.0, rwcldox=1.0,
               rcldlx=1.01, rcldix=1.0)
    return a, par


def test_synthetic_hand_derived_liquid_layer():
    """SYNTHETIC hand-derived check of one liquid layer (L=10 > DCL, TAUMCL=0 -> free-troposphere rescaling)."""
    a, par = _col()
    s, wmsum, cnt = sz.size_tail(a, par)
    i = 9
    fcld = 0.5 + 1e-30
    wtem = 1e5 * 1e-4 * 800.0 / (fcld * 270.0 * sz.RGAS + 1e-30)
    rcld = 1.01 * 100.0 * (wtem / (2.0 * sz.BY3 * sz.TWOPI * 59.68)) ** sz.BY3
    rclde = rcld / par["bybr"]
    tem = 50.0 * 1e-4 * 1e2 * sz.BYGRAV
    tau = 1.5e3 * tem / (fcld * rclde + 1e-30)
    assert s["csizel"][i] == pytest.approx(rclde, rel=1e-12)
    assert wmsum == pytest.approx(29 * tem, rel=1e-12)
    # rescaling for L>DCL: CLDSSL=min(CLDSSL**(2/3),FSSL), TAUSSL*=CLDSV1**(1/3)
    assert s["cldssl"][i] == pytest.approx(0.5 ** (2.0 / 3.0), rel=1e-12)
    assert s["taussl"][i] == pytest.approx(tau * 0.5 ** (1.0 / 3.0), rel=1e-12)
    assert s["cldsv1"][i] == 0.5
    assert s["qlss"][i] == 1e-4 and s["qiss"][i] == 0.0


def test_inputs_not_modified_and_layers_above_lmcld_pass_through():
    a, par = _col()
    a["taussl"] = [7.0] * 40
    a["csizel"] = [3.0] * 40
    before = {k: list(v) for k, v in a.items()}
    s, _, _ = sz.size_tail(a, par)
    assert a == before
    for i in range(29, 40):                       # L > LMCLD untouched
        assert s["taussl"][i] == 7.0 and s["csizel"][i] == 3.0 and s["cldssl"][i] == 0.5


def test_synthetic_negative_tau_branch():
    """SYNTHETIC (branch never reached in the real dumps): a negative TAUSSL zeroes CLDSSL/TAUSSL and the cloud water."""
    a, par = _col()
    a["qclx"] = [-1e-4] * 40
    s, wmsum, cnt = sz.size_tail(a, par)
    assert cnt["neg_tau"] == 29
    assert all(s["taussl"][i] == 0.0 and s["cldssl"][i] == 0.0 and s["qclx"][i] == 0.0 for i in range(29))
    assert all(s["qlss"][i] == 0.0 for i in range(29))     # not accumulated in the negative branch


def test_synthetic_vmp_liquid_layer_with_ice_precip_and_stop():
    """SYNTHETIC: ice precip under a liquid cloud (LHP=LHS, LHX=LHE) fills CSIZELIP/TAUSSLIP; liquid precip under an ice cloud
    (LHP=LHE, LHX=LHS) hits the Fortran stop_model('VMP: should not be here')."""
    a, par = _col(wmpr=[1e-5] * 40, lhp=[sz.LHS] * 41)
    s, _, cnt = sz.size_tail(a, par)
    assert cnt["vmp_ip_liq"] == 29 and cnt["tau_ip"] == 29
    assert all(s["csizelip"][i] > 0 and s["tausslip"][i] > 0 for i in range(29))
    a, par = _col(svlhxl=[sz.LHS] * 40, qcix=[1e-4] * 40, qclx=[0.0] * 40, wmpr=[1e-5] * 40, lhp=[sz.LHE] * 41)
    with pytest.raises(RuntimeError):
        sz.size_tail(a, par)


def test_synthetic_non_vmp_arm_adds_precip_optical_thickness():
    a, par = _col(wmpr=[1e-5] * 40, lhp=[sz.LHS] * 41)
    s1, _, _ = sz.size_tail(a, par, use_vmp=True)
    s0, _, _ = sz.size_tail(a, par, use_vmp=False)
    assert s0["taussl"][0] > s1["taussl"][0] and s0["tausslip"][0] == 0.0 and s1["tausslip"][0] > 0.0
