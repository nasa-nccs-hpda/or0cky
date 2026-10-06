"""Tests for the batched cloud physics: clouds_lscond_batch.py, clouds_mstcnv_batch.py, clouds_condse_batch.py -- D131-D132.

Real-Fortran-chain tests use the dumps ff_data/<date>/ffc_cse_*.bin (step 2 of nov26, itime 33313) and are skipped when absent.  The per-column
ports (validated against the real model in D107-D126) are the reference: the batch must reproduce them BIT FOR BIT in the same libm mode, on a
few full latitude rows (the rows with the most convecting columns and the most snow, so every branch family is exercised) run through the
per-column chain (about 20 s).  The whole-step results (3 dates, per-column vs batch vs the real exit state, libm and libimf) are in the ledger entry
and come from the three compare scripts.  Tests marked "synthetic" use made-up data for logic the real rows do not cover (labelled).
Mutation tests change one statement/constant of the batch code and require the comparison to get worse (the comparison is not vacuous)."""
import copy
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_condse_batch as cb  # noqa: E402
import clouds_condse_ff as cf  # noqa: E402
import clouds_condse_io as cio  # noqa: E402
import clouds_lscond_batch as lb  # noqa: E402
import clouds_lscond_batch_compare as lc  # noqa: E402
import clouds_lscond_ff as ls0  # noqa: E402
import clouds_mstcnv_batch as mb  # noqa: E402
import clouds_mstcnv_batch_compare as mcmp  # noqa: E402
import clouds_mstcnv_ff as mc  # noqa: E402
import dyn_adv1d_ff as adv  # noqa: E402

DATE, ITIME = "nov26", 33313
HAVE = cio.have_dumps(DATE) and os.path.exists(f"{cio.FF_DEFAULT}/{DATE}/ffc_cse_out_{ITIME}.bin")
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffc_cse_* dumps not present on this host")
LM, IM, JM = 40, 72, 46
_cache = {}


def _data():
    """Per-column reference on the chosen rows: trace (MSTCNV r/o, LSCOND S0/P0) and the per-column exit state X of those rows."""
    if "d" not in _cache:
        inp, ref = cio.load_step(DATE, ITIME)
        cfg = cf.make_cfg(DATE)
        G = cfg["geom"]
        conv_rows = (ref["LMC"][0] > 0).sum(axis=0)
        snow_rows = (ref["EPREC"] < 0).sum(axis=0)
        rows = sorted({int(np.argmax(conv_rows[2:-2])) + 2, int(np.argmax(snow_rows[2:-2])) + 2})
        order = [(i, j) for j in rows for i in range(int(G["IMAXJ"][j]))]
        trace = {k: None for k in order}
        cfg2 = dict(cfg)
        cfg2["trace"] = trace
        X, cnt = cf.condse_step(inp, cfg2, cols=order)
        _cache["d"] = (inp, ref, cfg, rows, order, trace, X)
    return _cache["d"]


def _ls_inputs():
    inp, ref, cfg, rows, order, trace, X = _data()
    return [trace[k]["S0"] for k in order], [trace[k]["P0"] for k in order]


def _mc_inputs():
    inp, ref, cfg, rows, order, trace, X = _data()
    return [trace[k]["r"] for k in order], [trace[k]["o"] for k in order]


def _ls_run(S_list, P_list):
    Sref, Wref, _ = lc.run_ref(S_list, P_list)
    carry = {k: S_list[0][k] for k in ("tausslip", "csizelip", "cldsal", "cldsv1")}
    S, P = lb.pack(S_list, P_list, carry)
    S, W = lb.lscond_batch(S, P)
    return lc.compare(Sref, Wref, S, W, len(S_list))


def _ndiff(res):
    return sum(v[0] for v in res.values())


# ------------------------------------------------------------------------------------------------ helpers (synthetic)
def test_pymax_pymin_match_python_including_zero_signs_synthetic():
    a = np.array([0.0, -0.0, 1.0, 2.0, -3.0, 0.0])
    b = np.array([-0.0, 0.0, 1.0, -2.0, -3.5, 0.0])
    for x, y, mx, mn in zip(a, b, lb.pymax(a, b), lb.pymin(a, b)):
        assert math.copysign(1, mx) == math.copysign(1, max(x, y)) and mx == max(x, y)
        assert math.copysign(1, mn) == math.copysign(1, min(x, y)) and mn == min(x, y)
    assert np.array_equal(mb.pymax(a, b), lb.pymax(a, b))


def test_exm_pwm_are_the_scalar_functions_and_skip_masked_columns_synthetic():
    x = np.array([0.3, 1e3, -2.0, 5.0])
    m = np.array([True, False, True, True])           # exp(1e3) would overflow in math.exp; it must not be evaluated
    r = lb.exm(x, m)
    assert r[0] == math.exp(0.3) and r[2] == math.exp(-2.0) and r[1] == 0.0
    p = lb.pwm(np.array([2.0, 3.0, 7.0]), 1.0 / 3.0, np.array([True, False, True]))
    assert p[0] == 2.0 ** (1.0 / 3.0) and p[1] == 0.0 and p[2] == 7.0 ** (1.0 / 3.0)


def test_ffill_carry_synthetic():
    vals = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    setm = np.array([[False, True, False, False, True]])
    out = lb._ffill(vals, setm, np.array([9.0]))
    assert out.tolist() == [[9.0, 2.0, 2.0, 2.0, 5.0]]


def test_adv_batch_equals_adv1d_per_line_including_qlimit_synthetic():
    rng = np.random.default_rng(3)
    g, nx = 6, 5
    s = rng.uniform(0.5, 2.0, (g, nx))
    smom = rng.normal(0, 0.2, (9, g, nx))
    mass = rng.uniform(50, 60, (g, nx))
    dm = rng.normal(0, 8, (g, nx))
    dm[:, -1] = 0.0
    for ql in (False, True):
        s1, m1, ma1, d1 = s.copy(), smom.copy(), mass.copy(), dm.copy()
        ie, ne = mb.adv_batch(s1, m1, ma1, d1, ql)
        for l in range(g):
            s2, m2, ma2 = s[l].copy(), smom[:, l].copy(), mass[l].copy()
            _, _, ie2, ne2 = adv.adv1d(s2, m2, ma2, dm[l].copy(), ql, adv.ZDIR)
            assert np.array_equal(s1[l], s2) and np.array_equal(m1[:, l], m2) and np.array_equal(ma1[l], ma2)
            assert ie[l] == ie2 and ne[l] == ne2


def test_conv_micro_equals_helper_on_random_inputs_synthetic():
    import clouds_helpers_ff as hp
    rng = np.random.default_rng(5)
    n = 40
    pl = rng.uniform(100, 1000, n)
    wcu = rng.uniform(0, 20, n)
    dwcu = rng.uniform(0, 1, n)
    lfrz = rng.integers(0, 5, n).astype(float)
    wfz = rng.uniform(0, 10, n)
    tp = rng.uniform(220, 300, n)
    pland = rng.uniform(0, 1, n)
    fl = [rng.uniform(1e3, 1e5, n) for _ in range(3)]
    tl0, tl1 = rng.uniform(250, 300, n), rng.uniform(250, 300, n)
    got = mb.conv_micro(pl, wcu, dwcu, lfrz, wfz, tp, pland, fl[0], fl[1], fl[2], tl0, tl1, np.zeros(n), np.zeros(n))
    full = lambda x: np.full(n, x)  # noqa: E731
    ref = hp.convective_microphysics(pl, wcu, dwcu, lfrz, wfz, tp, full(mc.TI), full(mc.FITMAX), pland, full(mc.CN0), full(mc.CN0I), full(mc.CN0G),
                                     fl[0], fl[1], fl[2], full(mc.RHOIP), full(mc.RHOG), full(mc.ITMAX), tl0, tl1, full(mc.WMAX), np.zeros(n), np.zeros(n))
    for a, b in zip(got, ref):
        assert np.array_equal(a, b)


# ------------------------------------------------------------------------------------------------ LSCOND (real rows)
@needs_dumps
def test_lscond_batch_bitwise_vs_per_column_real_rows():
    S_list, P_list = _ls_inputs()
    keep = [k for k, p in enumerate(P_list) if p["kmax"] <= 4]
    res = _ndiff(_ls_run([S_list[k] for k in keep], [P_list[k] for k in keep]))
    assert res == 0


@needs_dumps
def test_lscond_batch_columns_are_independent_real_rows():
    S_list, P_list = _ls_inputs()
    n = len(S_list)
    S, P = lb.pack(S_list, P_list, {k: S_list[0][k] for k in ("tausslip", "csizelip", "cldsal", "cldsv1")})
    S, W = lb.lscond_batch(S, P)
    half = n // 2
    S2, P2 = lb.pack(S_list[half:], P_list[half:], {k: S_list[half][k] for k in ("tausslip", "csizelip", "cldsal", "cldsv1")})
    S2, W2 = lb.lscond_batch(S2, P2)
    for k in ("tl", "ql", "cldssl", "taussl", "qclx", "qcix", "cldsal"):
        assert np.array_equal(S[k][:, half:], S2[k])


@needs_dumps
def test_lscond_batch_mutation_changes_result_real_rows(monkeypatch):
    S_list, P_list = _ls_inputs()
    keep = [k for k, p in enumerate(P_list) if p["kmax"] <= 4][:60]
    S_list, P_list = [S_list[k] for k in keep], [P_list[k] for k in keep]
    assert _ndiff(_ls_run(S_list, P_list)) == 0
    monkeypatch.setattr(lb, "pymin", lb.pymax)             # min -> max in every Python-min statement of the batch
    assert _ndiff(_ls_run(S_list, P_list)) > 0
    monkeypatch.undo()
    monkeypatch.setattr(lb, "COESIG", lb.COESIG * 1.0001)  # CTEI rate constant
    assert _ndiff(_ls_run(S_list, P_list)) > 0


# ------------------------------------------------------------------------------------------------ MSTCNV (real rows)
def _mc_run(r_list, o_ref):
    o = mb.mstcnv_batch(mb.stack_r(r_list), cf.RUN_TUNE)
    return mcmp.compare(o_ref, o)


@needs_dumps
def test_mstcnv_batch_bitwise_vs_per_column_real_rows():
    r_list, o_list = _mc_inputs()
    keep = [k for k, r in enumerate(r_list) if len(r["ra"]) == 4]
    assert sum(1 for k in keep if o_list[k]["lmcmin"] > 0) >= 20          # the rows really convect
    assert _ndiff(_mc_run([r_list[k] for k in keep], [o_list[k] for k in keep])) == 0


@needs_dumps
def test_mstcnv_batch_columns_are_independent_real_rows():
    """Lock-step masks must not leak between columns: every convecting column alone == the same column inside the batch (found two such bugs)."""
    r_list, o_list = _mc_inputs()
    conv = [k for k, o in enumerate(o_list) if o["lmcmin"] > 0 and len(r_list[k]["ra"]) == 4]
    pick = conv[::7][:12]
    for k in pick:
        alone = mb.mstcnv_batch(mb.stack_r([r_list[k]]), cf.RUN_TUNE)
        assert _ndiff(mcmp.compare([o_list[k]], alone)) == 0


@needs_dumps
def test_mstcnv_batch_mutations_change_result_real_rows(monkeypatch):
    r_list, o_list = _mc_inputs()
    keep = [k for k, r in enumerate(r_list) if len(r["ra"]) == 4 and o_list[k]["lmcmin"] > 0][:40]
    rl, ol = [r_list[k] for k in keep], [o_list[k] for k in keep]
    assert _ndiff(_mc_run(rl, ol)) == 0
    monkeypatch.setattr(mb, "pymin", mb.pymax)
    assert _ndiff(_mc_run(rl, ol)) > 0
    monkeypatch.undo()
    monkeypatch.setattr(mb, "CLDMIN", 0.2)                # minimum cloud fraction of the convective partition
    assert _ndiff(_mc_run(rl, ol)) > 0
    monkeypatch.undo()
    monkeypatch.setattr(mb, "FDDET", 0.5)                # downdraft detrainment fraction
    assert _ndiff(_mc_run(rl, ol)) > 0


# ------------------------------------------------------------------------------------------------ CONDSE chain
@needs_dumps
def test_condse_batch_vs_per_column_rows_and_real_state():
    inp, ref, cfg, rows, order, trace, Xpc = _data()
    Xb, cnt = cb.condse_step_batch(inp, cfg, with_momentum=False)
    assert cnt["columns"] == 3170
    skip = ("CSIZSSIP",)                      # module-state carry: the row run starts from zeros, the batch from the pole-0 column
    sel = lambda a, k: a[..., rows] if k in ("T", "Q", "QCL", "QCI") else a  # noqa: E731
    bad = []
    for k in Xpc:
        if k in skip or k in ("U", "V", "UALIJ", "VALIJ", "TLS", "QLS", "UKMSP", "VKMSP", "UKMNP", "VKMNP") or k not in Xb:
            continue
        a, b = Xb[k], Xpc[k]
        jax = {"T": 1, "Q": 1, "QCL": 1, "QCI": 1, "TMC": 1, "QMC": 1, "TMOM": 2, "QMOM": 2}.get(k, a.ndim - 1)
        for j in rows:
            sl = [slice(None)] * a.ndim
            sl[jax] = j
            if not np.array_equal(a[tuple(sl)], b[tuple(sl)]):
                bad.append((k, j))
    assert not bad, bad
    # and the batch is the same physics as the real model (libm: not bitwise, so only a sanity bound on a few fields)
    scale = np.abs(ref["PREC"]).max()
    assert (np.abs(Xb["PREC"] - ref["PREC"]) > 1e-9 * scale).mean() < 0.15      # libm threshold flips only (D126: a few % of columns)
    assert np.array_equal(Xb["LMC"], ref["LMC"])


@needs_dumps
def test_condse_batch_mutation_changes_result(monkeypatch):
    inp, ref, cfg, rows, order, trace, Xpc = _data()
    j = rows[0]
    Xb, _ = cb.condse_step_batch(inp, cfg, with_momentum=False)
    monkeypatch.setattr(cb, "TINY", 1e-3)                  # hand-off denominators
    Xm, _ = cb.condse_step_batch(inp, cfg, with_momentum=False)
    assert not np.array_equal(Xb["FRAC_ST_WATER"][:, :, j], Xm["FRAC_ST_WATER"][:, :, j])


def test_hand_off_b_matches_hand_off_per_column_synthetic():
    rng = np.random.default_rng(11)
    N = 30
    I, J = np.arange(N) % IM, np.full(N, 5)
    mk = lambda: np.zeros((LM, IM, JM))  # noqa: E731
    names = ("W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE DIM_ST_WATER "
             "DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV CSIZSS CSIZMC").split()
    Xa = {n: rng.uniform(0.1, 1, (LM, IM, JM)) for n in names}
    Xb = {n: v.copy() for n, v in Xa.items()}
    pick = lambda a: np.where(rng.uniform(size=a) < 0.5, rng.uniform(0.01, 1, a), 0.0)  # noqa: E731
    cldmcl, cldssl = pick((LM, N)), pick((LM, N))
    svlatl = np.where(rng.uniform(size=(LM, N)) < .5, cf.LHE, cf.LHS)
    svlhxl = np.where(rng.uniform(size=(LM, N)) < .5, cf.LHE, cf.LHS)
    qclx, qcix, cnv, cldsal = (rng.uniform(0, 1e-3, (LM, N)) for _ in range(4))
    cnv = np.where(rng.uniform(size=(LM, N)) < 0.3, 0.0, cnv)
    cb.hand_off_b(Xb, I, J, cldmcl, cldssl, svlatl, svlhxl, qclx, qcix, cnv, cldsal)
    cnt = cf.new_counts()
    for n in range(N):
        cf._hand_off(Xa, I[n], J[n], cldmcl[:, n], cldssl[:, n], svlatl[:, n], svlhxl[:, n], qclx[:, n], qcix[:, n], cnv[:, n], cldsal[:, n], cnt)
    for k in names:
        assert np.array_equal(Xa[k], Xb[k]), k
