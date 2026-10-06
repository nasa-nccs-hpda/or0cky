"""Tests for clouds_condse_ff.py (CONDSE column driver glue + init_CLD constants) -- D124-D126.

Real-Fortran tests use ff_data/<date>/ffc_cse_in_*.bin / ffc_cse_out_*.bin (+ the old ffd_*_pre_condse/post_condse dumps) and are skipped
when those are absent.  They run the port on a few full latitude rows (all columns of the row, Fortran order) of step 2 of nov26
(33313, the second step: the carried arrays are non-zero at entry); the full-population results (3 dates x 6 steps x 3170 columns, both libm and libimf modes) come from
`python3 clouds_condse_compare.py [--imf]` and are in the ledger entry.  Module-state carry (CSIZELIP/TAUSSIP etc., see the port docstring)
depends on the preceding column, so those two fields are not compared in the row tests.
libimf tests are skipped without the Intel runtime.  Tests marked "non-real" use synthetic data (labelled): they cover branches that real
data never reach (stop_model exits) or algebra with a closed-form answer.
Mutation tests change one statement of the port and require the real-dump comparison to get worse, i.e. the comparison is not vacuous."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_condse_ff as cf  # noqa: E402
import clouds_condse_io as cio  # noqa: E402
import clouds_mstcnv_ff as mc  # noqa: E402

DATE, ITIME = "nov26", 33313          # step 2 of the window: the carried arrays (FRAC_AREA_CNV, CSIZMC) are non-zero at entry
HAVE = cio.have_dumps(DATE) and os.path.exists(f"{cio.FF_DEFAULT}/{DATE}/ffc_cse_out_{ITIME}.bin")
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffc_cse_* dumps not present on this host")
needs_imf = pytest.mark.skipif(not mc.imf_available(), reason="Intel libimf not available")
LM, IM, JM = 40, 72, 46
SKIP_CARRY = ("CSIZSSIP", "TAUSSIP")
CMP = ("T Q QCL QCI TMOM QMOM TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV CLDSAV1 FSS TAUSS TAUMC CLDSS CLDMC CSIZMC CSIZSS QLSS QISS QLMC QIMC "
       "W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE DIM_ST_WATER DIM_ST_ICE "
       "DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV P_ACC PM_ACC PREC EPREC PRECSS DDM1 DDMS TDN1 QDN1 DDML AIRX SNOAGE LMC").split()
_cache = {}


def _data():
    if "d" not in _cache:
        inp, ref = cio.load_step(DATE, ITIME)
        cfg = cf.make_cfg(DATE)
        G = cfg["geom"]
        conv_rows = (ref["LMC"][0] > 0).sum(axis=0)
        snow_rows = (ref["EPREC"] < 0).sum(axis=0)
        rows = sorted({int(np.argmax(conv_rows[2:-2])) + 2, int(np.argmax(snow_rows[2:-2])) + 2})
        _cache["d"] = (inp, ref, cfg, G, rows)
    return _cache["d"]


def _jax(n):
    return {"T": 1, "Q": 1, "QCL": 1, "QCI": 1, "TMOM": 2, "QMOM": 2}.get(n)


def _row_sel(n, a, rows):
    ax = _jax(n)
    ax = a.ndim - 1 if ax is None else ax
    m = np.zeros(a.shape, bool)
    for j in rows:
        sl = [slice(None)] * a.ndim
        sl[ax] = j
        m[tuple(sl)] = True
    return m


def _run(rows, imf=False, mut=None, momentum=False):
    inp, ref, cfg, G, _ = _data()
    cols = [(i, j) for j in rows for i in range(int(G["IMAXJ"][j]))]
    cf.set_backend("imf" if imf else "numpy")
    try:
        X, cnt = cf.condse_step(inp, cfg, cols=cols, with_momentum=momentum, mut=mut)
    finally:
        cf.set_backend("numpy")
    return X, cnt


def _inexact(X, ref, rows):
    out = {}
    for n in CMP:
        m = _row_sel(n, X[n], rows)
        out[n] = int(((X[n] != ref[n]) & m).sum())
    return out


def _base(imf):
    key = ("base", imf)
    if key not in _cache:
        _, ref, _, _, rows = _data()
        X, cnt = _run(rows, imf)
        _cache[key] = (_inexact(X, ref, rows), cnt)
    return _cache[key]


# ---------------------------------------------------------------------------------------------------- constants and small pieces
def test_init_cld_constants_recorded():
    c = cf.init_cld_constants(1800.0)
    assert c["bydtsrc"] == 0.0005555555555555556          # recorded BYDTsrc (ffc_mc_cols)
    assert c["xmass"] == 1765.197                            # recorded XMASS = .1d0*DTsrc*GRAV
    assert abs(c["bybr"] - 0.7829735282337728) <= 2 * np.spacing(0.7829735282337728)   # libm pow: at most 1-2 ulp from the recorded libimf value


@needs_imf
def test_bybr_bitwise_with_libimf():
    cf.set_backend("imf")
    try:
        assert cf.init_cld_constants(1800.0)["bybr"] == 0.7829735282337728
    finally:
        cf.set_backend("numpy")


def test_ls_params_recorded_formulas():
    # recorded LSCOND parameter values (ffc_ls_bnd): WMUI, and the pearth dependence of WCONST / SCDNCW (59.68 over ocean, 174 over land)
    p0 = cf.ls_params(0.0, 5, 4, [0.25] * 4, 1 / 1800., 1800., 0.7829735282337728)
    p1 = cf.ls_params(1.0, 5, 4, [0.25] * 4, 1 / 1800., 1800., 0.7829735282337728)
    assert p0["wmui"] == 1.0000000474974512e-06
    assert (p0["wconst"], p1["wconst"]) == (0.25, 0.5)
    assert (p0["scdncw"], p1["scdncw"]) == (59.68, 174.0)
    pm = cf.ls_params(0.046875, 5, 4, [0.25] * 4, 1 / 1800., 1800., 0.78)
    assert pm["wconst"] == 0.25 * (1.0 - 0.046875) + 0.5 * 0.046875 == 0.26171875      # a recorded WCONST value
    assert pm["scdncw"] == pytest.approx(59.68 * (1.0 - 0.046875) + 174.0 * 0.046875, rel=0, abs=0)


def _burn(ix, n):
    """RANDOM:BURN_RANDOM (jump-ahead) re-implemented independently, 32-bit wrap."""
    a, b = 69069, 1
    nn = n
    while nn:
        if nn % 2 == 1:
            ix = (ix * a + b) & 0xFFFFFFFF
        b = ((a + 1) * b) & 0xFFFFFFFF
        a = (a * a) & 0xFFFFFFFF
        nn //= 2
    return ix


def test_randu_stream_matches_burn_random_jump_ahead():
    """non-real: the sequential draw count of the stream equals the BURN_RANDOM jump-ahead of the Fortran for a seed."""
    imaxj = [1 if j in (0, JM - 1) else IM for j in range(JM)]
    ix0 = 123456789
    _, seed = cf.randu_stream(ix0, imaxj, 29)
    nd = sum(imaxj) * 29 * 3
    assert seed == (lambda v: v - (1 << 32) if v >= (1 << 31) else v)(_burn(ix0, nd))


def test_randu_values_closed_form():
    """non-real: first draw for seed 0: ix=1; iand(1,ffffff00)=0 -> 0.0; seed -1 -> ix=-69068 -> negative branch 1+(iand*2**-32)."""
    r, _ = cf.randu_stream(0, [1] + [0] * (JM - 1), 1)
    assert r[0, 0, 0, 0] == 0.0
    ix = (-1 * 69069 + 1)
    s = ix
    expect = 1.0 + ((s & ~0xFF) * 2.0 ** -32)
    r2, _ = cf.randu_stream(0xFFFFFFFF, [1] + [0] * (JM - 1), 1)
    assert r2[0, 0, 0, 0] == expect and 0.0 < expect < 1.0


def test_avg_replicated_duv_nonpolar_algebra():
    """non-real: a unit tendency in every replicated slot adds 4 to u,v on the interior rows (the four neighbouring slots) and 3 on the two
    pole-adjacent rows (two slots are the .5-scaled pole entries); row J=1 gets nothing (J_0STG=2)."""
    u = np.zeros((IM, JM, LM))
    v = np.zeros((IM, JM, LM))
    ukm = np.ones((4, LM, IM, JM))
    vkm = np.ones((4, LM, IM, JM))
    sp = np.ones((IM, LM))
    cf.avg_replicated_duv_to_vgrid(u, v, ukm, vkm, sp, sp, sp, sp)
    assert np.all(u[:, 0, :] == 0.0) and np.all(v[:, 0, :] == 0.0)
    assert np.all(u[:, 2:JM - 1, :] == 4.0) and np.all(v[:, 2:JM - 1, :] == 4.0)
    assert np.all(u[:, 1, :] == 3.0) and np.all(u[:, JM - 1, :] == 3.0)        # .5+.5 from the pole slots, +1+1 own slots


def test_replicate_then_avg_roundtrip_of_constant_field():
    """non-real: replicate of a constant wind field gives the constant everywhere, so a zero tendency leaves u unchanged."""
    u = np.full((IM, JM, LM), 3.25)
    v = np.full((IM, JM, LM), -1.5)
    ukm, vkm, usp, vsp, unp, vnp = cf.replicate_uv_to_agrid(u, v)
    assert np.all(ukm[:, :, :, 1:JM - 1] == 3.25) and np.all(vkm[:, :, :, 1:JM - 1] == -1.5)
    assert np.all(usp == 3.25) and np.all(vnp == -1.5)
    zu = np.zeros_like(ukm)
    u2 = u.copy()
    v2 = v.copy()
    cf.avg_replicated_duv_to_vgrid(u2, v2, zu, zu.copy(), usp * 0, vsp * 0, unp * 0, vnp * 0)
    assert np.array_equal(u2, u) and np.array_equal(v2, v)


def _empty_X():
    return {k: np.zeros((LM, IM, JM)) for k in ("W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER "
                                                "MIX_CNV_ICE DIM_ST_WATER DIM_ST_ICE DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV CSIZSS CSIZMC").split()}


def test_hand_off_stop_model_branches_are_errors():
    """non-real (these exits are never reached on real data): a stratiform / convective cloud whose phase marker is neither LHE nor LHS stops."""
    X = _empty_X()
    z = np.zeros(LM)
    cld = z.copy()
    cld[3] = 0.5
    with pytest.raises(RuntimeError):
        cf._hand_off(X, 0, 0, z, cld, z, np.full(LM, 1.0), z, z, z, z, cf.new_counts())       # svlhxl neither LHE nor LHS
    with pytest.raises(RuntimeError):
        cf._hand_off(X, 0, 0, cld, z, np.full(LM, 1.0), np.full(LM, cf.LHE), z, z, z, z, cf.new_counts())    # svlatl neither


def test_hand_off_area_cnv_is_carried_not_reset():
    """non-real: FRAC_AREA_CNV keeps its previous-step value where there is no convective cloud (CLOUDS2_DRV.F90:2068-2078)."""
    X = _empty_X()
    X["FRAC_AREA_CNV"][:, 0, 0] = 0.7
    z = np.zeros(LM)
    cf._hand_off(X, 0, 0, z, z, z, np.full(LM, cf.LHE), z, z, z, z, cf.new_counts())
    assert np.all(X["FRAC_AREA_CNV"][:, 0, 0] == 0.7) and np.all(X["FRAC_CNV_WATER"][:, 0, 0] == 0.0)
    X2 = _empty_X()
    X2["FRAC_AREA_CNV"][:, 0, 0] = 0.7
    cf._hand_off(X2, 0, 0, z, z, z, np.full(LM, cf.LHE), z, z, z, z, cf.new_counts(), mut={"reset_area_cnv": True})
    assert np.all(X2["FRAC_AREA_CNV"][:, 0, 0] == 0.0)


def test_hand_off_zero_cnv_mass_ratio_renormalises_stratiform():
    """non-real: CNVMMRL==0 with a convective cloud fraction: convective fields zero, w_cloud=CLDSSL, stratiform water fraction renormalised to 1."""
    X = _empty_X()
    z = np.zeros(LM)
    cs, cm = z.copy(), z.copy()
    cs[5], cm[5] = 0.4, 0.2
    svl = np.full(LM, cf.LHE)
    qc = z.copy()
    qc[5] = 1e-4
    cf._hand_off(X, 0, 0, cm, cs, svl, svl, qc, z, z, z + 1.0, cf.new_counts())
    assert X["W_CLOUD"][5, 0, 0] == 0.4 and X["FRAC_CNV_WATER"][5, 0, 0] == 0.0 and X["FRAC_AREA_CNV"][5, 0, 0] == 0.0
    assert X["FRAC_ST_WATER"][5, 0, 0] == 1.0 and X["FRAC_ST_ICE"][5, 0, 0] == 0.0


# ---------------------------------------------------------------------------------------------------- real-Fortran checks
@needs_dumps
def test_randu_stream_equals_recorded_rndss():
    inp, ref, cfg, G, _ = _data()
    rn, seed = cf.randu_stream(inp["SEEDS"][0], [int(x) for x in G["IMAXJ"]], cfg["lmcld"])
    L = cfg["lmcld"]
    for j in range(JM):
        n = int(G["IMAXJ"][j])
        assert np.array_equal(rn[:, :L, :n, j], inp["RNDSS"][:, :L, :n, j]), j
    assert seed == int(inp["SEEDS"][1])


@needs_dumps
def test_replicate_uv_equals_recorded_ukm():
    inp, ref, cfg, G, _ = _data()
    ukm, vkm, usp, vsp, unp, vnp = cf.replicate_uv_to_agrid(inp["U"], inp["V"])
    assert np.array_equal(ukm[:, :, :, 1:JM - 1], inp["UKM"][:, :, :, 1:JM - 1])
    assert np.array_equal(vkm[:, :, :, 1:JM - 1], inp["VKM"][:, :, :, 1:JM - 1])
    assert np.array_equal(usp, inp["UKMSP"]) and np.array_equal(vsp, inp["VKMSP"])
    assert np.array_equal(unp, inp["UKMNP"]) and np.array_equal(vnp, inp["VKMNP"])


@needs_dumps
def test_momentum_back_transfer_from_recorded_tendencies_exact():
    """entry U,V + the recorded exit tendencies (UKM/VKM, pole arrays) -> avg_replicated_duv_to_vgrid + recalc_agrid_uv == recorded exit U,V,UALIJ,VALIJ."""
    import dyn_glue_ff as gf
    inp, ref, cfg, G, _ = _data()
    u, v = inp["U"].copy(), inp["V"].copy()
    ukm, vkm = ref["UKM"].copy(), ref["VKM"].copy()
    cf.avg_replicated_duv_to_vgrid(u, v, ukm, vkm, ref["UKMSP"], ref["VKMSP"], ref["UKMNP"], ref["VKMNP"])
    assert np.array_equal(u, ref["U"]) and np.array_equal(v, ref["V"])
    assert np.array_equal(ukm[:, :, :, 1:JM - 1], ref["UKM"][:, :, :, 1:JM - 1])      # pole rows were rescaled in place, as in the Fortran
    ua, va = gf.recalc_agrid_uv(u, v, cfg["glue_geom"])
    for j in range(JM):
        n = int(G["IMAXJ"][j])
        assert np.array_equal(ua[:, :n, j], ref["UALIJ"][:, :n, j]) and np.array_equal(va[:, :n, j], ref["VALIJ"][:, :n, j])


@needs_dumps
@pytest.mark.parametrize("imf", [False, pytest.param(True, marks=needs_imf)])
def test_rows_setup_matches_recorded_mstcnv_and_lscond_inputs(imf):
    """MSTCNV inputs built by the port from the CONDSE entry state == recorded MSTCNV boundary inputs; LSCOND parameters == recorded ones."""
    import clouds_lscond_io as lio
    import clouds_mstcnv_io as mio
    inp, ref, cfg, G, rows = _data()
    cols, bnd = mio.load_cols(DATE), lio.load_bnd(DATE)
    cfg = dict(cfg)
    h, I = cols["hdr"], cols["inp"]
    sel = {(int(h["i"][k]) - 1, int(h["j"][k]) - 1): k for k in np.where(h["itime"] == ITIME)[0] if int(h["j"][k]) - 1 in rows}
    bsel = {(int(bnd["i"][k]) - 1, int(bnd["j"][k]) - 1): k for k in np.where(bnd["itime"] == ITIME)[0] if int(bnd["j"][k]) - 1 in rows}
    assert sel and bsel
    cfg["trace"] = {key: None for key in list(sel) + list(bsel)}
    cf.set_backend("imf" if imf else "numpy")
    try:
        cf.condse_step(inp, cfg, cols=[(i, j) for j in rows for i in range(int(G["IMAXJ"][j]))], with_momentum=False)
    finally:
        cf.set_backend("numpy")
    nchk = 0
    for key, k in sel.items():
        r = cfg["trace"][key]["r"]
        for name in I:
            if name in ("ra", "um", "vm", "u0", "v0"):
                continue
            assert np.array_equal(np.asarray(r[name], float).reshape(np.shape(I[name][k])), I[name][k]), (key, name)
            nchk += 1
    for key, k in bsel.items():
        P0, S0 = cfg["trace"][key]["P0"], cfg["trace"][key]["S0"]
        for nm in ("wconst", "scdncw", "scdnci", "wmui", "bybr", "pearth", "dcl", "lmcld", "kmax", "rtemp", "cmx", "u00a", "bydtsrc"):
            assert float(P0[nm]) == float(bnd[nm][k]), (key, nm)
        for nm in ("qcll", "qcil", "sdl", "ttoldl", "aq", "dpdt", "pl", "plk", "airm", "byam", "pdsigl00"):
            assert np.array_equal(np.asarray(S0[nm], float), bnd[nm][k]), (key, nm)
        if imf:        # inputs that come out of MSTCNV: bitwise only with the libimf backend (libm may flip a threshold in a few columns)
            for nm in ("svlatl", "fssl", "u00l", "taumcl", "vsubl"):
                assert np.array_equal(np.asarray(S0[nm], float), bnd[nm][k]), (key, nm)
        for nm in ("th", "ql", "tl", "rh", "svlhxl", "cldsavl"):
            assert np.array_equal(np.asarray(S0[nm], float), bnd["in_" + nm][k]), (key, nm)
        nchk += 1
    assert nchk > 50


@needs_dumps
@needs_imf
def test_rows_libimf_nearly_bitwise():
    """Full latitude rows (conv + snow rows) in libimf mode: every compared field bitwise except columns that carry the known MSTCNV imf residual."""
    _, ref, _, G, rows = _data()
    nin, cnt = _base(True)
    ncells = {n: int(_row_sel(n, ref[n], rows).sum()) for n in CMP}
    bad = {n: v for n, v in nin.items() if v}
    assert sum(bad.values()) <= 0.001 * sum(ncells.values()), bad            # <= 0.1 % of the compared cells
    assert cnt["columns"] == len(rows) * IM and cnt["convecting"] > 0 and cnt["ddml_exit_early"] > 0


@needs_dumps
def test_rows_libm_mostly_exact_and_nonvacuous():
    """libm mode: a threshold flip may move a few columns; the large majority of cells is bitwise and the comparison has real content."""
    _, ref, _, G, rows = _data()
    nin, cnt = _base(False)
    ncells = {n: int(_row_sel(n, ref[n], rows).sum()) for n in CMP}
    tot_bad, tot = sum(nin.values()), sum(ncells.values())
    assert tot_bad < 0.05 * tot, (tot_bad, tot)
    assert cnt["convecting"] > 20 and cnt["st_water"] > 0 and cnt["cnv_none"] > 0 and cnt["snow_age"] > 0


# ---------------------------------------------------------------------------------------------------- mutation checks
def _mut_effect(mut, imf=False):
    _, ref, _, _, rows = _data()
    base, _ = _base(imf)
    X, _ = _run(rows, imf, mut=mut)
    mutd = _inexact(X, ref, rows)
    return base, mutd


@needs_dumps
@pytest.mark.parametrize("mut,fields", [
    ({"tprcp_post": True}, ("EPREC",)),                       # precipitation temperature test (CONDSE:848/1125)
    ({"no_snow_age": True}, ("SNOAGE",)),                     # SNOAGE*exp(-PRCP) (CONDSE:1485-1489)
    ({"ddml_loop_long": True}, ("DDML", "TDN1", "QDN1", "DDMS")),   # DO L=1,DCL downdraft-layer search (CONDSE:1164-1167)
    ({"merge_swap": True}, ("T", "Q")),                       # final FSSL merge (CONDSE:2152-2153)
    ({"reset_area_cnv": True}, ("FRAC_AREA_CNV",)),           # the unreset FRAC_AREA_CNV (CONDSE:2068-2078)
])
def test_mutations_are_detected(mut, fields):
    base, mutd = _mut_effect(mut)
    worse = [f for f in fields if mutd[f] > base[f]]
    assert worse, (mut, {f: (base[f], mutd[f]) for f in fields})


@needs_dumps
def test_mutation_real_literal_in_wturb_is_detected():
    """Replacing the REAL(4) literal .6666667 of WTURB by the double literal changes MSTCNV's inputs and hence outputs on the real rows."""
    _, ref, _, _, rows = _data()
    inp, _, cfg, G, _ = _data()
    cols = [(i, j) for j in rows for i in range(int(G["IMAXJ"][j]))]
    cfg = dict(cfg)
    cfg["trace"] = {c: None for c in cols[:200]}
    cf.condse_step(inp, cfg, cols=cols, with_momentum=False)
    cfg2 = dict(cfg)
    cfg2["trace"] = {c: None for c in cols[:200]}
    cf.condse_step(inp, cfg2, cols=cols, with_momentum=False, mut={"wturb_double": True})
    diff = sum(not np.array_equal(cfg["trace"][c]["r"]["wturb"], cfg2["trace"][c]["r"]["wturb"]) for c in cols[:200] if cfg["trace"][c])
    assert diff > 0
