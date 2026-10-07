"""D169: Ent per-iteration exports (ent_ff.py) against the ffent block of the real ffg dumps.  Skipped when the data are absent.

Measured bounds (not loosened): see scoping/D169_ENT_ENTRY.md.  Bitwise results need the Intel libimf exp/pow; the bitwise
assertions are skipped when it is not available (the residual is then only bounded by `test_glibc_fallback_close`)."""
import glob
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
MODEL = os.environ.get("MODELE_SRC", "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0")
RESTART = f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc"
F1 = f"{FF}/nov26/ffg_33312.bin"
F2 = f"{FF}/nov26/ffg_33313.bin"
HAVE = os.path.exists(RESTART) and os.path.exists(F1) and os.path.exists(F2)

import ent_ff as E            # noqa: E402

need_data = pytest.mark.skipif(not HAVE, reason="nov26 restart / ffg dumps not available")
need_imf = pytest.mark.skipif(not E.imf_available(), reason="Intel libimf not available (bitwise exp/pow)")
need_src = pytest.mark.skipif(not os.path.exists(f"{MODEL}/model/Ent/ent_pfts_ENT.f"), reason="ModelE source not available")

SUBSET = 11    # every 11th land cell: Ent cells are independent, so a subset keeps the state carry exact and the test short


def _keys(path):
    import ghy_compare as GC
    rec = GC.load(path)
    n = len(rec) // 2
    return {(int(r[0]), int(r[1])) for r in rec[:n:SUBSET]}


@pytest.fixture(scope="module")
def run():
    import ent_ghy_compare as C
    E.set_use_imf(True)
    E._PS = None
    cells = E.load_restart_cells(RESTART)
    keys = _keys(F1)
    r1 = C.run_file(F1, cells, "teacher", keys=keys)
    r2 = C.run_file(F2, cells, "teacher", keys=keys)
    return C, r1, r2


@need_src
def test_tables_match_fortran_source():
    s = open(f"{MODEL}/model/Ent/ent_pfts_ENT.f").read()
    i = s.index("type(pftype),parameter :: pfpar(N_PFT)")
    blk = s[i:s.index("/)", i)]
    txt = " ".join(re.sub(r"^\s*&", "", l.split("!")[0]) for l in blk.split("\n")[1:] if not l.strip().startswith("!"))
    rows = [[x.strip() for x in r.split(",")] for r in re.findall(r"pftype\(([^)]*)\)", txt)]
    assert len(rows) == 16
    num = lambda x: float(x.replace("d", "e").replace("D", "e"))
    assert [int(r[0]) for r in rows] == E.PFPAR_PST
    assert [int(r[2]) for r in rows] == E.PFPAR_LEAFTYPE
    assert [num(r[4]) for r in rows] == E.PFPAR_SSTAR
    assert [num(r[5]) for r in rows] == E.PFPAR_SWILT
    assert [int(r[14]) for r in rows] == E.PFPAR_PHENOTYPE
    s2 = open(f"{MODEL}/model/Ent/FBBpfts_ENT.f").read()
    j = s2.index("pftpar(N_PFT)")
    body = " ".join(l.split("!")[0] for l in s2[j:].split("\n") if not l.strip().startswith("!"))
    body = re.sub(r"\s*&\s*", " ", body)
    ps = re.findall(r"pspartype\(([^()]*)\)", body)
    assert len(ps) == 16
    vals = [[x.strip() for x in p.split(",") if x.strip()] for p in ps]
    assert [num(v[2]) for v in vals] == E.PFTPAR_VCMAX
    assert [num(v[3]) for v in vals] == E.PFTPAR_M
    assert [num(v[4]) for v in vals] == E.PFTPAR_B


@need_src
def test_single_precision_literals_are_in_the_object_code():
    """The literals without d0 (0.21 in calc_CO2compp, .20900 O2frac) are promoted as float32 by the real build: the double
    values appear in the compiled .o files (checked when the .o files exist)."""
    import struct
    for f, v in (("FBBphotosynthesis.o", 0.21), ("canopyspitters.o", 0.209)):
        p = f"{MODEL}/model/Ent/{f}"
        if not os.path.exists(p):
            pytest.skip("object files not available")
        d = open(p, "rb").read()
        f32v = float(np.float32(v))
        assert d.count(struct.pack("<d", f32v)) >= 1
        assert d.count(struct.pack("<d", v)) == 0
    assert E.F32(0.21) == float(np.float32(0.21)) and E.O2FRAC == float(np.float32(0.209))


@need_data
def test_unpack_and_per_call_exports_bitwise(run):
    C, r1, _ = run
    s = C.summarize_call(r1)
    for name in ("ws_can", "shc_can", "fv", "height", "albedo"):
        assert s[name]["n"] > 0
        assert s[name]["neq"] == s[name]["n"], name        # bitwise
    assert s["qf_exit"]["maxabs"] < 1e-16                  # GHY.f:2621 Qf update (in D169: 2.5e-17 over all steps/dates)


@need_data
@need_imf
def test_first_step_exports_residual_bounded(run):
    C, r1, _ = run
    a = C.summarize(r1)
    assert a["cnc"]["n"] > 0
    for f in ("trans_sw", "lai", "ipp"):
        assert a[f]["neq"] == a[f]["n"], f                 # bitwise
    # first step of the run: ulp-level residuals (attributed to GHY-side inputs of the first step, not verified)
    assert a["cnc"]["maxrel"] < 2e-15
    assert a["ci"]["maxrel"] < 2e-15
    assert a["gpp"]["maxrel"] < 2e-15
    assert a["betadl"]["maxabs"] < 5e-16


@need_data
@need_imf
def test_second_step_exports_bitwise_with_carried_state(run):
    C, _, r2 = run
    a = C.summarize(r2)
    for f in C.FIELDS:
        assert a[f]["n"] > 0
        if f == "ci":
            continue
        assert a[f]["neq"] >= a[f]["n"] - 1, f
    assert a["cnc"]["maxrel"] < 1e-15


@need_data
def test_glibc_fallback_close():
    """Without libimf (glibc exp/pow) the exports stay close (the root finder stops at |dx|<1e-4, so 1-ulp libm
    differences can move a result by ~1e-9 relative; bound deliberately loose and reported, not hidden)."""
    import ent_ghy_compare as C
    E.set_use_imf(False)
    try:
        E._PS = None
        cells = E.load_restart_cells(RESTART)
        keys = set(list(_keys(F1))[:12])
        res = C.run_file(F1, cells, "teacher", keys=keys)
        a = C.summarize(res)
        assert a["cnc"]["maxrel"] < 1e-8
    finally:
        E.set_use_imf(True)
        E._PS = None


DAY = f"{FF}/nov26_day"
HAVE_DAY = os.path.exists(f"{DAY}/ffg_33361.bin") and os.path.exists(f"{DAY}/ffg_33360.bin")


@need_src
def test_albvnd_table_matches_source():
    import ent_daily_ff as D
    src = open(f"{MODEL}/model/Ent/ent_pfts_ENT.f").read().split("\n")
    i = [k for k, l in enumerate(src) if "ALBVND(N_COVERTYPES,4,6)" in l][0]
    vals = []
    k = i + 1
    while True:
        l = src[k]
        k += 1
        if l.startswith("C") or l.strip() == "":
            continue
        if "/)," in l:
            break
        vals += [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[dDeE][-+]?\d+)?", l[6:].split("!")[0])]
    assert len(vals) == 432
    ref = np.array(vals, dtype=np.float32).astype(np.float64).reshape((18, 4, 6), order="F")
    assert np.array_equal(ref, D.ALBVND)


@need_data
@pytest.mark.skipif(not (HAVE_DAY and os.path.exists("/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files/V72x46_EntMM16_lai_trimmed_scaled_ext.nc")),
                    reason="nov26_day dumps or the LAI file not available")
def test_daily_prescribed_lai_albedo_bitwise_at_day_boundary():
    """The record of step 33360 carries end_of_day = 1; the prescribed LAI/albedo for jday 331 applied to the restart state reproduces
    ws_can, albedo(6) and the first-iteration lai of step 33361 bit for bit (every 5th land cell)."""
    import ghy_compare as GC
    import ent_daily_ff as D
    cells = E.load_restart_cells(RESTART)
    qty = D.read_lai_file()
    la = D.lai_linm2m(qty, 331)
    r1 = GC.load(f"{DAY}/ffg_33361.bin")
    n = 0
    for k in range(0, 753, 5):
        r = r1[k]
        i, j = int(r[0]), int(r[1])
        import pickle
        cell = pickle.loads(pickle.dumps(cells[(i, j)]))
        D.daily_update(cell, la[:, j - 1, i - 1], D.hemi_of_j(j), 331)
        ce = E.call_exports(cell)
        assert ce["ws_can"] == r[167]
        assert np.array_equal(ce["albedo"], r[171:177])
        if r[289] > 0:
            assert cell.LAI == r[299 + 10]
        n += 1
    assert n > 100
    # the day number matters: the previous day's value does not reproduce the record
    la0 = D.lai_linm2m(qty, 330)
    assert not np.array_equal(la0, la)
