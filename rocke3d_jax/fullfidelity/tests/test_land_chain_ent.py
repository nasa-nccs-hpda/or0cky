"""D171: Ent computed from our own state in the chained land path (land_chain_ent.py).  Skipped when the data are absent.

Tolerances are the measured ones of scoping/D171_ENT_WIRING_ENTRY.md (not loosened); bitwise statements need the Intel libimf."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
RESTART = f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc"
DAY = f"{FF}/nov26_day"
HAVE = all(os.path.exists(p) for p in (RESTART, f"{DAY}/ffg_33312.bin", f"{DAY}/ffg_33313.bin", f"{DAY}/ffg_33360.bin",
                                      f"{DAY}/ffp_33312.bin", f"{DAY}/fft_33312.bin"))
import ent_ff as E            # noqa: E402

need_data = pytest.mark.skipif(not HAVE, reason="nov26_day dumps / restart not available")
need_imf = pytest.mark.skipif(not E.imf_available(), reason="Intel libimf not available")

GHY = ["tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"]
SUBSET = 13


@pytest.fixture(scope="module")
def closed():
    import land_chain_ent as L
    import ghy_compare as GC
    E.set_use_imf(True)
    E._PS = None
    rec = GC.load(f"{DAY}/ffg_33312.bin")
    n = len(rec) // 2
    keys = {(int(r[0]), int(r[1])) for r in rec[:n:SUBSET]}
    ent = L.EntLand(RESTART, qf_mode="record", keys=keys)
    res = []
    for it in (33312, 33313):
        rec = GC.load(f"{DAY}/ffg_{it}.bin")
        for r in rec:
            key = (int(r[0]), int(r[1]))
            if key in keys:
                res.append(L.ghy_ent_call(r, ent.cells[key], None, compare=True))
    return res


@need_data
def test_closed_ghy_outputs_vs_record(closed):
    # measured on the full day: scaled differences <= 1e-12 (see the entry); 2 steps subset here
    for o in closed:
        for k in GHY:
            ref = o["refs"][k]
            assert abs(getattr(o["col"], k) - ref) <= 1e-9 * max(abs(ref), 1e-3), (k, getattr(o["col"], k), ref)
        assert o["nit"] == o["refs"]["ffnit"]
        assert np.abs(o["col"].w[:, :2] - o["refs"]["w_out"]).max() < 1e-12


@need_data
@need_imf
def test_closed_exports_close(closed):
    for o in closed:
        for it in o["comp"]:
            for f in ("cnc", "ci", "gpp", "lai", "ipp", "trans_sw"):
                got, ref = it[f]
                assert abs(got - ref) <= 1e-12 * max(abs(ref), 1e-30) + 1e-300, (f, got, ref)


@need_data
@need_imf
def test_substep_ent_matches_recorded_ent_substep():
    """land_substep_ent (Ent computed, scalar GHY) vs land_chain.land_substep (recorded Ent, batched JAX GHY) on substep 1 of 33312:
    same PBL, same forcing -> GHY outputs agree at the GHY-port level."""
    import land_chain as LC
    import land_chain_ent as L
    import ghy_compare as GC
    import pbl_compare as PC
    import tile_aggregate_ff as TA
    E.set_use_imf(True)
    E._PS = None
    p = PC.load(f"{DAY}/ffp_33312.bin")
    g = GC.load(f"{DAY}/ffg_33312.bin")
    fft = TA.load(f"{DAY}/fft_33312.bin")
    n, ng, B = len(p) // 2, len(g) // 2, len(fft) // 2
    blk1 = fft[:B]
    _, patch1, _ = TA.unpack(blk1)
    lut = {(int(a), int(b)): k for k, (a, b) in enumerate(blk1[:, :2])}
    idx = np.array([lut[(int(a), int(b))] for a, b in g[:ng, :2]])
    trup = LC.infer_trup(g[:ng], patch1["dth1"][idx, 3], 900.0)
    ph = p[:n]
    p4 = np.array(ph[ph[:, 2] == 4])
    gr = g[:ng]
    q1 = gr[:, 157] / gr[:, 165]
    ent = L.EntLand(RESTART)
    ent.begin_step(33312)
    a = LC.land_substep(p4, gr, q1, trup, 900.0, None)
    b = L.land_substep_ent(p4, gr, q1, trup, 900.0, None, ent)
    for k in ("tbcs", "tsns", "ashg", "alhg", "aevap"):
        sc = max(np.abs(a["ghy"][k]).max(), 1e-3)
        assert np.abs(a["ghy"][k] - b["ghy"][k]).max() <= 1e-9 * sc, k
    for k in ("uflux1", "vflux1", "dth1", "dq1"):
        sc = max(np.abs(a["patch"][k]).max(), 1e-30)
        assert np.abs(a["patch"][k] - b["patch"][k]).max() <= 1e-9 * sc, k
    assert np.abs(a["evap_max_ij"] - b["evap_max_ij"]).max() <= 1e-9 * max(np.abs(a["evap_max_ij"]).max(), 1e-30)


@need_data
@need_imf
def test_daily_update_reproduces_recorded_call_exports():
    """At the day boundary the daily update (applied by EntLand.maybe_daily) gives the recorded per-call exports of ffg_33360."""
    import land_chain_ent as L
    import ghy_compare as GC
    E.set_use_imf(True)
    rec = GC.load(f"{DAY}/ffg_33360.bin")
    n = len(rec) // 2
    keys = {(int(r[0]), int(r[1])) for r in rec[:n:29]}
    ent = L.EntLand(RESTART, keys=keys)
    ent.nsteps = 1
    ent.begin_step(33360)
    assert rec[0, L.EOD] == 1.0
    ent.maybe_daily(rec[:n])
    assert ent.cur["daily"] == 331
    for r in rec[:n]:
        key = (int(r[0]), int(r[1]))
        if key in keys:
            ce = E.call_exports(ent.cells[key])
            assert ce["ws_can"] == r[167] and ce["shc_can"] == r[168] and ce["height"] == r[170]
            assert np.array_equal(ce["albedo"], r[171:177])
