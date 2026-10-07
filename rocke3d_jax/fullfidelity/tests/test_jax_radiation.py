"""D185 tests for jax_radiation.py (radiation hand-off, stage S3).  Own unit only.

Without server or data: seed chain on device, surface_fields_jax vs drv_radpacket (synthetic inputs), cond/io_callback skip pattern with a fake server.
With data (nov26_day records): packet from arrays == NumPy path == live packet (33312, 33317).
With the persistent server binary (RADSRVP_SCRATCH): outputs via the callback inside jit equal the recorded rsv_n26_<it>_out.bin (21 fields
bitwise), repeat order-independence, toy step (about 2 min; skipped when the binary or data is absent).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_radiation as JR  # noqa: E402
import drv_radpacket as DRP  # noqa: E402
import drv_rng  # noqa: E402
import radiation_server as rs  # noqa: E402

D = rs.FF + "/nov26_day"
STEPS = (33312, 33317)
HAVE_DATA = all(os.path.exists(f"{D}/{n}_{it}.bin") for it in STEPS for n in ("rsv_n26", "ffc_cse_in", "ffc_cse_out")
                if n != "rsv_n26") and all(os.path.exists(f"{D}/rsv_n26_{it}_{k}.bin") for it in STEPS for k in ("in", "out")) \
    and all(os.path.exists(f"{D}/ffa_step_{it}_r.bin") for it in STEPS)


def _server_ok():
    try:
        import radiation_server_persist as P
        return P.server_available() and HAVE_DATA
    except Exception:
        return False


NEED_DATA = pytest.mark.skipif(not HAVE_DATA, reason="nov26_day records absent")
NEED_SERVER = pytest.mark.skipif(not _server_ok(), reason="persistent radiation server binary / records absent")


def test_seed_chain_device_equals_python():
    for s0 in (1, 123456789, 2**32 - 1, 832739605):
        assert int(JR.seed_chain_radia(jnp.uint32(s0))) == _signed(drv_rng.radia_seed(s0))
        for rad in (True, False):
            assert int(JR.seed_chain_next(jnp.uint32(s0), rad)) == drv_rng.next_seed(s0, rad)


def _signed(v):
    return v - 2**32 if v >= 2**31 else v


def test_surface_fields_jax_equals_numpy():
    rng = np.random.default_rng(3)
    shp = (72, 46)
    fwater = np.where(rng.random(shp) < 0.3, 0.0, rng.random(shp))
    flake = np.where(rng.random(shp) < 0.5, 0.0, rng.random(shp)) * (fwater > 0)
    flice = np.where(rng.random(shp) < 0.8, 0.0, rng.random(shp))
    rsi = np.where(rng.random(shp) < 0.5, 0.0, rng.random(shp))
    axyp = 1e9 * (1 + rng.random(shp))
    geo = dict(flake=flake, fwater=fwater)
    st = dict(axyp=axyp, flice=flice, fland=rng.random(shp), fearth=rng.random(shp), geo=geo)
    ice = dict(rsi=rsi, snowi=rng.random(shp), pond_melt=rng.random(shp), flag_dsws=(rng.random(shp) < 0.5))
    ag = dict(zsi=rng.random(shp), zsnowi=rng.random(shp), gtempr=270 + rng.random(shp))
    S = dict(lake=dict(mwl=1e3 * rng.random(shp)), atm=dict(gtempr=270 + rng.random(shp)),
             li=dict(tlandi=rng.random(shp + (2,)) - 5, snowli=rng.random(shp)))
    ref = DRP.ice_lake_landice_fields(S, st, ice, ag)
    got = JR.surface_fields_jax(ice["rsi"], ice["snowi"], ice["pond_melt"], ice["flag_dsws"], ag["zsi"], ag["zsnowi"], ag["gtempr"], fwater, flake,
                                S["lake"]["mwl"], axyp, flice, st["fland"], st["fearth"], S["atm"]["gtempr"], S["li"]["tlandi"][..., 0], S["li"]["snowli"])
    assert set(got) == set(ref) and len(ref) == 15
    bad = [k for k in ref if not np.array_equal(np.asarray(got[k]), ref[k])]
    assert not bad, bad


def test_cond_skips_callback_and_hold_is_applied():
    """Fake server: radiation steps call it, the other four of five do not; the held SRHR/TRHR are applied with the per-step COSZ1."""
    class Fake:
        binary = restart = "fake"
        rundir = "."
        n = 0

        def request(self, state, itime, seed):
            Fake.n += 1
            out = {k: np.full(JR.OUT_SHAPES[k], float(itime % 7)) for k in JR.OUT_FIELDS}
            out["AIJ"] = np.zeros((72, 46, 1660))
            out["_wall_s"] = out["_server_s"] = 0.0
            return out

    h = JR.RadiationHandoff(Fake())
    const = JR.RadConst(1800.0, 1000.0, np.ones((72, 46), bool), 33312)
    step = JR.make_handoff_step(h, const)
    z = lambda shp: jnp.ones(shp)
    state = {k: z(rs.INPUT_FIELDS[k]) for k in JR.ATM_IN + JR.CLOUD_IN}
    surf = {k: z(rs.INPUT_FIELDS[k]) for k in JR.SURF_IN}
    hold = JR.empty_hold({k: z(rs.INPUT_FIELDS[k]) for k in JR.CARRY_IN})
    cz = jnp.full((72, 46), 0.5)
    for k in range(6):
        T, Q, hold, c = step(state, surf, hold, jnp.int32(33312 + k), jnp.int32(5), cz)
        jax.block_until_ready(T)
    assert Fake.n == 2                                            # 33312 and 33317 only
    assert [c["itime"] for c in h.log.calls] == [33312, 33317]
    assert h.log.sync_points == 2


@NEED_DATA
def test_packet_from_arrays_equals_numpy_path_and_live():
    import atm_day_free_rad as FRM
    import jax_radiation_check as CK
    for it in STEPS:
        lv = CK.live(it, "in")
        S, X = CK.record_state(it)
        st, _ = FRM.assemble_packet(S, X, X["SNOAGE"], lv["RQT"], lv["KLIQ"], lv)
        pk, _, _ = CK.device_packet(it)
        assert list(pk) == list(rs.INPUT_FIELDS)
        for k in rs.INPUT_FIELDS:
            assert np.array_equal(np.asarray(pk[k]), st[k]), (it, k, "array path vs NumPy path")
            assert np.array_equal(np.asarray(pk[k]), lv[k]), (it, k, "array path vs live packet")


@NEED_DATA
def test_seed_chain_equals_recorded():
    s0 = JR.seed0_from_record(33312)
    seed0 = jnp.uint32(s0)
    for it in range(33312, 33318):
        assert int(JR.seed_chain_radia(seed0)) == JR.seed_from_record(it)
        assert int(seed0) == JR.seed0_from_record(it)
        seed0 = JR.seed_chain_next(seed0, (it - 33312) % 5 == 0)


@pytest.fixture(scope="module")
def served():
    import jax_radiation_check as CK
    import radiation_server_persist as P
    res = {}
    with P.PersistentServer("nov26") as srv:
        h = JR.RadiationHandoff(srv, keep_outputs=True)

        @jax.jit
        def inside(pk, itime, seed):
            return h.call(pk, itime, seed)

        pk12, _, _ = CK.device_packet(33312)
        pk17, _, _ = CK.device_packet(33317)
        res["outs"] = []
        for it, pk in ((33312, pk12), (33317, pk17), (33312, pk12)):
            o, aijc = inside(pk, jnp.int32(it), jnp.int32(JR.seed_from_record(it)))
            res["outs"].append((it, {k: np.asarray(v) for k, v in o.items()}, np.asarray(aijc)))
        res["log"] = h.log.totals()
        res["sentence"] = h.log.sentence()
        res["toy"] = CK.toy_step(srv, log=lambda *a: None)
        res["pid"] = srv.proc.pid
    return res


@NEED_SERVER
def test_callback_outputs_equal_recorded(served):
    cols = [c - 1 for c in JR.AIJ_COLS.values()]
    for it, o, aijc in served["outs"]:
        ref = rs.read_packet(f"{D}/rsv_n26_{it}_out.bin")
        bad = [k for k in JR.OUT_FIELDS if not np.array_equal(o[k], ref[k])]
        assert not bad, (it, bad)
        assert np.abs(aijc - ref["AIJD"][:, :, cols]).max() < 1e-9


@NEED_SERVER
def test_order_independence_and_log(served):
    (i0, o0, _), (_, _, _), (i2, o2, _) = served["outs"]
    assert i0 == i2 and all(np.array_equal(o0[k], o2[k]) for k in JR.OUT_FIELDS)
    lg = served["log"]
    assert lg["calls"] == 3 and lg["host_sync_points"] == 3 and lg["SOCRATES_modified"] is False
    assert lg["arrays_device_to_host"] == 3 * 54 and lg["bytes_device_to_host"] > 3 * 33e6
    assert served["sentence"].startswith(JR.SENTENCE)


@NEED_SERVER
def test_toy_step_pattern(served):
    t = served["toy"]
    assert t["n_callbacks"] == 2 and t["itimes_called"] == [33312, 33317]
    assert t["seeds_used"] == t["seeds_recorded"]                  # device seed chain == recorded SEEDS[1]
    assert t["first_call_outputs_unequal_vs_recorded"] == []
    assert t["log"]["steps_without_callback"] == 4
    assert JR.SENTENCE in t["sentence"]


@NEED_SERVER
def test_server_stopped(served):
    import time
    time.sleep(1)
    assert not os.path.exists(f"/proc/{served['pid']}") or open(f"/proc/{served['pid']}/stat").read().split(")")[-1].split()[0] == "Z"
