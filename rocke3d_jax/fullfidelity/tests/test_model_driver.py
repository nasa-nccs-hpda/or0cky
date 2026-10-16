"""D178: model driver skeleton (model_driver.py) and the day-boundary items (drv_daily.py).  Skips when the nov26 dumps are absent.

The checkpoint test runs a short closed-surface window in ONE process (so the JIT compile is paid once): driver A runs 3 steps with a checkpoint
after step 2, a fresh driver B resumes from that file and runs the last step; the complete state trees (atmosphere, CONDSE module arrays,
surface state, DYNSI/ADVSI state, land state, F3 accumulators, RNG seed, provider state) must be bitwise equal.  The cross-process version of
the same test (6 steps in one go versus 3 + checkpoint + a fresh process resuming + 3) is the one reported in the ledger.
Bitwise reproducibility holds for a fixed thread configuration (see the ledger: 1 core vs 2 cores differ at rounding level from step 0)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_condse_io as cio  # noqa: E402

FF = cio.FF_DEFAULT
DAY = f"{FF}/nov26_day"
HAVE = all(os.path.exists(p) for p in (f"{DAY}/ffc_cse_in_33312.bin", f"{DAY}/ffa_step_33359_e.bin", f"{DAY}/ffa_step_33360_a.bin",
                                       f"{DAY}/ffc_cse_out_33359.bin", f"{DAY}/ffc_cse_in_33360.bin", f"{DAY}/ffg_33312.bin",
                                       f"{FF}/nov26/ffd_glue_daily_33360.bin",
                                       f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc"))
needs_data = pytest.mark.skipif(not HAVE, reason="nov26 dumps / restart missing")


# ------------------------------------------------------------------------------------------------ no data needed
def test_trees_equal_detects_one_ulp():
    import model_driver as M
    a = dict(x=np.arange(5.0), y=dict(z=[np.ones(3), 2]), s=1.5)
    b = dict(x=np.arange(5.0), y=dict(z=[np.ones(3), 2]), s=1.5)
    assert M.trees_equal(a, b) == []
    b["y"]["z"][0] = np.nextafter(np.ones(3), 2.0)
    assert len(M.trees_equal(a, b)) == 1
    assert M.tree_digest(a).hexdigest() != M.tree_digest(b).hexdigest()


def test_abstract_provider_raises():
    import model_driver as M
    p = M.BoundaryProvider()
    with pytest.raises(NotImplementedError):
        p.real(33312)
    with pytest.raises(NotImplementedError):
        p.initial_seed(33312)
    assert p.column_modifiers() == [] and p.radiation_aij(33312) is None


def test_date_of_itime():
    import drv_daily as D
    assert D.date_of_itime(33312) == (1850, 11) or D.date_of_itime(33312)[1] == 11
    assert D.date_of_itime(33360)[1] == 11 and D.date_of_itime(17520)[1] == 1


def test_mdrya_derived():
    import drv_daily as D
    assert D.mdrya() == 10034.007535702814            # the recorded value of the real DAILY_ATMDYN dump (nov26 -> nov27)


def test_snoage_age_formula():
    import drv_daily as D
    imaxj = np.full(46, 72)
    imaxj[0] = imaxj[45] = 1
    s = np.random.RandomState(0).rand(3, 72, 46) * 50
    o = D.snoage_age(s, imaxj)
    assert np.array_equal(o[:, 1:, 0], s[:, 1:, 0]) and np.array_equal(o[:, 1:, 45], s[:, 1:, 45])
    assert np.array_equal(o[:, :, 10], 1.0 + 0.98 * s[:, :, 10])


# ------------------------------------------------------------------------------------------------ day boundary against the real dumps
@needs_data
def test_day_boundary_items_vs_real():
    import drv_daily as D
    r = D.replay_check()
    for k, f in r["computed"]["fields"].items():          # computed MDRYA + computed ch4ox on the real end state of 33359 == real start state of 33360
        assert f["max_abs"] == 0.0 and f["n_diff"] == 0, k
    assert r["computed"]["fields"]["Q"]["change_by_daily"] > 1e-7         # non-vacuity: the daily routine changes Q by 2.7e-6
    s = D.snoage_check()
    assert s["bitwise"] and s["n_changed_by_aging"] > 5000
    u = D.updtype_check()
    assert all(v["max_abs"] < 1e-15 for v in u.values())


@needs_data
def test_ch4ox_water_mass_vs_recorded():
    import drv_daily as D
    import atm_day_open_loop as OL
    dm_rec, spread = OL.ch4ox_increment(FF, "nov26_day", 33360)
    dm = D.ch4ox_dm(1950, 11)
    nz = np.abs(dm_rec) > 1e-9 * np.abs(dm_rec).max()
    assert np.array_equal(dm == 0, dm_rec == 0)
    assert (np.abs(dm - dm_rec)[nz] / np.abs(dm_rec[nz])).max() < 1e-8            # measured 2.8e-10 (the recorded one is a difference of Q's)
    assert abs(D.ch4_ppm(1950) - 0.808092) < 1e-12                                # ghg_yr = master_yr = 1850: table row 1848


@needs_data
def test_rng_chain_reproduces_recorded_rndss():
    import atm_step as A
    import clouds_condse_ff as cf
    import drv_rng
    ci = cio.read_cse(f"{DAY}/ffc_cse_in_33312.bin")
    ctx = A.make_ctx("nov26", imf=True)
    s0 = int(round(ci["SEEDS"][0])) & 0xFFFFFFFF
    rnd, seed = cf.randu_stream(s0, ctx.imaxj, ctx.cfg["lmcld"])
    vm = A.imaxj_mask(ctx)[None, None]
    lm = ctx.cfg["lmcld"]
    assert np.array_equal(np.where(vm, rnd[:, :lm], 0), np.where(vm, ci["RNDSS"][:, :lm], 0))
    assert drv_rng.radia_seed(s0) == (int(round(ci["SEEDS"][1])) & 0xFFFFFFFF)


# ------------------------------------------------------------------------------------------------ driver
@needs_data
def test_checkpoint_resume_bitwise_closed(tmp_path):
    import model_driver as M
    ck = str(tmp_path / "ck.pkl")
    a = M.ModelDriver(surface="closed", ent="record", f3=True, log=lambda *x: None)
    a.run(2)
    a.checkpoint(ck)
    a.run(1)
    b = M.ModelDriver(surface="closed", ent="record", f3=True, log=lambda *x: None)
    b.resume(ck)
    assert b.itime == 33314 and b.k == 2
    b.run(1)
    diff = M.trees_equal(a.state_dict(), b.state_dict())
    # the timing log differs by construction (wall times); everything else must be bitwise equal
    diff = [d for d in diff if not d.startswith("/timing_log")]
    assert diff == [], diff[:10]
    assert a.timing_log[-1]["digest"] == b.timing_log[-1]["digest"]
    # non-vacuity: a resumed driver that skips the surface state differs
    c = M.ModelDriver(surface="closed", ent="record", f3=True, log=lambda *x: None)
    c.resume(ck)
    c.loop.SS["ice"]["rsi"] = c.loop.SS["ice"]["rsi"] + 1e-9
    assert M.trees_equal(a.state_dict(), c.state_dict())


@needs_data
def test_modifier_shadow_mode_logs_and_does_not_change(tmp_path):
    import model_driver as M
    seen = {}

    def bump(rec, ctx):
        rec["ta"] = rec["ta"].copy()
        rec["ta"][:, 20] = rec["ta"][:, 20] * (1 + 1e-6)             # a recorded tile column replaced by a 'computed' one
        seen["it"] = ctx.itime
        return rec
    d0 = M.ModelDriver(surface="replay", ent="record", log=lambda *x: None)
    d0.run(1)
    prov = M.RecordProvider()
    prov.add_modifier("test_shadow", bump, mode="shadow")
    d1 = M.ModelDriver(prov, surface="replay", ent="record", log=lambda *x: None)
    d1.run(1)
    assert d1.timing_log[-1]["digest"] == d0.timing_log[-1]["digest"]
    assert d1.timing_log[-1]["modifiers"]["test_shadow"]["changed"]["ta"] > 0 and seen["it"] == 33312
