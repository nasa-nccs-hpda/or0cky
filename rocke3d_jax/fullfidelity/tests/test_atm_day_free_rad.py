"""D155-D157 tests for the free-running-radiation one-day run (atm_day_free_rad.py).

Skipped when the server binary, the nov26 restart, the day dumps or the dump-mode packets (ff_data/nov26_day/rsv_n26_*) are absent.
Runtime: about 1 min (one server call for the step-0 plumbing oracle; the day itself is NOT run here).  The saved-run tests additionally need
ff_data/nov26_day/ours_free_np (written by `python atm_day_free_rad.py run --tag free_np`) and skip otherwise.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radiation_server as rs  # noqa: E402
import atm_day_free_rad as D  # noqa: E402
import clouds_condse_io as cio  # noqa: E402

DAY = D.FF + "/" + D.DAYDIR
OURS = DAY + "/ours_free_np"
HAVE = rs.server_available() and all(os.path.exists(DAY + "/" + f) for f in (
    "rsv_n26_33312_in.bin", "rsv_n26_33312_out.bin", "rsv_n26_33362_in.bin", "ffa_step_33312_r.bin", "ffc_cse_in_33312.bin", "ffc_cse_out_33312.bin"))
NEED = pytest.mark.skipif(not HAVE, reason="radiation server / day dumps / dump-mode packets not present")
HAVE_RUN = HAVE and os.path.exists(OURS + "/run.json") and len(
    [f for f in os.listdir(OURS) if f.startswith("rad_out_")]) == 11
NEED_RUN = pytest.mark.skipif(not HAVE_RUN, reason="saved free-radiation run ours_free_np not present")


def test_radiation_step_list():
    assert D.rad_steps() == [33312 + 5 * i for i in range(11)]


@NEED
def test_assembler_on_real_state_equals_live_packet_all_steps():
    """the assembler applied to the REAL state (existing dumps) reproduces the live 52-field packet bitwise at the first and last radiation step."""
    for it in (33312, 33362):
        cin = cio.read_cse(f"{DAY}/ffc_cse_in_{it}.bin")
        cout = cio.read_cse(f"{DAY}/ffc_cse_out_{it}.bin")
        r = cio.read_cse(f"{DAY}/ffa_step_{it}_r.bin")
        live = D.load_live(it, "in")
        S = dict(T=cout["T"], Q=cout["Q"], PK=cin["PK"], PMID=cin["PMID"], PDSIG=cin["PDSIG"], PEDN=cin["PEDN"], MA=r["MA"])
        st, src = D.assemble_packet(S, cout, cout["SNOAGE"], live["RQT"], live["KLIQ"], live)
        pv = D.packet_vs_live(st, live)
        bad = [k for k, v in pv.items() if not v["equal"]]
        assert not bad, (it, bad)
        assert list(st) == list(rs.INPUT_FIELDS)


@NEED
def test_live_outputs_equal_recorded_radiation_all_steps():
    """the dump-mode run of the real model over the day reproduces the recorded real radiation (ffa_step_r) at every radiation step."""
    for it in D.rad_steps():
        r = cio.read_cse(f"{DAY}/ffa_step_{it}_r.bin")
        lo = D.load_live(it, "out")
        for k in ("T", "Q", "SRHR", "TRHR", "COSZ1"):
            assert np.array_equal(lo[k], r[k]), (it, k)


@NEED
def test_step0_server_bitwise_through_the_driver_assembler():
    res = D.check_step0(log=lambda *_: None)
    assert res["n_equal"] == res["n_fields"] == 52
    bad = [k for k, v in res["outputs"].items() if not v]
    assert not bad, bad
    assert max(res["aij_cols_vs_live_aijd_maxabs"].values()) < 1e-9       # exact AIJ delta vs rounded AIJ_after-AIJ_before
    assert res["wall"] < 150


@NEED_RUN
def test_saved_run_consistency():
    run = json.load(open(OURS + "/run.json"))
    assert run["nsteps"] == 54 and len(run["rows"]) == 54 and len(run["calls"]) == 11
    for k, row in enumerate(run["rows"]):
        assert row["radiation_step"] == (k % 5 == 0)
        assert (row["rad_call"] is not None) == (k % 5 == 0)
        assert os.path.exists(f"{OURS}/step_{33312 + k}.npz")
    for c in run["calls"]:
        assert c["T_apply_vs_server"] == 0.0           # radia_apply of the chain == the server's own T update
        assert c["cosz1_vs_recorded"] == 0.0
        assert c["cloudmask_vs_server"] == 0.0         # RADIA's taulim masking of OUR clouds == the server's CLDSS/CLDMC
    c0 = run["calls"][0]
    assert c0["itime"] == 33312 and c0["input_diff_vs_live"]["carried_RQT"] == 0.0
    assert c0["input_diff_vs_live"]["T"] < 1e-10       # step 0: our chained T equals the real state to rounding (D150: 7e-14 rms)
    # step 0 of the free run equals step 0 of the open-loop run to rounding: our chained state is 7e-14 K from the real one (D150), so the server's SRHR/TRHR
    # differ from the recorded ones at the 1e-14 relative level (not bitwise; the bitwise oracle is test_step0_server_bitwise_through_the_driver_assembler)
    if os.path.exists(DAY + "/ours_imf_np/step_33312.npz"):
        a, b = np.load(OURS + "/step_33312.npz"), np.load(DAY + "/ours_imf_np/step_33312.npz")
        for f in ("T", "Q", "U", "V", "P"):
            assert np.abs(a[f] - b[f]).max() < 1e-12 * max(1.0, np.abs(b[f]).max()), f
    # step-0 server output of the saved run equals the live (real) output bitwise where our step-0 state is bitwise real
    o = np.load(OURS + "/rad_out_33312.npz")
    lo = D.load_live(33312, "out")
    assert np.abs(o["SRHR"] - lo["SRHR"]).max() < 1e-9 * max(1.0, np.abs(lo["SRHR"]).max())


@NEED_RUN
def test_saved_run_flux_comparison_and_report_consistent():
    import dyn_glue_io as gio
    axyp = gio.load_g("nov26", D.FF)["axyp"]
    fl = D.flux_comparison(D.FF, D.DAYDIR, OURS, D.IT0, 54, axyp)
    assert len(fl) == 11 and fl[0]["k"] == 0
    assert abs(fl[0]["surf_abs_gmean_diff"]) < 1e-8 and abs(fl[0]["TOA_net_gmean_diff"]) < 1e-8
    if os.path.exists(OURS + "/report.json"):
        rep = json.load(open(OURS + "/report.json"))
        assert len(rep["flux"]) == 11
        assert abs(rep["flux"][5]["surf_abs_gmean_diff"] - fl[5]["surf_abs_gmean_diff"]) < 1e-9
