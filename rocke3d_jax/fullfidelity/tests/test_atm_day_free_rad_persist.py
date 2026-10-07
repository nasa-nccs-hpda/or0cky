"""D162 tests for atm_day_free_rad_persist.py (free-radiation day through the persistent radiation server).

Skipped when the persistent server binary (RADSRVP_SCRATCH / mE_persist), the nov26 restart, the day dumps or the dump-mode packets are absent.
Runtime about 1 min (one server start-up + two calls).  The day itself is not run here; the saved-run test needs ours_free_persist and
ours_free_np in ff_data/nov26_day and skips otherwise.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radiation_server as rs  # noqa: E402
import radiation_server_persist as P  # noqa: E402
import atm_day_free_rad as D  # noqa: E402
import atm_day_free_rad_persist as DP  # noqa: E402

DAY = D.FF + "/" + D.DAYDIR
HAVE = P.server_available("nov26") and all(os.path.exists(DAY + "/" + f) for f in (
    "rsv_n26_33312_in.bin", "rsv_n26_33312_out.bin", "rsv_n26_33317_in.bin", "rsv_n26_33317_out.bin", "ffc_cse_out_33312.bin", "ffc_cse_out_33317.bin"))
NEED = pytest.mark.skipif(not HAVE, reason="persistent radiation server / day dumps / dump-mode packets not present")
HAVE_RUNS = all(os.path.exists(f"{DAY}/ours_{t}/run.json") for t in ("free_persist", "free_np"))


@NEED
def test_seed_matches_entry_seed_and_wrapper_reproduces_live_output():
    srv = DP.PersistentRadiation()
    try:
        for it in (33317, 33312):            # out of order on purpose
            live_in, live_out = D.load_live(it, "in"), D.load_live(it, "out")
            out = srv(live_in, it)
            for k in ("T", "Q", "SRHR", "TRHR", "COSZ1", "CLDSS", "CLDMC", "SNOAGE", "RQT", "KLIQ"):
                assert np.array_equal(out[k], live_out[k]), (it, k)
        assert srv.seeds[33312] == srv.server.seed_entry          # entry seed of the server == SEEDS[1] of the real record
    finally:
        srv.stop()


@pytest.mark.skipif(not HAVE_RUNS, reason="saved days ours_free_persist / ours_free_np not present")
def test_persistent_day_equals_oneshot_day_bitwise():
    s = DP.compare("free_persist", "free_np")["summary"]
    assert s["state"]["n_compared"] > 0 and s["state"]["n_unequal"] == 0, s["state"]
    assert s["rad_in"]["n_unequal"] == 0, s["rad_in"]
    assert s["rad_out"]["n_unequal"] == 0, s["rad_out"]
