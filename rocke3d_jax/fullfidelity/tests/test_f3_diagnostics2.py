"""D172 tests: f3_diagnostics2 accumulators vs the real 54-step nov26 accumulation (real inputs).  Tolerance 1e-12 of each column's own
maximum increment (observed worst 3.2e-14).  The chained-run comparison (run_chained_window, ~2.5 h on one core) is not run here; set
F3_CHAINED_NPZ to a saved chained54.npz to test its bookkeeping."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f3_diagnostics as f3  # noqa: E402
import f3_diagnostics2 as d2  # noqa: E402

DAY = f"{f3.FF}/{f3.DAY}"


def _have():
    try:
        import intel_libm_ff
        if not intel_libm_ff.available():
            return False
    except Exception:
        return False
    need = [f.REAL_ACC if False else f3.REAL_ACC]
    for it in (f3.IT0, f3.IT0 + 53):
        need += [f"{DAY}/ffp_{it}.bin", f"{DAY}/ffs_{it}.bin", f"{DAY}/ffl_{it}.bin", f"{DAY}/ffg_{it}.bin", f"{DAY}/fft_{it}.bin",
                 f"{DAY}/ffc_cse_out_{it}.bin", f"{DAY}/ffc_cse_in_{it}.bin", f"{DAY}/ffa_step_{it}_r.bin"]
    return all(os.path.exists(p) for p in need)


pytestmark = pytest.mark.skipif(not _have(), reason="real window reference/dumps/libimf not present")


@pytest.fixture(scope="module")
def result():
    real = np.load(f3.REAL_ACC)["daij"]
    acc = d2.run_window2(log=lambda *a: None)
    return acc, real, d2.compare2(acc, real)


def test_all_columns_present_and_exact(result):
    acc, real, cmp = result
    assert set(cmp) == set(d2.COLS2)
    for n, r in cmp.items():
        assert r["real_maxabs"] > 0 or r["maxabs"] == 0, n
        assert r["rel"] < 1e-12, (n, r)


def test_column_names_match_acc_file():
    p = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc"
    if not os.path.exists(p):
        pytest.skip("acc file absent")
    meta = f3.aij_names_from_nc(p)
    for n, c in d2.COLS2.items():
        assert meta[c]["name"].strip().lower() == n.lower(), (n, c, meta[c]["name"])


def test_sampling_counters(result):
    acc = result[0]
    assert acc.idacc[3] == 36


def test_chained_bookkeeping_if_present():
    p = os.environ.get("F3_CHAINED_NPZ")
    if not p or not os.path.exists(p):
        pytest.skip("no chained54.npz")
    z = np.load(p)
    assert list(z["idacc"]) == [54, 11, 36, 4]
