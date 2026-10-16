"""D154 tests for the radiation server (radiation_server.py / radiation_server_compare.py).

Skipped when the server binary (scratch build, see instrumentation/ATM_DRV_radsrv.f.patch), the nov26 restart or the
ff_data dumps are absent.  Runtime: about 1 min for the dump-mode run (only if its packets are not already there) +
about 30 s for the oracle + about 1 min for the 4-field audit (concurrent runs).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radiation_server as rs  # noqa: E402

FFN = rs.FF + "/nov26"
HAVE = rs.server_available() and all(
    os.path.exists(FFN + "/" + f) for f in ("ffa_step_33312_r.bin", "ffc_cse_in_33312.bin", "ffc_cse_out_33312.bin",
                                            "ffc_cse_in_33313.bin", "ffs_33312.bin"))
NEED = pytest.mark.skipif(not HAVE, reason="radiation server binary / nov26 restart / ff_data dumps not present")


def _cmp():
    import radiation_server_compare as C
    return C


@NEED
def test_packet_roundtrip(tmp_path):
    st = {k: np.random.RandomState(0).rand(*shp) for k, shp in rs.INPUT_FIELDS.items()}
    p = str(tmp_path / "p.bin")
    rs.write_packet(p, st)
    back = rs.read_packet(p)
    assert list(back) == list(rs.INPUT_FIELDS)
    assert all(np.array_equal(back[k], st[k]) for k in st)


@NEED
def test_oracle_step0_bitwise():
    C = _cmp()
    r = C.oracle(33312, verbose=False)
    assert all(v["equal"] for v in r["input_vs_live"].values()), "existing dumps differ from the live RADIA inputs"
    must = [k for k in r["outputs"] if not k.startswith("AIJ")]
    bad = {k: r["outputs"][k] for k in must if not r["outputs"][k]["equal"]}
    assert not bad, bad
    a = r["outputs"]["AIJ (exact delta) vs dump-mode AIJD (rounded delta)"]
    assert a["max_rel_scaled"] < 1e-9      # difference of the two AIJ deltas is rounding of AIJ_after - AIJ_before
    assert r["wall_s"] < 120


@NEED
def test_audit_inputs_move_outputs():
    C = _cmp()
    res = C.audit(33312, fields=["T", "GTEMPR4", "W_CLOUD", "RQT"], max_parallel=4)
    for k, moved in res["fields"].items():
        assert "SRHR" in moved or "TRHR" in moved or "FSF" in moved, k
    assert "TRHR" in res["fields"]["T"]
    assert "FSF" in res["fields"]["GTEMPR4"] or "TRSURF" in res["fields"]["GTEMPR4"]
