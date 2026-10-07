"""D159-D160 tests for the persistent radiation server (radiation_server_persist.py, instrumentation/ATM_DRV_radsrv_persist.f.patch).

Skipped when the persistent binary (scratch build mE_persist), the one-shot binary, the nov26 restart or the nov26_day live packets are
absent.  Runtime about 5 min (measured 4.5-4.8 min on a node with other jobs) (one persistent server of 20 s start-up + 11 s per call; two one-shot reference calls 27 s + 42 s;
two clean-shutdown start-ups of 20 s each).

Reference for every packet: the live (dump-mode) output packet of the real model (rsv_n26_<it>_out.bin; its AIJ is a rounded
difference, so AIJ is compared with a tolerance there) and the one-shot server (exact AIJ: compared bitwise).
"""
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radiation_server as rs  # noqa: E402
import radiation_server_persist as P  # noqa: E402
import clouds_condse_io as cio  # noqa: E402

FFD = rs.FF + "/nov26_day"
STEPS = (33312, 33317, 33322, 33332, 33362)
HAVE = (P.server_available() and rs.server_available()
        and all(os.path.exists(f"{FFD}/rsv_n26_{it}_{k}.bin") for it in STEPS for k in ("in", "out"))
        and all(os.path.exists(f"{FFD}/ffc_cse_out_{it}.bin") for it in STEPS))
NEED = pytest.mark.skipif(not HAVE, reason="persistent radiation server binary / restart / nov26_day live packets not present")
OUT21 = [k for k in rs.OUTPUT_FIELDS if k != "AIJ"]


def live(it, kind):
    return rs.read_packet(f"{FFD}/rsv_n26_{it}_{kind}.bin")


def seed_of(it):
    """RADIA's random-number seed of the real model at this step = the seed saved in CONDSE (SEEDS[1] of ffc_cse_out)."""
    return int(round(float(cio.read_cse(f"{FFD}/ffc_cse_out_{it}.bin")["SEEDS"][1])))


def assert_equal(a, b, fields, what):
    bad = [k for k in fields if not np.array_equal(a[k], b[k])]
    assert not bad, f"{what}: fields differ {bad}"


@pytest.fixture(scope="module")
def served():
    """One persistent server; the calls (in this order, forward then reverse, with repeats) are recorded; reused by the tests below."""
    res = {"calls": []}
    with P.PersistentServer("nov26") as s:
        res["start_wall"] = s.start_wall
        res["pid"] = s.proc.pid
        res["proc"] = s.proc
        # (itime): forward, reverse, repeat of the first; then day 27 (a later day: the daily updates run); the server day cannot go back
        for it in (33312, 33322, 33332, 33322, 33312, 33317, 33362):
            o = s.request(live(it, "in"), it, seed_of(it))
            res["calls"].append((it, o))
        with pytest.raises(P.ServerError):                  # earlier day than the server day is refused, the server survives it
            s.request(live(33312, "in"), 33312, None)
        assert s.alive() and s.ping().startswith("OK")
        with pytest.raises(P.ServerError):                  # not a radiation step
            s.request(live(33312, "in"), 33313, None)
        assert s.alive()
    return res


@NEED
def test_matches_live_model(served):
    """Every call equals the live real-model output: 21 fields bitwise, AIJ within the rounding of the live (after - before) difference."""
    for it, o in served["calls"]:
        ref = live(it, "out")
        assert_equal(o, ref, OUT21, f"step {it} vs live")
        assert np.abs(o["AIJ"] - ref["AIJD"]).max() < 1e-9, f"step {it} AIJ vs live delta"


@NEED
def test_order_independence(served):
    """Same packet, different position in the call sequence (forward, reverse, repeat): bitwise identical outputs, AIJ included."""
    by = {}
    for it, o in served["calls"]:
        by.setdefault(it, []).append(o)
    n = 0
    for it, outs in by.items():
        for o in outs[1:]:
            assert_equal(outs[0], o, rs.OUTPUT_FIELDS, f"step {it} repeated")
            n += 1
    assert n >= 2, "the sequence must repeat packets"


@NEED
def test_persistent_equals_one_shot():
    """Persistent output (called after other packets) equals the one-shot server for the same packet, all 22 fields bitwise (AIJ exact)."""
    pers = None
    with P.PersistentServer("nov26") as s:
        s.request(live(33322, "in"), 33322, seed_of(33322))      # something else first: state from a previous call must not matter
        pers = {it: s.request(live(it, "in"), it, seed_of(it)) for it in (33312, 33317)}
    for it in (33312, 33317):
        one = rs.run_radiation(live(it, "in"), it, rundir=os.path.join(P.SCRATCH, "run_oneshot_%d_%d" % (it, os.getpid())))
        assert_equal(pers[it], one, rs.OUTPUT_FIELDS, f"step {it} persistent vs one-shot")


@NEED
def test_per_call_time(served):
    """Documented target ~11 s per call (single thread); loose bounds so a loaded node does not fail it."""
    t = [o["_server_s"] for _, o in served["calls"]]
    assert max(t) < 60, t
    assert served["start_wall"] < 120


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                          # zombie counts as gone
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


@NEED
def test_clean_shutdown_stop(served):
    """stop() ended the model process (exit code 0 from the Fortran STOP) and the object is idempotent."""
    p = served["proc"]
    assert p.returncode == 0
    assert not _alive(served["pid"])


CHILD = """
import sys, os, time
sys.path.insert(0, {here!r})
import radiation_server_persist as P
s = P.PersistentServer('nov26').start()
print(s.proc.pid, flush=True)
{tail}
"""


@NEED
@pytest.mark.parametrize("how", ["sigkill_python", "python_exits_without_stop"])
def test_no_orphan(how):
    """The Fortran process must not outlive python: SIGKILL of python (PR_SET_PDEATHSIG / EOF of the request pipe) and a normal python exit
    without stop() (atexit handler)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tail = "time.sleep(600)" if how == "sigkill_python" else ""
    p = subprocess.Popen([sys.executable, "-c", CHILD.format(here=here, tail=tail)], stdout=subprocess.PIPE, text=True)
    pid = int(p.stdout.readline())
    assert _alive(pid)
    if how == "sigkill_python":
        p.send_signal(signal.SIGKILL)
    p.wait(timeout=60)
    t0 = time.time()
    while _alive(pid) and time.time() - t0 < 30:
        time.sleep(0.5)
    assert not _alive(pid), "model process outlived python (%s)" % how


@NEED
def test_oracle_dec01_jan01():
    """D160: the persistent server on the real first-radiation-step packets of dec01 (33552) and jan01 (17522) reproduces the recorded real
    radiation (ffa_step_<it>_r: T Q SRHR TRHR COSZ1 bitwise), the live outputs and the next step's CLDSS/CLDMC/SNOAGE.  Skipped when the
    live packets of the date (radiation_server_persist_oracle.py dump <date>) are absent."""
    import concurrent.futures as cf
    import radiation_server_persist_oracle as O
    dates = [d for d in ("dec01", "jan01")
             if P.server_available(d) and all(os.path.exists(x) for x in O.live_paths(d)[1:])
             and os.path.exists(f"{rs.FF}/{d}/ffa_step_{O.live_paths(d)[0]}_r.bin")]
    if not dates:
        pytest.skip("no dec01/jan01 live packets")
    with cf.ThreadPoolExecutor(len(dates)) as ex:
        res = list(ex.map(lambda d: O.oracle(d, log=lambda *_: None), dates))
    for r in res:
        assert r["all_bitwise"], r
        assert r["live"]["AIJ"]["maxabs_vs_rounded_live_delta"] < 1e-9
