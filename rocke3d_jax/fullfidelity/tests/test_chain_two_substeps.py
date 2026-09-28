"""Tests (D19): two chained NIsurf substeps from real step-start state vs the real substep-2 ATURB exit state,
and every predicted substep-2 PBL input column vs the recorded one. Land (GHY/Ent) stays recorded."""
import os, sys
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = {"nov26": 33312, "dec01": 33552, "jan01": 17520}


def _have(d, it):
    need = [f"fft_{it}.bin", f"ffs_{it}.bin", f"ffp_{it}.bin", f"ffl_{it}.bin"] + \
           [f"ffa_{it}_c{n}_{s}.bin" for n in (1, 2) for s in ("in", "out")]
    return all(os.path.exists(f"{FF}/{d}/{f}") for f in need)


CASES = [(d, it) for d, it in DATES.items() if _have(d, it)]
pytestmark = pytest.mark.skipif(len(CASES) < 3, reason="real-Fortran dumps not available")

import chain_two_substeps as T2   # noqa: E402
import substep_chain as SC        # noqa: E402

_cache = {}


def _run(d, it):
    if (d, it) not in _cache:
        _cache[(d, it)] = T2.run_two_substeps(f"{FF}/{d}", it)
    return _cache[(d, it)]


@pytest.mark.parametrize("d,it", CASES)
def test_predicted_substep2_inputs_match_recorded(d, it):
    _, diag = _run(d, it)
    for k, v in diag.items():
        tol = 1e-6 if k.endswith(".dbl") else 1e-8
        assert v < tol, (k, v)
    assert len(diag) > 40


@pytest.mark.parametrize("d,it", CASES)
def test_two_chained_substeps_match_real_exit_state(d, it):
    rows, _ = _run(d, it)
    for k, r in rows.items():
        scale = r.get("fortran_change_rms", 1.0)
        assert r["max_abs"] < (1e-7 * scale if k != "pblht" else 1e-6), (k, r)
        if k != "pblht":
            assert scale > 1e-6          # non-vacuous: substep 2 really changes the field


def test_mutation_wrong_dbl_is_detected(monkeypatch):
    """A 5% error in get_dbl's PBL depth (which feeds PBL at substep 2) must show up in the final exit state."""
    d, it = CASES[0]
    base, _ = _run(d, it)
    orig = SC.get_dbl

    def bad(*a, **k):
        ug, vg, dbl = orig(*a, **k)
        return ug, vg, dbl * 1.05
    monkeypatch.setattr(SC, "get_dbl", bad)
    rows, _ = T2.run_two_substeps(f"{FF}/{d}", it)
    worst = max(rows[k]["max_abs"] / rows[k]["fortran_change_rms"] for k in ("t", "q", "e", "U", "V"))
    assert worst > 1e3 * max(base[k]["max_abs"] / base[k]["fortran_change_rms"] for k in ("t", "q", "e", "U", "V"))
    assert worst > 1e-6
