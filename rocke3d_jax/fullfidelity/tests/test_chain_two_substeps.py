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


def _full(d, it):
    if (d, it) not in _cache:
        _cache[(d, it)] = T2.run_two_substeps(f"{FF}/{d}", it, return_state=True)
    return _cache[(d, it)]


def _run(d, it):
    return _full(d, it)[:2]


_gs = {}


def _gs_stage(d, it):
    if (d, it) not in _gs:
        _gs[(d, it)] = T2.ground_si_stage(f"{FF}/{d}", it, _full(d, it)[2])
    return _gs[(d, it)]


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


@pytest.mark.parametrize("d,it", CASES)
def test_ground_si_after_two_substeps_matches_real(d, it):
    """GROUND_SI (once per step, after the NS loop) on OUR accumulated ice-tile fluxes: accumulated inputs agree with the
    recorded ones and the outputs are exactly as accurate as with recorded inputs (D10 residuals)."""
    in_err, out, base, gs = _gs_stage(d, it)
    assert in_err["f0dt"] < 1e-3 and in_err["f1dt"] < 1e-3 and in_err["evap"] < 1e-9 and in_err["srox0"] < 1e-6
    for k, v in out.items():
        assert v <= 2 * base[k] + 1e-11, (k, v, base[k])
    assert (gs["rec"][:, 15] != 0).sum() > 100 and (gs["rec"][:, 18] != 0).sum() > 50   # non-vacuous


@pytest.mark.parametrize("d,it", CASES)
def test_ground_lk_after_ground_si_matches_real(d, it):
    """GROUND_LK on OUR open-water accumulators and OUR chained GROUND_SI ice-lake fluxes (lake state recorded)."""
    st = _full(d, it)[2]
    gs = _gs_stage(d, it)[3]
    in_err, out, base = T2.ground_lk_stage(f"{FF}/{d}", it, st, gs)
    assert in_err["fodt"] < 1e-3 and in_err["evapo"] < 1e-9 and in_err["run0"] < 1e-6 and in_err["fidt"] < 1e-6
    for k, v in out.items():
        assert v < 1e-9, (k, v)
    assert base["mlake0"] < 1e-12


def test_mutation_missing_second_substep_is_detected():
    """Dropping substep 2's ice-tile flux from the accumulation must show up in GROUND_SI's recorded inputs."""
    d, it = CASES[0]
    st = dict(_full(d, it)[2])
    r2 = dict(st["r2"]); tile = dict(r2["tile"])
    tile["f0dt"] = np.zeros_like(np.asarray(tile["f0dt"]))
    r2["tile"] = tile; st["r2"] = r2
    in_err, out, base, _ = T2.ground_si_stage(f"{FF}/{d}", it, st)
    assert in_err["f0dt"] > 1e3


@pytest.mark.parametrize("d,it", CASES)
def test_lake_addice_after_ground_lk_matches_real(d, it):
    """FORM_SI/ADDICE for lake cells on OUR chained GROUND_SI state and OUR chained LKSOURC frazil fluxes."""
    st = _full(d, it)[2]
    gs = _gs_stage(d, it)[3]
    lk = T2.ground_lk_stage(f"{FF}/{d}", it, st, gs, return_arrays=True)[3]
    in_err, out, base = T2.form_si_lake_stage(f"{FF}/{d}", it, gs, lk)
    assert in_err["state"] < 1e-3 and in_err["fluxes"] < 1e-3
    for k, v in out.items():
        assert v < 1e-9, (k, v)
    assert np.abs(lk["src"]["acefo"]).max() > 0 or np.abs(lk["src"]["acefi"]).max() > 0   # non-vacuous: frazil ice forms
