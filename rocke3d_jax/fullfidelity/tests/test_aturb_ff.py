"""Tests for the full-fidelity ATURB port against REAL-Fortran dumps.

Data: ff_data/nov26_steps0-1 (ffa_*_in/out.bin + ffa_consts.txt), produced by the
instrumented ModelE (see fullfidelity/instrumentation/build_and_run.md). Tests are
skipped when the data are absent. Run in its own process (x64 is enabled on import):

    JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest fullfidelity/tests -q
"""
import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATA = f"{FF}/nov26"
CASES = [("nov26", "33312_c1"), ("nov26", "33313_c2"), ("dec01", "33552_c1"), ("dec01", "33553_c2"),
         ("jan01", "17520_c1"), ("jan01", "17521_c2")]
have = os.path.exists(f"{DATA}/ffa_33312_c1_in.bin") and os.path.exists(f"{FF}/jan01/ffa_17520_c1_in.bin")
pytestmark = pytest.mark.skipif(not have, reason="real-Fortran ATURB dumps not available")

import aturb_ff as A          # noqa: E402  (enables x64)
import aturb_compare as K     # noqa: E402


def consts():
    d = {}
    for line in open(f"{DATA}/ffa_consts.txt"):
        k, v = line.split()
        d[k] = float(v)
    return d


def test_constants_match_running_model():
    c = consts()
    for k, mine in dict(grav=A.GRAV, rgas=A.RGAS, deltx=A.DELTX, teeny=A.TEENY, sha=A.SHA, mb2kg=A.MB2KG,
                        psf=A.PSF, pmtop=A.PMTOP, kappa=A.KAPPA, zgs=A.ZGS, prt=A.PRT, b1=A.B1,
                        d1=A.D1, d2=A.D2, d3=A.D3, d4=A.D4, d5=A.D5, s0=A.S0, s1=A.S1, s2=A.S2,
                        s4=A.S4, s5=A.S5, s6=A.S6, s7=A.S7, s8=A.S8, c1=A.C1, c2=A.C2, c3=A.C3,
                        c4=A.C4, c5=A.C5, rimax=A.RIMAX, gm_at_rimax=A.GM_AT_RIMAX,
                        ghmin=A.GHMIN, ghmax=A.GHMAX, emin=A.EMIN, emax=A.EMAX, k_max=A.K_MAX,
                        kmmin=A.KMMIN, khmin=A.KHMIN, ustar_min=A.USTAR_MIN,
                        lmonin_min=A.LMONIN_MIN, lmonin_max=A.LMONIN_MAX).items():
        assert mine == pytest.approx(c[k], rel=1e-14, abs=0), k


@pytest.mark.parametrize("case,tag", CASES)
def test_tq_e_pbl_match_fortran_to_rounding(case, tag):
    rows = K.run(f"{FF}/{case}/ffa_{tag}_in.bin", f"{FF}/{case}/ffa_{tag}_out.bin")
    # non-vacuity: the Fortran routine actually changed the state
    assert rows["t"]["fortran_change_rms"] > 1e-3
    assert rows["e"]["fortran_change_rms"] > 1e-2
    # rounding-level agreement (observed max 7e-13 K on T~137 K rms, 9e-13 m on PBL height)
    assert rows["t"]["max_abs"] < 1e-10
    assert rows["q"]["max_abs"] < 1e-15
    assert rows["e"]["max_abs"] < 1e-12
    assert rows["w2"]["max_abs"] < 1e-12
    assert rows["pblht"]["max_abs"] < 1e-9
    assert rows["pblptop"]["max_abs"] < 1e-9
    assert rows["dclev"]["max_abs"] == 0.0


@pytest.mark.parametrize("case,tag", CASES)
def test_uv_grid_diffusion_and_agrid_winds_match_fortran(case, tag):
    rows = K.run_full(f"{FF}/{case}/ffa_{tag}_in.bin", f"{FF}/{case}/ffa_{tag}_out.bin")
    for k in ("U", "V", "UA", "VA"):
        assert rows[k]["fortran_change_rms"] > 1e-2, k        # non-vacuous: real change
        assert rows[k]["max_abs"] < 1e-10, k                   # observed <=1.4e-14 m/s


def test_geometry_matches_model():
    import aturb_uv_ff as UV
    g = UV.geometry()
    cur, d = None, {}
    for line in open(f"{DATA}/ffa_geom.txt"):
        line = line.strip()
        if not line or line.startswith("im_jm"):
            continue
        if line[0].isalpha():
            cur = line; d[cur] = []
        else:
            d[cur].append(float(line))
    for k in ("rapvs", "rapvn", "cosiv", "siniv"):
        assert np.abs(np.array(d[k]) - g[k]).max() < 1e-15, k


def test_mutations_are_detected():
    """The test above must fail if the physics is wrong (guards against vacuous validation)."""
    import jax
    for name, bad in (("DELTX", 0.608), ("GRAV", 9.81), ("B1", 19.0)):
        old = getattr(A, name)
        try:
            jax.clear_caches()
            setattr(A, name, bad)
            rows = K.run(f"{DATA}/ffa_33312_c1_in.bin", f"{DATA}/ffa_33312_c1_out.bin")
            assert rows["t"]["max_abs"] > 1e-6 or rows["e"]["max_abs"] > 1e-6, name
        finally:
            setattr(A, name, old)
            jax.clear_caches()
