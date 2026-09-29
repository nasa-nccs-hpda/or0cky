"""Tests for seaice_to_atmgrid_ff.py / seaice_to_atmgrid_jax.py (SEAICE_DRV.f seaice_to_atmgrid)
against real Fortran dumps -- Stage 1 of the DYNSI/ocean port, D31."""
import glob

import jax
import numpy as np
import pytest

from seaice_to_atmgrid_ff import seaice_to_atmgrid_cell as s2ag_ff
from seaice_to_atmgrid_jax import seaice_to_atmgrid_cell as s2ag_jax
from seaice_to_atmgrid_compare import read_s2ag

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
ALL_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_s2ag_*.bin"))

pytestmark = pytest.mark.skipif(not ALL_DUMPS, reason="ff_data s2ag dumps not present on this host")

ABS_TOL = {"gtemp": 1e-6, "gtemp2": 1e-6, "gtempr": 1e-6}  # degrees C; see seaice_to_atmgrid_compare.py
REL_TOL = {"zsnowi": 1e-8, "zsi": 1e-8, "fwsim": 1e-8}


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_ff_matches_real_fortran(path):
    d = read_s2ag(path)
    for idx in range(0, len(d["i"]), max(1, len(d["i"]) // 200)):  # subsample for speed
        out = s2ag_ff(d["rsi"][idx], d["snowi"][idx], d["msi"][idx], d["hsi1"][idx], d["hsi2"][idx],
                       d["ssi1"][idx], d["ssi2"][idx], d["ssi3"][idx], d["ssi4"][idx])
        for k, tol in ABS_TOL.items():
            assert abs(out[k] - d[k][idx]) < tol, f"{path} idx={idx} {k}"
        for k, tol in REL_TOL.items():
            ref = d[k][idx]
            assert abs(out[k] - ref) / max(abs(ref), 1e-6) < tol, f"{path} idx={idx} {k}"


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_jax_matches_real_fortran(path):
    d = read_s2ag(path)
    out = s2ag_jax(d["rsi"], d["snowi"], d["msi"], d["hsi1"], d["hsi2"], d["ssi1"], d["ssi2"],
                    d["ssi3"], d["ssi4"])
    for k, tol in ABS_TOL.items():
        assert np.max(np.abs(np.asarray(out[k]) - d[k])) < tol, f"{path} {k}"
    for k, tol in REL_TOL.items():
        ref = d[k]
        rel = np.abs(np.asarray(out[k]) - ref) / np.maximum(np.abs(ref), 1e-6)
        assert np.max(rel) < tol, f"{path} {k}"


def test_jit_compiles_and_matches_eager():
    jitted = jax.jit(s2ag_jax)
    d = read_s2ag(ALL_DUMPS[0])
    d = {k: v.astype(np.float64) for k, v in d.items()}  # big-endian dump -> native dtype for JAX
    eager = s2ag_jax(d["rsi"], d["snowi"], d["msi"], d["hsi1"], d["hsi2"], d["ssi1"], d["ssi2"],
                      d["ssi3"], d["ssi4"])
    jitted_out = jitted(d["rsi"], d["snowi"], d["msi"], d["hsi1"], d["hsi2"], d["ssi1"], d["ssi2"],
                         d["ssi3"], d["ssi4"])
    for k in eager:
        assert np.max(np.abs(np.asarray(eager[k]) - np.asarray(jitted_out[k]))) < 1e-9


def test_both_mice1_branches_are_exercised():
    """Mutation check: the dump must exercise BOTH the some-ice-in-layer-1 and some-snow-in-
    layer-2 branches (not a vacuous single-branch pass)."""
    from ice_props_ff import ACE1I
    from seaice_core_ff import XSI
    d = read_s2ag(ALL_DUMPS[0])
    msi1 = d["snowi"] + ACE1I
    n_branch_a = int(np.sum(ACE1I > XSI[1] * msi1))
    n_branch_b = int(np.sum(ACE1I <= XSI[1] * msi1))
    assert n_branch_a > 0
    assert n_branch_b > 0
