"""Tests for precip_oc_ff.py / precip_oc_jax.py (OCNDYN.f PRECIP_OC) against real Fortran dumps --
Stage 2 (first item) of the DYNSI/ocean port, D33."""
import glob

import jax
import numpy as np
import pytest

from precip_oc_ff import precip_oc_cell as precoc_ff
from precip_oc_jax import precip_oc_cell as precoc_jax
from precip_oc_compare import read_precoc

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
ALL_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_precoc_*.bin"))

pytestmark = pytest.mark.skipif(not ALL_DUMPS, reason="ff_data precoc dumps not present on this host")


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_ff_matches_real_fortran(path):
    d = read_precoc(path)
    for idx in range(len(d["i"])):
        out = precoc_ff(d["focean"][idx], d["oprec"][idx], d["orsi"][idx], d["orunpsi"][idx],
                         d["oeprec"][idx], d["oerunpsi"][idx], d["osrunpsi"][idx], d["dxypo"][idx],
                         d["mo0"][idx], d["g0m0"][idx], d["s0m0"][idx])
        for k, ref_key in (("mo", "mo1"), ("g0m", "g0m1"), ("s0m", "s0m1")):
            ref = d[ref_key][idx]
            assert abs(out[k] - ref) / max(abs(ref), 1e-10) < 1e-9, f"{path} idx={idx} {k}"


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_jax_matches_real_fortran(path):
    d = read_precoc(path)
    out = precoc_jax(d["focean"], d["oprec"], d["orsi"], d["orunpsi"], d["oeprec"], d["oerunpsi"],
                      d["osrunpsi"], d["dxypo"], d["mo0"], d["g0m0"], d["s0m0"])
    for k, ref_key in (("mo", "mo1"), ("g0m", "g0m1"), ("s0m", "s0m1")):
        ref = d[ref_key]
        rel = np.abs(np.asarray(out[k]) - ref) / np.maximum(np.abs(ref), 1e-10)
        assert np.max(rel) < 1e-9, f"{path} {k}"


def test_jit_compiles_and_matches_eager():
    jitted = jax.jit(precoc_jax)
    d = read_precoc(ALL_DUMPS[0])
    d = {k: v.astype(np.float64) for k, v in d.items()}
    eager = precoc_jax(d["focean"], d["oprec"], d["orsi"], d["orunpsi"], d["oeprec"],
                        d["oerunpsi"], d["osrunpsi"], d["dxypo"], d["mo0"], d["g0m0"], d["s0m0"])
    jitted_out = jitted(d["focean"], d["oprec"], d["orsi"], d["orunpsi"], d["oeprec"],
                         d["oerunpsi"], d["osrunpsi"], d["dxypo"], d["mo0"], d["g0m0"], d["s0m0"])
    for k in eager:
        assert np.max(np.abs(np.asarray(eager[k]) - np.asarray(jitted_out[k]))) < 1e-9


def test_precip_oc_is_sensitive_to_all_inputs():
    """Mutation check: every input must actually influence at least one output."""
    base = precoc_ff(1.0, 0.01, 0.3, 0.005, 1e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5)
    variants = [
        precoc_ff(0.8, 0.01, 0.3, 0.005, 1e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.02, 0.3, 0.005, 1e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.4, 0.005, 1e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.3, 0.010, 1e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.3, 0.005, 2e5, 2e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.3, 0.005, 1e5, 4e5, 0.002, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.3, 0.005, 1e5, 2e5, 0.004, 1e10, 1e8, 1e13, 1e5),
        precoc_ff(1.0, 0.01, 0.3, 0.005, 1e5, 2e5, 0.002, 2e10, 1e8, 1e13, 1e5),
    ]
    for v in variants:
        assert v != base
