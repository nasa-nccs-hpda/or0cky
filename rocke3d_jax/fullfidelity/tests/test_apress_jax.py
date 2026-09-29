"""Tests for apress_ff.py / apress_jax.py (SEAICE_DRV.f CALC_APRESS) against real Fortran dumps --
Stage 1 of the DYNSI/ocean port, D30."""
import glob
import os

import jax
import numpy as np
import pytest

from apress_ff import ACE1I, calc_apress as calc_apress_ff
from apress_jax import calc_apress as calc_apress_jax
from apress_compare import read_apress

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
ALL_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_apress_*.bin"))

pytestmark = pytest.mark.skipif(not ALL_DUMPS, reason="ff_data apress dumps not present on this host")


def _infer_grav(d):
    has_ice = d["rsi"] > 1e-6
    idx = np.argmax(has_ice)
    denom = d["rsi"][idx] * (d["snowi"][idx] + ACE1I + d["msi"][idx])
    return (d["apress"][idx] - 100.0 * (d["srfp"][idx] - 1013.25)) / denom


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_calc_apress_ff_matches_real_fortran(path):
    d = read_apress(path)
    grav = _infer_grav(d)
    mine = calc_apress_ff(d["srfp"], d["rsi"], d["snowi"], d["msi"], grav=grav)
    rel = np.abs(mine - d["apress"]) / np.maximum(np.abs(d["apress"]), 1e-6)
    assert rel.max() < 1e-9


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_calc_apress_jax_matches_real_fortran(path):
    d = read_apress(path)
    grav = _infer_grav(d)
    mine = np.asarray(calc_apress_jax(d["srfp"], d["rsi"], d["snowi"], d["msi"], grav=grav))
    rel = np.abs(mine - d["apress"]) / np.maximum(np.abs(d["apress"]), 1e-6)
    assert rel.max() < 1e-9


def test_jax_matches_ff_on_synthetic_inputs():
    rng = np.random.default_rng(0)
    n = 200
    srfp = rng.uniform(950.0, 1050.0, n)
    rsi = rng.uniform(0.0, 1.0, n)
    snowi = rng.uniform(0.0, 50.0, n)
    msi = rng.uniform(0.0, 500.0, n)
    ref = calc_apress_ff(srfp, rsi, snowi, msi)
    mine = np.asarray(calc_apress_jax(srfp, rsi, snowi, msi))
    assert np.max(np.abs(ref - mine)) < 1e-9


def test_jit_compiles_and_matches_eager():
    jitted = jax.jit(calc_apress_jax)
    rng = np.random.default_rng(1)
    srfp = rng.uniform(950.0, 1050.0, 50)
    rsi = rng.uniform(0.0, 1.0, 50)
    snowi = rng.uniform(0.0, 50.0, 50)
    msi = rng.uniform(0.0, 500.0, 50)
    eager = calc_apress_jax(srfp, rsi, snowi, msi)
    jitted_out = jitted(srfp, rsi, snowi, msi)
    assert np.max(np.abs(np.asarray(eager) - np.asarray(jitted_out))) < 1e-12


def test_apress_is_sensitive_to_all_inputs():
    """Mutation check: every input must actually influence the output."""
    base = calc_apress_ff(1013.25, 0.5, 10.0, 100.0)
    assert calc_apress_ff(1020.0, 0.5, 10.0, 100.0) != base
    assert calc_apress_ff(1013.25, 0.6, 10.0, 100.0) != base
    assert calc_apress_ff(1013.25, 0.5, 20.0, 100.0) != base
    assert calc_apress_ff(1013.25, 0.5, 10.0, 200.0) != base
