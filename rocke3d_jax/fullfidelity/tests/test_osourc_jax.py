"""Tests for osourc_ff.py / osourc_jax.py (OCNDYN.f OSOURC, called from GROUND_OC) against real
Fortran dumps -- Stage 2 of the DYNSI/ocean port, D34."""
import glob

import jax
import numpy as np
import pytest

from osourc_ff import osourc as osourc_ff, LMO, LSRPD, FSR, FSRZ
from osourc_jax import osourc as osourc_jax
from osourc_compare import read_osourc

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
ALL_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_osourc_*.bin"))

pytestmark = pytest.mark.skipif(not ALL_DUMPS, reason="ff_data osourc dumps not present on this host")

FIELDS = ("mo", "s0m", "dmoo", "deoo", "dmoi", "deoi", "dsoo", "dsoi")


def test_init_solar_derivation():
    """LSRPD/FSR/FSRZ are derived analytically (OCEAN_COM.f init_solar, L13 layering) -- this is
    proven correct implicitly by every real-record test below matching exactly, but also sanity-
    checked directly here: LSRPD=3 for this rundeck's L13/ZMAX_SOLAR=92m combination."""
    assert LSRPD == 3
    assert set(FSR) == {1, 2, 3}
    assert set(FSRZ) == {1, 2, 3}
    assert 0.0 < FSR[1] <= 1.0


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_ff_matches_real_fortran(path):
    d = read_osourc(path)
    for idx in range(len(d["i"])):
        out = osourc_ff(d["roice"][idx], d["mo0"][idx], d["g0ml0"][idx], d["gzml0"][idx],
                         d["so0"][idx], d["dxypj"][idx], d["bydxypj"][idx], int(d["lmij"][idx]),
                         d["runo"][idx], d["runi"][idx], d["eruno"][idx], d["eruni"][idx],
                         d["sruno"][idx], d["sruni"][idx], (d["srox1"][idx], d["srox2"][idx]))
        for k, ref_key in (("mo", "mo1"), ("s0m", "so1"), ("dmoo", "dmoo"), ("deoo", "deoo"),
                           ("dmoi", "dmoi"), ("deoi", "deoi"), ("dsoo", "dsoo"), ("dsoi", "dsoi")):
            ref = d[ref_key][idx]
            assert abs(out[k] - ref) / max(abs(ref), 1e-10) < 1e-8, f"{path} idx={idx} {k}"
        lmij = int(d["lmij"][idx])
        for k, ref_key in (("g0ml", "g0ml1"), ("gzml", "gzml1")):
            mine = np.array(out[k][:lmij])
            ref = d[ref_key][idx][:lmij]
            rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-10)
            assert np.max(rel) < 1e-8, f"{path} idx={idx} {k}"


@pytest.mark.parametrize("path", ALL_DUMPS)
def test_jax_matches_real_fortran(path):
    d = read_osourc(path)
    d = {k: (v.astype(np.float64) if hasattr(v, "astype") else v) for k, v in d.items()}
    out = osourc_jax(d["roice"], d["mo0"], d["g0ml0"], d["gzml0"], d["so0"], d["dxypj"],
                      d["bydxypj"], d["lmij"], d["runo"], d["runi"], d["eruno"], d["eruni"],
                      d["sruno"], d["sruni"], d["srox1"], d["srox2"])
    for k, ref_key in (("mo", "mo1"), ("s0m", "so1"), ("dmoo", "dmoo"), ("deoo", "deoo"),
                       ("dmoi", "dmoi"), ("deoi", "deoi"), ("dsoo", "dsoo"), ("dsoi", "dsoi"),
                       ("g0ml", "g0ml1"), ("gzml", "gzml1")):
        mine = np.asarray(out[k])
        ref = d[ref_key]
        rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-10)
        assert np.max(rel) < 1e-8, f"{path} {k}"


def test_jit_compiles_and_matches_eager():
    jitted = jax.jit(osourc_jax)
    d = read_osourc(ALL_DUMPS[0])
    d = {k: (v.astype(np.float64) if hasattr(v, "astype") else v) for k, v in d.items()}
    args = (d["roice"], d["mo0"], d["g0ml0"], d["gzml0"], d["so0"], d["dxypj"], d["bydxypj"],
            d["lmij"], d["runo"], d["runi"], d["eruno"], d["eruni"], d["sruno"], d["sruni"],
            d["srox1"], d["srox2"])
    eager = osourc_jax(*args)
    jitted_out = jitted(*args)
    for k in eager:
        # relative, not absolute: G0M-derived fields run ~1e13 (J), where jit-vs-eager fusion-
        # order noise is ordinary float64 rounding, not a correctness issue (same pattern as D27)
        e, j = np.asarray(eager[k]), np.asarray(jitted_out[k])
        rel = np.abs(e - j) / np.maximum(np.abs(e), 1e-10)
        assert np.max(rel) < 1e-9, k


def test_lsr_less_than_lsrpd_branch_is_exercised():
    """Mutation check: the 'shallow column, LSR<LSRPD' edge case (where OSOURC's insolation loop
    range shrinks -- the fix for a real off-by-one found while porting) must actually occur in
    the real 18-record window, not just in theory."""
    found = False
    for path in ALL_DUMPS:
        d = read_osourc(path)
        if np.any(d["lmij"] < LSRPD):
            found = True
            break
    assert found, "no real column ever has LMIJ < LSRPD -- the shallow-column branch is untested"


def test_freezing_branches_are_exercised():
    """Mutation check: both the open-ocean and under-ice freezing (frazil-ice-formation) branches
    must fire on real data (checked via nonzero DMOO/DMOI in the real dump)."""
    d = read_osourc(ALL_DUMPS[0])
    assert np.sum(d["dmoo"] != 0.0) > 0
    assert np.sum(d["dmoi"] != 0.0) > 0
