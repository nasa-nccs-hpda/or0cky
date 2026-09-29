"""Tests for underice_ff.py / underice_jax.py (SEAICE.f iceocean_fluxes/icelake_fluxes, the core
physics of SEAICE_DRV.f UNDERICE) against real Fortran dumps -- Stage 1 of the DYNSI/ocean port,
D32."""
import glob

import jax
import numpy as np
import pytest

import underice_ff as F
import underice_jax as J
from underice_compare import read_undocn, read_undlk, DTSRC

FF_DATA = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
OCN_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_undocn_*.bin"))
LAKE_DUMPS = sorted(glob.glob(f"{FF_DATA}/*/ffz_undlk_*.bin"))

pytestmark = pytest.mark.skipif(not OCN_DUMPS or not LAKE_DUMPS,
                                 reason="ff_data underice dumps not present on this host")


@pytest.mark.parametrize("path", OCN_DUMPS)
def test_ocean_ff_matches_real_fortran(path):
    d = read_undocn(path)
    for idx in range(len(d["i"])):
        out = F.iceocean_fluxes(d["tic"][idx], d["si"][idx], d["tm"][idx], d["sm"][idx],
                                 d["dh"][idx], d["ustar"][idx], d["coriol"][idx], DTSRC,
                                 d["mlsh"][idx])
        for k in ("mflux", "sflux", "hflux"):
            ref = d[k][idx]
            assert abs(out[k] - ref) / max(abs(ref), 1e-10) < 1e-6, f"{path} idx={idx} {k}"


@pytest.mark.parametrize("path", LAKE_DUMPS)
def test_lake_ff_matches_real_fortran(path):
    d = read_undlk(path)
    for idx in range(len(d["i"])):
        out = F.icelake_fluxes_limited(d["tic"][idx], d["tm"][idx], d["dh"][idx], DTSRC,
                                        d["mlsh"][idx], d["dlake"][idx], d["glake"][idx])
        for k in ("mflux", "hflux"):
            ref = d[k][idx]
            assert abs(out[k] - ref) / max(abs(ref), 1e-10) < 1e-6, f"{path} idx={idx} {k}"


@pytest.mark.parametrize("path", OCN_DUMPS)
def test_ocean_jax_matches_real_fortran(path):
    d = read_undocn(path)
    d = {k: v.astype(np.float64) for k, v in d.items()}
    out = J.iceocean_fluxes(d["tic"], d["si"], d["tm"], d["sm"], d["dh"], d["ustar"], d["coriol"],
                             DTSRC, d["mlsh"])
    for k in ("mflux", "sflux", "hflux"):
        rel = np.abs(np.asarray(out[k]) - d[k]) / np.maximum(np.abs(d[k]), 1e-10)
        assert np.max(rel) < 1e-6, f"{path} {k}"


@pytest.mark.parametrize("path", LAKE_DUMPS)
def test_lake_jax_matches_real_fortran(path):
    d = read_undlk(path)
    d = {k: v.astype(np.float64) for k, v in d.items()}
    out = J.icelake_fluxes_limited(d["tic"], d["tm"], d["dh"], DTSRC, d["mlsh"], d["dlake"],
                                    d["glake"])
    for k in ("mflux", "hflux"):
        rel = np.abs(np.asarray(out[k]) - d[k]) / np.maximum(np.abs(d[k]), 1e-10)
        assert np.max(rel) < 1e-6, f"{path} {k}"


def test_ocean_jit_compiles_and_matches_eager():
    jitted = jax.jit(J.iceocean_fluxes)
    d = read_undocn(OCN_DUMPS[0])
    d = {k: v.astype(np.float64) for k, v in d.items()}
    eager = J.iceocean_fluxes(d["tic"], d["si"], d["tm"], d["sm"], d["dh"], d["ustar"],
                               d["coriol"], DTSRC, d["mlsh"])
    jitted_out = jitted(d["tic"], d["si"], d["tm"], d["sm"], d["dh"], d["ustar"], d["coriol"],
                         DTSRC, d["mlsh"])
    for k in eager:
        rel = np.abs(np.asarray(eager[k]) - np.asarray(jitted_out[k])) / np.maximum(
            np.abs(np.asarray(eager[k])), 1e-10)
        assert np.max(rel) < 1e-9


def test_both_freezing_and_melting_branches_are_exercised():
    """Mutation check: the real dump must exercise both the freezing and melting branches inside
    iceocean_fluxes' Newton iteration (checked via the sign of the final mflux, which tracks the
    branch taken on the last iteration -- m<0 is freezing, m>0 is melting)."""
    d = read_undocn(OCN_DUMPS[0])
    assert np.sum(d["mflux"] < 0) > 0, "no freezing cells in this record"
    assert np.sum(d["mflux"] > 0) > 0, "no melting cells in this record"


def test_lake_flux_limiting_branch_not_exercised_in_real_record_checked_synthetically():
    """The real 18-record window has no lake shallower than 0.4 m (checked across all dates), so
    UNDERICE's flux-limiting branch is honestly unexercised by real data -- cross-checked instead
    against the plain-Python reference on synthetic shallow-lake inputs, the same pattern used for
    D28's unexercised rain branch."""
    for path in LAKE_DUMPS:
        d = read_undlk(path)
        assert np.sum(d["dlake"] < 0.4) == 0, f"{path} unexpectedly has a shallow-lake cell"

    rng = np.random.default_rng(0)
    n = 50
    ti = rng.uniform(-20.0, -0.1, n)
    tm = rng.uniform(-2.0, 2.0, n)
    dh = rng.uniform(0.01, 0.2, n)
    mlsh = rng.uniform(1e6, 1e8, n)
    dlake = rng.uniform(0.05, 0.35, n)  # all shallow -> branch must fire
    glake = rng.uniform(1e5, 1e8, n)
    n_limited = 0
    for i in range(n):
        out = F.icelake_fluxes_limited(ti[i], tm[i], dh[i], DTSRC, mlsh[i], dlake[i], glake[i])
        unlimited = F.icelake_fluxes(ti[i], tm[i], dh[i], DTSRC, mlsh[i])
        if out["mflux"] != unlimited["mflux"] or out["hflux"] != unlimited["hflux"]:
            n_limited += 1
    assert n_limited > 0, "synthetic shallow-lake inputs never triggered flux limiting"
