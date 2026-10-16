"""Tests for odhorz_ff.py's D44 SMU/SMV accumulation (OCNDYN2.f's OCEAN_DYN integrated
horizontal mass fluxes, consumed by OFLUXV/OADVT2's tracer advection) against real Fortran
dumps -- Stage 2 of the DYNSI/ocean port, D44. JAX vectorization deliberately deferred, same as
ODHORZ itself (D42's open item)."""
import glob

import numpy as np
import pytest

from odhorz_ff import odhorz, IM, JM, LMO
from odhorz_compare import load_records, load_lmm, load_lmv, load_lmu, load_hocean, FF_DEFAULT
from odhorz_smuv_compare import load_smuv_records, load_smfinal

DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
SMUV_PATHS = sorted(glob.glob(f"{FF_DEFAULT}/*/ffz_odhorz_smuv_*.bin"))

pytestmark = pytest.mark.skipif(not SMUV_PATHS, reason="ff_data odhorz_smuv dumps not present on this host")


def _load(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    records = load_records(f"{ff}/ffz_odhorz_{itime}.bin")
    smuv_records = load_smuv_records(f"{ff}/ffz_odhorz_smuv_{itime}.bin")
    return lmm, lmu, lmv, hocean, records, smuv_records


@pytest.mark.parametrize("date,itime", DATES)
def test_per_call_matches_real_fortran(date, itime):
    lmm, lmu, lmv, hocean, records, smuv_records = _load(date, itime)
    assert len(records) == len(smuv_records) == 5, f"{date}: expected 5 ODHORZ calls"
    for rec, srec in zip(records, smuv_records):
        qeven = srec["xeven"] > 0.5
        _, _, _, _, _, _, smu, smv = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"],
            qeven=qeven, smu0=srec["smu0"], smv0=srec["smv0"])
        assert np.allclose(smu, srec["smu1"], atol=1e-6, rtol=1e-6), f"{date} SMU"
        assert np.allclose(smv, srec["smv1"], atol=1e-6, rtol=1e-6), f"{date} SMV"


@pytest.mark.parametrize("date,itime", DATES)
def test_chained_accumulation_matches_smfinal(date, itime):
    """End-to-end: replay all 5 ODHORZ calls starting from SMU=SMV=0 (the real reset at the top
    of the NOCEAN loop, NOCEAN=1 for this rundeck), using the port's own output as each next
    call's input -- the same way OFLUXV/OADVT2 will consume SMU/SMV downstream."""
    ff = f"{FF_DEFAULT}/{date}"
    lmm, lmu, lmv, hocean, records, smuv_records = _load(date, itime)
    smu_final_real, smv_final_real = load_smfinal(f"{ff}/ffz_smfinal_{itime}.bin")

    smu = np.zeros((IM + 1, JM + 1, LMO + 1))
    smv = np.zeros((IM + 1, JM + 1, LMO + 1))
    for rec, srec in zip(records, smuv_records):
        qeven = srec["xeven"] > 0.5
        _, _, _, _, _, _, smu, smv = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"],
            qeven=qeven, smu0=smu, smv0=smv)

    assert np.allclose(smu, smu_final_real, atol=1e-6, rtol=1e-6), f"{date} chained SMU final"
    assert np.allclose(smv, smv_final_real, atol=1e-6, rtol=1e-6), f"{date} chained SMV final"


@pytest.mark.parametrize("date,itime", DATES)
def test_xeven_pattern_matches_leapfrog_structure(date, itime):
    """Verified from source (D44): of the 5 ODHORZ calls per itime, only the 2 calls inside the
    'do n=1,neven' leapfrog loop with qeven=.true. contribute to SMU/SMV (xeven=1); the 2 initial
    odd-state-init calls and the 1 odd leapfrog-corrector call all have xeven=0."""
    _, _, _, _, _, smuv_records = _load(date, itime)
    xevens = [srec["xeven"] for srec in smuv_records]
    assert sum(x > 0.5 for x in xevens) == 2, f"{date}: expected exactly 2 xeven=1 calls, got {xevens}"
    assert sum(x < 0.5 for x in xevens) == 3, f"{date}: expected exactly 3 xeven=0 calls, got {xevens}"


@pytest.mark.parametrize("date,itime", DATES)
def test_smu_smv_nonvacuous(date, itime):
    """SMU/SMV must actually be nonzero somewhere in the final accumulated field -- confirms real
    mass-flux data flows through, not a silently-zero accumulator."""
    ff = f"{FF_DEFAULT}/{date}"
    itime_int = itime
    smu_final, smv_final = load_smfinal(f"{ff}/ffz_smfinal_{itime_int}.bin")
    assert np.any(smu_final != 0.0), f"{date}: SMU is all zero"
    assert np.any(smv_final != 0.0), f"{date}: SMV is all zero"


def test_qeven_none_preserves_original_signature():
    """Backward-compatible: calling odhorz() without qeven/smu0/smv0 (D42's original signature)
    still returns the original 6-tuple, unaffected by D44's addition."""
    lmm = np.ones((IM + 1, JM + 1), dtype=int)
    lmu = np.ones((IM + 1, JM + 1), dtype=int)
    lmv = np.ones((IM + 1, JM + 1), dtype=int)
    hocean = np.zeros((IM + 1, JM + 1))
    z3 = np.zeros((IM + 1, JM + 1, LMO + 1))
    z2 = np.zeros((IM + 1, JM + 1))
    vbar = np.ones((IM + 1, JM + 1, LMO + 1))
    result = odhorz(lmm, lmu, lmv, hocean, 1800.0,
                     z3, z3, z3, z3, z3, z2,
                     z3, z3, z3, z3, z3, z2,
                     vbar, z3, z3, z3)
    assert len(result) == 6
