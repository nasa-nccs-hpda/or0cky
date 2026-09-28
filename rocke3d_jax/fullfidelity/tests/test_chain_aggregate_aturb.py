"""Tests: composition link  ocean/ice tiles (our PBL+tile chain) + recorded land-ice/land -> our tile
aggregation -> SURFACE.f flux conversion -> our ATURB, vs the real ATURB exit state (D18).

Also checks the conversion identities against the recorded ATURB *entry* arrays exactly, and that the test
is not vacuous (a perturbed flux input or a dropped patch makes ATURB visibly wrong)."""
import os, sys, glob
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = {"nov26": 33312, "dec01": 33552, "jan01": 17520}


def _have(d, it):
    need = [f"fft_{it}.bin", f"ffs_{it}.bin", f"ffp_{it}.bin"] + [f"ffa_{it}_c{n}_{s}.bin" for n in (1, 2) for s in ("in", "out")]
    return all(os.path.exists(f"{FF}/{d}/{f}") for f in need)


CASES = [(d, it, ns) for d, it in DATES.items() if _have(d, it) for ns in (1, 2)]
pytestmark = pytest.mark.skipif(len(CASES) < 6, reason="real-Fortran fft/ffs/ffp/ffa dumps not available")

import chain_aggregate_aturb as C   # noqa: E402
import tile_aggregate_ff as TA      # noqa: E402
from ffdump_reader import read_dump  # noqa: E402

_cache = {}


def _run(d, it, ns, chained=True):
    key = (d, it, ns, chained)
    if key not in _cache:
        _cache[key] = C.run_to_aturb(f"{FF}/{d}", it, ns, chained)
    return _cache[key]


@pytest.mark.parametrize("d,it,ns", CASES)
def test_flux_conversion_matches_recorded_aturb_inputs_exactly(d, it, ns):
    """SURFACE.f:1091-1092: the composite -> ATURB flux arrays are an exact algebraic map (0.0 error)."""
    fft = TA.load(f"{FF}/{d}/fft_{it}.bin")
    B = len(fft) // C.NISURF
    blk = fft[(ns - 1) * B:ns * B]
    ftype, patch, comp_ref = TA.unpack(blk)
    din = read_dump(f"{FF}/{d}/ffa_{it}_c{ns}_in.bin", 1)
    i, j = blk[:, 0].astype(int) - 1, blk[:, 1].astype(int) - 1
    fl = C.aturb_flux_arrays(comp_ref, np.asarray(din["MA"])[0, i, j], float(din["header"][0]))
    for k, v in fl.items():
        assert np.array_equal(np.asarray(v), np.asarray(din[k])[i, j]), k
    assert (np.abs(comp_ref["dth1"]) > 0).mean() > 0.5      # non-vacuous: fluxes are genuinely nonzero


@pytest.mark.parametrize("d,it,ns", CASES)
def test_recorded_patches_through_aggregation_and_aturb_match_fortran(d, it, ns):
    rows, info = _run(d, it, ns, chained=False)
    assert info["n_tiles_chained"] == 0
    for k, e in info["flux_err"].items():
        assert e <= 1e-10 * max(info["flux_scale"][k], 1e-30), k
    for k, r in rows.items():
        scale = r.get("fortran_change_rms", 1.0)
        assert r["max_abs"] < (1e-7 * scale if k != "pblht" else 1e-7), (k, r)


@pytest.mark.parametrize("d,it,ns", CASES)
def test_chained_ocean_ice_through_aggregation_and_aturb_match_fortran(d, it, ns):
    """Ocean/ice patches from OUR PBL + tile fluxes (no recorded PBL/tile outputs), aggregated and run through
    OUR ATURB: final T/Q/TKE/PBL height/U/V agree with the real exit state at roundoff."""
    rows, info = _run(d, it, ns, chained=True)
    assert info["n_tiles_chained"] > 3000                    # both ocean and ice tiles genuinely chained
    for k, e in info["flux_err"].items():
        assert e <= 1e-8 * max(info["flux_scale"][k], 1e-30), (k, e)
    for k, r in rows.items():
        scale = r.get("fortran_change_rms", 1.0)
        assert r["max_abs"] < (1e-7 * scale if k != "pblht" else 1e-6), (k, r)
        if k != "pblht":
            assert scale > 1e-6                              # non-vacuous: the step really changes the field


def test_mutation_perturbed_flux_is_detected():
    """0.1% error in the conversion changes T by far more than the pass tolerance -> the check can fail."""
    d, it, ns = CASES[0]
    rows0, info0 = _run(d, it, ns, chained=False)
    fft = TA.load(f"{FF}/{d}/fft_{it}.bin")
    B = len(fft) // C.NISURF
    blk = fft[(ns - 1) * B:ns * B]
    _, _, comp_ref = TA.unpack(blk)
    din = read_dump(f"{FF}/{d}/ffa_{it}_c{ns}_in.bin", 1)
    i, j = blk[:, 0].astype(int) - 1, blk[:, 1].astype(int) - 1
    fl = C.aturb_flux_arrays(comp_ref, np.asarray(din["MA"])[0, i, j], float(din["header"][0]))
    fl["TFLUX1"] = fl["TFLUX1"] * 1.001
    rows, _ = C.run_to_aturb(f"{FF}/{d}", it, ns, chained=False, fluxes_override=fl)
    assert rows["t"]["max_abs"] > 1e3 * max(rows0["t"]["max_abs"], 1e-13)
    assert rows["t"]["max_abs"] > 1e-8


def test_mutation_dropped_land_patch_is_detected():
    """Removing the (recorded) land patch from the aggregation must be visible in the ATURB result."""
    d, it, ns = CASES[0]
    fft = TA.load(f"{FF}/{d}/fft_{it}.bin")
    B = len(fft) // C.NISURF
    blk = fft[(ns - 1) * B:ns * B].copy()
    _, _, comp_ref = TA.unpack(blk)
    din = read_dump(f"{FF}/{d}/ffa_{it}_c{ns}_in.bin", 1)
    ftype, patch, _ = TA.unpack(blk)
    ft2 = np.array(ftype); ft2[:, 3] = 0.0                   # drop land
    comp = {k: np.sum(np.asarray(patch[k]) * ft2, axis=-1) for k in TA.FIELDS}
    comp["tsavg"], comp["qsavg"] = comp_ref["tsavg"], comp_ref["qsavg"]   # keep states finite; drop land from the fluxes
    i, j = blk[:, 0].astype(int) - 1, blk[:, 1].astype(int) - 1
    fl = C.aturb_flux_arrays(comp, np.asarray(din["MA"])[0, i, j], float(din["header"][0]))
    rows, _ = C.run_to_aturb(f"{FF}/{d}", it, ns, chained=False, fluxes_override=fl)
    assert rows["t"]["max_abs"] > 1e-6
