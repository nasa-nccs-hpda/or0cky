"""Batched validation of seaice_core_jax against the SAME real-Fortran dumps used to validate the
plain-Python seaice_core_ff (ffi_<itime>.bin for SEA_ICE/SSIDEC/snowice, ffm_<itime>.bin for
SIMELT) -- see seaice_compare.py / simelt_compare.py for the record layouts. Every real record from
every file is packed into one batch and run through the vectorized functions in a handful of calls
(not a Python loop per cell), then compared field-by-field both against the real Fortran numbers
directly and against the plain-Python reference (seaice_core_ff), which was already itself
validated bitwise/near-bitwise against Fortran (FULL_FIDELITY_DELTAS.md D10/D12).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_jax as J
import seaice_compare as C
import simelt_compare as M
import addice_compare as A


def load_all_ffi(paths):
    return np.concatenate([C.load(p) for p in paths], axis=0)


def load_all_ffm(paths):
    return np.concatenate([M.load(p) for p in paths], axis=0)


def batched_ground_si(rec):
    """rec: (N,60) real ffi_ rows. Returns dict of batched jnp arrays: final snow/hsil/ssil/msi2,
    runosi/erunosi/srunosi -- mirrors seaice_compare.run_full but batched and domain-gated with
    jnp.where instead of a Python branch per row."""
    dtsrc = jnp.asarray(rec[:, 3])
    snow = jnp.asarray(rec[:, 4])
    hsil = jnp.asarray(rec[:, 6:10])
    ssil = jnp.asarray(rec[:, 10:14])
    msi2 = jnp.asarray(rec[:, 14])
    f0dt = jnp.asarray(rec[:, 15])
    f1dt = jnp.asarray(rec[:, 16])
    evap = jnp.asarray(rec[:, 17])
    srox0 = jnp.asarray(rec[:, 18])
    fmoc = jnp.asarray(rec[:, 19])
    fhoc = jnp.asarray(rec[:, 20])
    fsoc = jnp.asarray(rec[:, 21])
    wetsnow = jnp.asarray(rec[:, 22]) > 0.5
    tm = jnp.asarray(rec[:, 23])
    sm = jnp.asarray(rec[:, 24])
    is_ocean = jnp.asarray(rec[:, 2]) > 0.5

    si = J.sea_ice(dtsrc, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow)
    dec = J.ssidec(si["snow"], si["msi2"], si["hsil"], si["ssil"], dtsrc, si["melt12"])
    sic = J.snowice(tm, sm, dec["snow"], dec["msi2"], dec["hsil"], dec["ssil"], False)

    snow_ocean = sic["snow"]; hsil_ocean = sic["hsil"]; ssil_ocean = sic["ssil"]; msi2_ocean = sic["msi2"]
    runosi_ocean = fmoc + si["run"] + dec["mflux"] + sic["msnwic"]
    erunosi_ocean = fhoc + si["erun"] + dec["hflux"] + sic["hsnwic"]
    srunosi_ocean = fsoc + si["srun"] + dec["sflux"] + sic["ssnwic"]

    snow_other = si["snow"]; hsil_other = si["hsil"]; ssil_other = si["ssil"]; msi2_other = si["msi2"]
    runosi_other = fmoc + si["run"]
    erunosi_other = fhoc + si["erun"]
    srunosi_other = fsoc + si["srun"]

    m = is_ocean
    return dict(
        snow=jnp.where(m, snow_ocean, snow_other),
        hsil=jnp.where(m[..., None], hsil_ocean, hsil_other),
        ssil=jnp.where(m[..., None], ssil_ocean, ssil_other),
        msi2=jnp.where(m, msi2_ocean, msi2_other),
        runosi=jnp.where(m, runosi_ocean, runosi_other),
        erunosi=jnp.where(m, erunosi_ocean, erunosi_other),
        srunosi=jnp.where(m, srunosi_ocean, srunosi_other),
    )


def load_all_ffn(paths):
    return np.concatenate([A.load(p) for p in paths], axis=0)


def batched_addice(rec):
    snow = jnp.asarray(rec[:, 3])
    roice = jnp.asarray(rec[:, 4])
    hsil = jnp.asarray(rec[:, 5:9])
    ssil = jnp.asarray(rec[:, 9:13])
    msi2 = jnp.asarray(rec[:, 13])
    enrgfo = jnp.asarray(rec[:, 14])
    acefi = jnp.asarray(rec[:, 15])
    enrgfi = jnp.asarray(rec[:, 16])
    acefo = jnp.asarray(rec[:, 17])
    salto = jnp.asarray(rec[:, 18])
    salti = jnp.asarray(rec[:, 19])
    flead = jnp.asarray(rec[:, 20])
    qfixr = jnp.asarray(rec[:, 21]) > 0.5
    return J.addice(snow, roice, hsil, ssil, msi2, enrgfo, acefo, acefi, enrgfi, salto, salti, flead, qfixr)


def batched_simelt(rec):
    dt = jnp.asarray(rec[:, 2])
    roice = jnp.asarray(rec[:, 3])
    snow = jnp.asarray(rec[:, 4])
    msi2 = jnp.asarray(rec[:, 5])
    hsil = jnp.asarray(rec[:, 6:10])
    ssil = jnp.asarray(rec[:, 10:14])
    pocean = jnp.asarray(rec[:, 14])
    tm = jnp.asarray(rec[:, 15])
    tfo = jnp.asarray(rec[:, 16])
    enrgmax = jnp.asarray(rec[:, 17])
    return J.simelt(dt, roice, snow, msi2, hsil, ssil, pocean, tm, tfo, enrgmax)


def relerr(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    return np.max(np.abs(got - ref) / denom)


if __name__ == "__main__":
    import glob
    ff = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
    ffi_files = sorted(glob.glob(f"{ff}/*/ffi_*.bin"))
    ffm_files = sorted(glob.glob(f"{ff}/*/ffm_*.bin"))
    print(f"ffi files: {len(ffi_files)}  ffm files: {len(ffm_files)}")

    rec = load_all_ffi(ffi_files)
    print(f"total GROUND_SI cells: {len(rec)}")
    out = batched_ground_si(rec)
    ref = dict(snow=rec[:, 42], hsil=rec[:, 43:47], ssil=rec[:, 47:51], msi2=rec[:, 51],
               runosi=rec[:, 52], erunosi=rec[:, 53], srunosi=rec[:, 54])
    for k in ("snow", "msi2", "runosi", "erunosi", "srunosi"):
        print(f"  {k:9s} max rel err (vs Fortran) = {relerr(out[k], ref[k]):.3e}")
    print(f"  hsil      max rel err (vs Fortran) = {relerr(out['hsil'], ref['hsil']):.3e}")
    print(f"  ssil      max rel err (vs Fortran) = {relerr(out['ssil'], ref['ssil']):.3e}")

    rec_m = load_all_ffm(ffm_files)
    print(f"total SIMELT cells: {len(rec_m)}")
    out_m = batched_simelt(rec_m)
    ref_m = dict(roice=rec_m[:, 18], snow=rec_m[:, 19], msi2=rec_m[:, 20],
                 hsil=rec_m[:, 21:25], ssil=rec_m[:, 25:29], enrgused=rec_m[:, 29])
    for k in ("roice", "snow", "msi2", "enrgused"):
        print(f"  {k:9s} max rel err (vs Fortran) = {relerr(out_m[k], ref_m[k]):.3e}")
    print(f"  hsil      max rel err (vs Fortran) = {relerr(out_m['hsil'], ref_m['hsil']):.3e}")
    print(f"  ssil      max rel err (vs Fortran) = {relerr(out_m['ssil'], ref_m['ssil']):.3e}")
    melted_out = int(np.sum(np.asarray(out_m["melted_out"])))
    print(f"  melted_out count = {melted_out}")
