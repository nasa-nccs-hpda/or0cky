"""
Real-data I/O for the P2SAoM40 physics-only JAX comparison workflow
=====================================================================

Reads the actual NASA GISS ModelE2/ROCKE-3D production run at
P2SAoM40 (72x46 horizontal x 40 vertical levels) so the JAX physics
orchestrator (p2saom40_driver.py) can be driven from real restart
state instead of synthetic data.

Uses netCDF4 directly (not xarray -- xarray's pandas import is broken
in this environment).

All paths default to the live run directory (on the "forest204" node's
filesystem); nothing here writes to that directory. Override with the
P2SAOM40_RUN_DIR / P2SAOM40_TOPO_PATH environment variables when running
from elsewhere (e.g. the discover cluster, which does not mount that
filesystem) -- copy fort.1.nc, PARTIAL.accP2SAoM40.nc, and the topo file
to somewhere reachable first.
"""

import os
import numpy as np
import netCDF4 as nc

RUN_DIR = os.environ.get(
    "P2SAOM40_RUN_DIR",
    "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/huge_space/P2SAoM40",
)
TOPO_PATH = os.environ.get(
    "P2SAOM40_TOPO_PATH",
    "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_input_files/Z72X46N_gas.1_nocasp.nc",
)

IM, JM, LM = 72, 46, 40

# Fields stored (L, J, I) in the restart file -- transposed to (J, I, L) on load
_PROFILE_VARS = ["u", "v", "t", "q", "qcl", "qci"]

# Fields already stored (J, I) -- loaded as-is
_SURFACE_2D_VARS = [
    "p", "tearth", "snowe", "wearth", "fr_sat_ij", "qg_ij",
    "tlake", "mwl", "gml", "flake",
    "rsi_atm", "snowi_atm", "msi_atm",
    "uosurf_icdyn", "vosurf_icdyn",
]

# Fields stored (J, I, L) already -- loaded as-is (L last)
_SURFACE_PROFILE_VARS = ["ma"]

# Fields stored (J, I, 4) -- per sea-ice layer
_SEAICE_4L_VARS = ["hsi_atm", "ssi_atm"]


def load_restart_state(path=None):
    """Load real 72x46x40 restart state from a P2SAoM40 fort.*.nc file.

    Returns a dict of numpy arrays. Profile fields (u,v,t,q,qcl,qci) are
    returned as (JM, IM, LM) -- vertical level last, matching the
    convention expected by drycnv.py / atm_com_jax.py. 2D fields are
    (JM, IM).
    """
    path = path or f"{RUN_DIR}/fort.1.nc"
    state = {}
    with nc.Dataset(path) as ds:
        state["itime"] = int(ds.variables["itime"][...])
        for name in _PROFILE_VARS:
            arr = np.array(ds.variables[name][:], dtype=np.float64)  # (L,J,I)
            state[name] = np.transpose(arr, (1, 2, 0))  # -> (J,I,L)
        for name in _SURFACE_2D_VARS:
            state[name] = np.array(ds.variables[name][:], dtype=np.float64)
        for name in _SURFACE_PROFILE_VARS:
            state[name] = np.array(ds.variables[name][:], dtype=np.float64)  # already (J,I,L)
        for name in _SEAICE_4L_VARS:
            state[name] = np.array(ds.variables[name][:], dtype=np.float64)  # (J,I,4)
    assert state["t"].shape == (JM, IM, LM), state["t"].shape
    assert state["ma"].shape == (JM, IM, LM), state["ma"].shape
    return state


def itime_to_datetime(itime, dtsrc=1800.0):
    """Convert a ModelE internal clock tick (DTsrc-steps since 1/1/1949 00:00
    UTC) to a real calendar datetime. Use this rather than assuming any
    particular restart file's date -- restart files in this run directory
    are not guaranteed to align with any single run's logged start/end.
    """
    import datetime
    base = datetime.datetime(1949, 1, 1)
    return base + datetime.timedelta(seconds=itime * dtsrc)


def load_surface_fractions(topo_path=None):
    """Load real surface-type fractions and lat/lon from the TOPO file.

    Returns dict with focean, flake, fgrnd, fgice (JM, IM) and lat, lon
    (1D, degrees -- the real GISS 72x46 grid, not an approximation).
    """
    topo_path = topo_path or TOPO_PATH
    with nc.Dataset(topo_path) as ds:
        out = {
            "focean": np.array(ds.variables["focean"][:], dtype=np.float64),
            "flake": np.array(ds.variables["flake"][:], dtype=np.float64),
            "fgrnd": np.array(ds.variables["fgrnd"][:], dtype=np.float64),
            "fgice": np.array(ds.variables["fgice"][:], dtype=np.float64),
            "zatmo": np.array(ds.variables["zatmo"][:], dtype=np.float64),
            "lat": np.array(ds.variables["lat"][:], dtype=np.float64),
            "lon": np.array(ds.variables["lon"][:], dtype=np.float64),
        }
    return out


def build_itype(frac, rsi_atm):
    """Derive a single dominant surface type per cell (1=ocean, 2=seaice,
    3=landice, 4=land) from real TOPO fractions + sea-ice concentration.

    Simplification: real ModelE area-weights fluxes across sub-tile
    fractions within a cell; this orchestrator picks one dominant type
    per cell instead (documented limitation, not a bug).
    """
    focean, fgice, fgrnd = frac["focean"], frac["fgice"], frac["fgrnd"]
    is_ocean_dominant = focean > 0.5
    # Land cells: fgice/fgrnd are sub-fractions of (1-focean); pick whichever
    # land sub-type dominates (landice vs bare/vegetated ground).
    itype = np.where(fgice > fgrnd, 3, 4).astype(np.int32)
    itype = np.where(is_ocean_dominant & (rsi_atm > 0.5), 2, itype)
    itype = np.where(is_ocean_dominant & (rsi_atm <= 0.5), 1, itype)
    return itype


# ---------------------------------------------------------------------------
# Accumulated-diagnostics decoding (PARTIAL.accP2SAoM40.nc)
# ---------------------------------------------------------------------------

def _decode_names(char_array):
    """Decode a (N, 30) fixed-width byte-char array into a list of strings."""
    out = []
    for row in char_array:
        chars = [c.decode() if isinstance(c, bytes) else c for c in row]
        out.append("".join(chars).strip())
    return out


def decode_aij(varname, acc_path=None):
    """Decode one named 2D (aij) diagnostic from the real accumulator file
    into physical units, using its embedded scale/denom/ia metadata.

    This is a period-mean (accumulated-over-the-run) field, not a
    per-timestep value -- see FINDINGS.md for why (SUBDD was disabled
    in this rundeck). Useful for qualitative/pattern accuracy checks
    only.
    """
    acc_path = acc_path or f"{RUN_DIR}/PARTIAL.accP2SAoM40.nc"
    with nc.Dataset(acc_path) as ds:
        names = _decode_names(ds.variables["sname_aij"][:])
        if varname not in names:
            raise KeyError(f"{varname!r} not found in sname_aij. Use list_aij_names() to search.")
        idx = names.index(varname)
        raw = np.array(ds.variables["aij"][idx, :, :], dtype=np.float64)
        scale = float(np.array(ds.variables["scale_aij"][idx]))
        ia = int(np.array(ds.variables["ia_aij"][idx]))
        idacc = np.array(ds.variables["idacc"][:], dtype=np.float64)
        denom_idx = int(np.array(ds.variables["denom_aij"][idx]))
        n = idacc[ia] if 0 <= ia < idacc.shape[0] and idacc[ia] > 0 else 1.0
        value = raw * scale / n
        if denom_idx > 0:
            denom_raw = np.array(ds.variables["aij"][denom_idx - 1, :, :], dtype=np.float64)
            denom_scale = float(np.array(ds.variables["scale_aij"][denom_idx - 1]))
            denom_ia = int(np.array(ds.variables["ia_aij"][denom_idx - 1]))
            dn = idacc[denom_ia] if 0 <= denom_ia < idacc.shape[0] and idacc[denom_ia] > 0 else 1.0
            denom = denom_raw * denom_scale / dn
            value = np.divide(value, denom, out=np.zeros_like(value), where=np.abs(denom) > 1e-20)
    return value


def list_aij_names(acc_path=None, contains=None):
    """List available aij short-names (optionally filtered by substring)."""
    acc_path = acc_path or f"{RUN_DIR}/PARTIAL.accP2SAoM40.nc"
    with nc.Dataset(acc_path) as ds:
        names = _decode_names(ds.variables["sname_aij"][:])
    names = [n for n in names if n]
    if contains:
        names = [n for n in names if contains.lower() in n.lower()]
    return names
