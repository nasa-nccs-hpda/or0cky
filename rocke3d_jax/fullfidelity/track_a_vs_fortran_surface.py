"""First delta-ledger row: Track A driver vs the REAL Fortran SURFACE (incl. ATURB).

For each dumped step, feed the Fortran state *just before SURFACE* (pre_surface)
into the Track A driver and compare against the Fortran state *just after
SURFACE* (post_surface). Same inputs, so any difference is Track A's physics.

Usage: JAX_PLATFORMS=cpu python fullfidelity/track_a_vs_fortran_surface.py DUMP_DIR [out.json]
Scope caveat: Track A also runs its own DRYCNV on layers 2..LM (which real
P2SAoM40 never executes) and a layer-1 shortcut instead of ATURB; radiation is
disabled here so only the surface/turbulence part is compared.
"""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffdump_reader import read_dump
import p2saom40_io as io, p2saom40_driver as drv

dump_dir = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None

base = io.load_restart_state(f"{io.RUN_DIR}/fort.1.nc")
frac = io.load_surface_fractions()
itype = io.build_itype(frac, base["rsi_atm"])
sf = drv.build_static_fields(itype)


def to_state(d):
    s = dict(base)
    for nm, key in (("T", "t"), ("Q", "q"), ("U", "u"), ("V", "v")):
        s[key] = np.ascontiguousarray(np.transpose(d[nm], (1, 0, 2)))       # (I,J,L)->(J,I,L)
    s["ma"] = np.ascontiguousarray(np.transpose(d["MA"], (2, 1, 0)))        # (L,I,J)->(J,I,L)
    s["p"] = np.ascontiguousarray(d["P"].T)                                 # (I,J)->(J,I)
    return s


def metrics(a, b):
    d = a - b
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d ** 2))),
                rms_change_fortran=None, corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))


rows = []
files = sorted(glob.glob(f"{dump_dir}/ffd_*_pre_surface.bin"))
for f in files:
    itime = int(os.path.basename(f).split("_")[1])
    pre = read_dump(f)
    post = read_dump(f.replace("pre_surface", "post_surface"))
    s_in, s_ref = to_state(pre), to_state(post)
    dt = io.itime_to_datetime(itime)
    new, _ = drv.run_dtsrc_step(s_in, itype, sf, frac["lat"], frac["lon"], dt, step_index=1, do_radiation=False)
    row = dict(itime=itime)
    for nm, sl in (("T_layer1", np.s_[..., 0]), ("T_layers2_40", np.s_[..., 1:]),
                   ("Q_layer1", np.s_[..., 0]), ("Q_layers2_40", np.s_[..., 1:]),
                   ("U_layer1", np.s_[..., 0]), ("V_layer1", np.s_[..., 0])):
        key = nm[0].lower()
        got, ref, ini = new[key][sl], s_ref[key][sl], s_in[key][sl]
        m = metrics(got, ref)
        # how large is the Fortran change itself (the signal Track A must reproduce)?
        m["rms_change_fortran"] = float(np.sqrt(np.mean((ref - ini) ** 2)))
        m["rms_change_trackA"] = float(np.sqrt(np.mean((got - ini) ** 2)))
        row[nm] = m
    rows.append(row)
    print(f"itime {itime}: " + "  ".join(
        f"{k} rms={row[k]['rms']:.3e} (Fortran change {row[k]['rms_change_fortran']:.3e}, TrackA {row[k]['rms_change_trackA']:.3e})"
        for k in ("T_layer1", "T_layers2_40", "U_layer1", "Q_layer1")))
if out_json:
    json.dump(rows, open(out_json, "w"), indent=1)
