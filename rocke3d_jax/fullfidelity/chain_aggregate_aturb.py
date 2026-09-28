"""Track B composition link: surface tile outputs -> tile aggregation -> ATURB fluxes -> ATURB.

Closes the one link in the SURFACE.f per-substep chain that no earlier delta checked as a *composition*:
D4 fed ATURB the model's own recorded entry fluxes, D6 stopped at the ocean/ice tile outputs, and D11
aggregated recorded per-tile fields. Here the ocean and sea-ice patch fields (uflux1, vflux1, dth1, dq1,
tsavg, qsavg) come from OUR PBL `advanc` + tile-flux chain (`surface_chain_ff.run_chain`, no recorded PBL
outputs), are aggregated with OUR `tile_aggregate_ff.aggregate` together with the RECORDED land-ice and
land patches (land is dump-fed for the same reason as D9: Ent is not ported), converted to the ATURB
flux arrays exactly as SURFACE.f:1091-1092 does, and run through OUR `aturb_ff`/`aturb_uv_ff`. The result
is compared with the real ATURB exit state.

Flux conversion (SURFACE.f "UPDATE FIRST LAYER QUANTITIES"): uflux1/vflux1 pass through unchanged,
tflux1 = -dth1*MA(1)/dtsurf, qflux1 = -dq1*MA(1)/dtsurf; tsavg/qsavg are the composite exports. Verified
against the recorded ATURB entry arrays with 0.0 error on every cell (see tests).
"""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_aggregate_ff as TA
import surface_tile_ff as S
import surface_chain_ff as CH
import pbl_compare as PC
import aturb_compare as AC
import aturb_ff as A
import aturb_uv_ff as UV
from ffdump_reader import read_dump

NISURF = 2
XDELT = 0.0     # P2SAoM40 (pbl_ff.py header): TS = TSV/(1+QSRF*xdelt) = TSV


def aturb_flux_arrays(comp, ma1, dtsurf):
    """Composite tile exports -> ATURB flux/state inputs (SURFACE.f:1091-1092 and pass-throughs)."""
    return dict(UFLUX1=comp["uflux1"], VFLUX1=comp["vflux1"],
                TFLUX1=-comp["dth1"] * ma1 / dtsurf, QFLUX1=-comp["dq1"] * ma1 / dtsurf,
                TSAVG=comp["tsavg"], QSAVG=comp["qsavg"])


def _cell_lookup(blk):
    lut = np.full((int(blk[:, 0].max()) + 1, int(blk[:, 1].max()) + 1), -1, int)
    lut[blk[:, 0].astype(int), blk[:, 1].astype(int)] = np.arange(len(blk))
    return lut


def chained_ocean_ice_patches(dd, itime, ns, blk, patch):
    """Overwrite the ocean (k=0) and sea-ice (k=1) patch fields in `patch` (in place) with values computed
    by our PBL + tile-flux chain from the real inputs of substep `ns`. Returns the tile-record count."""
    pbl = PC.load(f"{dd}/ffp_{itime}.bin")
    tiles = S.load(f"{dd}/ffs_{itime}.bin")
    p = pbl[pbl[:, 2] <= 2]
    assert len(p) == len(tiles), "PBL (itype<=2) and SURFACE tile record counts differ"
    sel = tiles[:, 3].astype(int) == ns
    got, pout = CH.run_chain(p[sel], tiles[sel], return_pbl=True)
    t = tiles[sel]
    idx = _cell_lookup(blk)[t[:, 0].astype(int), t[:, 1].astype(int)]
    assert (idx >= 0).all()
    k = t[:, 2].astype(int) - 1
    patch["uflux1"][idx, k] = np.asarray(got["dmua"])
    patch["vflux1"][idx, k] = np.asarray(got["dmva"])
    patch["dth1"][idx, k] = np.asarray(got["dth1"])
    patch["dq1"][idx, k] = np.asarray(got["dq1"])
    patch["tsavg"][idx, k] = np.asarray(pout["tsv"]) / (1.0 + np.asarray(pout["qsrf"]) * XDELT)
    patch["qsavg"][idx, k] = np.asarray(pout["qsrf"])
    return len(t)


def run_to_aturb(dd, itime, ns, chained=True, fluxes_override=None):
    """One NIsurf substep: aggregate -> ATURB flux inputs -> ATURB (A-grid + velocity) vs the real exit state.

    chained=True : ocean+ice patches from our PBL/tile chain; land-ice/land recorded.
    chained=False: all four patches recorded (aggregation + conversion + ATURB only).
    fluxes_override: optional dict of ATURB flux arrays (cell-ordered like `blk`) replacing the composed ones
    (used by the mutation test). Returns (rows, info)."""
    fft = TA.load(f"{dd}/fft_{itime}.bin")
    B = len(fft) // NISURF
    blk = fft[(ns - 1) * B:ns * B]
    ftype, patch, comp_ref = TA.unpack(blk)
    patch = {k: np.array(v) for k, v in patch.items()}
    n_tiles = chained_ocean_ice_patches(dd, itime, ns, blk, patch) if chained else 0
    comp = {k: np.asarray(v) for k, v in TA.aggregate(jnp.asarray(ftype), {k: jnp.asarray(v) for k, v in patch.items()}).items()}

    path_in = f"{dd}/ffa_{itime}_c{ns}_in.bin"
    path_out = f"{dd}/ffa_{itime}_c{ns}_out.bin"
    args, dt, din = AC.load_inputs(path_in)
    dout = read_dump(path_out, 1)
    m = AC.valid_mask(args["T"].shape[:2])
    for k in ("UFLUX1", "VFLUX1", "TFLUX1", "QFLUX1", "TSAVG", "QSAVG"):
        args[k] = np.where(m, args[k], {"TSAVG": 280.0, "QSAVG": 0.0}.get(k, 1e-3))
    i = blk[:, 0].astype(int) - 1
    j = blk[:, 1].astype(int) - 1
    ma1 = np.asarray(din["MA"])[0, i, j]
    fl = aturb_flux_arrays(comp, ma1, dt) if fluxes_override is None else fluxes_override
    fl_rec = {k: np.asarray(args[k])[j, i] for k in fl}
    for k, v in fl.items():
        a = np.array(args[k])
        a[j, i] = v
        args[k] = a

    res = A.aturb_grid(**{k: jnp.asarray(v) for k, v in args.items()}, dtime=dt)
    geo = UV.geometry()
    U = jnp.asarray(np.transpose(din["U"], (1, 0, 2)))
    V = jnp.asarray(np.transpose(din["V"], (1, 0, 2)))
    Un, Vn = UV.diffuse_uv(U, V, res["uflxa"], res["vflxa"], res["km"], res["uw_nl"], res["vw_nl"],
                           res["rho"], res["rhoe"], res["dz"], res["dze"], dt, geo)
    ua, va = UV.recalc_agrid_uv(Un, Vn, geo)
    got = {"t": res["t"], "q": res["q"], "e": res["e"], "pblht": res["pblht"], "U": Un, "V": Vn, "UA": ua, "VA": va}
    ref = {"t": np.transpose(dout["T"], (1, 0, 2)), "q": np.transpose(dout["Q"], (1, 0, 2)),
           "e": np.transpose(dout["EGCM"], (2, 1, 0)), "pblht": dout["PBLHT"].T,
           "U": np.transpose(dout["U"], (1, 0, 2)), "V": np.transpose(dout["V"], (1, 0, 2)),
           "UA": np.transpose(dout["UALIJ"], (2, 1, 0)), "VA": np.transpose(dout["VALIJ"], (2, 1, 0))}
    ini = {"t": np.transpose(din["T"], (1, 0, 2)), "q": np.transpose(din["Q"], (1, 0, 2)),
           "e": np.transpose(din["EGCM"], (2, 1, 0)), "U": np.transpose(din["U"], (1, 0, 2)),
           "V": np.transpose(din["V"], (1, 0, 2)), "UA": np.transpose(din["UALIJ"], (2, 1, 0)),
           "VA": np.transpose(din["VALIJ"], (2, 1, 0))}
    rows = {}
    for k, r in ref.items():
        g = np.asarray(got[k])
        if k in ("U", "V"):
            mm = np.ones(r.shape, bool)
            mm[0] = False
        else:
            mm = np.broadcast_to(m if r.ndim == 2 else m[..., None], r.shape)
        d = (g - r)[mm]
        row = dict(max_abs=float(np.abs(d).max()), n_exact=float((d == 0).mean()))
        if k in ini:
            row["fortran_change_rms"] = float(np.sqrt(((r - ini[k])[mm] ** 2).mean()))
        rows[k] = row
    flux_err = {k: float(np.abs(np.asarray(fl[k]) - fl_rec[k]).max()) for k in fl}
    flux_scale = {k: float(np.abs(fl_rec[k]).max()) for k in fl}
    info = dict(n_cells=len(blk), n_tiles_chained=n_tiles, flux_err=flux_err, flux_scale=flux_scale)
    return rows, info


if __name__ == "__main__":
    dd = sys.argv[1]
    itime = int(sys.argv[2])
    for ns in (1, 2):
        for chained in (False, True):
            rows, info = run_to_aturb(dd, itime, ns, chained)
            print(f"ns={ns} chained={chained} tiles={info['n_tiles_chained']} flux_err="
                  + " ".join(f"{k}:{v:.2e}" for k, v in info["flux_err"].items()))
            for k, r in rows.items():
                print(f"   {k:6s} max_abs={r['max_abs']:.3e} exact={r['n_exact']:.2f} "
                      f"change_rms={r.get('fortran_change_rms', float('nan')):.3e}")
