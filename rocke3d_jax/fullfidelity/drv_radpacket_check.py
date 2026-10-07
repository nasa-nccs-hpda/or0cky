"""D176: check of drv_radpacket against the live RADIA input packets rsv_n26_<it>_in.bin of ff_data/nov26_day (11 radiation steps, 33312 + 5k).

Teacher inputs (stated per field group):
  land group (GTEMPR4 BARESW SNOWD FRSNOW) : the ffg record of the PREVIOUS step, second substep (the GHY outputs of the last call before RADIA), as a stand-in for OUR
                                             GHY state; not available for the first step 33312 (no previous record).
  TSAVG, WSAVG                              : composite of the previous step's substep-2 tile records (fft 'composite' column / sum ftype*ws).
  ice / lake / land-ice / ocean-water group : surface state at the start of the step: the restart state (init_surface_state) at 33312; at later radiation steps the state
                                             carried by OUR replay surface loop (state npz written by the capture script), only where such an npz is given.

Usage: python drv_radpacket_check.py [--state <npz of entry states>] [--out <json>]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import radiation_server as rs  # noqa: E402
import drv_radpacket as RP  # noqa: E402

FF = rs.FF
DAY = "nov26_day"
RAD_STEPS = [33312 + 5 * k for k in range(11)]


def live(it):
    return rs.read_packet(f"{FF}/{DAY}/rsv_n26_{it}_in.bin")


def cmp(a, b, sel):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim == 3:
        d = np.stack([np.abs(a[k] - b[k])[sel] for k in range(a.shape[0])])
        nd = int(sum((a[k][sel] != b[k][sel]).sum() for k in range(a.shape[0])))
        n = int(d.size)
        sc = float(np.abs(b[:, sel]).max()) if n else 0.0
    else:
        d = np.abs(a - b)[sel]
        nd = int((a[sel] != b[sel]).sum())
        n = int(d.size)
        sc = float(np.abs(b[sel]).max()) if n else 0.0
    return dict(max=float(d.max()) if n else 0.0, ndiff=nd, n=n, scale=sc)


def land_from_record(it_prev, ff=FF):
    """GHY outputs of the last call of step it_prev (ffg g2) + TSAVG/WSAVG composites."""
    import ghy_compare as GC
    import tile_aggregate_ff as TA
    import drv_radcols_day as D
    g = GC.load(f"{ff}/{DAY}/ffg_{it_prev}.bin")
    n = len(g) // 2
    g2 = g[n:]
    N = len(g2)
    tbcs = g2[:, 245]
    w = np.zeros((N, 7, 2)); nsn = np.zeros((N, 2), int); dz = np.zeros((N, 3, 2)); wsn = np.zeros((N, 3, 2)); fr = np.zeros((N, 2))
    for k in range(N):
        _s, _d, _f, _e, refs, _sm = GC.unpack(g2[k])
        w[k] = refs["w_out"]; nsn[k] = refs["nsn_out"]; fr[k] = refs["fr_snow_out"]
        dz[k] = g2[k, 224:230].reshape(3, 2, order="F"); wsn[k] = g2[k, 230:236].reshape(3, 2, order="F")
    rec = D.records(it_prev, ff)
    ft, _patch, comp = TA.unpack(rec["blk2"])
    tsavg = np.zeros((72, 46))
    tsavg[rec["blk2"][:, 0].astype(int) - 1, rec["blk2"][:, 1].astype(int) - 1] = comp["tsavg"]
    ws = RP.ws_tiles_from_records(rec["blk2"], rec["tb"], rec["lb"], rec["g2"])
    wsavg = RP.wsavg_composite(ft, ws, rec["blk2"][:, :2])
    return g2, tbcs, w, nsn, dz, wsn, fr, tsavg, wsavg


def check_step(it, S, st, snowbv_prev=None, ff=FF, sbv_out=None):
    lv = live(it)
    pole = RP.pole_mask()
    ok = ~pole
    res = {}
    if S is not None:
        f = RP.ice_lake_landice_fields(S, st)
        geo = st["geo"]
        poice = (f["RSI"] * geo["fwater"] > 0) & ok
        fw = (geo["fwater"] > 0) & ok
        fli = (st["flice"] > 0) & ok
        sel = dict(RSI=ok, SNOWI=ok, POND_MELT=ok, FLAG_DSWS=ok, ZSI=poice, ZSNOWI=poice, GTEMPR2=poice, FLAKE=ok, DLAKE=ok, FLICE=ok, FLAND=ok, FEARTH=ok,
                   GTEMPR1=fw, GTEMPR3=fli, SNOWLI=fli)
        for k, s in sel.items():
            res[k] = cmp(f[k], lv[k], s)
    if it == 33312:
        import ghy_compare as GC
        g = GC.load(f"{ff}/{DAY}/ffg_{it}.bin")
        g2 = g[:len(g) // 2]
        ra = RP.restart_land_arrays(f"{ff}/_pristine_restarts/fort1_nov26_itime33312.nc", g2)
        tsavg, wsavg = ra["tsavg"], ra["wsavg"]
        lf, sbv = RP.land_fields(g2, ra["tbcs"], ra["w"], ra["nsn"], ra["dzsn"], ra["wsn"], ra["fr_snow"], ra["snowbv"], update_snowbv=False)
    else:
        g2, tbcs, w, nsn, dz, wsn, fr, tsavg, wsavg = land_from_record(it - 1, ff)
        lf, sbv = RP.land_fields(g2, tbcs, w, nsn, dz, wsn, fr, snowbv_prev)
    if sbv_out is not None:
        sbv_out["sbv"] = sbv
    if True:
        land = np.zeros((72, 46), bool)
        land[g2[:, 0].astype(int) - 1, g2[:, 1].astype(int) - 1] = True
        land &= ok
        fb0 = np.zeros((72, 46), bool)
        fvv = np.zeros((72, 46))
        fvv[g2[:, 0].astype(int) - 1, g2[:, 1].astype(int) - 1] = np.where(g2[:, 169] < 1e-6, 0.0, np.where(g2[:, 169] > 1 - 1e-6, 1.0, g2[:, 169]))
        for k in ("GTEMPR4", "BARESW"):
            res[k] = cmp(lf[k], lv[k], land)
        res["SNOWD"] = cmp(lf["SNOWD"], lv["SNOWD"], land)
        # FRSNOW: bare-soil part only where fb>0 (fb = 0: persistent value, not computable from the last step alone); vegetated part where fv>0
        res["FRSNOW_bare(fb>0)"] = cmp(lf["FRSNOW"][0], lv["FRSNOW"][0], land & (fvv < 1.0))
        res["FRSNOW_bare(fb=0)"] = cmp(lf["FRSNOW"][0], lv["FRSNOW"][0], land & (fvv >= 1.0))
        res["FRSNOW_veg(fv>0)"] = cmp(lf["FRSNOW"][1], lv["FRSNOW"][1], land & (fvv > 0.0))
        res["FRSNOW_veg(fv=0)"] = cmp(lf["FRSNOW"][1], lv["FRSNOW"][1], land & (fvv <= 0.0))
        cells = np.ones((72, 46), bool)
        res["TSAVG"] = cmp(tsavg, lv["TSAVG"], cells & ok)
        res["WSAVG"] = cmp(wsavg, lv["WSAVG"], cells & ok)
    return res


def main(argv):
    state_npz, out = None, None
    a = argv
    while a:
        if a[0] == "--state":
            state_npz = a[1]; a = a[2:]
        elif a[0] == "--out":
            out = a[1]; a = a[2:]
        else:
            raise SystemExit("bad arg")
    import surface_loop as L
    st = L.load_statics("nov26", L.FF)
    st["ctx"] = L.make_ocean_ctx("nov26", L.FF)
    S0 = L.init_surface_state("nov26", L.FF, st=st)
    z = np.load(state_npz) if state_npz else None
    rows = {}
    import ghy_compare as GC
    import netCDF4 as nc
    d0 = nc.Dataset(f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc")
    sbv_carry = np.transpose(np.array(d0.variables["snowbv"][:])[:, :, :2], (2, 1, 0))
    d0.close()
    for it in range(33312, 33366):
        # carry snowbv through every step (needed for the cells with fb = 0 / fv = 0)
        holder = {}
        if it not in RAD_STEPS:
            if it > 33312:
                g2, tbcs, w, nsn, dz, wsn, fr, _ts, _ws = land_from_record(it - 1, FF)
                _lf, sbv_carry = RP.land_fields(g2, tbcs, w, nsn, dz, wsn, fr, sbv_carry)
            continue
        S = None
        if it == 33312:
            S = S0
        elif z is not None and f"ice.rsi_{it}" in z.files:
            S = {g: {k.split(".", 1)[1].rsplit("_", 1)[0]: z[k] for k in z.files if k.startswith(g + ".") and k.endswith(f"_{it}")} for g in ("ice", "lake", "li", "atm")}
        rows[it] = check_step(it, S, st, snowbv_prev=sbv_carry, sbv_out=holder)
        if it > 33312:
            sbv_carry = holder['sbv']
        print(it, {k: (v["max"], v["ndiff"]) for k, v in rows[it].items()}, flush=True)
    if out:
        json.dump(rows, open(out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
