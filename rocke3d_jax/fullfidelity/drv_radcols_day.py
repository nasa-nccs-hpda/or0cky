"""D176: column-by-column validation of drv_radcols against the 54 real SURFACE records of ff_data/nov26_day.

For every step the radiation-derived columns are COMPUTED (drv_radcols) from
  * the radiation-server outputs of the last radiation step (rsv_n26_<it>_out.bin, steps 33312, 33317, ... 33362),
  * the sea-ice fraction legs of the RESET_SURF_FLUXES calls between steps (sources below),
  * COSZ1 (input; the server output on radiation steps, the recorded COSZ1 (ffa_step_*_r) on the other steps: NOT computed, see D177),
and compared with the columns recorded in ffs/ffp/ffl/ffg of that step.

Sources of the ice-fraction legs (`legs=`):
  'dump' for OCEAN cells: rsisave / rsi(ADVSI entry) / rsi(ADVSI exit) of the real ADVSI dumps ffadv_in/out (advsi_dumps/nov26, 54 steps, bitwise real values).
  lake cells: 'ours' = npz written by lake_legs_capture (the v2 replay surface loop, OUR carried ice state: melt, form_si results), or 'record' = a single
  composite leg between the entry RSI of consecutive records (RSI = ptype(ice)/FLAKE; an approximation, the order melt/form is lost).

Usage (fullfidelity/): python drv_radcols_day.py run <out.json> [--lake ours:<npz>|record|none] [--nsteps 54]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import atm_step as A  # noqa: E402
import radiation_server as rs  # noqa: E402
import drv_radcols as DR  # noqa: E402

FF = rs.FF
DAY = "nov26_day"
IT0 = 33312


class _R:
    pass


def records(it, ff=FF, day=DAY):
    r = _R(); r.ff = ff; r.date = day; r.itime = it
    return A.surface_records(r)


def cosz_record(it, ff=FF, day=DAY):
    return np.array(A.Real(day, it, ff).site("r")["COSZ1"])


def server_out(it, ff=FF, day=DAY):
    return rs.read_packet(f"{ff}/{day}/rsv_n26_{it}_out.bin")


def advsi_legs(it, ff=FF):
    import advsi_ff as AD
    d, o = AD.read_dump(f"{ff}/advsi_dumps/nov26/ffadv_in_{it}.bin", f"{ff}/advsi_dumps/nov26/ffadv_out_{it}.bin")
    return dict(focean=d["focean"] > 0, melt=d["rsisave"], preadv=d["rsi"], end=o["rsi"])


def lake_rsi_from_record(ta):
    """RSI and lake mask of a step from the ffs tile records: RSI = ptype(ice)/FLAKE; lake cell = FOCEAN == 0 and FLAKE > 0."""
    i, j = ta[:, 0].astype(int) - 1, ta[:, 1].astype(int) - 1
    fl = np.zeros((72, 46)); fo = np.zeros((72, 46)); pi = np.zeros((72, 46))
    fl[i, j] = ta[:, 26]; fo[i, j] = ta[:, 27]
    m = ta[:, 2] == 2
    pi[i[m], j[m]] = ta[m, 4]
    return np.where(fl > 0, pi / np.where(fl > 0, fl, 1.0), 0.0), (fo == 0) & (fl > 0)


def _grid(t, c):
    g = np.zeros((72, 46)); g[t[:, 0].astype(int) - 1, t[:, 1].astype(int) - 1] = t[:, c]; return g


def land_trup_vs_inferred(rec, rad):
    """Our land TRUP_in_rad (TRSURF(4)) against the value the old path inferred from the recorded land patch (land_chain.infer_trup; the record has no
    TRUP column for land: this is the only available reference, itself reconstructed from recorded GHY outputs)."""
    import land_chain as LC
    M = A._surf_mods()
    TA = M["TA"]
    g1 = rec["g1"]
    _, patch1, _ = TA.unpack(rec["blk1"])
    lut = {(int(a), int(b)): k for k, (a, b) in enumerate(rec["blk1"][:, :2])}
    idx = np.array([lut[(int(a), int(b))] for a, b in g1[:, :2]])
    old = LC.infer_trup(g1, patch1["dth1"][idx, 3], 900.0)
    new = DR.land_trup(g1, rad)
    d = np.abs(new - old)
    return dict(max=float(d.max()), ndiff=int((new != old).sum()), n=int(d.size), scale=float(np.abs(old).max()))


def compare_step(rec, rad, cosz1, ca):
    """Per column: dict(max abs diff, number of differing values, n, scale, split ocean/lake where meaningful)."""
    out = {}
    new = DR.fill_records(rec, rad, cosz1, ca)

    def cmp(name, a, b, sel=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if sel is not None:
            a, b = a[sel], b[sel]
        d = np.abs(a - b)
        out[name] = dict(max=float(d.max()) if d.size else 0.0, ndiff=int((a != b).sum()), n=int(d.size), scale=float(np.abs(b).max()) if d.size else 0.0)

    for sub, key in (("ffs", "ta"), ("ffs_sub2", "tb")):
        t, n = rec[key], new[key]
        lake = t[:, 27] == 0
        for nm, c in (("srheat", DR.FFS["srheat"]), ("trhr0", DR.FFS["trhr0"]), ("trup_in_rad", DR.FFS["trup"])):
            cmp(f"{sub}.{nm}.ocean_domain", n[:, c], t[:, c], ~lake)
            cmp(f"{sub}.{nm}.lake", n[:, c], t[:, c], lake)
    for sub, key in (("ffp", "pa"), ("ffp_sub2", "pb")):
        p, n = rec[key], new[key]
        lake = np.zeros(len(p), bool)
        # lake cells of ffp: type 1/2 rows whose cell is a lake cell in ffs
        ta = rec["ta"]
        lk = {(int(r[0]), int(r[1])) for r in ta if r[27] == 0}
        lake = np.array([(int(r[0]), int(r[1])) in lk and int(r[2]) <= 2 for r in p])
        for nm, c in (("trhr0", DR.FFP["trhr0"]), ("qsol", DR.FFP["qsol"])):
            cmp(f"{sub}.{nm}.ocean_domain_and_land", n[:, c], p[:, c], ~lake)
            cmp(f"{sub}.{nm}.lake", n[:, c], p[:, c], lake)
    for sub, key in (("ffl", "la"), ("ffl_sub2", "lb")):
        for nm, c in (("srheat", DR.FFL["srheat"]), ("flong", DR.FFL["flong"]), ("trup_in_rad", DR.FFL["trup"])):
            cmp(f"{sub}.{nm}", new[key][:, c], rec[key][:, c])
    for sub, key in (("ffg", "g1"), ("ffg_sub2", "g2")):
        for nm in ("Ca", "cosz1", "vis_rad", "dvis", "srheat", "trheat"):
            c = DR.FFG[nm]
            cmp(f"{sub}.{nm}", new[key][:, c], rec[key][:, c])
    out["land.trup_vs_inferred"] = land_trup_vs_inferred(rec, rad)
    return out


def run(nsteps=54, lake="record", ff=FF, log=print, ice_legs=True, daily_lake=True):
    ca = DR.ghg_ca()
    lk_npz = None
    if lake.startswith("ours:"):
        lk_npz = np.load(lake[5:])
    rad = DR.RadSurf()
    rows = []
    prev_end = None            # RSI at the end of the previous step (ocean domain, from the dumps)
    prev_lake = None
    for k in range(nsteps):
        it = IT0 + k
        lg = advsi_legs(it, ff)
        rec = records(it, ff)
        foc = lg["focean"]
        if prev_end is None:
            prev_end = lg["melt"]
        # (a) melt legs at the head of the step
        if rad.ready and ice_legs:
            rad.ice_leg(prev_end, lg["melt"], foc)
            if lake != "none":
                rsi_k, lkm = lake_rsi_from_record(rec["ta"])
                if lk_npz is not None:
                    rad.ice_leg(lk_npz[f"entry_{it}"], lk_npz[f"melt_{it}"], lkm)
                else:
                    rad.ice_leg(prev_lake[0], rsi_k, lkm)       # composite leg entry(k-1) -> entry(k)
        rsi_k, lkm = lake_rsi_from_record(rec["ta"])
        if A.is_radiation_step(it):
            rad.from_server(server_out(it, ff))
        cz = rad.cosz1_server if A.is_radiation_step(it) else cosz_record(it, ff)
        st = compare_step(rec, rad, cz, ca)
        cz_rec = cosz_record(it, ff)
        st["cosz1_input_vs_record"] = dict(max=float(np.abs(cz - cz_rec).max()), source="server" if A.is_radiation_step(it) else "record")
        rows.append(dict(itime=it, radiation_step=bool(A.is_radiation_step(it)), cols=st))
        worst = max(((v["max"] / v["scale"] if v["scale"] else v["max"]), n) for n, v in st.items() if "ndiff" in v)
        log(f"step {k} it={it} rad={A.is_radiation_step(it)} worst rel {worst[0]:.3g} ({worst[1]})")
        # (b), (c) ocean-domain legs and (F) lake form leg at the end of the step
        if ice_legs:
            rad.ice_leg(lg["melt"], lg["preadv"], foc)
            rad.ice_leg(lg["preadv"], lg["end"], foc)
            if lake != "none" and lk_npz is not None:
                rad.ice_leg(lk_npz[f"melt_{it}"], lk_npz[f"gsi_{it}"], lkm)
        prev_end = lg["end"]
        prev_lake = (rsi_k, lkm)
        if (it + 1) % 48 == 0 and k + 1 < nsteps and daily_lake:
            # day boundary: daily_LAKE changes FLAKE/FEARTH (inputs here: the next record), RESET_SURF_FLUXES for the changes
            import clouds_condse_io as cio
            nxt = records(it + 1, ff)
            rsi_n, lkn = lake_rsi_from_record(nxt["ta"])
            fl = lambda t: _grid(t, 26)
            fe_old = cio.read_cse(f"{ff}/{DAY}/ffc_cse_in_{it}.bin")["FEARTH"]
            fe_new = cio.read_cse(f"{ff}/{DAY}/ffc_cse_in_{it + 1}.bin")["FEARTH"]
            rsi_dl = DR.daily_lake_rsi(fl(rec["ta"]), fl(nxt["ta"]), rsi_k)
            rad.daily_lake(fl(rec["ta"]), fl(nxt["ta"]), rsi_k, rsi_dl, fe_old, fe_new)
            prev_lake = (rsi_dl, lkn)
            prev_end = prev_end
    return rows


def main(argv):
    if len(argv) < 2 or argv[0] != "run":
        print(__doc__); return 2
    out = argv[1]
    lake = "record"
    ns = 54
    a = argv[2:]
    while a:
        if a[0] == "--lake":
            lake = a[1]; a = a[2:]
        elif a[0] == "--nsteps":
            ns = int(a[1]); a = a[2:]
        else:
            raise SystemExit("bad arg " + a[0])
    rows = run(ns, lake)
    json.dump(rows, open(out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
