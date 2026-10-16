"""Track B: two consecutive NIsurf substeps chained from real step-start state (SURFACE.f `DO NS=1,NIsurf`, D19).

Substep 1 runs on the REAL step-start inputs (recorded PBL/tile/ATURB entry state). Substep 2 is then run on inputs
that are built ONLY from substep 1's results computed by our own code (no recorded substep-2 input is used except
where stated): PBL profiles and cm/ch/cq carried over, the layer-1 atmosphere scalars and get_dbl from our ATURB
exit state (`substep_chain`), ice/land-ice ground state from our tile outputs, e0/evapor accumulated. The final
result (our ATURB exit state after substep 2) is compared with the real one, and every predicted substep-2 input
column is diffed against the recorded value.

Not chained (recorded, stated plainly): the land patch (GHY, needs Ent) at both substeps, its PBL outputs used in the
composite ustar/lmonin for get_dbl, and the ocean tile's own state (unchanged inside the loop in the real model).
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_aggregate_ff as TA
import surface_tile_ff as S
import surface_chain_ff as CH
import landice_tile_ff as LI
import pbl_compare as PC
import pbl_ff as P
import aturb_compare as AC
import aturb_ff as A
import aturb_uv_ff as UV
import substep_chain as SC
import chain_aggregate_aturb as C
import land_chain as LC
from ffdump_reader import read_dump

TF = S.TF


def load_atm(path_in):
    """Atmosphere state + statics in the ATURB (J,I,L) layout from an ffa_*_in dump."""
    args, dt, din = AC.load_inputs(path_in)
    atm = {k: np.array(args[k]) for k in ("T", "Q", "UA", "VA", "E", "PMID", "PEDN", "PK", "PEK1", "PDSIG")}
    atm["U"] = np.transpose(np.asarray(din["U"]), (1, 0, 2))
    atm["V"] = np.transpose(np.asarray(din["V"]), (1, 0, 2))
    atm["MA1"] = np.asarray(din["MA"])[0].T
    return atm, dt


_GEO = None


def _aturb_uv(args, U, V, dt):
    global _GEO
    if _GEO is None:
        _GEO = UV.geometry()
    res = A.aturb_grid(**args, dtime=dt)
    Un, Vn = UV.diffuse_uv(U, V, res["uflxa"], res["vflxa"], res["km"], res["uw_nl"], res["vw_nl"], res["rho"],
                           res["rhoe"], res["dz"], res["dze"], dt, _GEO)
    ua, va = UV.recalc_agrid_uv(Un, Vn, _GEO)
    return res, Un, Vn, ua, va


_aturb_uv_jit = jax.jit(_aturb_uv)   # un-jitted, the scans inside re-trace every call (~2 s per substep)


def run_aturb(atm, fl, cells, dt):
    """fl: dict of per-cell ATURB flux arrays for `cells` (i,j 1-based columns); other columns get the same inert fill
    as aturb_compare. Returns exit-state dict in (J,I,L) layout."""
    i = cells[:, 0].astype(int) - 1
    j = cells[:, 1].astype(int) - 1
    shape = atm["T"].shape[:2]
    m = AC.valid_mask(shape)
    full = {}
    for k in ("UFLUX1", "VFLUX1", "TFLUX1", "QFLUX1", "TSAVG", "QSAVG"):
        a = np.full(shape, {"TSAVG": 280.0, "QSAVG": 0.0}.get(k, 1e-3))
        a[j, i] = np.asarray(fl[k])
        full[k] = a
    args = {k: jnp.asarray(atm[k]) for k in ("T", "Q", "UA", "VA", "E", "PMID", "PEDN", "PK", "PEK1", "PDSIG")}
    args.update({k: jnp.asarray(v) for k, v in full.items()})
    res, Un, Vn, ua, va = _aturb_uv_jit(args, jnp.asarray(atm["U"]), jnp.asarray(atm["V"]), dt)
    return dict(T=res["t"], Q=res["q"], E=res["e"], pblht=res["pblht"], dclev=res["dclev"], U=Un, V=Vn, UA=ua, VA=va,
                m=m)


def next_atm(atm, ex):
    new = dict(atm)
    for k in ("T", "Q", "E", "U", "V", "UA", "VA"):
        new[k] = np.asarray(ex[k])
    return new


def substep(pbl12, tile, pbl3, li, blk, atm, dt, land_patch_rec, land=None):
    """One substep on the given records. Returns dict with our tile/PBL outputs, aggregated fluxes and ATURB exit."""
    ftype, patch, _ = TA.unpack(land_patch_rec)
    patch = {k: np.array(v) for k, v in patch.items()}
    got, pout = CH.run_chain(pbl12, tile, return_pbl=True)
    lut = C._cell_lookup(blk)
    idx = lut[tile[:, 0].astype(int), tile[:, 1].astype(int)]
    k = tile[:, 2].astype(int) - 1
    patch["uflux1"][idx, k] = np.asarray(got["dmua"]); patch["vflux1"][idx, k] = np.asarray(got["dmva"])
    patch["dth1"][idx, k] = np.asarray(got["dth1"]); patch["dq1"][idx, k] = np.asarray(got["dq1"])
    patch["tsavg"][idx, k] = np.asarray(pout["tsv"]); patch["qsavg"][idx, k] = np.asarray(pout["qsrf"])
    gli, pli = C.landice_chain(pbl3, li)
    idl = lut[li[:, 0].astype(int), li[:, 1].astype(int)]
    patch["uflux1"][idl, 2] = np.asarray(gli["uflux1"]); patch["vflux1"][idl, 2] = np.asarray(gli["vflux1"])
    patch["dth1"][idl, 2] = np.asarray(gli["dth1"]); patch["dq1"][idl, 2] = np.asarray(gli["dq1"])
    patch["tsavg"][idl, 2] = np.asarray(pli["tsv"]); patch["qsavg"][idl, 2] = np.asarray(pli["qsrf"])
    lr = None
    if land is not None:
        gi = land["g"][:, 0].astype(int) - 1
        gj = land["g"][:, 1].astype(int) - 1
        lr = LC.land_substep(land["p4"], land["g"], atm["Q"][gj, gi, 0], land["trup"], dt, land.get("dyn"))
        idg = lut[gi + 1, gj + 1]
        for k_ in ("uflux1", "vflux1", "dth1", "dq1", "tsavg", "qsavg"):
            patch[k_][idg, 3] = lr["patch"][k_]
    comp = {k_: np.asarray(v) for k_, v in TA.aggregate(jnp.asarray(ftype), {k_: jnp.asarray(v) for k_, v in patch.items()}).items()}
    i = blk[:, 0].astype(int) - 1
    j = blk[:, 1].astype(int) - 1
    fl = C.aturb_flux_arrays(comp, atm["MA1"][j, i], dt)
    ex = run_aturb(atm, fl, blk, dt)
    return dict(tile=got, pbl=pout, li=gli, pbl_li=pli, comp=comp, ex=ex, ftype=ftype, land=lr)


def predict_ns2(a12, a3, a4, b12, b3, ta, tb, la, lb, r1, blk, atm2, ftype, b_all, b4=None):
    """Predicted substep-2 records (PBL itype<=2, PBL itype 3, tile, land-ice tile) from substep-1 results r1 and the
    substep-2 atmosphere `atm2` (our ATURB exit). Start from the recorded substep-2 rows so that columns we do not model
    (constants, ddml flags, ...) stay as recorded; overwrite every column we predict."""
    e = SC.layer1_exports(atm2["T"], atm2["Q"], atm2["UA"], atm2["VA"], atm2["MA1"], atm2["PEK1"], atm2["PMID"][..., 0])
    # composite ustar/lmonin from substep-1 PBL outputs (land recorded)
    shape = atm2["T"].shape[:2]
    ust = np.zeros(shape); lmo = np.zeros(shape)
    lut = C._cell_lookup(blk)
    ft = r1["ftype"]
    def acc(rows, ustar, lmonin, itp):
        idx = lut[rows[:, 0].astype(int), rows[:, 1].astype(int)]
        j = rows[:, 1].astype(int) - 1; i = rows[:, 0].astype(int) - 1
        np.add.at(ust, (j, i), ft[idx, itp - 1] * ustar)
        np.add.at(lmo, (j, i), ft[idx, itp - 1] * lmonin)
    for itp in (1, 2):
        m = a12[:, 2] == itp
        acc(a12[m], r1["pbl"]["ustar"][m], r1["pbl"]["lmonin"][m], itp)
    acc(a3, r1["pbl_li"]["ustar"], r1["pbl_li"]["lmonin"], 3)
    if r1["land"] is not None:
        acc(a4, r1["land"]["pbl"]["ustar"], r1["land"]["pbl"]["lmonin"], 4)
    else:
        acc(a4, a4[:, 99], a4[:, 100], 4)
    cor = np.zeros(shape)
    cor[b_all[:, 1].astype(int) - 1, b_all[:, 0].astype(int) - 1] = b_all[:, 36]   # per-cell constant (sinlat*omega2)
    ug, vg, dbl = SC.get_dbl(jnp.asarray(ust), jnp.asarray(lmo), jnp.asarray(cor), jnp.asarray(atm2["pblht"]),
                             jnp.asarray(atm2["dclev"]), jnp.asarray(atm2["T"]), jnp.asarray(atm2["Q"]),
                             jnp.asarray(atm2["UA"]), jnp.asarray(atm2["VA"]), jnp.asarray(atm2["PMID"]),
                             jnp.asarray(atm2["PK"]), jnp.asarray(e["ztop"]))
    cellv = dict(utop=e["utop"], vtop=e["vtop"], qtop=e["qtop"], tkv=e["tkv"], zs1=e["zs1"], ztop=e["ztop"],
                 dbl=dbl, ug=ug, vg=vg)
    cols = dict(zs1=5, tkv=7, dbl=29, ug=31, vg=32, utop=37, vtop=38, qtop=39, ztop=40)

    def fill(rec, out_pbl, tg1_out=None, itype_rows=None):
        rec = np.array(rec)
        j = rec[:, 1].astype(int) - 1; i = rec[:, 0].astype(int) - 1
        for nm, c in cols.items():
            rec[:, c] = np.asarray(cellv[nm])[j, i]
        rec[:, 50:58] = out_pbl["u"]; rec[:, 58:66] = out_pbl["v"]; rec[:, 66:74] = out_pbl["t"]
        rec[:, 74:82] = out_pbl["q"]; rec[:, 82:89] = out_pbl["e"]
        rec[:, 33] = out_pbl["cm"]; rec[:, 34] = out_pbl["ch"]; rec[:, 35] = out_pbl["cq"]
        return rec

    p12 = fill(b12, r1["pbl"])
    p3 = fill(b3, r1["pbl_li"])
    p4 = None
    if r1["land"] is not None:
        p4 = LC.next_land_pbl_columns(fill(b4, r1["land"]["pbl"]), r1["land"])
    # ice / land-ice ground state carried from tile outputs (ocean tile state is unchanged in the loop)
    ice = p12[:, 2] == 2
    tg_ice = np.asarray(r1["tile"]["tg1"])[ice] + TF
    tg_li = np.asarray(r1["li"]["tg1"]) + TF
    for rec, sel, tg in ((p12, ice, tg_ice), (p3, slice(None), tg_li)):
        lh = rec[sel, 19]; ps = rec[sel, 16]
        qs = np.asarray(P.qsat(jnp.asarray(tg), jnp.asarray(lh), jnp.asarray(ps)))
        rec[sel, 18] = tg; rec[sel, 6] = tg; rec[sel, 8] = qs; rec[sel, 9] = qs
    # tile records
    tnew = np.array(tb)
    j = tnew[:, 1].astype(int) - 1; i = tnew[:, 0].astype(int) - 1
    tnew[:, S.IN["q1"]] = np.asarray(e["qtop"])[j, i]
    tnew[:, S.IN["thv1"]] = np.asarray(atm2["T"])[j, i, 0] * (1.0 + np.asarray(atm2["Q"])[j, i, 0] * SC.XDELT)
    tnew[:, S.IN["e0"]] = ta[:, S.IN["e0"]] + np.asarray(r1["tile"]["f0dt"])
    tnew[:, S.IN["evapor"]] = ta[:, S.IN["evapor"]] + np.asarray(r1["tile"]["evap"])
    ii = tnew[:, 2] == 2
    tnew[ii, S.IN["tg1"]] = np.asarray(r1["tile"]["tg1"])[ii]
    tnew[ii, S.IN["tg2"]] = np.asarray(r1["tile"]["tg2"])[ii]
    tnew[ii, S.IN["tr4"]] = np.asarray(r1["tile"]["tr4"])[ii]
    lnew = np.array(lb)
    j = lnew[:, 1].astype(int) - 1; i = lnew[:, 0].astype(int) - 1
    lnew[:, LI.IN["q1"]] = np.asarray(e["qtop"])[j, i]
    lnew[:, LI.IN["tg1"]] = np.asarray(r1["li"]["tg1"])
    return p12, p3, tnew, lnew, p4


PRED_COLS = dict(zs1=5, tgv=6, tkv=7, qg_sat=8, qg_aver=9, tg=18, dbl=29, ug=31, vg=32, cm=33, ch=34, cq=35,
                 utop=37, vtop=38, qtop=39, ztop=40)


def run_two_substeps(dd, it, return_state=False, land=False):
    """Returns (rows vs the real substep-2 ATURB exit state, diag of predicted-vs-recorded substep-2 input columns)
    [+ a state dict with both substeps' results if return_state]."""
    p = PC.load(f"{dd}/ffp_{it}.bin"); t = S.load(f"{dd}/ffs_{it}.bin"); l = LI.load(f"{dd}/ffl_{it}.bin")
    fft = TA.load(f"{dd}/fft_{it}.bin")
    n = len(p) // 2; nt = len(t) // 2; nl = len(l) // 2; B = len(fft) // 2
    (pa, pb), (ta, tb), (la, lb) = (p[:n], p[n:]), (t[:nt], t[nt:]), (l[:nl], l[nl:])
    blk1, blk2 = fft[:B], fft[B:]
    a12, a3, a4 = pa[pa[:, 2] <= 2], pa[pa[:, 2] == 3], pa[pa[:, 2] == 4]
    b12, b3 = pb[pb[:, 2] <= 2], pb[pb[:, 2] == 3]
    atm1, dt = load_atm(f"{dd}/ffa_{it}_c1_in.bin")
    land1 = land2 = None
    b4 = pb[pb[:, 2] == 4]
    if land:
        import ghy_compare as GC
        g = GC.load(f"{dd}/ffg_{it}.bin")
        ng = len(g) // 2
        g1, g2 = g[:ng], g[ng:]
        ftype1, patch1, _ = TA.unpack(blk1)
        lut1 = C._cell_lookup(blk1)
        idx1 = lut1[g1[:, 0].astype(int), g1[:, 1].astype(int)]
        trup = LC.infer_trup(g1, patch1["dth1"][idx1, 3], dt)
        land1 = dict(p4=a4, g=g1, trup=trup)
    r1 = substep(a12, ta, a3, la, blk1, atm1, dt, blk1, land1)
    atm2 = next_atm(atm1, r1["ex"])
    atm2["pblht"] = np.asarray(r1["ex"]["pblht"]); atm2["dclev"] = np.asarray(r1["ex"]["dclev"])
    p12, p3, tnew, lnew, p4 = predict_ns2(a12, a3, a4, b12, b3, ta, tb, la, lb, r1, blk1, atm2, r1["ftype"], pb, b4)
    diag = {}
    for nm, c in PRED_COLS.items():
        for lab, pr, rc, sel in (("itype1", p12, b12, b12[:, 2] == 1), ("itype2", p12, b12, b12[:, 2] == 2),
                                 ("itype3", p3, b3, np.ones(len(b3), bool))):
            if sel.sum() == 0:
                continue
            d = np.abs(pr[sel, c] - rc[sel, c])
            diag[f"{lab}.{nm}"] = float(d.max())
    for nm, pr, rc in (("profiles", p12[:, 50:89], b12[:, 50:89]), ("profiles3", p3[:, 50:89], b3[:, 50:89])):
        diag[nm] = float(np.abs(pr - rc).max())
    if land:
        land2 = dict(p4=p4, g=g2, trup=trup, dyn=r1["land"]["dyn_next"])
        for nm, c in dict(zs1=5, tgv=6, tkv=7, qg_sat=8, qg_aver=9, tr4=11, evap_max=12, fr_sat=13, tg=18, dbl=29, ug=31,
                          vg=32, cm=33, ch=34, cq=35, utop=37, vtop=38, qtop=39, ztop=40).items():
            diag[f"itype4.{nm}"] = float(np.abs(p4[:, c] - b4[:, c]).max())
        diag["profiles4"] = float(np.abs(p4[:, 50:89] - b4[:, 50:89]).max())
    r2 = substep(p12, tnew, p3, lnew, blk2, atm2, dt, blk2, land2)
    # compare with the real substep-2 exit
    dout = read_dump(f"{dd}/ffa_{it}_c2_out.bin", 1)
    din2 = read_dump(f"{dd}/ffa_{it}_c2_in.bin", 1)
    ex = r2["ex"]; m = ex["m"]
    got = {"t": ex["T"], "q": ex["Q"], "e": ex["E"], "pblht": ex["pblht"], "U": ex["U"], "V": ex["V"],
           "UA": ex["UA"], "VA": ex["VA"]}
    ref = {"t": np.transpose(dout["T"], (1, 0, 2)), "q": np.transpose(dout["Q"], (1, 0, 2)),
           "e": np.transpose(dout["EGCM"], (2, 1, 0)), "pblht": dout["PBLHT"].T,
           "U": np.transpose(dout["U"], (1, 0, 2)), "V": np.transpose(dout["V"], (1, 0, 2)),
           "UA": np.transpose(dout["UALIJ"], (2, 1, 0)), "VA": np.transpose(dout["VALIJ"], (2, 1, 0))}
    ini = {"t": np.transpose(din2["T"], (1, 0, 2)), "q": np.transpose(din2["Q"], (1, 0, 2)),
           "e": np.transpose(din2["EGCM"], (2, 1, 0)), "U": np.transpose(din2["U"], (1, 0, 2)),
           "V": np.transpose(din2["V"], (1, 0, 2)), "UA": np.transpose(din2["UALIJ"], (2, 1, 0)),
           "VA": np.transpose(din2["VALIJ"], (2, 1, 0))}
    rows = {}
    for k, r in ref.items():
        g = np.asarray(got[k])
        if k in ("U", "V"):
            mm = np.ones(r.shape, bool); mm[0] = False
        else:
            mm = np.broadcast_to(m if r.ndim == 2 else m[..., None], r.shape)
        d = (g - r)[mm]
        row = dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())), n_exact=float((d == 0).mean()))
        if k in ini:
            row["fortran_change_rms"] = float(np.sqrt(((r - ini[k])[mm] ** 2).mean()))
        rows[k] = row
    if return_state:
        return rows, diag, dict(r1=r1, r2=r2, ta=ta, tb=tnew)
    return rows, diag


def ground_si_stage(dd, it, state=None):
    """Once-per-step GROUND_SI (SURFACE.f:1230, after the NS loop) fed by OUR two-substep ice-tile fluxes.
    Replaces the recorded accumulated inputs f0dt, f1dt, evap (sum over substeps of the ice tile outputs) and srox0
    (sum of srheat*dtsurf) by ours; everything else (ocean fluxes fmoc/fhoc/fsoc, mixed-layer tm/sm, sea-ice state) is
    the recorded step-start value. Returns (worst input error dict, worst output error dict, baseline output error
    with fully recorded inputs), all vs the real GROUND_SI outputs, plus our chained outputs for downstream stages."""
    import seaice_compare as SCMP
    import seaice_core_jax as J
    if state is None:
        _, _, state = run_two_substeps(dd, it, return_state=True)
    rec = SCMP.load(f"{dd}/ffi_{it}.bin")
    acc = {}
    for tiles, r in ((state["ta"], state["r1"]), (state["tb"], state["r2"])):
        m = tiles[:, 2] == 2
        ij = tiles[m][:, :2].astype(int)
        f0 = np.asarray(r["tile"]["f0dt"])[m]; f1 = np.asarray(r["tile"]["f1dt"])[m]; ev = np.asarray(r["tile"]["evap"])[m]
        sr = tiles[m][:, S.IN["srheat"]] * tiles[m][:, S.IN["dtsurf"]]
        for k, a, b, c, d in zip(map(tuple, ij), f0, f1, ev, sr):
            v = acc.setdefault(k, np.zeros(4)); v += [a, b, c, d]
    mine = np.array([acc[(int(x[0]), int(x[1]))] for x in rec])
    in_err = {nm: float(np.abs(mine[:, q] - rec[:, c]).max()) for q, (nm, c) in enumerate((("f0dt", 15), ("f1dt", 16), ("evap", 17), ("srox0", 18)))}

    def run(recx):
        a = lambda c: jnp.asarray(recx[:, c])
        return J.ground_si(a(2) > 0.5, a(3), a(4), jnp.asarray(recx[:, 6:10]), jnp.asarray(recx[:, 10:14]), a(14), a(15), a(16),
                           a(17), a(18), a(19), a(20), a(21), a(22) > 0.5, a(23), a(24))
    ref = dict(snow=rec[:, 42], hsil=rec[:, 43:47], ssil=rec[:, 47:51], msi2=rec[:, 51], runosi=rec[:, 52],
               erunosi=rec[:, 53], srunosi=rec[:, 54])
    rec2 = np.array(rec); rec2[:, 15:19] = mine
    def errs(out):
        return {k: float(np.max(np.abs(np.asarray(out[k]) - ref[k]) / np.maximum(np.abs(ref[k]), 1e-6))) for k in ref}
    out2 = run(rec2)
    return in_err, errs(out2), errs(run(rec)), dict(rec=rec, out={k: np.asarray(v) for k, v in out2.items()})


def ground_lk_stage(dd, it, state, gs, return_arrays=False):
    """Once-per-step GROUND_LK (SURFACE.f:1232) fed by OUR chain: open-lake accumulators fodt/evapo/srox(1) from the two
    substeps' open-water tile outputs, ice-lake fluxes run0/fidt/srox(2) from our chained GROUND_SI outputs. Lake state
    (mlake/elake, incl. land runoff already added by GROUND_LK itself), roice, fsr2, hlake stay recorded. Returns
    (input errors, output errors vs real LKSOURC+LKMIX outputs, baseline errors with recorded inputs)."""
    import lakes_compare as LC
    import lakes_core_jax as LJ
    rec = LC.load(f"{dd}/ffl2_{it}.bin")
    acc = {}
    for tiles, r in ((state["ta"], state["r1"]), (state["tb"], state["r2"])):
        m = tiles[:, 2] == 1
        for row, a, b, c in zip(tiles[m], np.asarray(r["tile"]["f0dt"])[m], np.asarray(r["tile"]["evap"])[m],
                                 tiles[m][:, S.IN["srheat"]] * tiles[m][:, S.IN["dtsurf"]]):
            v = acc.setdefault((int(row[0]), int(row[1])), np.zeros(3)); v += [a, b, c]
    gsr, gso = gs["rec"], gs["out"]
    gmap = {(int(x[0]), int(x[1])): q for q, x in enumerate(gsr)}
    mine = np.zeros((len(rec), 6))
    for q, x in enumerate(rec):
        k = (int(x[0]), int(x[1]))
        f0, ev, so = acc.get(k, (0.0, 0.0, 0.0))   # fully ice-covered lake cell: no open-water tile
        g = gmap.get(k)
        run0, fidt, s2 = (gso["runosi"][g], gso["erunosi"][g], gso["solar_io"][g]) if g is not None else (0.0, 0.0, 0.0)
        mine[q] = [f0, ev, so, run0, fidt, s2]
    cols = dict(fodt=8, evapo=13, srox0=10, run0=7, fidt=9, srox1=11)
    order = ["fodt", "evapo", "srox0", "run0", "fidt", "srox1"]
    in_err = {nm: float(np.abs(mine[:, q] - rec[:, cols[nm]]).max()) for q, nm in enumerate(order)}

    def run(recx):
        a = lambda c: jnp.asarray(recx[:, c])
        src = LJ.lksourc_full(a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), a(11), a(12), a(13))
        mix = LJ.lkmix(src["mlake0"], src["mlake1"], src["elake0"], src["elake1"], a(14), jnp.zeros_like(a(2)), a(2), a(15))
        return dict(src=src, mix=mix)
    ref = dict(enrgfo=rec[:, 20], acefo=rec[:, 21], acefi=rec[:, 22], enrgfi=rec[:, 23],
               mlake0=rec[:, 24], mlake1=rec[:, 25], elake0=rec[:, 26], elake1=rec[:, 27])

    def errs(o):
        got = dict(o["src"]); got.update({k: o["mix"][k] for k in ("mlake0", "mlake1", "elake0", "elake1")})
        return {k: float(np.max(np.abs(np.asarray(got[k]) - ref[k]) / np.maximum(np.abs(ref[k]), 1e-6))) for k in ref}
    rec2 = np.array(rec)
    for q, nm in enumerate(order):
        rec2[:, cols[nm]] = mine[:, q]
    out2 = run(rec2)
    if return_arrays:
        return in_err, errs(out2), errs(run(rec)), dict(rec=rec, src={k: np.asarray(v) for k, v in out2["src"].items()})
    return in_err, errs(out2), errs(run(rec))


def form_si_lake_stage(dd, it, gs, lk):
    """FORM_SI/ADDICE (D12/D14) for the lake cells after GROUND_LK: sea-ice state from OUR chained GROUND_SI (where the
    cell had an ice record, else the recorded state), frazil fluxes enrgfo/acefo/acefi/enrgfi from OUR chained LKSOURC.
    Ocean cells need the ocean model's fluxes and stay out. Returns (input errors, output errors, baseline errors)."""
    import addice_compare as AD
    import seaice_core_jax as J
    ffn = AD.load(f"{dd}/ffn_{it}.bin")
    rec = ffn[ffn[:, 2] == 0]
    gmap = {(int(x[0]), int(x[1])): q for q, x in enumerate(gs["rec"])}
    lmap = {(int(x[0]), int(x[1])): q for q, x in enumerate(lk["rec"])}
    new = np.array(rec)
    for q, x in enumerate(rec):
        k = (int(x[0]), int(x[1]))
        g = gmap.get(k)
        if g is not None:
            new[q, 3] = gs["out"]["snow"][g]; new[q, 5:9] = gs["out"]["hsil"][g]
            new[q, 9:13] = gs["out"]["ssil"][g]; new[q, 13] = gs["out"]["msi2"][g]
        l_ = lmap[k]
        new[q, 14] = lk["src"]["enrgfo"][l_]; new[q, 17] = lk["src"]["acefo"][l_]
        new[q, 15] = lk["src"]["acefi"][l_]; new[q, 16] = lk["src"]["enrgfi"][l_]
    in_err = dict(state=float(np.abs(new[:, 3:14] - rec[:, 3:14]).max()), fluxes=float(np.abs(new[:, 14:18] - rec[:, 14:18]).max()))

    def run(r):
        a = lambda c: jnp.asarray(r[:, c])
        return J.addice(a(3), a(4), jnp.asarray(r[:, 5:9]), jnp.asarray(r[:, 9:13]), a(13), a(14), a(17), a(15), a(16),
                        a(18), a(19), a(20), a(21) > 0.5)
    ref = dict(snow=rec[:, 22], roice=rec[:, 23], hsil=rec[:, 24:28], ssil=rec[:, 28:32], msi2=rec[:, 32],
               dmimp=rec[:, 33], dhimp=rec[:, 34], dsimp=rec[:, 35])

    def errs(o):
        return {k: float(np.max(np.abs(np.asarray(o[k]) - ref[k]) / np.maximum(np.abs(ref[k]), 1e-6))) for k in ref}
    return in_err, errs(run(new)), errs(run(rec))


def form_si_ocean_stage(dd, it, gs):
    """FORM_SI/ADDICE for OCEAN cells: sea-ice state from OUR chained GROUND_SI (cells with an ice record; others keep their
    unchanged state), frazil/ocean fluxes enrgfo/acefo/acefi/enrgfi/salto/salti/flead RECORDED (they come from the ocean
    model, which is not ported). Returns (state input error, output errors vs real post-ADDICE, baseline errors)."""
    import addice_compare as AD
    import seaice_core_jax as J
    ffn = AD.load(f"{dd}/ffn_{it}.bin")
    rec = ffn[ffn[:, 2] == 1]
    gmap = {(int(x[0]), int(x[1])): q for q, x in enumerate(gs["rec"])}
    new = np.array(rec)
    n_chained = 0
    for q, x in enumerate(rec):
        g = gmap.get((int(x[0]), int(x[1])))
        if g is not None:
            n_chained += 1
            new[q, 3] = gs["out"]["snow"][g]; new[q, 5:9] = gs["out"]["hsil"][g]
            new[q, 9:13] = gs["out"]["ssil"][g]; new[q, 13] = gs["out"]["msi2"][g]
    in_err = dict(state=float(np.abs(new[:, 3:14] - rec[:, 3:14]).max()), n_chained=n_chained,
                  state_rel=float(np.abs(new[:, 3:14] - rec[:, 3:14]).max() / np.abs(rec[:, 3:14]).max()))

    def run(r):
        a = lambda c: jnp.asarray(r[:, c])
        return J.addice(a(3), a(4), jnp.asarray(r[:, 5:9]), jnp.asarray(r[:, 9:13]), a(13), a(14), a(17), a(15), a(16),
                        a(18), a(19), a(20), a(21) > 0.5)
    ref = dict(snow=rec[:, 22], roice=rec[:, 23], hsil=rec[:, 24:28], ssil=rec[:, 28:32], msi2=rec[:, 32],
               dmimp=rec[:, 33], dhimp=rec[:, 34], dsimp=rec[:, 35])

    def errs(o):
        return {k: float(np.max(np.abs(np.asarray(o[k]) - ref[k]) / np.maximum(np.abs(ref[k]), 1e-6))) for k in ref}
    return in_err, errs(run(new)), errs(run(rec))


if __name__ == "__main__":
    dd, it = sys.argv[1], int(sys.argv[2])
    rows, diag = run_two_substeps(dd, it)
    print("predicted substep-2 input columns vs recorded (max abs):")
    for k, v in diag.items():
        print(f"  {k:16s} {v:.3e}")
    print("our substep-2 ATURB exit vs real:")
    for k, r in rows.items():
        print(f"  {k:6s} max_abs={r['max_abs']:.3e} exact={r['n_exact']:.2f} change_rms={r.get('fortran_change_rms', float('nan')):.3e}")
