"""Track B land tile in the surface chain: our PBL (itype 4) -> our JAX GHY `advnc` -> land patch fields (D22).

Mirrors GHY_DRV.f `earth`: PBL is called with the land inputs, its outputs become GHY's forcing (`ts=tsv/(1+qs*xdelt)`,
`qs=qsrf`, `rho=100*ps/(R*tsv)`, `ch`, `vs=ws`, `tprime/qprime`, `qm1=q1*ma1`), GHY runs, then the atmosphere-facing
patch fields are formed exactly as `earth` does after `advnc`:
    uflux1 = cm*ws*rho*us, vflux1 = cm*ws*rho*vs, dth1 = -(SHDT + dLWDT)/(sha*ma1) with SHDT = -ashg and
    dLWDT = dtsurf*(TRUP_in_rad - stbo*(tbcs+tf)^4), dq1 = aevap/ma1, tsavg = tsv, qsavg = qsrf.
Recorded (dump-fed) inputs, stated plainly: Ent's per-substep exports (canopy conductance, betadl, LAI, dts), the
precipitation/radiation forcing, and `TRUP_in_rad` for land -- the radiation input is not dumped for land cells, so it is
reconstructed once per step from substep 1's recorded land patch (constant across substeps to 6e-14, checked in tests).
"""
import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_jax as J
import ghy_advnc_test as AT
import pbl_compare as PC
import pbl_ff as P

RGAS = 287.048730386149032
SHA = 1002.88097573814161
STBO = 5.67037320999999984e-8
TF = 273.15
XDELT = 0.0
C001 = float(np.float32(0.001))     # `0.001` in GHY_DRV.f is a REAL*4 literal (0.0010000000474974513)
DYN_KEYS = ("w", "ht", "nsn", "dzsn", "wsn", "hsn", "fr_snow")
_advnc_jit = jax.jit(J.advnc)   # un-jitted, the lax.scan inside is re-traced and re-compiled on every call (~30 s)


def infer_trup(g, patch_dth1, dtsurf):
    """TRUP_in_rad for land from a recorded substep (dth1 patch value and GHY outputs). Returns (N,)."""
    ma1 = g[:, 165]
    shdt = -g[:, 247]
    dlw = -patch_dth1 * SHA * ma1 - shdt
    return dlw / dtsurf + STBO * (g[:, 245] + TF) ** 4


def run_ghy(g, forcing_over=None, dyn_over=None):
    """Run ghy_jax.advnc on ffg records `g` (recorded Ent exports/forcing), overriding forcing/dynamic entries."""
    (s0, d0, f, edts, ecnc, ebet, elai, ns, dt, snowm, wsc, shc, refs) = AT.build_batch(g)
    f = dict(f)
    if forcing_over:
        f.update(forcing_over)
    d0 = dict(d0)
    if dyn_over:
        d0.update(dyn_over)
    st = dict(J.init_static(jnp.asarray(s0["dz"]), jnp.asarray(s0["q"]), jnp.asarray(s0["qk"]),
                            jnp.asarray(f["fb"]), jnp.asarray(f["fv"])))
    st["ws"] = st["ws"].at[:, 0, 1].set(jnp.asarray(wsc))
    st["shc"] = st["shc"].at[:, 0, 1].set(jnp.asarray(shc))
    st = J.init_xklh_static(st)
    st["sl"] = jnp.asarray(s0["sl"])
    out = _advnc_jit(st, {k: jnp.asarray(v) for k, v in d0.items()}, {k: jnp.asarray(v) for k, v in f.items()},
                  jnp.asarray(edts), jnp.asarray(ecnc), jnp.asarray(ebet), jnp.asarray(elai), jnp.asarray(ns),
                  jnp.asarray(dt), jnp.asarray(snowm))
    return {k: np.asarray(v) for k, v in out.items()}, refs


def land_substep(p4, g, q1, trup, dtsurf=900.0, dyn=None):
    """p4: PBL records (itype 4) for this substep; g: matching ffg records; q1: layer-1 specific humidity at the cells;
    trup: (N,) TRUP_in_rad; dyn: carried GHY state (dict of arrays) or None to use the recorded state in `g`.
    Returns dict(patch, pbl, ghy, rho, dyn_next, evap_max_ij, fr_sat_ij)."""
    assert np.array_equal(p4[:, :2], g[:, :2])
    out = PC.run(p4)
    ps = p4[:, 16]
    tsv, qsrf = out["tsv"], out["qsrf"]
    rho = 100.0 * ps / (RGAS * tsv)
    ddml = p4[:, 23] > 0.5
    ma1 = g[:, 165]
    forcing = dict(ts=tsv / (1.0 + qsrf * XDELT), qs=qsrf, rho=rho, ch=out["ch"], vs=out["ws"],
                   tprime=np.where(ddml, p4[:, 25] - p4[:, 7], 0.0), qprime=np.where(ddml, p4[:, 26] - p4[:, 39], 0.0),
                   qm1=q1 * ma1)
    ghy, _ = run_ghy(g, forcing, dyn)
    rcdmws = out["cm"] * out["ws"] * rho
    dlw = dtsurf * (trup - STBO * (ghy["tbcs"] + TF) ** 4)
    patch = dict(uflux1=rcdmws * out["us"], vflux1=rcdmws * out["vs"],
                 dth1=-(-ghy["ashg"] + dlw) / (SHA * ma1), dq1=ghy["aevap"] / ma1, tsavg=tsv, qsavg=qsrf)
    return dict(patch=patch, pbl=out, ghy=ghy, rho=rho,
                dyn_next={k: ghy[k] for k in DYN_KEYS}, evap_max_ij=ghy["evap_max_ij"], fr_sat_ij=ghy["fr_sat_ij"])


def next_land_pbl_columns(rec, land):
    """Overwrite the land-ground PBL input columns of substep-2 records `rec` (itype 4) from substep-1 GHY results:
    tg/tgv = tsns+tf, qg_sat = qsat(tg), qg_aver = qg_ij (evap-limited non-saturated blend, with the REAL*4 0.001),
    tr4 = (tbcs+tf)^4, evap_max = evap_max_ij*1000/rhosrf0, fr_sat = fr_sat_ij."""
    rec = np.array(rec)
    gh = land["ghy"]
    ps = rec[:, 16]
    tg = gh["tsns"] + TF
    qg_sat = np.asarray(P.qsat(jnp.asarray(tg), jnp.asarray(rec[:, 19]), jnp.asarray(ps)))
    qs = land["pbl"]["qsrf"]
    rcdhws = land["pbl"]["ch"] * land["pbl"]["ws"] * land["rho"]
    em, fr = land["evap_max_ij"], land["fr_sat_ij"]
    qn = np.array(qs)
    m = rcdhws > 1e-30
    qn[m] = qs[m] + em[m] / (C001 * rcdhws[m])
    qn = np.minimum(qn, qg_sat)
    qg = fr * qg_sat + (1.0 - fr) * qn
    rhosrf0 = 100.0 * ps / (RGAS * tg * (1.0 + qg * XDELT))
    rec[:, 18] = tg
    rec[:, 6] = tg * (1.0 + qg * XDELT)
    rec[:, 8] = qg_sat
    rec[:, 9] = qg
    rec[:, 11] = (gh["tbcs"] + TF) ** 4
    rec[:, 12] = em * 1000.0 / rhosrf0
    rec[:, 13] = fr
    return rec


def run_land_multistep(dd, its):
    """Land-only chain over consecutive DTsrc steps `its` (e.g. [33312, 33313] -> 4 substeps). Atmosphere-dependent inputs at
    every substep (utop, tkv, dbl, ug, ..., Ent, precipitation/radiation forcing, q1) are the RECORDED ones; what is carried by
    our own code is the land state: GHY prognostic state (w, ht, snow layers), the PBL land profiles and cm/ch/cq, and the
    ground columns (tg, tr4, qg, evap_max, fr_sat) built from GHY's outputs. Returns a list (one per substep) of dicts with the
    max/rms errors of GHY's outputs vs the recorded outputs of that substep."""
    import ghy_compare as GC
    import tile_aggregate_ff as TA
    res = []
    prev = None
    for it in its:
        p = PC.load(f"{dd}/ffp_{it}.bin")
        g = GC.load(f"{dd}/ffg_{it}.bin")
        fft = TA.load(f"{dd}/fft_{it}.bin")
        n, ng, B = len(p) // 2, len(g) // 2, len(fft) // 2
        blk1 = fft[:B]
        _, patch1, _ = TA.unpack(blk1)
        lut = {(int(a), int(b)): k for k, (a, b) in enumerate(blk1[:, :2])}
        idx = np.array([lut[(int(a), int(b))] for a, b in g[:ng, :2]])
        trup = infer_trup(g[:ng], patch1["dth1"][idx, 3], 900.0)
        for ns in (0, 1):
            ph = p[ns * n:(ns + 1) * n]
            p4 = np.array(ph[ph[:, 2] == 4])
            gr = g[ns * ng:(ns + 1) * ng]
            dyn = None
            if prev is not None:
                assert np.array_equal(prev["p4_ij"], p4[:, :2])
                p4[:, 50:89] = np.concatenate([prev["pbl"][k] for k in ("u", "v", "t", "q", "e")], axis=1)
                p4[:, 33], p4[:, 34], p4[:, 35] = prev["pbl"]["cm"], prev["pbl"]["ch"], prev["pbl"]["cq"]
                p4 = next_land_pbl_columns(p4, prev)
                dyn = prev["dyn_next"]
            q1 = gr[:, 157] / gr[:, 165]
            land = land_substep(p4, gr, q1, trup, 900.0, dyn)
            land["p4_ij"] = p4[:, :2]
            gh = land["ghy"]
            row = dict(step=it, ns=ns + 1)
            for k, c in (("tbcs", 245), ("tsns", 246), ("ashg", 247), ("alhg", 248), ("aevap", 249)):
                d = gh[k] - gr[:, c]
                row[k] = (float(np.abs(d).max()), float(np.sqrt((d ** 2).mean())), float(np.abs(gr[:, c]).max()), float(np.quantile(np.abs(d), 0.99)), int((np.abs(d) > 1e-3 * np.abs(gr[:, c]).max()).sum()))
            wref = gr[:, 180:201].reshape(-1, 7, 3, order="F")[:, :, :2]
            d = gh["w"][:, :7, :] - wref
            row["w"] = (float(np.abs(d).max()), float(np.sqrt((d ** 2).mean())), float(np.abs(wref).max()), float(np.quantile(np.abs(d).max(axis=(1, 2)), 0.99)), int((np.abs(d).max(axis=(1, 2)) > 1e-3 * np.abs(wref).max()).sum()))
            res.append(row)
            prev = land
    return res
