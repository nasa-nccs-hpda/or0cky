"""Validate clouds_helpers_ff.py (D91 PRECIP_MP / ANVIL_OPTICAL_THICKNESS / MC_CLOUD_FRACTION / MC_PRECIP_PHASE and
D92 CONVECTIVE_MICROPHYSICS) against the instrumented real model.

Dumps: ff_data/<date>/ffc_{pmp,anv,mcf,cmp,mpp}_<itime>.bin, big-endian f8 records, header itime,site,ncall then
  pmp (8):  rho,flam,dc,cn | out                                   site 1-8 = PRECIP_MP call in CONVECTIVE_MICROPHYSICS
            (1 liq lower CONDP1, 2 liq upper CONDP, 3 ice lower, 4 ice upper, 5/6 mixed lower CONDIP/CONDGP,
             7/8 mixed upper CONDIP/CONDGP)
  anv (15): svlatl,rcldlx,rcldix,mcdncw,mcdnci,rimax,bybr,fcld,tem,wtem | rcld,taumc     (CLOUDS2.F90:3114)
  mcf (18): L,tl,pl,ccm,wcu,plemin,plel2,plemax,ccmmin,wcumin,lmin,lmax,ccmul,ccmul2 | fcloud   (2675)
  cmp (29): pl,wcu,dwcu,lfrz,wcufrz,tp,ti,fitmax,pland,cn0,cn0i,cn0g,flamw,flamg,flami,rhoip,rhog,itmax,tlmin,
            tlmin1,wmax,condip_in,condgp_in | condp,condp1,condip,condgp                         (1938)
  mpp (19): L,airm,fevap,lhp1,prcp_in,told,told1,vlat,cond,lmin,heat1_in,mc_revp_abv_cldbase | prcp,lhp,mcloud,heat1 (2703)
Sampling strides (ffc_h_consts.txt): every Nth call per (itime,site) plus the first two calls of each site.

Usage: python3 clouds_helpers_compare.py
"""
import glob
import sys
import numpy as np
import clouds_helpers_ff as ch

FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
COLS = {
    "pmp": "rho flam dc cn out".split(),
    "anv": "svlatl rcldlx rcldix mcdncw mcdnci rimax bybr fcld tem wtem rcld taumc".split(),
    "mcf": "L tl pl ccm wcu plemin plel2 plemax ccmmin wcumin lmin lmax ccmul ccmul2 fcloud".split(),
    "cmp": ("pl wcu dwcu lfrz wcufrz tp ti fitmax pland cn0 cn0i cn0g flamw flamg flami rhoip rhog itmax tlmin "
            "tlmin1 wmax condip_in condgp_in condp condp1 condip condgp").split(),
    "mpp": "L airm fevap lhp1 prcp_in told told1 vlat cond lmin heat1_in mcrevp prcp lhp mcloud heat1".split(),
}
HDR = ["itime", "site", "ncall"]


def read_consts(path):
    out = {}
    for line in open(path):
        k, v = line.split()
        out[k] = float(v) if not k.startswith("stride") else int(v)
    return out


def load_date(date, rt, ff=FF_DEFAULT):
    cols = HDR + COLS[rt]
    files = sorted(glob.glob(f"{ff}/{date}/ffc_{rt}_[0-9]*.bin"))
    if not files:
        return None
    parts = []
    for f in files:
        raw = np.fromfile(f, dtype=">f8")
        assert raw.size % len(cols) == 0, (f, raw.size, len(cols))
        parts.append(raw.reshape(-1, len(cols)).astype(np.float64))
    r = np.concatenate(parts)
    return {c: r[:, i] for i, c in enumerate(cols)}


def run_pmp(d, **kw):
    return ch.precip_mp(d["rho"], d["flam"], d["dc"], d["cn"], **kw)


def run_anv(d, **kw):
    return ch.anvil_optical_thickness(d["svlatl"], d["rcldlx"], d["rcldix"], d["mcdncw"], d["mcdnci"], d["rimax"],
                                      d["bybr"], d["fcld"], d["tem"], d["wtem"], **kw)


def run_mcf(d, dtsrc=ch.DTSRC, **kw):
    return ch.mc_cloud_fraction(d["L"], d["tl"], d["pl"], d["ccm"], d["wcu"], d["plemin"], d["plel2"], d["plemax"],
                                d["ccmmin"], d["wcumin"], d["lmin"], d["lmax"], d["ccmul"], d["ccmul2"],
                                dtsrc=dtsrc, **kw)


def run_cmp(d, **kw):
    return ch.convective_microphysics(d["pl"], d["wcu"], d["dwcu"], d["lfrz"], d["wcufrz"], d["tp"], d["ti"],
                                      d["fitmax"], d["pland"], d["cn0"], d["cn0i"], d["cn0g"], d["flamw"],
                                      d["flamg"], d["flami"], d["rhoip"], d["rhog"], d["itmax"], d["tlmin"],
                                      d["tlmin1"], d["wmax"], d["condip_in"], d["condgp_in"], **kw)


def run_mpp(d, **kw):
    return ch.mc_precip_phase(d["L"], d["airm"], d["fevap"], d["lhp1"], d["prcp_in"], d["told"], d["told1"],
                              d["vlat"], d["cond"], d["lmin"], d["heat1_in"], int(d["mcrevp"][0]), **kw)


def rel(a, b, floor=1e-300):
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def summarize(name, pairs, extra=""):
    """pairs: list of (label, port, real).  Prints bitwise count, max abs, max rel."""
    ok = True
    for lab, a, b in pairs:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        fin = np.isfinite(a) & np.isfinite(b)
        mabs = float(np.max(np.abs(a - b)[fin])) if fin.any() else 0.0
        mrel = float(np.max(rel(a, b)[fin & (b != 0)])) if (fin & (b != 0)).any() else 0.0
        good = bool(same.all() or mrel < 1e-12)
        ok &= good
        print(f"    {name} {lab:<8} n={a.size:6d} bitwise={int(same.sum()):6d} ({100.0*same.mean():6.2f}%) "
              f"max_abs={mabs:.2e} max_rel={mrel:.2e} [{'OK' if good else 'FAIL'}]")
    return ok


def branch_report(date, data):
    """Branch/exercise counts for each routine (non-vacuity)."""
    out = {}
    p = data["pmp"]
    out["pmp"] = {int(s): int((p["site"] == s).sum()) for s in range(1, 9)}
    a = data["anv"]
    out["anv"] = dict(liq=int((a["svlatl"] == ch.LHE).sum()), ice=int((a["svlatl"] != ch.LHE).sum()),
                      rimax_cap=int(((a["svlatl"] != ch.LHE) & (a["rcld"] == a["rimax"])).sum()),
                      tau_cap=int((a["taumc"] == 100.0).sum()), n=a["taumc"].size)
    m = data["mcf"]
    deep = (m["plemin"] - m["plel2"]) >= 450
    shallow = (m["plemin"] - m["plemax"]) < 450
    below = m["L"] < m["lmin"]
    out["mcf"] = dict(n=m["fcloud"].size, deep_x5=int(deep.sum()), below_virga_deep=int((below & ~shallow).sum()),
                      shallow=int(shallow.sum()), shallow_detr=int((shallow & (m["L"] == m["lmax"] - 1)).sum()),
                      shallow_below_zero=int((shallow & below).sum()), capped1=int((m["fcloud"] == 1.0).sum()),
                      zero=int((m["fcloud"] == 0).sum()))
    c = data["cmp"]
    _, _, _, _, dg = ch.convective_microphysics(*[c[k] for k in (
        "pl wcu dwcu lfrz wcufrz tp ti fitmax pland cn0 cn0i cn0g flamw flamg flami rhoip rhog itmax tlmin tlmin1 "
        "wmax condip_in condgp_in").split()], return_diag=True)
    wat = dg["water"]
    out["cmp"] = dict(n=c["condp"].size, water=int(wat.sum()), ice=int(dg["ice"].sum()), mixed=int(dg["mixed"].sum()),
                      mixed_fg0=int((dg["mixed"] & (dg["fg"] == 0)).sum()),
                      mixed_fg1=int((dg["mixed"] & (dg["fg"] == 1)).sum()),
                      mixed_fg_mid=int((dg["mixed"] & (dg["fg"] > 0) & (dg["fg"] < 1)).sum()),
                      lfrz0=int((c["lfrz"] == 0).sum()), lfrz_pos=int((c["lfrz"] > 0).sum()),
                      pland_lt_half=int((wat & (c["pland"] < 0.5)).sum()), pland_ge_half=int((wat & (c["pland"] >= 0.5)).sum()),
                      water_exit1_lo=int((dg["exit1"] == 1).sum()), water_exit2_wmax_lo=int((dg["exit1"] == 2).sum()),
                      water_noexit_lo=int((dg["exit1"] == 0).sum()), water_exit1_up=int((dg["exit2"] == 1).sum()),
                      water_exit2_wmax_up=int((dg["exit2"] == 2).sum()), water_noexit_up=int((dg["exit2"] == 0).sum()),
                      wv_clamped=int((c["wcu"] - c["dwcu"] < 0).sum()), dcg_capped=int((ch._dcg(np.maximum(c["wcu"] - c["dwcu"], 0), c["pl"]) == 1e-2).sum()),
                      dci_capped=int((ch._dci(np.maximum(c["wcu"] - c["dwcu"], 0), c["pl"]) == 1e-2).sum()),
                      tig_floor=int((dg["tig"] == c["ti"] - 10).sum()), mixed_blocked_by_tlmin=int((dg["mixed"] & ((c["tlmin"] <= ch.TF) | (c["tlmin1"] <= ch.TF))).sum()))
    q = data["mpp"]
    told, told1, lhp1 = q["told"], q["told1"], q["lhp1"]
    melt = (lhp1 == ch.LHS) & (told > ch.TF) & (told1 <= ch.TF)
    frz = (lhp1 == ch.LHE) & (told <= ch.TF) & (told1 > ch.TF)
    out["mpp"] = dict(n=q["lhp"].size, melt=int(melt.sum()), refreeze=int(frz.sum()),
                      phase_conv=int(((q["lhp"] != q["vlat"]) & (q["cond"] > 0)).sum()),
                      mcloud_capped=int((q["mcloud"] == q["airm"]).sum()), mcloud_zero=int((q["mcloud"] == 0).sum()),
                      lhp_is_lhs=int((q["lhp1"] == ch.LHS).sum()), lhp_is_lhe=int((q["lhp1"] == ch.LHE).sum()),
                      heat1_in_nonzero=int((q["heat1_in"] != 0).sum()), prcp_pos=int((q["prcp_in"] > 0).sum()),
                      L_le_lmin=int((q["L"] <= q["lmin"]).sum()), L_gt_lmin=int((q["L"] > q["lmin"]).sum()))
    return out


if __name__ == "__main__":
    ok_all = True
    try:
        c = read_consts(f"{FF_DEFAULT}/nov26/ffc_h_consts.txt")
        mine = dict(rgas=ch.RGAS, grav=ch.GRAV, teeny=ch.TEENY, pi=ch.PI, by3=ch.BY3, by6=ch.BY6, twopi=ch.TWOPI,
                    lhe=ch.LHE, lhs=ch.LHS, lhm=ch.LHM, tf=ch.TF, bysha=ch.BYSHA, rhow=ch.RHOW, dtsrc=ch.DTSRC)
        for k, v in mine.items():
            print(f"const {k:6s} port={v:.17e} real={c[k]:.17e} {'same' if v == c[k] else 'DIFF'}")
            ok_all &= v == c[k]
    except OSError:
        pass
    for date, _ in DATES:
        data = {rt: load_date(date, rt) for rt in COLS}
        if any(v is None for v in data.values()):
            print(f"{date}: no dumps")
            continue
        print(f"== {date}: " + ", ".join(f"{rt}={data[rt]['site'].size}" for rt in COLS))
        p = data["pmp"]
        ok = summarize("PRECIP_MP", [("out", run_pmp(p), p["out"])])
        for s in range(1, 9):
            m = p["site"] == s
            ok &= summarize(f"PMP s{s}", [("out", run_pmp({k: v[m] for k, v in p.items()}), p["out"][m])])
        a = data["anv"]
        rc, tm = run_anv(a)
        ok &= summarize("ANVIL", [("rcld", rc, a["rcld"]), ("taumc", tm, a["taumc"])])
        m = data["mcf"]
        ok &= summarize("MCF", [("fcloud", run_mcf(m, dtsrc=read_consts(f"{FF_DEFAULT}/{date}/ffc_h_consts.txt")["dtsrc"]), m["fcloud"])])
        cm = data["cmp"]
        cp, cp1, cip, cgp = run_cmp(cm)
        ok &= summarize("CMP", [("condp", cp, cm["condp"]), ("condp1", cp1, cm["condp1"]),
                                ("condip", cip, cm["condip"]), ("condgp", cgp, cm["condgp"])])
        q = data["mpp"]
        lhp, mc, h1 = run_mpp(q)
        ok &= summarize("MPP", [("lhp", lhp, q["lhp"]), ("mcloud", mc, q["mcloud"]), ("heat1", h1, q["heat1"]),
                                ("prcp", q["prcp_in"], q["prcp"])])
        ok_all &= ok
        for rt, br in branch_report(date, data).items():
            print(f"  branches {rt}: {br}")
    print("ALL MATCH" if ok_all else "MISMATCH FOUND")
    sys.exit(0 if ok_all else 1)
