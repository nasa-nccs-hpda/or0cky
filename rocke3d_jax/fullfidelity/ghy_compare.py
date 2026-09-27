"""Compare ghy_ref.GhyColumn against real-Fortran GHY (land) dumps (ffg_<itime>.bin).

Record layout (450 doubles, 1-based Fortran offsets, see instrumentation/GHY_DRV.f.patch and GHY.f.patch):
see the field map below (F) built once at import.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_ref as G

NREC = 450


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, NREC)


def _sl(a, b):
    return slice(a - 1, b)


def unpack(rec):
    """rec: 1-D array of 450 doubles (one grid cell). Returns (static, dynamic, forcing, ent_iters, refs)."""
    r = rec
    w_in = r[_sl(9, 29)].reshape(7, 3, order="F")[:, :2]
    ht_in = r[_sl(30, 50)].reshape(7, 3, order="F")[:, :2]
    nsn = r[_sl(51, 52)].astype(int)
    dzsn = r[_sl(53, 58)].reshape(3, 2, order="F")
    wsn = r[_sl(59, 64)].reshape(3, 2, order="F")
    hsn = r[_sl(65, 70)].reshape(3, 2, order="F")
    fr_snow = r[_sl(71, 72)]
    top_index, top_dev = r[72], r[73]
    dz = r[_sl(75, 80)]
    q = r[_sl(81, 110)].reshape(5, 6, order="F")
    qk = r[_sl(111, 140)].reshape(5, 6, order="F")
    sl = r[140]
    fb_in, fv_in = r[141], r[142]
    pr, htpr, prs = r[143], r[144], r[145]
    irrig_tot, htirrig_tot = r[147], r[148]
    srheat, trheat, fgeotherm = r[149], r[150], r[151]
    ts_in, qs_in, pres_in, rho_in, ch_in = r[152], r[153], r[154], r[155], r[156]
    qm1_in, vs_in, vs0_in, gusti_in, tprime_in, qprime_in = r[157], r[158], r[159], r[160], r[161], r[162]

    static = dict(dz=dz, q=q, qk=qk, sl=sl, top_index=top_index, top_stdev=top_dev)
    dynamic = dict(w=w_in, ht=ht_in, nsn=nsn, dzsn=dzsn, wsn=wsn, hsn=hsn, fr_snow=fr_snow)
    fv = r[169]                    # ffent0(3): fv as returned by Ent (used inside advnc, not fv_in)
    if fv < 1e-6:
        fv = 0.0
    if fv > 1.0 - 1e-6:
        fv = 1.0
    fb = 1.0 - fv
    height_can = r[170]
    ws_can, shc_can = r[167], r[168]     # ffent0(1), ffent0(2)
    forcing = dict(geothermal_heat=fgeotherm, pr=pr, htpr=htpr, prs=prs, htprs=0.0,
                   irrig=(irrig_tot / fv if fv > 0 else 0.0), htirrig=(htirrig_tot / fv if fv > 0 else 0.0),
                   srht=srheat, trht=trheat, ts=ts_in, qs=qs_in, pres=pres_in, rho=rho_in, ch=ch_in,
                   qm1=qm1_in, vs=vs_in, vs0=vs0_in, gusti=gusti_in, tprime=tprime_in, qprime=qprime_in,
                   fb=fb, fv=fv, snowm=height_can * 0.1, ws_can=ws_can, shc_can=shc_can)

    ffnit = int(round(r[289]))
    n_avail = min(ffnit, 11)     # writer overflow guard: only ff1<=11 is within the 450-slot record
    ent_iters = []
    for ff1 in range(1, n_avail + 1):
        base = 299 + 13 * (ff1 - 1)     # 1-based index of ffent(1, ff1)
        vals = r[_sl(base + 1, base + 13)]
        ent_iters.append(dict(cnc=vals[0], betadl=vals[1:7], trans_sw=vals[7], ci=vals[8], gpp=vals[9],
                              lai=vals[10], ipp=vals[11], dts=vals[12]))

    refs = dict(
        w_out=r[_sl(181, 201)].reshape(7, 3, order="F")[:, :2],
        ht_out=r[_sl(202, 222)].reshape(7, 3, order="F")[:, :2],
        nsn_out=r[_sl(223, 224)].astype(int),
        fr_snow_out=r[_sl(243, 244)],
        tbcs=r[245], tsns=r[246], ashg=r[247], alhg=r[248], aevap=r[249],
        aruns=r[250], arunu=r[251], aeruns=r[252], aerunu=r[253], ae0=r[254], abetad=r[255],
        tp_out=r[_sl(259, 272)].reshape(7, 2, order="F"),
        fice_out=r[_sl(273, 286)].reshape(7, 2, order="F"),
        ffnit=ffnit,
    )
    return static, dynamic, forcing, ent_iters, refs, forcing['snowm']


def run_cell(rec):
    static, dynamic, forcing, ent_iters, refs, snowm = unpack(rec)
    col = G.GhyColumn(static, dynamic, forcing)
    col.fb, col.fv = forcing['fb'], forcing['fv']
    dt = sum(it['dts'] for it in ent_iters) if ent_iters else 1800.0 / 2
    col.advnc(ent_iters, dt, snowm)
    return col, refs


SCALARS = ["tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"]


def compare_cell(rec):
    col, refs = run_cell(rec)
    rows = {}
    for k in SCALARS:
        got = getattr(col, k)
        ref = refs[k]
        rows[k] = dict(got=got, ref=ref, abs=abs(got - ref))
    rows["w_out"] = dict(abs=float(np.abs(col.w[:, :2] - refs["w_out"]).max()))
    rows["ht_out"] = dict(abs=float(np.abs(col.ht[:, :2] - refs["ht_out"]).max()))
    rows["tp_out"] = dict(abs=float(np.abs(col.tp[:, :2] - refs["tp_out"]).max()))
    rows["ffnit_used"] = refs["ffnit"]
    return rows


if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rec = load(path)
    idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).astype(int)
    for i in idx:
        try:
            rows = compare_cell(rec[i])
        except Exception as e:
            print(f"cell {i}: EXCEPTION {type(e).__name__}: {e}")
            continue
        print(f"cell {i} (ffnit={rows['ffnit_used']}):")
        for k in SCALARS:
            r = rows[k]
            print(f"  {k:8s} got={r['got']: .6e} ref={r['ref']: .6e} abs={r['abs']:.3e}")
        print(f"  w_out max_abs={rows['w_out']['abs']:.3e}  ht_out max_abs={rows['ht_out']['abs']:.3e}"
              f"  tp_out max_abs={rows['tp_out']['abs']:.3e}")
