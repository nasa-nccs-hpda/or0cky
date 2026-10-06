"""Validate the full ghy_jax.advnc() pipeline (all substeps, batched) against the REAL Fortran
ffg_*.bin ground truth -- the only point in this vectorization where real-Fortran comparison is
possible (ghy_compare.py's `refs`), everything upstream was cross-checked against the plain-Python
ghy_ref.py instead (see ghy_jax_compare.py / ghy_flux_chain_test.py / ghy_snow_test.py)."""
import glob
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_jax as J
import ghy_compare as GC

MAX_SUBSTEPS = 11
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


def build_batch(rec):
    unpacked = [GC.unpack(r) for r in rec]
    N = len(unpacked)
    dz = np.stack([u[0]["dz"] for u in unpacked])
    q = np.stack([u[0]["q"] for u in unpacked])
    qk = np.stack([u[0]["qk"] for u in unpacked])
    sl = np.array([u[0]["sl"] for u in unpacked])
    fb = np.array([u[2]["fb"] for u in unpacked])
    fv = np.array([u[2]["fv"] for u in unpacked])
    w = np.stack([u[1]["w"] for u in unpacked])
    ht = np.stack([u[1]["ht"] for u in unpacked])
    nsn = np.stack([u[1]["nsn"] for u in unpacked])
    wsn = np.stack([u[1]["wsn"] for u in unpacked])
    hsn = np.stack([u[1]["hsn"] for u in unpacked])
    dzsn_raw = np.stack([u[1]["dzsn"] for u in unpacked])
    dzsn = np.concatenate([dzsn_raw, np.zeros((N, 1, 2))], axis=1)
    fr_snow = np.stack([u[1]["fr_snow"] for u in unpacked])
    ws_can = np.array([u[2]["ws_can"] for u in unpacked])
    shc_can = np.array([u[2]["shc_can"] for u in unpacked])
    snowm = np.array([u[2]["snowm"] for u in unpacked])

    def farr(key):
        return np.array([u[2][key] for u in unpacked])

    ent_dts = np.zeros((N, MAX_SUBSTEPS)); ent_cnc = np.zeros((N, MAX_SUBSTEPS))
    ent_lai = np.zeros((N, MAX_SUBSTEPS)); ent_betadl = np.zeros((N, MAX_SUBSTEPS, 6))
    n_substeps = np.zeros(N, dtype=int)
    for i, u in enumerate(unpacked):
        ent_iters = u[3]
        ns = len(ent_iters)
        n_substeps[i] = ns
        for j in range(MAX_SUBSTEPS):
            src = ent_iters[j] if j < ns else (ent_iters[-1] if ns > 0 else
                                               dict(dts=900.0, cnc=0.0, betadl=np.zeros(6), lai=0.0))
            ent_dts[i, j] = src["dts"]; ent_cnc[i, j] = src["cnc"]; ent_lai[i, j] = src["lai"]
            ent_betadl[i, j] = src["betadl"]

    static0 = dict(dz=dz, q=q, qk=qk, sl=sl)
    dynamic0 = dict(w=w, ht=ht, nsn=nsn, dzsn=dzsn, wsn=wsn, hsn=hsn, fr_snow=fr_snow)
    forcing = dict(fb=fb, fv=fv, pr=farr("pr"), htpr=farr("htpr"), prs=farr("prs"), htprs=np.zeros(N),
                  srht=farr("srht"), trht=farr("trht"), ts=farr("ts"), qs=farr("qs"), pres=farr("pres"),
                  rho=farr("rho"), ch=farr("ch"), qm1=farr("qm1"), vs=farr("vs"), vs0=farr("vs0"),
                  gusti=farr("gusti"), tprime=farr("tprime"), qprime=farr("qprime"),
                  geothermal_heat=farr("geothermal_heat"))
    # Forcing conditioning and irrigation as GHY.f does them (D135, D136): pr>=0, 0<=prs<=pr, htprs=htpr/pr*prs (ghy_ref.py:597-599, 648);
    # irrig(2)=irrig_in/fv, htirrig(2)=htirrig_in/fv for the vegetated tile (giss_LSM/GHY.f:2230-2234). Both were missing before 2026-10-06.
    _pr = np.maximum(forcing["pr"], 0.0)
    _prs = np.minimum(np.maximum(forcing["prs"], 0.0), _pr)
    forcing["pr"], forcing["prs"] = _pr, _prs
    forcing["htprs"] = np.where(_pr <= 0.0, 0.0, forcing["htpr"] / np.where(_pr <= 0.0, 1.0, _pr) * _prs)
    forcing["irrig"] = farr("irrig")
    forcing["htirrig"] = farr("htirrig")
    refs = dict(
        w_out=np.stack([r[4]["w_out"] for r in unpacked]), ht_out=np.stack([r[4]["ht_out"] for r in unpacked]),
        tp_out=np.stack([r[4]["tp_out"] for r in unpacked]),
        **{k: np.array([r[4][k] for r in unpacked])
           for k in ("tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0",
                     "abetad", "ffnit")})
    dt_total = np.array([sum(it["dts"] for it in u[3]) if u[3] else 900.0 for u in unpacked])
    return static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm, \
        ws_can, shc_can, refs


def run(n_cells=None, file_idx=0, verbose=True):
    files = sorted(glob.glob(f"{FF}/*/ffg_*.bin"))
    rec = GC.load(files[file_idx])
    if n_cells:
        rec = rec[:n_cells]
    (static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm,
     ws_can, shc_can, refs) = build_batch(rec)

    static = J.init_static(jnp.asarray(static0["dz"]), jnp.asarray(static0["q"]), jnp.asarray(static0["qk"]),
                           jnp.asarray(forcing["fb"]), jnp.asarray(forcing["fv"]))
    static = dict(static)
    static["ws"] = static["ws"].at[:, 0, 1].set(jnp.asarray(ws_can))
    static["shc"] = static["shc"].at[:, 0, 1].set(jnp.asarray(shc_can))
    static = J.init_xklh_static(static)
    static["sl"] = jnp.asarray(static0["sl"])

    dynamic0_j = {k: jnp.asarray(v) for k, v in dynamic0.items()}
    forcing_j = {k: jnp.asarray(v) for k, v in forcing.items()}

    out = J.advnc(static, dynamic0_j, forcing_j, jnp.asarray(ent_dts), jnp.asarray(ent_cnc),
                  jnp.asarray(ent_betadl), jnp.asarray(ent_lai), jnp.asarray(n_substeps),
                  jnp.asarray(dt_total), jnp.asarray(snowm))

    n = static0["dz"].shape[0]
    def relerr(mine, ref):
        mine = np.asarray(mine); ref = np.asarray(ref)
        denom = np.maximum(np.abs(ref), 1e-6)
        return np.abs(mine - ref) / denom

    worst = {}
    for k in ("tbcs", "tsns", "ashg", "alhg", "aevap", "aruns", "arunu", "aeruns", "aerunu", "ae0", "abetad"):
        worst[k] = float(np.max(relerr(out[k], refs[k])))
    worst["w_out"] = float(np.max(relerr(out["w"][:, :7, :], refs["w_out"])))
    worst["ht_out"] = float(np.max(relerr(out["ht"][:, :7, :], refs["ht_out"])))
    worst["tp_out"] = float(np.max(relerr(np.asarray(out["tp"])[:, :7, :], refs["tp_out"])))
    if verbose:
        for k, v in sorted(worst.items()):
            print(f"{k:10s} max rel err = {v:.3e}")
    return worst, n


if __name__ == "__main__":
    run(400)
