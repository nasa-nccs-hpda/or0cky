"""Cross-check ghy_jax (batched/vectorized GHY) against the plain-Python ghy_ref.py on real cells'
recorded input state. Only the FULL advnc() pipeline has a real-Fortran ground truth available
(ghy_compare.py's `refs`, from ffg_<itime>.bin -- no intermediate per-substep dump exists), so
individual functions here are validated by cross-check against ghy_ref (already validated against
Fortran, FULL_FIDELITY_DELTAS.md D9), not against Fortran directly.
"""
import glob
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_ref as G
import ghy_jax as J
import ghy_compare as GC

FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


def load_batch(n=400, file_idx=0):
    files = sorted(glob.glob(f"{FF}/*/ffg_*.bin"))
    rec = GC.load(files[file_idx])[:n]
    unpacked = [GC.unpack(r) for r in rec]
    dz_b = np.stack([u[0]["dz"] for u in unpacked])
    q_b = np.stack([u[0]["q"] for u in unpacked])
    qk_b = np.stack([u[0]["qk"] for u in unpacked])
    fb_b = np.array([u[2]["fb"] for u in unpacked])
    fv_b = np.array([u[2]["fv"] for u in unpacked])
    w_b = np.stack([u[1]["w"] for u in unpacked])
    ht_b = np.stack([u[1]["ht"] for u in unpacked])
    nsn_b = np.stack([u[1]["nsn"] for u in unpacked])
    wsn_b = np.stack([u[1]["wsn"] for u in unpacked])
    hsn_b = np.stack([u[1]["hsn"] for u in unpacked])
    dzsn_b = np.stack([u[1]["dzsn"] for u in unpacked])
    fr_snow_b = np.stack([u[1]["fr_snow"] for u in unpacked])
    ws_can_b = np.array([u[2]["ws_can"] for u in unpacked])
    shc_can_b = np.array([u[2]["shc_can"] for u in unpacked])
    snowm_b = np.array([u[2]["snowm"] for u in unpacked])
    return dict(rec=rec, unpacked=unpacked, dz=dz_b, q=q_b, qk=qk_b, fb=fb_b, fv=fv_b, w=w_b, ht=ht_b,
                nsn=nsn_b, wsn=wsn_b, hsn=hsn_b, dzsn=dzsn_b, fr_snow=fr_snow_b, ws_can=ws_can_b,
                shc_can=shc_can_b, snowm=snowm_b)


def build_static(batch):
    static = J.init_static(jnp.asarray(batch["dz"]), jnp.asarray(batch["q"]), jnp.asarray(batch["qk"]),
                           jnp.asarray(batch["fb"]), jnp.asarray(batch["fv"]))
    static = dict(static)
    static["ws"] = static["ws"].at[:, 0, 1].set(jnp.asarray(batch["ws_can"]))
    static["shc"] = static["shc"].at[:, 0, 1].set(jnp.asarray(batch["shc_can"]))
    static = J.init_xklh_static(static)
    return static


def ref_cell(u):
    static_i, dynamic_i, forcing_i, ent_iters, refs, snowm = u
    cell = G.GhyColumn(static_i, dynamic_i, forcing_i)
    cell.fb, cell.fv = forcing_i["fb"], forcing_i["fv"]
    cell.snowm = snowm
    cell._bounds()
    return cell, ent_iters, refs


if __name__ == "__main__":
    batch = load_batch(400)
    static = build_static(batch)
    reth_out = J.reth(static, jnp.asarray(batch["w"]), jnp.asarray(batch["nsn"]), jnp.asarray(batch["wsn"]),
                      jnp.asarray(batch["fr_snow"]), jnp.asarray(batch["snowm"]))
    retp_out = J.retp(static, jnp.asarray(batch["w"]), jnp.asarray(batch["ht"]), jnp.asarray(batch["wsn"]),
                      jnp.asarray(batch["hsn"]))
    hydra_out = J.hydra(static, reth_out["theta"], retp_out["fice"])
    xklh_out = J.xklh(static, jnp.asarray(batch["w"]), retp_out["fice"], reth_out["theta"])

    worst = {}
    for i, u in enumerate(batch["unpacked"]):
        cell, _, _ = ref_cell(u)
        cell.reth(); cell.retp(); cell.hydra(); cell.xklh()
        n = cell.n

        def upd(name, mine, ref):
            worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(np.asarray(mine) - ref))))

        upd("theta", reth_out["theta"][i, :n + 1], cell.theta)
        upd("tp", retp_out["tp"][i, :n + 1], cell.tp)
        upd("fice", retp_out["fice"][i, :n + 1], cell.fice)
        upd("h", hydra_out["h"][i, :n + 1], cell.h[:n + 1])
        upd("d", hydra_out["d"][i, :n + 1], cell.d[:n + 1])
        upd("xku", hydra_out["xku"][i, :n + 1], cell.xku[:n + 1])
        upd("xk", hydra_out["xk"][i, 1:n + 1], cell.xk[1:n + 1])
        upd("xkh", xklh_out["xkh"][i, :n + 1], cell.xkh[:n + 1])
        upd("xkhm", xklh_out["xkhm"][i, :n + 1], cell.xkhm[:n + 1])
    for k, v in worst.items():
        print(f"{k:8s} max abs diff = {v:.3e}")
