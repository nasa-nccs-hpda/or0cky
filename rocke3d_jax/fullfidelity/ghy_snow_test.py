"""Validate ghy_jax.snow() (and its snow_drv/snow_adv_1/snow_redistr/heat_eq building blocks) against
ghy_ref's cell.snow() on real land-cell substep data, using a NaN-aware comparison (max() silently
swallows NaN mismatches in a naive comparison, as discovered while chasing a degenerate 0/0 case in a
fully-emptied snow layer -- see PHASE0_LOG.md)."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_jax as J
import ghy_jax_compare as JC
import ghy_flux_chain_test as T


def nan_aware_diff(mine, ref):
    mine = np.asarray(mine, dtype=float); ref = np.asarray(ref, dtype=float)
    nm, nr = np.isnan(mine), np.isnan(ref)
    if not np.array_equal(nm, nr):
        return float("inf"), "nan-mismatch"
    safe = ~nm
    if not np.any(safe):
        return 0.0, "all-nan-both"
    return float(np.max(np.abs(mine[safe] - ref[safe]))), None


def run(n_cells=400, file_idx=0):
    batch = JC.load_batch(n_cells, file_idx=file_idx)
    static = JC.build_static(batch)
    w = jnp.asarray(batch["w"]); ht = jnp.asarray(batch["ht"])
    nsn = jnp.asarray(batch["nsn"]); wsn = jnp.asarray(batch["wsn"]); hsn = jnp.asarray(batch["hsn"])
    # raw dump only records NLSN=3 slots; GhyColumn.__init__ pads to NLSN+1=4 (self.dzsn[:NLSN]=dynamic['dzsn'])
    dzsn_raw = jnp.asarray(batch["dzsn"])
    dzsn_in = jnp.concatenate([dzsn_raw, jnp.zeros((dzsn_raw.shape[0], 1, 2))], axis=1)
    fr_snow = jnp.asarray(batch["fr_snow"]); snowm = jnp.asarray(batch["snowm"])
    reth_out = J.reth(static, w, nsn, wsn, fr_snow, snowm)
    retp_out = J.retp(static, w, ht, wsn, hsn)

    dts = jnp.array([u[3][0]["dts"] if u[3] else 900.0 for u in batch["unpacked"]])
    cnc = jnp.array([u[3][0]["cnc"] if u[3] else 0.0 for u in batch["unpacked"]])
    lai = jnp.array([u[3][0]["lai"] if u[3] else 0.0 for u in batch["unpacked"]])
    betadl = jnp.stack([jnp.asarray(u[3][0]["betadl"]) if u[3] else jnp.zeros(6) for u in batch["unpacked"]])
    pr = T.arrf(batch, "pr"); prs = T.arrf(batch, "prs"); htpr = T.arrf(batch, "htpr")
    ch = T.arrf(batch, "ch"); vs = T.arrf(batch, "vs"); rho = T.arrf(batch, "rho"); pres = T.arrf(batch, "pres")
    qs = T.arrf(batch, "qs"); gusti = T.arrf(batch, "gusti"); qprime = T.arrf(batch, "qprime")
    qm1 = T.arrf(batch, "qm1"); ts = T.arrf(batch, "ts"); tprime = T.arrf(batch, "tprime")
    srht = T.arrf(batch, "srht"); trht = T.arrf(batch, "trht")
    htprs = jnp.zeros_like(pr)

    hydra_out = J.hydra(static, reth_out["theta"], retp_out["fice"])
    evap_out = J.evap_limits(static, w, reth_out["theta"], hydra_out["d"], retp_out["tp"], retp_out["fice"],
                             retp_out["tsn1"], nsn, wsn, fr_snow, dts, pr, betadl, cnc, ch, vs, rho,
                             pres, qs, gusti, qprime, qm1, lai, reth_out["fm"])
    drip_out = J.drip_from_canopy(static, w, htpr, htprs, pr, prs, evap_out["evapvw"], evap_out["fw"],
                                  reth_out["fm"], fr_snow, reth_out["fd0"], dts, retp_out["tp"])
    sh_out = J.sensible_heat(retp_out["tp"], retp_out["tsn1"], ts, vs, ch, rho, gusti, tprime)

    snow_out = J.snow(static, retp_out["tp"], sh_out["snshs"], srht, trht, drip_out["drips"],
                      drip_out["dripw"], drip_out["htdrips"], drip_out["htdripw"], evap_out["devapbs_dt"],
                      evap_out["devapvs_dt"], sh_out["dsnsh_dt"], evap_out["evap_min"], dts,
                      static["dz"], dzsn_in, wsn, hsn, nsn, fr_snow, evap_out["evapbs"], evap_out["evapvs"],
                      reth_out["fm"])

    worst = {}
    nan_mismatches = []
    for i, u in enumerate(batch["unpacked"]):
        cell, ent_iters, refs = JC.ref_cell(u)
        cell.reth(); cell.retp(); cell.hydra(); cell.xklh()
        it = ent_iters[0] if ent_iters else dict(dts=900.0, cnc=0.0, betadl=np.zeros(6), lai=0.0)
        cell.dts = it["dts"]; cell.dt = cell.dts
        cell.cnc = it["cnc"] if cell.process_vege else 0.0
        cell.betadl = np.asarray(it["betadl"]) if cell.process_vege else np.zeros(cell.n)
        cell.lai = it["lai"] if cell.process_vege else 0.0
        cell.evap_limits(True)
        cell.drip_from_canopy()
        cell.sensible_heat()
        try:
            cell.snow()
        except Exception as e:
            continue   # reference itself raised (e.g. RuntimeWarning-as-error, degenerate 0/0) -- skip

        n = cell.n
        for name, mine, ref in (
            ("flmlt", snow_out["flmlt"][i], cell.flmlt),
            ("fhsng", snow_out["fhsng"][i], cell.fhsng),
            ("flmlt_scale", snow_out["flmlt_scale"][i], cell.flmlt_scale),
            ("fhsng_scale", snow_out["fhsng_scale"][i], cell.fhsng_scale),
            ("thrmsn", snow_out["thrmsn"][i], cell.thrmsn),
            ("nsn", snow_out["nsn"][i], cell.nsn),
            ("fr_snow", snow_out["fr_snow"][i], cell.fr_snow),
            ("evapbs", snow_out["evapbs"][i], cell.evapbs),
            ("evapvs", snow_out["evapvs"][i], cell.evapvs),
            ("snshs", snow_out["snshs"][i], cell.snshs),
        ):
            d, flag = nan_aware_diff(mine, ref)
            if flag == "nan-mismatch":
                nan_mismatches.append((i, name))
            else:
                worst[name] = max(worst.get(name, 0.0), d)

    for k, v in sorted(worst.items()):
        print(f"{k:14s} {v:.3e}")
    print(f"NaN-pattern mismatches: {len(nan_mismatches)} / {len(batch['unpacked'])} cells")
    if nan_mismatches[:10]:
        print("first few:", nan_mismatches[:10])


if __name__ == "__main__":
    run()
