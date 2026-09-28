"""Standalone validation of the non-snow flux chain (fl, flh, flg, flhg, runoff, fllmt, apply_fluxes)
against ghy_ref, using ghy_ref's own snow() output as a temporary stand-in for the not-yet-built
vectorized snow model."""
import sys
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_jax as J
import ghy_jax_compare as JC


def slice_static(static, i):
    return {k: v[i:i + 1] for k, v in static.items()}


def arrf(batch, key):
    return jnp.array([u[2][key] for u in batch["unpacked"]])


def run(n_cells=300):
    batch = JC.load_batch(n_cells)
    static = JC.build_static(batch)
    w = jnp.asarray(batch["w"]); ht = jnp.asarray(batch["ht"])
    nsn = jnp.asarray(batch["nsn"]); wsn = jnp.asarray(batch["wsn"]); hsn = jnp.asarray(batch["hsn"])
    fr_snow = jnp.asarray(batch["fr_snow"]); snowm = jnp.asarray(batch["snowm"])
    reth_out = J.reth(static, w, nsn, wsn, fr_snow, snowm)
    retp_out = J.retp(static, w, ht, wsn, hsn)
    hydra_out = J.hydra(static, reth_out["theta"], retp_out["fice"])
    xklh_out = J.xklh(static, w, retp_out["fice"], reth_out["theta"])

    dts = jnp.array([u[3][0]["dts"] if u[3] else 900.0 for u in batch["unpacked"]])
    cnc = jnp.array([u[3][0]["cnc"] if u[3] else 0.0 for u in batch["unpacked"]])
    lai = jnp.array([u[3][0]["lai"] if u[3] else 0.0 for u in batch["unpacked"]])
    betadl = jnp.stack([jnp.asarray(u[3][0]["betadl"]) if u[3] else jnp.zeros(6) for u in batch["unpacked"]])
    pr = arrf(batch, "pr"); prs = arrf(batch, "prs"); htpr = arrf(batch, "htpr")
    ch = arrf(batch, "ch"); vs = arrf(batch, "vs"); rho = arrf(batch, "rho"); pres = arrf(batch, "pres")
    qs = arrf(batch, "qs"); gusti = arrf(batch, "gusti"); qprime = arrf(batch, "qprime")
    qm1 = arrf(batch, "qm1"); ts = arrf(batch, "ts"); tprime = arrf(batch, "tprime")
    srht = arrf(batch, "srht"); trht = arrf(batch, "trht"); geothermal = arrf(batch, "geothermal_heat")
    sl = jnp.array([u[0]["sl"] for u in batch["unpacked"]])
    htprs = jnp.zeros_like(pr)

    evap_out = J.evap_limits(static, w, reth_out["theta"], hydra_out["d"], retp_out["tp"], retp_out["fice"],
                             retp_out["tsn1"], nsn, wsn, fr_snow, dts, pr, betadl, cnc, ch, vs, rho,
                             pres, qs, gusti, qprime, qm1, lai, reth_out["fm"])
    drip_out = J.drip_from_canopy(static, w, htpr, htprs, pr, prs, evap_out["evapvw"], evap_out["fw"],
                                  reth_out["fm"], fr_snow, reth_out["fd0"], dts, retp_out["tp"])
    sh_out = J.sensible_heat(retp_out["tp"], retp_out["tsn1"], ts, vs, ch, rho, gusti, tprime)
    f_out = J.fl(static, hydra_out["h"], hydra_out["xk"])

    worst = {}

    def upd(name, mine, ref):
        worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(np.asarray(mine) - np.asarray(ref)))))

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
        upd("dripw", drip_out["dripw"][i], cell.dripw)
        upd("drips", drip_out["drips"][i], cell.drips)
        # snshs must be checked HERE, before snow() overwrites self.snshs with snow_drv's adjusted value
        upd("snshs", sh_out["snshs"][i], cell.snshs)
        upd("tsn1", retp_out["tsn1"][i], cell.tsn1)
        cell.snow()

        n = cell.n
        st_i = slice_static(static, i)
        flg_out = J.flg(st_i, f_out["f"][i:i + 1], jnp.asarray(cell.flmlt)[None],
                        jnp.asarray(cell.flmlt_scale)[None], jnp.asarray(cell.dripw)[None],
                        jnp.asarray(cell.drips)[None], jnp.asarray(cell.evapb)[None],
                        jnp.asarray(cell.evapvg)[None], jnp.asarray(cell.fr_snow)[None],
                        jnp.asarray(cell.pr)[None], jnp.asarray(cell.evapvw)[None],
                        jnp.asarray(cell.evapbs)[None], jnp.asarray(cell.evapvs)[None],
                        jnp.asarray(cell.evapvd)[None], jnp.asarray(cell.fw)[None],
                        jnp.asarray(cell.fd)[None], jnp.asarray(cell.fm)[None])
        cell.fl(); cell.flg()
        upd("f_after_flg", flg_out["f"][0, :n + 1], cell.f[:n + 1])
        upd("fc", flg_out["fc"][0], cell.fc)
        upd("evap_tot", flg_out["evap_tot"][0], cell.evap_tot)

        flh_out = J.flh(st_i, xklh_out["xkhm"][i:i + 1], retp_out["tp"][i:i + 1], flg_out["f"],
                        geothermal[i:i + 1])
        cell.flh()
        upd("fh_after_flh", flh_out["fh"][0, :n + 1], cell.fh[:n + 1])

        flhg_out = J.flhg(st_i, flh_out["fh"], retp_out["tp"][i:i + 1], jnp.asarray(cell.fhsng)[None],
                          jnp.asarray(cell.fhsng_scale)[None], jnp.asarray(cell.htdripw)[None],
                          jnp.asarray(cell.htdrips)[None], jnp.asarray(cell.evapb)[None],
                          jnp.asarray(cell.evapvg)[None], jnp.asarray(cell.evapvw)[None],
                          jnp.asarray(cell.evapvd)[None], jnp.asarray(cell.snshg)[None],
                          jnp.asarray(cell.snshv)[None], jnp.asarray(cell.snshs)[None],
                          jnp.asarray(cell.thrmsn)[None], jnp.asarray(cell.fr_snow)[None],
                          srht[i:i + 1], trht[i:i + 1], htpr[i:i + 1], jnp.asarray(cell.fw)[None],
                          jnp.asarray(cell.fd)[None], jnp.asarray(cell.fm)[None])
        cell.flhg()
        upd("fh_after_flhg", flhg_out["fh"][0, :n + 1], cell.fh[:n + 1])
        upd("fch", flhg_out["fch"][0], cell.fch)
        upd("thrm_tot", flhg_out["thrm_tot"][0], cell.thrm_tot)
        upd("snsh_tot", flhg_out["snsh_tot"][0], cell.snsh_tot)

        runoff_out = J.runoff(st_i, w[i:i + 1], flg_out["f"], f_out["xinfc"][i:i + 1],
                              jnp.asarray(cell.dripw)[None], jnp.asarray(cell.dripw_scale)[None],
                              jnp.asarray(cell.evapb)[None], jnp.asarray(cell.evapvg)[None],
                              jnp.asarray(cell.fr_snow)[None], pr[i:i + 1], hydra_out["xku"][i:i + 1],
                              sl[i:i + 1])
        cell.runoff()
        upd("rnf", runoff_out["rnf"][0], cell.rnf)
        upd("rnff", runoff_out["rnff"][0, :n], cell.rnff[:n])

        fllmt_out = J.fllmt(st_i, w[i:i + 1], flh_out["fh"] * 0.0 + flg_out["f"], runoff_out["rnff"],
                            runoff_out["rnf"], jnp.asarray(cell.evapdl)[None] if cell.process_vege
                            else jnp.zeros((1, 6)), jnp.asarray(cell.fr_snow)[None], jnp.asarray(cell.fm)[None],
                            dts[i:i + 1])
        cell.fllmt()
        upd("f_after_fllmt", fllmt_out["f"][0, :n + 1], cell.f[:n + 1])
        upd("rnff_after_fllmt", fllmt_out["rnff"][0, :n], cell.rnff[:n])
        upd("rnf_after_fllmt", fllmt_out["rnf"][0], cell.rnf)

        apply_out = J.apply_fluxes(st_i, w[i:i + 1], ht[i:i + 1], fllmt_out["f"], flhg_out["fh"],
                                   flg_out["fc"], flhg_out["fch"], fllmt_out["rnf"], fllmt_out["rnff"],
                                   retp_out["tp"][i:i + 1], jnp.asarray(cell.evapdl)[None] if cell.process_vege
                                   else jnp.zeros((1, 6)), jnp.asarray(cell.fr_snow)[None],
                                   jnp.asarray(cell.fm)[None], dts[i:i + 1])
        cell.apply_fluxes()
        upd("w_final", apply_out["w"][0, :n + 1], cell.w[:n + 1])
        upd("ht_final", apply_out["ht"][0, :n + 1], cell.ht[:n + 1])

    for k, v in sorted(worst.items()):
        print(f"{k:18s} {v:.3e}")


if __name__ == "__main__":
    run()
