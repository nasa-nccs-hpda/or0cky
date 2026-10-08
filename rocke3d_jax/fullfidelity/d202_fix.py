"""D202 PROPOSED FIX (new file, tracked files untouched): ghy_jax.advnc with the Fortran time loop of GHY.f:2389-2416 inside the lax.scan.

The port (ghy_jax.advnc as called from jax_surface._land) takes the GHY sub-iteration schedule (nsub = recorded ffnit, dts = recorded Ent dts) from the REAL
record of the step.  The real model derives it from the CURRENT state: `do while (dtr > 0): hydra; xklh; gdtm(dtm); dts = dtr if dtm >= dtr else min(dtm, dtr/2)`
(giss_LSM/GHY.f:2389-2416, gdtm at 3057-3145; stability limit 0.5*ak2/xk2 with xk2 = sha*rho*ch*vs + betas*rho3*ch*vs*elh*dqdt + 8*stbo*T^3).  For the
free-running port the state, and with it ch*vs, leaves the recorded one; a recorded one-iteration (dts = 900 s) schedule is then explicit-Euler beyond the limit.

advnc_gdtm has the signature of ghy_jax.advnc and replaces `ent_dts`/`n_substeps` by the in-loop gdtm schedule.  Ent exports of iteration i are those of
recorded iteration min(i, W0-1) (the template pads with the last recorded iteration; the same assumption ghy_ref_nit.advnc_full makes for iterations > 11).
If the loop would need more than `width` iterations the last lane takes the remainder dtr in one step (Fortran stops the model there); `nit_exhausted` flags it.
    import d202_fix; d202_fix.install(width=40)      # monkeypatches ghy_jax.advnc (jax_surface calls J.advnc at call time)
"""
import jax
import jax.numpy as jnp
import ghy_jax as J

_orig_advnc = J.advnc
WIDTH = 40


def _ep(tp, forcing):
    rho3 = forcing["rho"] / J.RHOW
    vq = forcing["gusti"] * forcing["qprime"]
    qb = J.qsat(tp[:, 1, 0] + J.TFRZ, J.LHE, forcing["pres"])
    qv = J.qsat(tp[:, 0, 1] + J.TFRZ, J.LHE, forcing["pres"])
    epb = rho3 * forcing["ch"] * (forcing["vs"] * (qb - forcing["qs"]) - vq)
    epv = rho3 * forcing["ch"] * (forcing["vs"] * (qv - forcing["qs"]) - vq)
    return epb, epv


def gdtm(static, forcing, w, fice, tp, d, xkh, fw, evapb, epb, evapvw, evapvd, epv):
    """GHY.f gdtm (giss_LSM/GHY.f:3057), batched; mirrors ghy_ref.GhyColumn.gdtm."""
    kmask = static["kmask"]
    ibv_active = jnp.stack([static["process_bare"], static["process_vege"]], axis=-1)       # (N,2)
    act = kmask[:, :, None] & ibv_active[:, None, :]                                        # (N,NGM,2)
    dz = static["dz"]
    dz_s = jnp.where(dz > 0.0, dz, 1.0)[:, :, None]
    dqdt = J.dqsatdt(forcing["ts"], J.LHE) * J.qsat(forcing["ts"], J.LHE, forcing["pres"])
    dldz2 = jnp.max(jnp.where(act, d[:, 1:, :] / dz_s ** 2, 0.0), axis=(1, 2))
    dtm = 1.0 / (dldz2 + 1e-12)
    dtm = jnp.where(static["q"][:, 3, 0] > 0.0, jnp.minimum(dtm, 450.0), dtm)
    ak1 = (static["shc"][:, 1:, :] + ((1.0 - fice[:, 1:, :]) * J.SHW + fice[:, 1:, :] * J.SHI) * w[:, 1:, :]) / dz_s
    t1 = 0.5 * ak1 * dz_s ** 2 / (xkh[:, 1:, :] + 1e-12)
    dtm = jnp.minimum(dtm, jnp.min(jnp.where(act, t1, jnp.inf), axis=(1, 2)))
    cna = forcing["ch"] * forcing["vs"]
    rho3 = 0.001 * forcing["rho"]
    betas0 = jnp.where(epb <= 0.0, 1.0, evapb / jnp.where(epb <= 0.0, 1.0, epb))
    betas1 = jnp.where(epv <= 0.0, 1.0, (evapvw * fw + evapvd * (1.0 - fw)) / jnp.where(epv <= 0.0, 1.0, epv))
    for ibv, betas in ((0, betas0), (1, betas1)):
        k = 1 - ibv
        xk2 = J.SHA * forcing["rho"] * cna + betas * rho3 * cna * J.ELH * dqdt + 8.0 * J.STBO * (tp[:, k, ibv] + J.TFRZ) ** 3
        ak2 = static["shc"][:, k, ibv] + ((1.0 - fice[:, k, ibv]) * J.SHW + fice[:, k, ibv] * J.SHI) * w[:, k, ibv]
        dtm = jnp.where(ibv_active[:, ibv], jnp.minimum(dtm, 0.5 * ak2 / (xk2 + 1e-12)), dtm)
    return dtm


def advnc_gdtm(static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt, snowm, max_substeps=11):
    W = max(int(max_substeps), WIDTH)
    W0 = ent_dts.shape[1]
    N = dynamic0["w"].shape[0]
    fb, fv = forcing["fb"], forcing["fv"]
    static = dict(static0, process_bare=fb > 0.0, process_vege=fv > 0.0, fb=fb, fv=fv, sl=static0["sl"])
    w0 = dynamic0["w"]; ht0 = dynamic0["ht"]; nsn0 = dynamic0["nsn"]; dzsn0 = dynamic0["dzsn"]
    wsn0 = dynamic0["wsn"]; hsn0 = dynamic0["hsn"]; fr_snow0 = dynamic0["fr_snow"]
    reth0 = J.reth(static, w0, nsn0, wsn0, fr_snow0, snowm)
    retp0 = J.retp(static, w0, ht0, wsn0, hsn0)
    one = jnp.ones(N)
    init_carry = dict(
        w=w0, ht=ht0, nsn=nsn0, dzsn=dzsn0, wsn=wsn0, hsn=hsn0, fr_snow=fr_snow0,
        theta=reth0["theta"], fice=retp0["fice"], tp=retp0["tp"], tsn1=retp0["tsn1"],
        fw=reth0["fw"], fd=reth0["fd"], fm=reth0["fm"], fw0=reth0["fw0"], fd0=reth0["fd0"],
        abetad=jnp.zeros(N), snsh_tot_carry=jnp.zeros((N, 2)), evap_tot_carry=jnp.zeros((N, 2)),
        acc=J.accm_zero(N),
        dtr=900.0 * one,
        nit=jnp.zeros(N, dtype=jnp.int32), exhausted=jnp.zeros(N, dtype=bool),
        epb=one, epv=one, evapb=one, evapvw=one, evapvd=one)
    # `dt` of the template is the total time of the recorded schedule (sum of recorded dts, 900 s for the cells it covers); the Fortran loop runs over the
    # whole DTsrc/2 = 900 s, as ghy_ref_nit.advnc_full does (dt=900)
    dt_run = 900.0 * one

    def _gather(a, i):
        j = jnp.minimum(i, W0 - 1)
        return a[:, j]

    def _substep_body(carry, i):
        w, ht, nsn, dzsn, wsn, hsn, fr_snow = (carry["w"], carry["ht"], carry["nsn"], carry["dzsn"], carry["wsn"], carry["hsn"], carry["fr_snow"])
        theta, fice, tp, tsn1 = carry["theta"], carry["fice"], carry["tp"], carry["tsn1"]
        fw, fd, fm, fw0, fd0 = carry["fw"], carry["fd"], carry["fm"], carry["fw0"], carry["fd0"]
        acc = carry["acc"]
        dtr = carry["dtr"]
        active_i = dtr > 0.0
        cnc = jnp.where(static["process_vege"], _gather(ent_cnc, i), 0.0)
        betadl = jnp.where(static["process_vege"][:, None], _gather(ent_betadl, i), 0.0)
        lai = jnp.where(static["process_vege"], _gather(ent_lai, i), 0.0)
        hydra_out = J.hydra(static, theta, fice)
        _z = jnp.zeros_like(forcing["pr"])
        irrig2 = jnp.stack([_z, forcing["irrig"] if "irrig" in forcing else _z], axis=-1)
        htirrig2 = jnp.stack([_z, forcing["htirrig"] if "htirrig" in forcing else _z], axis=-1)
        xklh_out = J.xklh(static, w, fice, theta)
        # ---- the Fortran time step (GHY.f:2408-2415)
        dtm = gdtm(static, forcing, w, fice, tp, hydra_out["d"], xklh_out["xkh"], fw, carry["evapb"], carry["epb"], carry["evapvw"], carry["evapvd"], carry["epv"])
        last_lane = i == W - 1
        full = (dtm >= dtr) | last_lane
        dts = jnp.where(full, dtr, jnp.minimum(dtm, dtr * 0.5))
        dtr_new = jnp.where(full, 0.0, dtr - dts)
        dts = jnp.where(active_i, dts, 1.0)
        evap_out = J.evap_limits(static, w, theta, hydra_out["d"], tp, fice, tsn1, nsn, wsn, fr_snow, dt_run, forcing["pr"], betadl, cnc, forcing["ch"],
                                 forcing["vs"], forcing["rho"], forcing["pres"], forcing["qs"], forcing["gusti"], forcing["qprime"], forcing["qm1"], lai, fm)
        epb, epv = _ep(tp, forcing)
        fw_i, fd_i = evap_out["fw"], evap_out["fd"]
        drip_out = J.drip_from_canopy(static, w, forcing["htpr"], forcing["htprs"], forcing["pr"], forcing["prs"], evap_out["evapvw"], fw_i, fm, fr_snow, fd0, dts, tp)
        sh_out = J.sensible_heat(tp, tsn1, forcing["ts"], forcing["vs"], forcing["ch"], forcing["rho"], forcing["gusti"], forcing["tprime"])
        snow_out = J.snow(static, tp, sh_out["snshs"], forcing["srht"], forcing["trht"], drip_out["drips"], drip_out["dripw"], drip_out["htdrips"],
                          drip_out["htdripw"], evap_out["devapbs_dt"], evap_out["devapvs_dt"], sh_out["dsnsh_dt"], evap_out["evap_min"], dts, static["dz"],
                          dzsn, wsn, hsn, nsn, fr_snow, evap_out["evapbs"], evap_out["evapvs"], fm)
        f_out = J.fl(static, hydra_out["h"], hydra_out["xk"])
        flg_out = J.flg(static, f_out["f"], snow_out["flmlt"], snow_out["flmlt_scale"], drip_out["dripw"], drip_out["drips"], evap_out["evapb"],
                        evap_out["evapvg"], snow_out["fr_snow"], forcing["pr"], evap_out["evapvw"], snow_out["evapbs"], snow_out["evapvs"], evap_out["evapvd"],
                        fw_i, fd_i, fm, irrig2)
        runoff_out = J.runoff(static, w, flg_out["f"], f_out["xinfc"], drip_out["dripw"], drip_out["dripw_scale"], evap_out["evapb"], evap_out["evapvg"],
                              snow_out["fr_snow"], forcing["pr"], hydra_out["xku"], static["sl"])
        fllmt_out = J.fllmt(static, w, flg_out["f"], runoff_out["rnff"], runoff_out["rnf"], evap_out["evapdl"], snow_out["fr_snow"], fm, dts)
        flh_out = J.flh(static, xklh_out["xkhm"], tp, fllmt_out["f"], forcing["geothermal_heat"])
        flhg_out = J.flhg(static, flh_out["fh"], tp, snow_out["fhsng"], snow_out["fhsng_scale"], drip_out["htdripw"], drip_out["htdrips"], evap_out["evapb"],
                          evap_out["evapvg"], evap_out["evapvw"], evap_out["evapvd"], sh_out["snshg"], sh_out["snshv"], snow_out["snshs"], snow_out["thrmsn"],
                          snow_out["fr_snow"], forcing["srht"], forcing["trht"], forcing["htpr"], fw_i, fd_i, fm, htirrig2)
        apply_out = J.apply_fluxes(static, w, ht, fllmt_out["f"], flhg_out["fh"], flg_out["fc"], flhg_out["fch"], fllmt_out["rnf"], fllmt_out["rnff"], tp,
                                   evap_out["evapdl"], snow_out["fr_snow"], fm, dts)
        acc_new = J.accm(acc, static, tp, flhg_out["thrm_tot"], flhg_out["snsh_tot"], flg_out["evap_tot"], fllmt_out["rnf"], fllmt_out["rnff"], fllmt_out["f"],
                         flhg_out["fh"], forcing["srht"], forcing["trht"], forcing["htpr"], dts)
        w_new, ht_new = apply_out["w"], apply_out["ht"]
        nsn_new, dzsn_new = snow_out["nsn"], snow_out["dzsn"]
        wsn_new, hsn_new, fr_snow_new = snow_out["wsn"], snow_out["hsn"], snow_out["fr_snow"]
        reth_new = J.reth(static, w_new, nsn_new, wsn_new, fr_snow_new, snowm)
        retp_new = J.retp(static, w_new, ht_new, wsn_new, hsn_new)
        S = lambda new, old: J._sel(active_i, new, old)
        new_carry = dict(
            w=S(w_new, w), ht=S(ht_new, ht), nsn=S(nsn_new, nsn), dzsn=S(dzsn_new, dzsn), wsn=S(wsn_new, wsn), hsn=S(hsn_new, hsn),
            fr_snow=S(fr_snow_new, fr_snow), theta=S(reth_new["theta"], theta), fice=S(retp_new["fice"], fice), tp=S(retp_new["tp"], tp),
            tsn1=S(retp_new["tsn1"], tsn1), fw=S(reth_new["fw"], fw), fd=S(reth_new["fd"], fd), fm=S(reth_new["fm"], fm),
            fw0=S(reth_new["fw0"], fw0), fd0=S(reth_new["fd0"], fd0), abetad=S(evap_out["abetad"], carry["abetad"]),
            snsh_tot_carry=S(flhg_out["snsh_tot"], carry["snsh_tot_carry"]), evap_tot_carry=S(flg_out["evap_tot"], carry["evap_tot_carry"]),
            acc={k: S(acc_new[k], acc[k]) for k in acc},
            dtr=jnp.where(active_i, dtr_new, dtr), nit=carry["nit"] + active_i.astype(jnp.int32),
            exhausted=carry["exhausted"] | (active_i & last_lane & (dtm < dtr)),
            epb=S(epb, carry["epb"]), epv=S(epv, carry["epv"]), evapb=S(evap_out["evapb"], carry["evapb"]),
            evapvw=S(evap_out["evapvw"], carry["evapvw"]), evapvd=S(evap_out["evapvd"], carry["evapvd"]))
        return new_carry, None

    final_carry, _ = jax.lax.scan(_substep_body, init_carry, jnp.arange(W), length=W)
    acc = final_carry["acc"]
    final = J.accm_final(acc, static, fb, fv, final_carry["snsh_tot_carry"], final_carry["evap_tot_carry"], dt_run, forcing["rho"], forcing["ch"], forcing["ts"],
                         forcing["gusti"], forcing["tprime"], forcing["vs"])
    rows = jnp.arange(N)
    last = jnp.clip(final_carry["nit"] - 1, 0, W0 - 1)
    pv = static["process_vege"]
    hyd_f = J.hydra(static, final_carry["theta"], final_carry["fice"])
    ev_f = J.evap_limits(static, final_carry["w"], final_carry["theta"], hyd_f["d"], final_carry["tp"], final_carry["fice"], final_carry["tsn1"],
                         final_carry["nsn"], final_carry["wsn"], final_carry["fr_snow"], dt_run, forcing["pr"],
                         jnp.where(pv[:, None], ent_betadl[rows, last], 0.0), jnp.where(pv, ent_cnc[rows, last], 0.0), forcing["ch"], forcing["vs"],
                         forcing["rho"], forcing["pres"], forcing["qs"], forcing["gusti"], forcing["qprime"], forcing["qm1"],
                         jnp.where(pv, ent_lai[rows, last], 0.0), final_carry["fm"])
    bad = jnp.isnan(ev_f["evap_max_out"])
    return dict(w=final_carry["w"], ht=final_carry["ht"], nsn=final_carry["nsn"], dzsn=final_carry["dzsn"], wsn=final_carry["wsn"], hsn=final_carry["hsn"],
                fr_snow=final_carry["fr_snow"], tp=final_carry["tp"], fice=final_carry["fice"], tbcs=final["tbcs"], tsns=final["tsns"], ashg=acc["ashg"],
                alhg=acc["alhg"], aevap=final["aevap"], aruns=final["aruns"], arunu=final["arunu"], aeruns=acc["aeruns"], aerunu=acc["aerunu"], ae0=acc["ae0"],
                abetad=final_carry["abetad"], evap_max_ij=jnp.where(bad, 0.0, ev_f["evap_max_out"]), fr_sat_ij=jnp.where(bad, 0.0, ev_f["fr_sat"]),
                nit_gdtm=final_carry["nit"], nit_exhausted=final_carry["exhausted"])


def install(width=40):
    global WIDTH
    WIDTH = int(width)
    J.advnc = advnc_gdtm


# ====================================================================================================== FIX 1 (the demonstrated defect): GHY gusti = the gusti the PBL used
_orig_land = None


def install_gusti():
    """jax_surface._land builds the GHY forcing from the RECORDED ffg batch and overrides ts, qs, rho, ch, vs, tprime, qprime, qm1 with the port's PBL values,
    but NOT `gusti`: GHY keeps the recorded ffg gusti_in (column 160).  GHY_DRV.f:1267 passes pbl_args%gusti, the gusti the PBL just used (the real records
    agree bitwise: ffg col 160 == ffp col 113 on every land cell).  Whenever the port's downdraft flag/DDMS differs from the real one (ddml=1 and gusti>0 in the
    port, 0 in the record) PBL (gusti) and GHY (0) are inconsistent: the PBL surface humidity qsrf contains the downdraft term gusti*(qtop-qdns)/ws, GHY
    sees it without the compensating -gusti*qprime in epb/epv and snsh.  This sets forcing['gusti'] = p4[:, 113] (the PBL gusti column)."""
    global _orig_land
    import jax_surface as JS
    if _orig_land is None:
        _orig_land = JS._land

    def _land_gusti(p4, out, gb, dyn, q1_land, ma1_land, trup, max_substeps):
        fo = dict(gb['forcing'])
        fo['gusti'] = p4[:, JS.PBL_GUSTI_OUT]
        gb2 = dict(gb, forcing=fo)
        return _orig_land(p4, out, gb2, dyn, q1_land, ma1_land, trup, max_substeps)
    JS._land = _land_gusti
