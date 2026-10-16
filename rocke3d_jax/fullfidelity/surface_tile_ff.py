"""Full-fidelity port of the SURFACE.f open-ocean/lake and sea-ice tile flux logic -- Track B.

Transcription of modelE2_planet_2.0/model/SURFACE.f (ITYPE_OCEAN, ITYPE_OCEANICE branches, after the
PBL call): skin-effect adjustment of the ground state, sensible/latent/thermal fluxes (explicit for
open water, implicit two-layer for sea ice), evaporation limits (lake minimum depth, dew limit),
lake heat-flux limit, and the flux/tendency outputs handed to the atmosphere (DTH1, DQ1, DMUA, DMVA).
Inputs are the tile state at PBL entry and the PBL outputs (from `pbl_ff.advanc`). Float64.

The ice thermal properties (dF1dTG, HCG1, HCG2, FSRI) are computed in SURFACE.f from SEAICE.f
helper functions; they are taken here as inputs (`ice_props` ports them separately).
"""
import numpy as np
import jax.numpy as jnp
import pbl_ff as P

TF, STBO, SHA, SHV, LHE, RGAS = P.TF, P.STBO, P.SHA, P.SHV, P.LHE, P.RGAS
_C_DQSAT = 1.0 / P.RVAP


def dqsatdt(tm, lh):
    return lh * _C_DQSAT / (tm * tm)


# column names of the ffs_*.bin record (0-based column indices; see instrumentation/SURFACE.f.patch)
IN = dict(i=0, j=1, itype=2, ns=3, ptype=4, tg1=5, tg2=6, tr4=7, snow=8, msi2=9, dF1dTG=10, hcg1=11, hcg2=12,
          fsri1=13, fsri2=14, srheat=15, elhx=16, uocean=17, vocean=18, sss=19, ps=20, ma1=21, q1=22, thv1=23,
          trhr0=24, dtsurf=25, flake=26, focean=27, evaplim=28, htlim=29, e0=30, evapor=31, byma1=32, ssi1=33,
          ssi2=34, flag_dsws=35, tgo=36, mwl=37, gml=38, axyp=39,
          us=40, vs=41, ws=42, gusti=43, qsrf=44, cm=45, ch=46, cq=47, ts=48, dskin=49, tprime=50, qprime=51,
          tsv=52, trup_in_rad=80)
OUT = dict(tg1=59, tg2=60, tr4=61, dth1=62, dq1=63, dmua=64, dmva=65, shdt=66, evhdt=67, trhdt=68, evap=69,
           f0dt=70, f1dt=71, qg_sat=72, tg=73, rcdmws=74, rhosrf=75, dq1x=76, dlwdt=77)


def tile_fluxes(d):
    """d: dict of arrays (N,) from IN names. Returns dict of outputs (N,) named as OUT."""
    itype = d["itype"]
    ocean_t = itype == 1
    dtsurf = d["dtsurf"]
    tg = d["tg1"] + TF
    # skin effect adjustment of ground variables (after the PBL call)
    tg = tg + d["dskin"]
    ocean_flag = ocean_t & (d["focean"] > 0)
    qg_sat = P.qsat(tg, d["elhx"], d["ps"])
    qg_sat = jnp.where(ocean_flag, 0.98 * qg_sat, qg_sat)
    tg1 = tg - TF
    tr4 = (jnp.sqrt(jnp.sqrt(d["tr4"])) + d["dskin"]) ** 4

    rhosrf = 100. * d["ps"] / (RGAS * d["tsv"])
    rcdmws = d["cm"] * d["ws"] * rhosrf
    rcdhws = d["ch"] * d["ws"] * rhosrf
    rcdqws = d["cq"] * d["ws"] * rhosrf
    rcdhdws = d["ch"] * d["gusti"] * rhosrf
    rcdqdws = d["cq"] * d["gusti"] * rhosrf
    sheat = SHA * (rcdhws * (d["ts"] - tg) + rcdhdws * d["tprime"])
    evheat = (LHE + tg1 * SHV) * (rcdqws * (d["qsrf"] - qg_sat) + rcdqdws * d["qprime"])
    trheat = d["trhr0"] - STBO * tr4

    # open water: explicit
    shdt_o, evhdt_o, trhdt_o = dtsurf * sheat, dtsurf * evheat, dtsurf * trheat

    # sea ice: implicit two-layer
    dF1dTG, hcg1, hcg2, srheat = d["dF1dTG"], d["hcg1"], d["hcg2"], d["srheat"]
    f1 = (tg1 - d["tg2"]) * dF1dTG + srheat * d["fsri1"]
    f2 = srheat * d["fsri2"]
    evheat_i = LHE * (rcdqws * (d["qsrf"] - qg_sat) + rcdqdws * d["qprime"])
    f0 = srheat + trheat + sheat + evheat_i
    dsndtg = -rcdhws * SHA
    dqgdtg = qg_sat * dqsatdt(tg, d["elhx"])
    devdtg = -dqgdtg * LHE * rcdqws
    dtrdtg = -4 * STBO * jnp.sqrt(jnp.sqrt(tr4)) ** 3
    df0dtg = dsndtg + devdtg + dtrdtg
    t2den = hcg2 + dtsurf * dF1dTG
    t2con = dtsurf * (f1 - f2) / t2den
    t2mul = dtsurf * dF1dTG / t2den
    dfdtg = df0dtg - dF1dTG
    dtg = (f0 - f1) * dtsurf / (hcg1 - dtsurf * dfdtg)
    dtg = jnp.where(tg1 + dtg > 0., -tg1, dtg)
    dt2 = t2con + t2mul * dtg
    shdt_i = dtsurf * (sheat + dtg * dsndtg)
    evhdt_i = dtsurf * (evheat_i + dtg * devdtg)
    trhdt_i = dtsurf * (trheat + dtg * dtrdtg)
    f1dt_i = dtsurf * (f1 + (dtg * dF1dTG - dt2 * dF1dTG))

    shdt = jnp.where(ocean_t, shdt_o, shdt_i)
    evhdt = jnp.where(ocean_t, evhdt_o, evhdt_i)
    trhdt = jnp.where(ocean_t, trhdt_o, trhdt_i)
    f1dt = jnp.where(ocean_t, 0.0, f1dt_i)
    tg1_n = jnp.where(ocean_t, tg1, tg1 + dtg)
    tg2_n = jnp.where(ocean_t, d["tg2"], d["tg2"] + dt2)

    # evaporation and limits
    ma1 = d["ma1"]
    dq1x = evhdt / ((LHE + tg1_n * SHV) * ma1)
    evhdt0 = evhdt
    lake = ocean_t & (d["flake"] > 0) & ((d["evapor"] - dq1x * ma1) > d["evaplim"])
    dew = (~lake) & (dq1x > d["q1"])
    dq1x = jnp.where(lake, (d["evapor"] - d["evaplim"]) * d["byma1"], jnp.where(dew, d["q1"], dq1x))
    lim = lake | dew
    evhdt = jnp.where(lim, dq1x * (LHE + tg1_n * SHV) * ma1, evhdt)
    tg1_n = jnp.where(lim & ~ocean_t, tg1_n + (evhdt - evhdt0) / hcg1, tg1_n)
    evap = -dq1x * ma1

    f0dt = dtsurf * srheat + trhdt + shdt + evhdt
    # lake heat-flux limit
    lakelim = ocean_t & (d["flake"] > 0) & (d["e0"] + f0dt + d["htlim"] < 0) & (d["e0"] + f0dt < 0)
    shdt = jnp.where(lakelim, -(jnp.maximum(0., d["htlim"]) + d["e0"] + dtsurf * srheat + trhdt + evhdt), shdt)
    f0dt = jnp.where(lakelim, -d["e0"] - jnp.maximum(0., d["htlim"]), f0dt)

    dlwdt = dtsurf * (d["trup_in_rad"] - d["trhr0"]) + trhdt
    dth1 = -(shdt + dlwdt) / (SHA * ma1)
    dmua = rcdmws * (d["us"] - d["uocean"])
    dmva = rcdmws * (d["vs"] - d["vocean"])
    return dict(tg1=tg1_n, tg2=tg2_n, tr4=tr4, dth1=dth1, dq1=-dq1x, dmua=dmua, dmva=dmva, shdt=shdt,
                evhdt=evhdt, trhdt=trhdt, evap=evap, f0dt=f0dt, f1dt=f1dt, qg_sat=qg_sat, tg=tg,
                rcdmws=rcdmws, rhosrf=rhosrf, dq1x=dq1x, dlwdt=dlwdt)


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, 90)


def to_inputs(rec):
    return {k: jnp.asarray(rec[:, c]) for k, c in IN.items()}
