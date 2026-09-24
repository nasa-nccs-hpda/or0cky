"""Full-fidelity port of SURFACE_LANDICE.f tile flux logic (implicit two-layer land ice) -- Track B, float64.
Inputs: tile state at PBL entry + PBL outputs; outputs: fluxes and atmosphere-facing tendencies."""
import numpy as np
import jax.numpy as jnp
import pbl_ff as P
from ice_props_ff import RHOI, SHI, ALAMI0, BYRLS

TF, STBO, SHA, SHV, LHE, LHS, RGAS = P.TF, P.STBO, P.SHA, P.SHV, P.LHE, P.LHS, P.RGAS
Z1E, Z2LI = 0.1, 2.9
ACE1LI = Z1E * RHOI
HC1LI = ACE1LI * SHI
Z2LI3L = Z2LI / (3. * ALAMI0)
Z1LIBYL = Z1E / ALAMI0
BY6 = 1.0 / 6.0
_C = 1.0 / P.RVAP

IN = dict(i=0, j=1, ihc=2, ptype=3, tg1=4, tg2=5, tr4=6, snow=7, srheat=8, ps=9, ma1=10, q1=11, flong=12,
          dtsurf=13, trup_in_rad=14, us=15, vs=16, ws=17, gusti=18, qsrf=19, cm=20, ch=21, cq=22, ts=23,
          dskin=24, tprime=25, qprime=26, tsv=27)
OUT = dict(tg1=29, dth1=30, dq1=31, uflux1=32, vflux1=33, shdt=34, evhdt=35, trhdt=36, evap=37, f0dt=38,
           f1dt=39, dlwdt=40, dq1x=41, qg_sat=42, tg=43, rcdmws=44)


def load(path):
    return np.fromfile(path, ">f8").astype(np.float64).reshape(-1, 60)


def to_inputs(rec):
    return {k: jnp.asarray(rec[:, c]) for k, c in IN.items()}


def tile_fluxes(d):
    dts = d["dtsurf"]
    tg = d["tg1"] + TF + d["dskin"]
    qg_sat = P.qsat(tg, LHS, d["ps"])
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
    trheat = d["flong"] - STBO * tr4
    snow = d["snow"]
    z1by6l = (Z1LIBYL + snow * BYRLS) * BY6
    cdterm = d["tg2"]
    cdenom = 1. / (2. * z1by6l + Z2LI3L)
    hcg1 = HC1LI + snow * SHI
    f0 = d["srheat"] + trheat + sheat + evheat
    f1 = (tg1 - cdterm - f0 * z1by6l) * cdenom
    dshdtg = -rcdhws * SHA
    dqgdtg = qg_sat * (LHS * _C / (tg * tg))
    devdtg = -rcdqws * LHE * dqgdtg
    dtrdtg = -4. * STBO * jnp.sqrt(jnp.sqrt(tr4)) ** 3
    df0dtg = dshdtg + devdtg + dtrdtg
    dfdtg = df0dtg - (1. - df0dtg * z1by6l) * cdenom
    dtg = (f0 - f1) * dts / (hcg1 - dts * dfdtg)
    shdt = dts * (sheat + dtg * dshdtg)
    evhdt = dts * (evheat + dtg * devdtg)
    trhdt = dts * (trheat + dtg * dtrdtg)
    f1dt = dts * (tg1 - cdterm - (f0 + dtg * dfdtg) * z1by6l) * cdenom
    tg1n = tg1 + dtg
    ma1 = d["ma1"]
    dq1x = evhdt / ((LHE + tg1n * SHV) * ma1)
    evhdt0 = evhdt
    dew = dq1x > d["q1"]
    dq1x = jnp.where(dew, d["q1"], dq1x)
    evhdt = jnp.where(dew, dq1x * (LHE + tg1n * SHV) * ma1, evhdt)
    tg1n = jnp.where(dew, tg1n + (evhdt - evhdt0) / hcg1, tg1n)
    evap = -dq1x * ma1
    f0dt = dts * d["srheat"] + trhdt + shdt + evhdt
    dlwdt = dts * (d["trup_in_rad"] - d["flong"]) + trhdt
    dth1 = -(shdt + dlwdt) / (SHA * ma1)
    return dict(tg1=tg1n, dth1=dth1, dq1=-dq1x, uflux1=rcdmws * d["us"], vflux1=rcdmws * d["vs"], shdt=shdt,
                evhdt=evhdt, trhdt=trhdt, evap=evap, f0dt=f0dt, f1dt=f1dt, dlwdt=dlwdt, dq1x=dq1x,
                qg_sat=qg_sat, tg=tg, rcdmws=rcdmws)
