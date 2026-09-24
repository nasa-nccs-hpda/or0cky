"""
Physics-only JAX orchestrator for the P2SAoM40 comparison
=============================================================

Chains the *validated* ROCKE-3D JAX physics modules (pbl.py,
fluxes_jax.py / surface_jax.py, radiation_jax.py, drycnv.py,
atm_com_jax.compute_pk_pek) together, in the real Fortran call order
and cadence (NIsurf=2 surface substeps per DTsrc, NRAD=5 radiation
gating), driven by real 72x46x40 restart state (see p2saom40_io.py).

Explicitly out of scope (not ported / not faithfully ported -- see
STATUS.md for the fidelity notes):
  - Atmospheric dynamical core, moist convection (CONDSE), ocean GCM.
  - seaice_jax's/lakes_jax's state-evolving thermodynamics (prec_si,
    addice, simelt, sea_ice, lkmix) are documented placeholders/no-ops
    in the existing port and are NOT called here -- sea ice / lake
    surface temperatures are held fixed at their restart values for
    the surface-flux calculation, they are not prognostically updated.
  - aturb_jax's tridiagonal PBL turbulence solver is not used (it has
    placeholder PBL-top-finding); layer-1 turbulent tendencies are
    instead applied directly and energy-consistently from the computed
    surface fluxes, which is the physically-correct leading-order
    coupling for a single explicit DTsrc step.

Bug found while driving fluxes_jax.py/surface_jax.py with real data:
their compute_momentum_flux/compute_heat_flux/compute_moisture_flux
formulas derive the wind-speed factor `ws` from the *surface* reference
wind (us, vs) alone (`ws = sqrt(us**2+vs**2)`), not the atmosphere-
relative wind. Since the physically correct value is us=vs=0 for
land/land-ice (no-slip surface), this makes `ws` -- and therefore every
flux -- identically zero for those cells; ocean/seaice cells are also
wrong (ws should be |V_atm - V_surface|, not |V_surface| alone). Their
existing Fortran-comparison unit tests apparently used arbitrary
nonzero (us, vs) test values, so this never surfaced there. Rather than
modify the shared, previously-validated module files, this driver
reimplements the same bulk-formula physics locally in
`_surface_fluxes_relative_wind()` below with the corrected `ws`.

Grid convention: horizontal fields are (JM, IM) = (46, 72); profile
fields are (JM, IM, LM) with level index 0 = lowest atmospheric layer,
matching drycnv.py's expected level-last layout.
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

import constant_jax as const
import pbl
import fluxes_jax
import surface_jax
import radiation_jax
import drycnv
from atm_com_jax import compute_pk_pek

DTsrc = 1800.0
NIsurf = 2
NRAD = 5
DTSURF = DTsrc / NIsurf

# ModelE's restart "p" field is surface pressure MINUS PTOP (confirmed
# from model/AtmL40p.F90: LM=40, LS1=24, PLBOT(24)=150 mb, default
# PSF=984 mb -- this is the standard 40-layer "top at .1mb" config that
# P2SAoM40 uses). True surface pressure = p + PTOP.
PTOP = 150.0  # mb

# Per-itype constants: simplified, declared stand-ins for the real
# spectral/vegetation-dependent GISS albedo and roughness schemes.
# itype: 1=ocean, 2=seaice, 3=landice, 4=land
Z0_BY_ITYPE = {1: 2e-4, 2: 1e-3, 3: 1e-3, 4: 0.1}
ALBEDO_BY_ITYPE = {1: 0.06, 2: 0.55, 3: 0.8, 4: 0.2}
EMISSIVITY_BY_ITYPE = {1: 0.98, 2: 0.98, 3: 0.98, 4: 0.96}
Z_REF = 10.0  # reference height (m) for surface-layer similarity, approximating layer-1 midpoint height


def build_static_fields(itype):
    """Per-cell roughness/albedo/emissivity maps from the itype mask."""
    shape = itype.shape
    z0 = np.zeros(shape)
    albedo = np.zeros(shape)
    emis = np.zeros(shape)
    for k, v in Z0_BY_ITYPE.items():
        z0 = np.where(itype == k, v, z0)
    for k, v in ALBEDO_BY_ITYPE.items():
        albedo = np.where(itype == k, v, albedo)
    for k, v in EMISSIVITY_BY_ITYPE.items():
        emis = np.where(itype == k, v, emis)
    return z0, albedo, emis


def surface_skin_temperature(itype, tearth, tlake, t1_actual):
    """Real per-cell ground/skin temperature (K), selected by itype from
    actual restart fields (not fabricated): land -> tearth (C), lake ->
    tlake (C), sea ice -> standard freezing-point proxy (hsi_atm/ssi_atm
    are enthalpy/salt content, not temperature directly -- using them
    properly is a v2 refinement), ocean -> not separately available in
    this atmosphere-side restart (no ocean GCM ported), approximated by
    the actual (already PK-converted) layer-1 air temperature minus a
    small stable offset. See STATUS.md for the full list of declared
    simplifications.

    `t1_actual` must already be in real Kelvin (i.e. state["t"][...,0]
    multiplied by PK -- GISS's prognostic T is T_actual/PK, not actual
    temperature, see run_dtsrc_step). Pure jnp (traced inside the fused
    per-step computation, see _step_core) -- tearth/tlake are passed in
    directly rather than pulled from a state dict so this composes as a
    plain array function under jax.jit.
    """
    tg_c = jnp.where(itype == 4, tearth, 0.0)
    tg_c = jnp.where(itype == 3, jnp.minimum(tearth, 0.0), tg_c)  # landice: cap at freezing
    tg_c = jnp.where(itype == 2, -1.8, tg_c)  # seaice surface: standard sea-water freezing point proxy
    tg = tg_c + const.TF
    # Ocean (itype==1): no ocean-model SST in this atm-only restart;
    # approximate with layer-1 air temperature minus a small stable
    # offset, a declared simplification (see STATUS.md).
    tg = jnp.where(itype == 1, t1_actual - 0.5, tg)
    return tg


def solve_surface_layer(z, z0, t1, tg, ws, n_iter=6):
    """Simplified fixed-point Monin-Obukhov solve on top of the
    *validated* pbl.getcm / pbl.getchq (which implement the real
    find_dpsim/find_dpsih similarity functions). This is a stand-in
    for PBL.f's internal Newton iteration on ustar/tstar/qstar -- same
    physical idea (iterate drag/transfer coefficients against a
    consistent Monin-Obukhov length), not a byte-for-byte port of the
    Fortran algorithm. Declared simplification.

    One jax.lax.fori_loop (traced once) rather than a Python loop (see
    STATUS.md "GPU optimization: Phase 1"). Each iteration computes cm/ch
    from the *incoming* lmonin before updating it, so the final cm/ch
    reflect the second-to-last lmonin estimate, exactly as originally.

    Round 2 (2026-09): the loop-invariant pieces (log(z/z0), tg - t1,
    0.5*(t1+tg), kappa*g) are hoisted out of the loop, and pbl.getcm/getchq
    now use vectorizable math (sqrt(sqrt), a Cephes atan, exp(log/3) in place
    of scalar-libm pow/atan/cbrt) -- same functions to float32 rounding.
    """
    kappa = 0.4
    ws_safe = jnp.maximum(ws, 0.5)
    lmonin0 = jnp.full_like(t1, 1.0e5)  # start near-neutral (large |L|)
    cm0 = jnp.zeros_like(t1)  # shape/dtype placeholder; overwritten on iteration 0
    ch0 = jnp.zeros_like(t1)
    logzz0 = jnp.log(z / z0)           # loop-invariant
    dtg = tg - t1                      # loop-invariant
    thetabar = 0.5 * (t1 + tg)         # loop-invariant
    kg = kappa * const.GRAV            # loop-invariant

    def body(_, carry):
        lmonin, _cm_prev, _ch_prev = carry
        dm, dpsim, cm = pbl.getcm(z, z0, lmonin, logzz0)
        dpsih, ch = pbl.getchq(z, z0, lmonin, dm, logzz0)
        ustar = jnp.sqrt(cm) * ws_safe
        # Kinematic heat flux (surface -> atm positive when tg > t1),
        # tstar defined via flux = -ustar*tstar (standard convention).
        wtheta = ch * ws_safe * dtg
        tstar = -wtheta / jnp.maximum(ustar, 1e-3)
        tstar_safe = jnp.sign(tstar) * jnp.maximum(jnp.abs(tstar), 1e-4) + 1e-8
        l_raw = (ustar ** 2) * thetabar / (kg * tstar_safe)
        lmonin_new = jnp.sign(l_raw) * jnp.clip(jnp.abs(l_raw), 1.0, 1.0e5)
        return (lmonin_new, cm, ch)

    _, cm, ch = jax.lax.fori_loop(0, n_iter, body, (lmonin0, cm0, ch0))

    # pbl.py's getcm/getchq clip to [cmin=0.001, cmax=1.0] -- a safety
    # bound carried over from the Fortran source, not a realistic bulk
    # transfer-coefficient range. Real neutral drag/Stanton/Dalton
    # numbers over Earth's surface are ~1e-3 to ~3e-3; this fixed-point
    # solve can drive cm/ch toward the 1.0 ceiling in extreme cells
    # (e.g. very unstable conditions), which would make fluxes 100-
    # 1000x too large. Apply an additional physically-motivated clip.
    cm = jnp.clip(cm, 5e-4, 3e-3)
    ch = jnp.clip(ch, 5e-4, 3e-3)
    cq = ch  # Dalton number approximated equal to Stanton number (declared simplification)
    return cm, ch, cq


def _surface_fluxes_relative_wind(itype, tg, qg, rho, cm, ch, cq, u1, v1, t1_actual, q1,
                                   fsf, flong, albedo, emis, uocean, vocean):
    """Same bulk-formula physics as fluxes_jax.py/surface_jax.py, with
    the corrected atmosphere-relative wind speed (see module docstring
    for the bug this works around). us/vs = 0 for land/land-ice
    (no-slip); = ocean/ice drift velocity for ocean/seaice.
    """
    is_water = (itype == 1) | (itype == 2)
    us = jnp.where(is_water, uocean, 0.0)
    vs = jnp.where(is_water, vocean, 0.0)
    ws_rel = jnp.sqrt((u1 - us) ** 2 + (v1 - vs) ** 2)
    uflux = rho * cm * ws_rel * (us - u1)
    vflux = rho * cm * ws_rel * (vs - v1)
    tflux = rho * ch * ws_rel * const.SHA * (tg - t1_actual)
    lh = jnp.where((itype == 2) | (itype == 3), const.LHS, const.LHE)
    qflux = rho * cq * ws_rel * lh * (qg - q1)
    solar = (1.0 - albedo) * fsf
    lw_net = emis * (flong - const.STBO * tg ** 4)
    net_energy = solar + lw_net - tflux - qflux
    return uflux, vflux, tflux, qflux, solar, lw_net, net_energy


def compute_zenith_cosz(lat_dg, lon_dg, dt_utc):
    """Simplified solar-zenith cosine (declination + hour angle), not the
    full orbital-mechanics ORBIT module used by the real RADIA driver.
    Declared simplification -- adequate for a physics-parameterization
    test, not for exact insolation accuracy.
    """
    doy = dt_utc.timetuple().tm_yday
    decl = np.radians(23.44) * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0
    lat_r = np.radians(lat_dg)[:, None] * np.ones((1, lon_dg.shape[0]))
    lon_r = np.radians(lon_dg)[None, :] * np.ones((lat_dg.shape[0], 1))
    hour_angle = np.radians(15.0 * (hour_utc - 12.0)) + lon_r
    cosz = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(cosz, 0.0, None)


def build_pressure_profile(p_sfc, ma):
    """pedn/pmid (mb) built directly from real surface pressure + real
    per-layer air mass, rather than atm_com_jax.compute_pmid_pedn's
    placeholder sigma-coordinate formula (which needs a DSIG input this
    restart doesn't directly provide). dp(L) [mb] = MA(L)*GRAV/100.

    Vectorized with jnp.cumsum rather than a Python for-loop over LM
    layers -- mathematically identical (pedn[l+1] = p_sfc - sum(dp[0:l+1])
    is exactly what the loop computed one layer at a time), but this
    version traces as part of one fused computation instead of running
    as untraceable host-side NumPy on every step (previously ~2.3ms/step
    of pure Python/NumPy that never touched JAX at all -- see STATUS.md's
    profiling section).
    """
    dp = ma * float(const.GRAV) / 100.0  # (J,I,L), mb
    pedn_rest = p_sfc[..., None] - jnp.cumsum(dp, axis=-1)  # (J,I,L), layers 1..LM
    pedn = jnp.concatenate([p_sfc[..., None], pedn_rest], axis=-1)  # (J,I,LM+1)
    pmid = 0.5 * (pedn[..., :-1] + pedn[..., 1:])
    return pmid, pedn


# ===========================================================================
# Round 2 (2026-09): prepared static context + device-resident stepping
# ===========================================================================
#
# In this physics-only driver the surface pressure `p` and air mass `ma` never
# change, so everything derived from them -- the pressure profile, PK = PMID**
# KAPA (a 132k-element pow, ~3 ms on CPU), and PDSIG -- is loop-invariant and
# is computed ONCE in prepare_static(), on the host in float64 (more accurate
# than the original float32 cumsum: the original reconstructed thin-layer
# PDSIG as a difference of ~1000 mb float32 numbers, ~6e-5 mb rounding on
# layers only ~0.1-1 mb thick). The time-varying state (t, q, u1, v1) then
# stays on the device in LAYER-FIRST layout (L, J, I) -- byte-for-byte
# Fortran's (I, J, L) memory order -- and steps chain with no host round trip.

class Prepared(NamedTuple):
    """Device-resident, loop-invariant inputs for the fast stepping path."""
    pk: jnp.ndarray        # (L, J, I)  PMID**KAPA
    pk1: jnp.ndarray       # (J, I)     pk[0]
    pdsig: jnp.ndarray     # (L, J, I)  layer pressure thickness (mb)
    ma1: jnp.ndarray       # (J, I)     layer-1 air mass
    p_sfc_pa: jnp.ndarray  # (J, I)     surface pressure (Pa)
    z0: jnp.ndarray
    albedo: jnp.ndarray
    emis: jnp.ndarray
    itype: jnp.ndarray
    tearth: jnp.ndarray
    tlake: jnp.ndarray
    uocean: jnp.ndarray
    vocean: jnp.ndarray
    sin_lat: jnp.ndarray   # (J, 1)
    cos_lat: jnp.ndarray   # (J, 1)
    sin_lon: jnp.ndarray   # (1, I)
    cos_lon: jnp.ndarray   # (1, I)


def _lf(a):
    """(J, I, L) -> contiguous float32 layer-first (L, J, I) device array."""
    return jnp.asarray(np.ascontiguousarray(np.moveaxis(np.asarray(a), -1, 0)), dtype=jnp.float32)


def prepare_static(state, itype, static_fields, lat_dg, lon_dg, precision="float64"):
    """Compute the loop-invariant device arrays once.

    precision="float64" (default): pressure profile / PK / PDSIG in float64 on
    the host. precision="float32": reproduce the original float32 arithmetic
    (used to verify the restructuring is regression-clean against the original
    driver to ~1e-6; not recommended otherwise).
    """
    z0, albedo, emis = static_fields
    dt = np.float64 if precision == "float64" else np.float32
    p_sfc = np.asarray(state["p"], dtype=dt) + dt(PTOP)
    ma = np.asarray(state["ma"], dtype=dt)
    dp = ma * dt(float(const.GRAV)) / dt(100.0)
    pedn = np.empty(ma.shape[:-1] + (ma.shape[-1] + 1,), dtype=dt)
    pedn[..., 0] = p_sfc
    pedn[..., 1:] = p_sfc[..., None] - np.cumsum(dp, axis=-1)
    pmid = dt(0.5) * (pedn[..., :-1] + pedn[..., 1:])
    pk = pmid ** dt(float(const.KAPA))
    pdsig = pedn[..., :-1] - pedn[..., 1:]
    J = jnp.asarray
    lat_r = np.radians(np.asarray(lat_dg, dtype=np.float64))[:, None]
    lon_r = np.radians(np.asarray(lon_dg, dtype=np.float64))[None, :]
    pk_lf = _lf(pk)
    return Prepared(
        pk=pk_lf, pk1=pk_lf[0], pdsig=_lf(pdsig), ma1=J(np.asarray(state["ma"])[..., 0], dtype=jnp.float32),
        p_sfc_pa=J(p_sfc * 100.0, dtype=jnp.float32),
        z0=J(z0, dtype=jnp.float32), albedo=J(albedo, dtype=jnp.float32), emis=J(emis, dtype=jnp.float32),
        itype=J(itype), tearth=J(state["tearth"], dtype=jnp.float32), tlake=J(state["tlake"], dtype=jnp.float32),
        uocean=J(state["uosurf_icdyn"], dtype=jnp.float32), vocean=J(state["vosurf_icdyn"], dtype=jnp.float32),
        sin_lat=J(np.sin(lat_r), dtype=jnp.float32), cos_lat=J(np.cos(lat_r), dtype=jnp.float32),
        sin_lon=J(np.sin(lon_r), dtype=jnp.float32), cos_lon=J(np.cos(lon_r), dtype=jnp.float32),
    )


def solar_scalars(dt_utc):
    """Per-step scalars for the on-device solar-zenith formula (same
    declination + hour-angle formula as compute_zenith_cosz, evaluated on the
    host for just four Python floats): [sin(decl), cos(decl), cos(ha0), sin(ha0)]."""
    doy = dt_utc.timetuple().tm_yday
    decl = np.radians(23.44) * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    ha0 = np.radians(15.0 * (dt_utc.hour + dt_utc.minute / 60.0 - 12.0))
    return np.array([np.sin(decl), np.cos(decl), np.cos(ha0), np.sin(ha0)], dtype=np.float32)


def _cosz_device(prep, sol):
    sd, cd, cha, sha = sol[0], sol[1], sol[2], sol[3]
    cos_ha = cha * prep.cos_lon - sha * prep.sin_lon            # cos(ha0 + lon)
    return jnp.maximum(prep.sin_lat * sd + prep.cos_lat * cd * cos_ha, 0.0)


def _step_dev(dyn, prep, sol):
    """One DTsrc step on device-resident, layer-first state.

    dyn = (t, q, u1, v1): t, q (L, J, I); u1, v1 (J, I) (only layer 1 of u/v
    evolves). Same physics, formulas and operation order as the original
    per-call driver; returns (new_dyn, diagnostics tuple).
    """
    t, q, u1, v1 = dyn
    pk1 = prep.pk1
    t1_actual = t[0] * pk1
    tg = surface_skin_temperature(prep.itype, prep.tearth, prep.tlake, t1_actual)
    thv1, rho, qsat = surface_jax.compute_surface_properties(t1_actual, q[0], prep.p_sfc_pa)
    qg = jnp.where(prep.itype == 1, qsat, qsat * 0.9)

    cosz = _cosz_device(prep, sol)
    fsf = const.SOLAR_CONSTANT * cosz
    flong = radiation_jax.STBO * t1_actual ** 4  # graybody proxy (emissivity 1.0); radiation_jax.STBO (5.67e-8), as the original stefan_boltzmann_jit

    ma1 = prep.ma1
    cp = const.SHA
    zref = jnp.asarray(Z_REF, dtype=jnp.float32)
    for _ in range(NIsurf):
        ws1 = jnp.sqrt(u1 ** 2 + v1 ** 2)
        cm, ch, cq = solve_surface_layer(zref, prep.z0, t1_actual, tg, ws1)
        uflux, vflux, tflux, qflux, solar, lw_net, net_energy = _surface_fluxes_relative_wind(
            prep.itype, tg, qg, rho, cm, ch, cq,
            u1, v1, t1_actual, q[0], fsf, flong, prep.albedo, prep.emis, prep.uocean, prep.vocean,
        )
        dT1 = ((tflux / (ma1 * cp)) * DTSURF) / pk1
        dQ1 = (qflux / (ma1 * const.LHE)) * DTSURF
        t = t.at[0].add(dT1)
        q = q.at[0].add(dQ1)
        t1_actual = t[0] * pk1
        u1 = u1 + (uflux / ma1) * DTSURF
        v1 = v1 + (vflux / ma1) * DTSURF

    t, q = drycnv.dry_convection_mixing_lf(t, q, prep.pk, prep.pdsig)
    diag = dict(tg=tg, cosz=cosz, fsf=fsf, flong=flong, sensible_heat_flux=tflux,
                latent_heat_flux=qflux, net_energy_flux=net_energy, momentum_flux_u=uflux)
    return (t, q, u1, v1), diag


step_device = jax.jit(_step_dev)


@jax.jit
def run_steps_device(dyn, prep, sol_seq):
    """N chained DTsrc steps in ONE dispatch (lax.scan); state never leaves the
    device. sol_seq: (N, 4) from solar_scalars(). Returns (final_dyn,
    diagnostics of the last step)."""
    def body(d, sol):
        return _step_dev(d, prep, sol)
    dyn_f, diags = jax.lax.scan(body, dyn, sol_seq)
    return dyn_f, jax.tree_util.tree_map(lambda a: a[-1], diags)


def dyn_from_state(state):
    """Host restart state -> device layer-first dynamic state (t, q, u1, v1)."""
    return (_lf(state["t"]), _lf(state["q"]),
            jnp.asarray(state["u"][..., 0], dtype=jnp.float32),
            jnp.asarray(state["v"][..., 0], dtype=jnp.float32))


def state_from_dyn(state, dyn):
    """Device dynamic state -> host restart-state dict (level-last, like the original)."""
    t, q, u1, v1 = dyn
    new_state = dict(state)
    new_state["t"] = np.ascontiguousarray(np.moveaxis(np.asarray(t), 0, -1))
    new_state["q"] = np.ascontiguousarray(np.moveaxis(np.asarray(q), 0, -1))
    new_state["u"] = np.asarray(state["u"]).copy()
    new_state["u"][..., 0] = np.asarray(u1)
    new_state["v"] = np.asarray(state["v"]).copy()
    new_state["v"][..., 0] = np.asarray(v1)
    return new_state


# Single-entry cache so repeated single-step calls on the SAME input arrays (the
# usual benchmark loop) don't redo the static preparation. Keyed on object
# identity of the inputs it depends on; in-place mutation of those arrays is not
# detected -- pass prepared= explicitly or call clear_prepare_cache() if you do.
_PREP_CACHE = {"key": None, "refs": None, "prep": None}


def clear_prepare_cache():
    _PREP_CACHE.update(key=None, refs=None, prep=None)


def _cached_prepare(state, itype, static_fields, lat_dg, lon_dg):
    refs = (state["p"], state["ma"], state["tearth"], state["tlake"], state["uosurf_icdyn"],
            state["vosurf_icdyn"], itype, static_fields[0], static_fields[1], static_fields[2],
            lat_dg, lon_dg)
    if _PREP_CACHE["refs"] is not None and all(a is b for a, b in zip(refs, _PREP_CACHE["refs"])):
        return _PREP_CACHE["prep"]
    prep = prepare_static(state, itype, static_fields, lat_dg, lon_dg)
    _PREP_CACHE.update(refs=refs, prep=prep)      # holds refs => ids can't be recycled
    return prep


def run_dtsrc_step(state, itype, static_fields, lat_dg, lon_dg, dt_utc, step_index=0,
                   do_radiation=None, prepared=None):
    """Advance the real restart state by one DTsrc=1800s step through the
    ported physics subset, in the real call order:
      zenith angle -> RADIATION (NRAD-gated) -> SURFACE (NIsurf substeps:
      PBL similarity + flux dispatch + layer-1 tendency) -> DRYCNV.

    Returns (new_state, diagnostics) exactly as before (host NumPy, level-last).
    Compatibility wrapper over the device-resident fast path: it pays host<->
    device conversion every call. For real multi-step runs use prepare_static()
    + dyn_from_state() + run_steps_device() and keep the state on the device.
    """
    if do_radiation is None:
        do_radiation = (step_index % NRAD == 0)   # gating changes no computation (see docs)
    prep = prepared if prepared is not None else _cached_prepare(state, itype, static_fields, lat_dg, lon_dg)
    dyn, diag = step_device(dyn_from_state(state), prep, jnp.asarray(solar_scalars(dt_utc)))
    new_state = state_from_dyn(state, dyn)
    new_state["itime"] = state["itime"] + 1  # itime is a DTsrc-tick counter, not seconds
    diagnostics = {k: np.asarray(v) for k, v in diag.items()}
    diagnostics["radiation_computed_this_step"] = bool(do_radiation)
    return new_state, diagnostics
