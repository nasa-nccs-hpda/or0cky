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

    Implemented as a single jax.lax.fori_loop (traced once) rather than
    a Python for-loop, so this whole 6-iteration solve compiles to one
    fused XLA computation instead of dispatching pbl.getcm/getchq as 12
    separate calls -- see STATUS.md's "Regression verification" /
    profiling section for why this mattered (this loop, x NIsurf=2, was
    24 of the ~33 separate JIT dispatches in the pre-fusion driver).
    Numerically identical to the original Python-loop version: each
    iteration still computes cm/ch from the *incoming* lmonin before
    updating it, so the final cm/ch reflect the second-to-last lmonin
    estimate, exactly as before.
    """
    kappa = 0.4
    ws_safe = jnp.maximum(ws, 0.5)
    lmonin0 = jnp.full_like(t1, 1.0e5)  # start near-neutral (large |L|)
    cm0 = jnp.zeros_like(t1)  # shape/dtype placeholder; overwritten on iteration 0
    ch0 = jnp.zeros_like(t1)

    def body(_, carry):
        lmonin, _cm_prev, _ch_prev = carry
        dm, dpsim, cm = pbl.getcm(z, z0, lmonin)
        dpsih, ch = pbl.getchq(z, z0, lmonin, dm)
        ustar = jnp.sqrt(cm) * ws_safe
        # Kinematic heat flux (surface -> atm positive when tg > t1),
        # tstar defined via flux = -ustar*tstar (standard convention).
        wtheta = ch * ws_safe * (tg - t1)
        tstar = -wtheta / jnp.maximum(ustar, 1e-3)
        tstar_safe = jnp.sign(tstar) * jnp.maximum(jnp.abs(tstar), 1e-4) + 1e-8
        thetabar = 0.5 * (t1 + tg)
        l_raw = (ustar ** 2) * thetabar / (kappa * const.GRAV * tstar_safe)
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


@jax.jit
def _step_core(t, q, u1, v1, p_sfc, ma, itype_j, z0, albedo, emis,
                tearth, tlake, uocean, vocean, cosz):
    """The full per-DTsrc physics computation (pressure profile ->
    PK/PEK -> skin temp -> surface properties -> RADIATION ->
    SURFACE[NIsurf substeps] -> DRYCNV), as ONE fused jax.jit
    computation instead of ~33 separately-dispatched calls.

    This is a pure restructuring for dispatch efficiency, not an
    algorithm change: every formula, iteration count, and operation
    order below is identical to the original per-call version (see git
    history / STATUS.md's profiling section for the before/after). All
    arguments are already jnp arrays (or convert trivially); the host
    wrapper run_dtsrc_step() below handles the one-time NumPy<->JAX
    conversion at the boundary instead of doing it call-by-call inside
    the step.
    """
    p_sfc = jnp.asarray(p_sfc)
    ma = jnp.asarray(ma)

    pmid, pedn = build_pressure_profile(p_sfc, ma)
    pk, pek = compute_pk_pek(pmid, pedn)
    pk1 = pk[..., 0]
    # Real GISS ModelE convention (confirmed empirically against this
    # restart: raw surface-layer T has global mean ~38.9, and 38.9*PK
    # ~=280K, a physically sane Dec global-mean surface air temp): the
    # prognostic "t" array is T_actual/PK, NOT actual temperature.
    # drycnv.py's mixing formula already expects this convention
    # directly (its THM computation multiplies by PK internally), so
    # only the *surface-physics* calls below need actual Kelvin.
    t1_actual = t[..., 0] * pk1
    tg = surface_skin_temperature(itype_j, tearth, tlake, t1_actual)

    thv1, rho, qsat = surface_jax.compute_surface_properties(t1_actual, q[..., 0], p_sfc * 100.0)
    qg = jnp.where(itype_j == 1, qsat, qsat * 0.9)  # near-saturated proxy over ocean/moist land (declared simplification)

    # --- RADIATION (real cadence: full computation every NRAD steps) ---
    # NOTE: the NRAD-gating itself (do_radiation) doesn't change any
    # computation here -- both branches evaluate the same cheap formulas
    # (see run_dtsrc_step's docstring) -- so it's resolved on the host,
    # not threaded through this jitted core.
    fsf, srdflb, srnflb = radiation_jax.compute_solar_flux_jit(const.SOLAR_CONSTANT, cosz, albedo)
    flong = radiation_jax.stefan_boltzmann_jit(t1_actual, 1.0)  # graybody downwelling LW proxy from lowest layer (declared simplification)

    # --- SURFACE: NIsurf=2 substeps at DTSURF=900s ---
    # NIsurf is a small, fixed Python int, so this unrolls at trace time
    # into NIsurf copies inside the same fused program (still ONE
    # dispatch) rather than NIsurf separate calls from the host.
    ma1 = ma[..., 0]
    cp = const.SHA
    tflux = qflux = uflux = net_energy = None
    for _ in range(NIsurf):
        ws1 = jnp.sqrt(u1 ** 2 + v1 ** 2)
        cm, ch, cq = solve_surface_layer(jnp.asarray(Z_REF), z0, t1_actual, tg, ws1)
        uflux, vflux, tflux, qflux, solar, lw_net, net_energy = _surface_fluxes_relative_wind(
            itype_j, tg, qg, rho, cm, ch, cq,
            u1, v1, t1_actual, q[..., 0], fsf, flong, albedo, emis, uocean, vocean,
        )
        # Layer-1 tendency applied directly from the computed fluxes,
        # normalized by the real layer-1 air mass (mass- and
        # energy-consistent leading-order surface coupling; stands in
        # for aturb_jax's full tridiagonal layer-1 diffusion solve --
        # declared simplification, see module docstring).
        # d(M1*cp*T1_actual)/dt = tflux => dT1_actual = tflux/(M1*cp)*dt,
        # then converted back to the stored T_actual/PK convention.
        # d(M1*q1)/dt*LHE = qflux       => dq1 = qflux/(M1*LHE)*dt
        # d(M1*u1)/dt = uflux (uflux already carries the correct sign:
        # negative when the atmosphere is faster than the surface, i.e.
        # drag decelerates the flow)
        dT1_actual = (tflux / (ma1 * cp)) * DTSURF
        dT1 = dT1_actual / pk1
        dQ1 = (qflux / (ma1 * const.LHE)) * DTSURF
        t = t.at[..., 0].add(dT1)
        q = q.at[..., 0].add(dQ1)
        t1_actual = t[..., 0] * pk1
        u1 = u1 + (uflux / ma1) * DTSURF
        v1 = v1 + (vflux / ma1) * DTSURF

    # --- DRYCNV proper, layers 2..LM (0-indexed 1..LM-1), once per DTsrc ---
    pdsig = pedn[..., :-1] - pedn[..., 1:]
    t_after, q_after = drycnv.dry_convection_mixing_jit(t, q, pk, pdsig)

    return t_after, q_after, u1, v1, tg, fsf, flong, tflux, qflux, net_energy, uflux


def run_dtsrc_step(state, itype, static_fields, lat_dg, lon_dg, dt_utc, step_index=0, do_radiation=None):
    """Advance the real restart state by one DTsrc=1800s step through the
    ported physics subset, in the real call order:
      zenith angle -> RADIATION (NRAD-gated) -> SURFACE (NIsurf substeps:
      PBL similarity + flux dispatch + layer-1 tendency) -> DRYCNV
      (layers 2..LM, once).

    Returns (new_state, diagnostics) where diagnostics holds the
    per-cell fields used for the accuracy/functionality comparison.

    Host-side thin wrapper: converts the restart state's NumPy arrays to
    device arrays ONCE, calls the single fused _step_core jax.jit
    computation, then converts the results back to NumPy ONCE for the
    returned dicts. All the actual physics is in _step_core.
    """
    z0, albedo, emis = static_fields

    if do_radiation is None:
        do_radiation = (step_index % NRAD == 0)
    # compute_zenith_cosz depends on Python datetime attributes (doy,
    # hour-of-day), which aren't traceable JAX values, so it's computed
    # here on the host -- cheap (~0.2ms), not part of the dispatch-count
    # problem this fusion targets -- and passed into _step_core as plain
    # array data.
    cosz = compute_zenith_cosz(lat_dg, lon_dg, dt_utc)

    t_after, q_after, u1, v1, tg, fsf, flong, tflux, qflux, net_energy, uflux = _step_core(
        jnp.asarray(state["t"]), jnp.asarray(state["q"]),
        jnp.asarray(state["u"][..., 0]), jnp.asarray(state["v"][..., 0]),
        jnp.asarray(state["p"] + PTOP), jnp.asarray(state["ma"]),
        jnp.asarray(itype),
        jnp.asarray(z0), jnp.asarray(albedo), jnp.asarray(emis),
        jnp.asarray(state["tearth"]), jnp.asarray(state["tlake"]),
        jnp.asarray(state["uosurf_icdyn"]), jnp.asarray(state["vosurf_icdyn"]),
        jnp.asarray(cosz),
    )

    new_state = dict(state)
    new_state["t"] = np.asarray(t_after)
    new_state["q"] = np.asarray(q_after)
    new_state["u"] = np.asarray(state["u"]).copy()
    new_state["u"][..., 0] = np.asarray(u1)
    new_state["v"] = np.asarray(state["v"]).copy()
    new_state["v"][..., 0] = np.asarray(v1)
    new_state["itime"] = state["itime"] + 1  # itime is a DTsrc-tick counter, not seconds

    diagnostics = {
        "tg": np.asarray(tg),
        "cosz": np.asarray(cosz),
        "fsf": np.asarray(fsf),
        "flong": np.asarray(flong),
        "sensible_heat_flux": np.asarray(tflux),
        "latent_heat_flux": np.asarray(qflux),
        "net_energy_flux": np.asarray(net_energy),
        "momentum_flux_u": np.asarray(uflux),
        "radiation_computed_this_step": bool(do_radiation),
    }
    return new_state, diagnostics
