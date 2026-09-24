"""
Regression tests for the Round-2 (2026-09) optimizations: vectorizable PBL math,
layer-first DRYCNV scan, and the prepared / device-resident driver path.

The references here are independent float64 NumPy implementations of the
*original* formulas (transcribed from the pre-optimization pbl.py / drycnv.py),
so a passing test bounds the optimized float32 code against the original
algorithm rather than against itself.
"""
import os
import sys
import datetime

import numpy as np
import pytest
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pbl  # noqa: E402
import drycnv  # noqa: E402

KAPPA, ZET1, SLOPE1, GAMAMS, GAMAMU, GAMAHS, GAMAHU = 0.4, 1.0, 5.0, 4.7, 16.0, 4.7, 16.0
ZETM, BY3 = -1.0, 1.0 / 3.0


def ref_atan(x):
    return np.arctan(x)


def ref_dpsim(zet, zet0):
    """Original find_dpsim, float64 NumPy."""
    with np.errstate(all="ignore"):
        x = (1.0 - GAMAMU * zet) ** 0.25
        x0 = (1.0 - GAMAMU * zet0) ** 0.25
        xm = (1.0 - GAMAMU * ZETM) ** 0.25
        gt = np.log(((1 + x) ** 2 * (1 + x * x)) / ((1 + x0) ** 2 * (1 + x0 * x0))) - 2 * (np.arctan(x) - np.arctan(x0))
        le = (np.log(((1 + xm) ** 2 * (1 + xm * xm)) / ((1 + x0) ** 2 * (1 + x0 * x0)))
              - 2 * (np.arctan(xm) - np.arctan(x0)) + np.log(zet / ZETM)
              - 1.140125 * ((-zet) ** BY3 - (-ZETM) ** BY3))
        s1 = -GAMAMS * (zet - zet0)
        s2 = (-GAMAMS * (ZET1 - zet0) + ZET1 * (SLOPE1 - GAMAMS) * np.log(zet / ZET1)
              - SLOPE1 * (zet - ZET1))
        stable = zet >= 0
        return np.where(stable & (zet <= ZET1), s1, np.where(stable, s2, np.where(zet > ZETM, gt, le)))


def ref_dpsih(zet, zet0, z, z0):
    """Original find_dpsih, float64 NumPy (sigma = sigma1 = 1)."""
    with np.errstate(all="ignore"):
        stable = zet >= 0
        return np.where(
            stable & (zet <= ZET1),
            np.log(z / z0) - GAMAHS * (zet - zet0),
            np.where(stable,
                     np.log(ZET1 / z0) - GAMAHS * (ZET1 - zet0)
                     + (1 + (ZET1 * (SLOPE1 - GAMAHS) - 1)) * np.log(zet / ZET1) - SLOPE1 * (zet - ZET1),
                     np.log(z / z0) - GAMAHU * (zet - zet0)))


def _zet_grid():
    mag = np.concatenate([[0.0], np.logspace(-6, 2, 400)])
    zet = np.concatenate([mag, -mag[1:], [1.0, -1.0, 1.0 + 1e-6, -1.0 - 1e-6, 1.0 - 1e-6, -1.0 + 1e-6]])
    return zet


def _cases():
    zet = _zet_grid()
    zs, z0s, zets, zet0s = [], [], [], []
    for ratio in (1e-5, 1e-4, 1e-3, 1e-2, 0.1):
        zs.append(np.full_like(zet, 10.0))
        z0s.append(np.full_like(zet, 10.0 * ratio))
        zets.append(zet)
        zet0s.append(zet * ratio)
    return tuple(np.concatenate(a) for a in (zs, z0s, zets, zet0s))


def test_cephes_atan_accuracy():
    x = np.concatenate([np.linspace(-50, 50, 20001), np.logspace(-8, 8, 4001), -np.logspace(-8, 8, 4001), [0.0]])
    got = np.asarray(pbl._atan(jnp.asarray(x, dtype=jnp.float32)), dtype=np.float64)
    assert np.max(np.abs(got - np.arctan(x))) < 3e-7


def test_find_dpsim_matches_float64_reference_all_branches():
    z, z0, zet, zet0 = _cases()
    got = np.asarray(pbl.find_dpsim(jnp.asarray(zet, jnp.float32), jnp.asarray(zet0, jnp.float32)), dtype=np.float64)
    ref = ref_dpsim(zet, zet0)
    ok = np.isfinite(ref)
    err = np.abs(got[ok] - ref[ok])
    assert np.isfinite(got[ok]).all()
    # float32 arithmetic on values up to |dpsim| ~ 10: allow a few ulp, relative + absolute
    assert np.all(err <= 2e-5 + 5e-5 * np.abs(ref[ok])), f"max err {err.max():.3e}"


def test_find_dpsih_matches_float64_reference_all_branches():
    z, z0, zet, zet0 = _cases()
    got = np.asarray(pbl.find_dpsih(*(jnp.asarray(a, jnp.float32) for a in (zet, zet0, z, z0))), dtype=np.float64)
    ref = ref_dpsih(zet, zet0, z, z0)
    ok = np.isfinite(ref)
    err = np.abs(got[ok] - ref[ok])
    assert np.all(err <= 2e-5 + 5e-5 * np.abs(ref[ok])), f"max err {err.max():.3e}"


def test_getcm_getchq_optional_logzz0_is_consistent():
    rng = np.random.default_rng(0)
    z = jnp.asarray(10.0, jnp.float32)
    z0 = jnp.asarray(rng.uniform(1e-4, 0.1, 500), jnp.float32)
    L = jnp.asarray(rng.choice([-1, 1], 500) * rng.uniform(1.0, 1e5, 500), jnp.float32)
    lz = jnp.log(z / z0)
    a = pbl.getcm(z, z0, L)
    b = pbl.getcm(z, z0, L, lz)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)
    dm = a[0]
    a2 = pbl.getchq(z, z0, L, dm)
    b2 = pbl.getchq(z, z0, L, dm, lz)
    for x, y in zip(a2, b2):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)


def ref_drycnv(T, Q, PK, PDSIG, deltx=0.608):
    """Original algorithm, straightforward float64 loop over (level-last) columns."""
    T = T.astype(np.float64).copy(); Q = Q.astype(np.float64).copy()
    PK = PK.astype(np.float64); PD = PDSIG.astype(np.float64)
    nl = T.shape[-1]
    for L in range(nl - 1):
        TV0 = T[..., L] * (1 + Q[..., L] * deltx)
        TV1 = T[..., L + 1] * (1 + Q[..., L + 1] * deltx)
        unstable = TV0 > TV1
        pkms = PK[..., L] * PD[..., L] + PK[..., L + 1] * PD[..., L + 1]
        tvms = TV0 * PK[..., L] * PD[..., L] + TV1 * PK[..., L + 1] * PD[..., L + 1]
        qms = Q[..., L] * PD[..., L] + Q[..., L + 1] * PD[..., L + 1]
        rdp = 1.0 / (PD[..., L] + PD[..., L + 1])
        thm = tvms / (pkms * (1 + qms * rdp * deltx))
        qm = qms * rdp
        for k in (L, L + 1):
            T[..., k] = np.where(unstable, thm, T[..., k])
            Q[..., k] = np.where(unstable, qm, Q[..., k])
    return T, Q


def _drycnv_inputs(seed=3, shape=(20, 15, 40)):
    rng = np.random.default_rng(seed)
    T = rng.uniform(200, 300, shape)
    Q = rng.uniform(0.0, 0.02, shape)
    PK = rng.uniform(0.3, 1.0, shape)
    PD = rng.uniform(0.5, 30.0, shape)
    return T, Q, PK, PD


def test_drycnv_matches_float64_reference():
    T, Q, PK, PD = _drycnv_inputs()
    ref_t, ref_q = ref_drycnv(T, Q, PK, PD)
    t, q = drycnv.dry_convection_mixing_jit(*(jnp.asarray(a, jnp.float32) for a in (T, Q, PK, PD)))
    assert np.max(np.abs(np.asarray(t) - ref_t)) < 5e-4          # T ~ 250 K, float32 eps*T ~ 1.5e-5
    assert np.max(np.abs(np.asarray(q) - ref_q)) < 1e-7
    assert np.mean(ref_t != T) > 0.5                                # the test really exercises mixing


def test_drycnv_layer_first_equals_level_last():
    T, Q, PK, PD = _drycnv_inputs(seed=5)
    a = [jnp.asarray(x, jnp.float32) for x in (T, Q, PK, PD)]
    t_ll, q_ll = drycnv.dry_convection_mixing_jit(*a)
    lf = [jnp.moveaxis(x, -1, 0) for x in a]
    t_lf, q_lf = drycnv.dry_convection_mixing_lf_jit(*lf)
    np.testing.assert_array_equal(np.asarray(t_ll), np.asarray(jnp.moveaxis(t_lf, 0, -1)))
    np.testing.assert_array_equal(np.asarray(q_ll), np.asarray(jnp.moveaxis(q_lf, 0, -1)))


def test_drycnv_stable_column_unchanged():
    nl = 40
    T = np.tile(np.linspace(250, 350, nl), (4, 3, 1))     # temperature rising with height -> all stable
    Q = np.zeros_like(T); PK = np.ones_like(T); PD = np.ones_like(T)
    t, q = drycnv.dry_convection_mixing_jit(*(jnp.asarray(a, jnp.float32) for a in (T, Q, PK, PD)))
    np.testing.assert_allclose(np.asarray(t), T, rtol=1e-6)


# ---------------------------------------------------------------- driver (needs real restart data)
try:
    import p2saom40_io as io
    import p2saom40_driver as drv
    _HAVE_DATA = os.path.exists(f"{io.RUN_DIR}/fort.1.nc")
except Exception:  # pragma: no cover
    _HAVE_DATA = False

needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="real P2SAoM40 restart not available")


@pytest.fixture(scope="module")
def world():
    state = io.load_restart_state(f"{io.RUN_DIR}/fort.1.nc")
    frac = io.load_surface_fractions()
    itype = io.build_itype(frac, state["rsi_atm"])
    sf = drv.build_static_fields(itype)
    return state, itype, sf, frac["lat"], frac["lon"], io.itime_to_datetime(state["itime"])


@needs_data
def test_wrapper_matches_device_path(world):
    state, itype, sf, lat, lon, t0 = world
    drv.clear_prepare_cache()
    ns, diag = drv.run_dtsrc_step(state, itype, sf, lat, lon, t0, step_index=1)
    prep = drv.prepare_static(state, itype, sf, lat, lon)
    dyn, d = drv.step_device(drv.dyn_from_state(state), prep, jnp.asarray(drv.solar_scalars(t0)))
    for k in ("tg", "sensible_heat_flux", "latent_heat_flux", "net_energy_flux", "fsf", "flong"):
        np.testing.assert_allclose(diag[k], np.asarray(d[k]), rtol=1e-6, atol=1e-6)
    fin = drv.state_from_dyn(state, dyn)
    np.testing.assert_allclose(ns["t"], fin["t"], rtol=1e-6)
    assert ns["t"].shape == state["t"].shape and ns["itime"] == state["itime"] + 1


@needs_data
def test_scan_equals_repeated_single_steps(world):
    state, itype, sf, lat, lon, t0 = world
    prep = drv.prepare_static(state, itype, sf, lat, lon)
    dts = [t0 + datetime.timedelta(seconds=1800 * k) for k in range(4)]
    sols = [jnp.asarray(drv.solar_scalars(d)) for d in dts]
    dyn = drv.dyn_from_state(state)
    for s in sols:
        dyn, d = drv.step_device(dyn, prep, s)
    dyn2, d2 = drv.run_steps_device(drv.dyn_from_state(state), prep, jnp.stack(sols))
    for a, b in zip(dyn, dyn2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


@needs_data
def test_prepare_cache_invalidates_on_new_inputs(world):
    state, itype, sf, lat, lon, t0 = world
    drv.clear_prepare_cache()
    a = drv._cached_prepare(state, itype, sf, lat, lon)
    assert drv._cached_prepare(state, itype, sf, lat, lon) is a       # same arrays -> reused
    state2 = dict(state); state2["p"] = state["p"].copy()             # new array object -> re-prepared
    assert drv._cached_prepare(state2, itype, sf, lat, lon) is not a


@needs_data
def test_solar_zenith_on_device_matches_numpy(world):
    state, itype, sf, lat, lon, t0 = world
    prep = drv.prepare_static(state, itype, sf, lat, lon)
    for k in (0, 5, 11, 23):
        dt = t0 + datetime.timedelta(hours=k)
        ref = drv.compute_zenith_cosz(lat, lon, dt)
        got = np.asarray(drv._cosz_device(prep, jnp.asarray(drv.solar_scalars(dt))))
        assert np.max(np.abs(got - ref)) < 5e-7
