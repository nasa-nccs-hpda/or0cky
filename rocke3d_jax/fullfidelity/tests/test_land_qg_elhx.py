"""D199: the ground humidity qg_ij that GHY_DRV.f:1304-1312 hands to the next substep is qg_sat = qsat(tg1+tf, elhx, ps) with the elhx of the
substep that just ran (set from the ENTRY tg1, GHY_DRV.f:1071-1075), not the elhx of the next substep's row.  They differ when tsns
crosses 0 C inside the substep.  (1) synthetic: NumPy and JAX versions agree bitwise and use the current elhx; the old behaviour is
kept when the land result carries no 'elhx'.  (2) real dec01 dumps (skipped when absent): substep-1 land -> substep-2 qg_aver (record
column 9) is reproduced at <= 1e-12 in all 753 cells including (57,34) and (21,33), which crossed 0 C."""
import os, sys
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import land_chain as LC
import jax_surface as JS
import pbl_ff as P

FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
LHE, LHS = 2500000.0, 2834000.0


def _synthetic(n=6):
    rng = np.random.default_rng(3)
    rec = np.zeros((n, 130))
    rec[:, 16] = 900.0 + 50 * rng.random(n)          # ps
    rec[:, 19] = np.where(np.arange(n) % 2 == 0, LHS, LHE)   # elhx of the NEXT substep row
    tsns = np.array([-0.044, 0.011, 5.0, -5.0, 0.2, -0.3])[:n]
    el_cur = np.array([LHE, LHS, LHE, LHS, LHE, LHE])[:n]  # elhx of the substep that ran
    ghy = dict(tsns=tsns, tbcs=tsns + 0.1)
    qsrf = 0.003 + 1e-4 * rng.random(n)
    land = dict(ghy=ghy, pbl=dict(qsrf=qsrf, ch=0.01 + 0.0 * qsrf, ws=5.0 + 0.0 * qsrf), rho=1.2 + 0.0 * qsrf,
                evap_max_ij=np.array([3.8e-8, 1e-9, 0.0, 1e-8, 5e-9, 0.0])[:n], fr_sat_ij=np.array([0.39, 0.34, 0.0, 1.0, 0.5, 0.2])[:n],
                elhx=el_cur)
    return rec, land, tsns


def test_next_land_pbl_columns_uses_current_elhx_numpy_and_jax():
    rec, land, tsns = _synthetic()
    out = LC.next_land_pbl_columns(rec, land)
    outj = np.asarray(JS.next_land_pbl_columns(jnp.asarray(rec), jax.tree_util.tree_map(jnp.asarray, land)))
    tg = tsns + 273.15
    qsc = np.asarray(P.qsat(jnp.asarray(tg), jnp.asarray(land['elhx']), jnp.asarray(rec[:, 16])))
    qsn = np.asarray(P.qsat(jnp.asarray(tg), jnp.asarray(rec[:, 19]), jnp.asarray(rec[:, 16])))
    assert np.array_equal(out[:, 8], qsn)                      # PBL column qg_sat: elhx of the new substep (unchanged)
    # qg_aver bounded by the CURRENT-elhx saturation value
    assert np.all(out[:, 9] <= np.maximum(qsc, 0) * (1 + 1e-15) + 1e-300)
    fr = land['fr_sat_ij']
    full = fr * qsc + (1 - fr) * np.minimum(land['pbl']['qsrf'] + land['evap_max_ij'] / (LC.C001 * (0.01 * 5.0 * 1.2)), qsc)
    np.testing.assert_allclose(out[:, 9], full, rtol=1e-14, atol=0)
    np.testing.assert_allclose(outj[:, 9], out[:, 9], rtol=1e-14, atol=0)
    # the crossing rows differ from the old behaviour, so the test is not vacuous
    old = LC.next_land_pbl_columns(rec, {k: v for k, v in land.items() if k != 'elhx'})
    assert np.abs(old[:, 9] - out[:, 9]).max() > 1e-8


def test_substep2_qg_aver_vs_record_dec01():
    d, it = 'dec01', 33552
    need = [f"{FF}/{d}/{f}" for f in (f"ffp_{it}.bin", f"ffg_{it}.bin", f"fft_{it}.bin", f"ffa_{it}_c1_in.bin")]
    if not all(os.path.exists(p) for p in need):
        pytest.skip("real-Fortran dumps not available")
    import pbl_compare as PC, ghy_compare as GC, tile_aggregate_ff as TA, chain_two_substeps as T2, chain_aggregate_aturb as C
    dd = f"{FF}/{d}"
    p = PC.load(f"{dd}/ffp_{it}.bin"); n = len(p) // 2
    pa, pb = p[:n], p[n:]
    a4, b4 = pa[pa[:, 2] == 4], pb[pb[:, 2] == 4]
    g = GC.load(f"{dd}/ffg_{it}.bin"); g1 = g[:len(g) // 2]
    fft = TA.load(f"{dd}/fft_{it}.bin"); blk1 = fft[:len(fft) // 2]
    atm1, dt = T2.load_atm(f"{dd}/ffa_{it}_c1_in.bin")
    _, patch1, _ = TA.unpack(blk1)
    idx1 = C._cell_lookup(blk1)[g1[:, 0].astype(int), g1[:, 1].astype(int)]
    trup = LC.infer_trup(g1, patch1["dth1"][idx1, 3], dt)
    gi, gj = g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1
    r1 = LC.land_substep(a4, g1, atm1["Q"][gj, gi, 0], trup, dt, None)
    new = LC.next_land_pbl_columns(np.array(b4), r1)
    assert np.abs(new[:, 9] - b4[:, 9]).max() < 1e-12            # before the fix: 1.9e-6 at (57,34), 1.5e-7 at (21,33)
    for ci, cj in ((57, 34), (21, 33)):
        k = np.where((b4[:, 0] == ci) & (b4[:, 1] == cj))[0][0]
        assert a4[k, 19] != b4[k, 19]                             # these two cells changed elhx class inside the substep
        assert abs(new[k, 9] - b4[k, 9]) < 1e-12
