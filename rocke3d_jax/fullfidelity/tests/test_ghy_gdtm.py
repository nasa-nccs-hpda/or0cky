"""D204: the GHY sub-iteration schedule computed from the state (ghy_jax.advnc_gdtm, Fortran GHY.f:2389-2416) on REAL nov26_day records.
Checks: (1) computed nit == recorded ffnit for every cell (4-substep and >= 11 cells included); (2) outputs equal the recorded-schedule advnc at rounding level
(<= 1e-12 scaled), and the 'recorded' mode of advnc_sched is bitwise the old advnc; (3) the NumPy ghy_ref_nit loop (use_recorded_dts=False) agrees on a cell sample;
(4) a stiff forcing (large ch*vs) makes the computed schedule take more iterations than the recorded one and stay finite."""
import glob, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DAY = sorted(glob.glob(f"{FF}/nov26_day/ffg_*.bin"))
pytestmark = pytest.mark.skipif(len(DAY) < 54, reason="nov26_day ffg dumps not available")

import jax            # noqa: E402
import jax.numpy as jnp   # noqa: E402
import ghy_compare as GC       # noqa: E402
import ghy_advnc_test as AT    # noqa: E402
import jax_surface as JS       # noqa: E402
import ghy_jax as J            # noqa: E402

OUT = ('w', 'ht', 'tp', 'tbcs', 'tsns', 'ashg', 'alhg', 'aevap', 'aruns', 'arunu', 'aeruns', 'aerunu', 'ae0', 'abetad')


def _batch(step, ns):
    g = GC.load(f"{FF}/nov26_day/ffg_{33312 + step}.bin")
    n = len(g) // 2
    rec = g[ns * n:(ns + 1) * n]
    gb = JS._ghy_batch(rec, AT)
    return rec, gb


def _args(gb, forcing=None):
    f = {k: jnp.asarray(v) for k, v in gb['forcing'].items()}
    if forcing:
        f.update(forcing)
    return ({k: jnp.asarray(v) for k, v in gb['static'].items()}, {k: jnp.asarray(v) for k, v in gb['dyn0'].items()}, f, jnp.asarray(gb['edts']),
            jnp.asarray(gb['ecnc']), jnp.asarray(gb['ebet']), jnp.asarray(gb['elai']), jnp.asarray(gb['nsub']), jnp.asarray(gb['dt']), jnp.asarray(gb['snowm']))


@pytest.fixture(scope="module")
def case():
    rec, gb = _batch(14, 0)          # step 14: 185 cells with ffnit >= 4
    ms = gb['edts'].shape[1]
    base = jax.jit(J.advnc, static_argnames=('max_substeps',))(*_args(gb), max_substeps=ms)
    new = jax.jit(lambda *a, max_substeps: J.advnc_sched(*a, max_substeps=max_substeps, mode='computed'), static_argnames=('max_substeps',))(*_args(gb), max_substeps=ms)
    return rec, gb, ms, base, new


def test_computed_nit_equals_recorded_ffnit(case):
    rec, gb, ms, base, new = case
    ffnit = np.round(rec[:, 289]).astype(int)
    assert (ffnit >= 4).sum() > 100 and (ffnit == 4).any()
    assert np.array_equal(np.asarray(new['nit_gdtm']), ffnit)
    assert not np.asarray(new['nit_exhausted']).any()


def test_outputs_equal_recorded_schedule_at_rounding_level(case):
    rec, gb, ms, base, new = case
    for k in OUT:
        a, b = np.asarray(new[k], float), np.asarray(base[k], float)
        assert np.nanmax(np.abs(a - b) / np.maximum(1.0, np.abs(b))) <= 1e-12, k


def test_recorded_mode_is_the_old_advnc_bitwise(case):
    rec, gb, ms, base, new = case
    old = jax.jit(lambda *a, max_substeps: J.advnc_sched(*a, max_substeps=max_substeps, mode='recorded'), static_argnames=('max_substeps',))(*_args(gb), max_substeps=ms)
    for k in OUT:
        assert np.array_equal(np.asarray(old[k]), np.asarray(base[k])), k


def test_numpy_loop_agrees_on_sample(case):
    import ghy_ref_nit as N
    rec, gb, ms, base, new = case
    ffnit = np.round(rec[:, 289]).astype(int)
    idx = list(np.where(ffnit >= 4)[0][:6]) + list(np.where(ffnit == 1)[0][:4])
    for i in idx:
        col, refs, info = N.run_cell_full(rec[i], dt=900.0, use_recorded_dts=False)
        assert info['nit'] == int(new['nit_gdtm'][i])
        assert abs(col.tsns - float(new['tsns'][i])) < 1e-9
        assert abs(col.aevap - float(new['aevap'][i])) < 1e-12
        assert abs(col.ashg - float(new['ashg'][i])) < 1e-6 * max(1.0, abs(col.ashg))


def test_stiff_forcing_needs_more_iterations_and_stays_finite(case):
    rec, gb, ms, base, new = case
    ch = np.asarray(gb['forcing']['ch']) * 20.0         # large ch*vs: the stability limit 0.5*ak2/xk2 of gdtm drops
    st = jax.jit(lambda *a, max_substeps: J.advnc_sched(*a, max_substeps=max_substeps, mode='computed'), static_argnames=('max_substeps',))(
        *_args(gb, dict(ch=jnp.asarray(ch))), max_substeps=ms)
    ffnit = np.round(rec[:, 289]).astype(int)
    nit = np.asarray(st['nit_gdtm'])
    assert (nit > ffnit).sum() > 50
    land = np.asarray(gb['forcing']['fb']) + np.asarray(gb['forcing']['fv']) > 0
    for k in ('tsns', 'ashg', 'aevap'):
        assert np.isfinite(np.asarray(st[k])[land]).all(), k
