"""D218: the device radiation-column fill (drv_radcols_dev.fill_tpl) reproduces the RECORDED template columns bitwise on the radiation steps of ff_data/nov26_day
(there the recorded columns are FSF/TRSURF/TRHR0 of the step's server output times the step's COSZ1: D176), and the device ice legs equal the NumPy ones."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clouds_jax_env  # noqa: F401,E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import atm_step as A  # noqa: E402
import jax_surface as JS  # noqa: E402
import radiation_server as rs  # noqa: E402
import drv_radcols as DR  # noqa: E402
import drv_radcols_dev as DD  # noqa: E402

DAY = 'nov26_day'
pytestmark = pytest.mark.skipif(not os.path.exists(f'{rs.FF}/{DAY}/rsv_n26_33312_out.bin'), reason='nov26_day radiation packets not available')


def _leaves(t, p=''):
    if isinstance(t, dict):
        for k in sorted(t):
            yield from _leaves(t[k], f'{p}/{k}')
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t):
            yield from _leaves(v, f'{p}/{i}')
    else:
        yield p, np.asarray(t)


@pytest.mark.parametrize('it', [33312, 33317])
def test_fill_tpl_equals_recorded_template_on_radiation_steps(it):
    R = A.Real(DAY, it)
    tpl, host = JS.build_template(A.surface_records(R))
    out = rs.read_packet(f'{rs.FF}/{DAY}/rsv_n26_{it}_out.bin')
    rst = DD.rs_from_server(out)
    new = DD.fill_tpl(tpl, rst, np.asarray(out['COSZ1']), host)
    n, bad = 0, []
    for (p, a), (_, b) in zip(_leaves(tpl), _leaves(new)):
        n += 1
        if p == '/trup':           # recorded value is INFERRED from the recorded land patch (land_chain.infer_trup); the device value is TRSURF(4) itself: rounding-level difference
            if not np.allclose(a, b, rtol=1e-14, atol=0.0):
                bad.append((p, float(np.max(np.abs(a - b)))))
            continue
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append((p, float(np.max(np.abs(a - b))) if a.shape == b.shape else 'shape'))
    assert not bad, f'{len(bad)} of {n} leaves differ: {bad[:6]}'


def test_device_ice_leg_equals_numpy_leg():
    rng = np.random.default_rng(3)
    F = rng.random((4, 72, 46)) * 300.0
    old, new = rng.random((72, 46)), rng.random((72, 46))
    new[::5] = old[::5]
    mask = rng.random((72, 46)) > 0.3
    a = DR.reset_surf_fluxes_ice(F, old, new, mask)
    b = np.asarray(DD._leg(jax.numpy.asarray(F), jax.numpy.asarray(old), jax.numpy.asarray(new), jax.numpy.asarray(mask)))
    assert np.array_equal(a, b)
