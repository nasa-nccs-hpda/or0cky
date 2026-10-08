"""D194: GHY substep schedule of ffnit >= 12 cells from OUR precipitation (diagnosis D194_STEP7_DIAGNOSIS.md).

Finding: for a cell whose recorded GHY iteration count is > 11 the Ent/dts rows of iterations 12.. are not in the ffg record; ghy_advnc_test.build_batch (D158)
reconstructs ALL dts of that cell with the real loop (ghy_ref_nit.run_cell_full) from the ROW.  The NumPy chain (surface_loop.Loop.stage_surface) overwrites the row's
precipitation columns 143-145 with OUR PREC/EPREC/PRECSS before build_batch, so the reconstructed dts follow our precipitation.  The assembled step
(jax_coupled.Coupled) builds the batch once in jax_surface.build_template from the RECORDED row and only overwrites the pr/htpr/prs forcing on the device afterwards,
so the dts of those cells stay on the recorded precipitation.

This module reproduces the reference order WITHOUT editing any existing file: install() wraps jax_surface.build_template (keeps the recorded g1/g2 rows and the
ffnit>11 cell list) and jax_tpl_state.make_apply_state (before the apply, only when such cells exist: ONE declared device->host read of PREC/EPREC/PRECSS, the
batch is rebuilt from the rows with our precipitation, and edts/ecnc/elai/ebet/nsub/dt of exactly those cells are replaced on the device).  No tolerance, no
category is touched; an assertion of build_batch_nit (recomputed nit != recorded ffnit) is NOT caught, exactly as in the reference."""
import clouds_jax_env  # noqa: F401
import numpy as np
import jax.numpy as jnp

import jax_surface as JS
import jax_tpl_state as TS
import ghy_advnc_test as AT

DTSRC, RHOW = 1800.0, 1000.0
CTX = dict(g=None, hit=None, width=None, log=[])
_ORIG = {}
KEYS = ('edts', 'ecnc', 'elai', 'ebet', 'nsub', 'dt')


def _bt(rec, *a, **k):
    tpl, host = _ORIG['bt'](rec, *a, **k)
    CTX['g'] = (np.array(rec['g1'], copy=True), np.array(rec['g2'], copy=True))
    CTX['hit'] = [np.nonzero(np.round(np.asarray(rec[g])[:, 289]).astype(int) > 11)[0] for g in ('g1', 'g2')]
    CTX['width'] = host['max_substeps']
    return tpl, host


def _fix(tpl, prec, eprec, precss):
    P, E, Ps = (np.asarray(x) for x in (prec, eprec, precss))      # the declared device->host read (3 arrays, only on steps with ffnit>11 cells)
    ghy = list(tpl['ghy'])
    for k, g0 in enumerate(CTX['g']):
        idx = CTX['hit'][k]
        if not len(idx):
            continue
        g = np.array(g0, dtype=np.float64, copy=True)
        i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
        g[:, 143] = P[i, j] / (DTSRC * RHOW)                       # as surface_loop.Loop.stage_surface
        g[:, 144] = E[i, j] / DTSRC
        g[:, 145] = Ps[i, j] / (DTSRC * RHOW)
        gb = JS._ghy_batch(g, AT)                                  # build_batch -> build_batch_nit: asserts nit == ffnit like the reference
        w = CTX['width']
        assert gb['edts'].shape[1] <= w, ('batch wider than the compiled max_substeps', gb['edts'].shape, w)
        new = dict(ghy[k])
        rep = {}
        for key in KEYS:
            v = np.asarray(gb[key])[idx]
            if v.ndim >= 2 and v.shape[1] < w:
                v = np.pad(v, [(0, 0), (0, w - v.shape[1])] + [(0, 0)] * (v.ndim - 2))
            old = np.asarray(new[key])[idx]
            rep[key] = float(np.abs(v - old).max())
            new[key] = new[key].at[jnp.asarray(idx)].set(jnp.asarray(v).astype(new[key].dtype))
        rep['dprec_rel_max'] = float(np.abs(g[idx, 143] - np.asarray(g0)[idx, 143]).max())
        CTX['log'].append(dict(substep=k + 1, cells=[(int(i[c]) + 1, int(j[c]) + 1) for c in idx], nsub=[int(x) for x in np.asarray(gb['nsub'])[idx]], max_abs_change=rep))
        ghy[k] = new
    return dict(tpl, ghy=ghy)


def _mk(host, K):
    inner = _ORIG['ma'](host, K)

    def apply(tpl, S1, ag, ps, prec, eprec, precss, *rest):
        if CTX['hit'] is not None and any(len(h) for h in CTX['hit']):
            tpl = _fix(tpl, prec, eprec, precss)
        return inner(tpl, S1, ag, ps, prec, eprec, precss, *rest)
    return apply


def install():
    if 'bt' in _ORIG:
        return
    _ORIG['bt'], _ORIG['ma'] = JS.build_template, TS.make_apply_state
    JS.build_template, TS.make_apply_state = _bt, _mk


def uninstall():
    if 'bt' in _ORIG:
        JS.build_template, TS.make_apply_state = _ORIG.pop('bt'), _ORIG.pop('ma')
