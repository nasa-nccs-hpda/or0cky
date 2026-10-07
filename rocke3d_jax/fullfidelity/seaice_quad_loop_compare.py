"""D173 main question: is the D166 'our ice already differs from the real ADVSI entry at step 0' explained by the float64 Ti/Ti2b?

Runs the D164/D166 free surface loop (surface_loop_advsi) for the nov26 window with the batched JAX sea-ice core's Ti/Ti2b (seaice_core_jax, the
version surface_loop actually uses; float64) either unchanged or replaced by binary128 emulations (seaice_quad_ff.ti_quad/ti2b_quad evaluated per
element through jax.pure_callback; module attributes swapped for the duration of the call, no file modified).  At every step the ice state that
surface_post hands to ADVSI (ocean cells, pole rows i > 1 excluded as in D166) is compared with the real ADVSI entry dump ffadv_in_<it>.bin
(rsi, msi, snowi, hsi, ssi): number of differing cells, max abs, and field scale.  Step 0 starts from the real restart, so it is the clean test;
later steps start from our own carried state.
Usage: python seaice_quad_loop_compare.py [nsteps=1]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import seaice_core_jax as SJ  # noqa: E402
import seaice_quad_ff as Q  # noqa: E402
import surface_loop as L  # noqa: E402
import surface_loop_advsi as LA  # noqa: E402
import advsi_ff as A  # noqa: E402


def _np_ti(e, s):
    e, s = np.broadcast_arrays(np.asarray(e, float), np.asarray(s, float))
    out = np.zeros(e.shape)
    for idx in np.ndindex(e.shape):
        try:
            out[idx] = Q.ti_quad(float(e[idx]), float(s[idx]))
        except Exception:      # masked lanes of the batched code (e.g. det < 0, 0/0): value is discarded by jnp.where upstream
            out[idx] = 0.0
    return out


def _np_ti2b(e, s, sn, mi):
    e, s, sn, mi = np.broadcast_arrays(*(np.asarray(x, float) for x in (e, s, sn, mi)))
    out = np.zeros(e.shape)
    for idx in np.ndindex(e.shape):
        try:
            out[idx] = Q.ti2b_quad(float(e[idx]), float(s[idx]), float(sn[idx]), float(mi[idx]))
        except Exception:
            out[idx] = 0.0
    return out


def ti_cb(e, s):
    e, s = jnp.broadcast_arrays(jnp.asarray(e, jnp.float64), jnp.asarray(s, jnp.float64))
    return jax.pure_callback(_np_ti, jax.ShapeDtypeStruct(e.shape, jnp.float64), e, s)


def ti2b_cb(e, s, sn, mi):
    e, s, sn, mi = jnp.broadcast_arrays(*(jnp.asarray(x, jnp.float64) for x in (e, s, sn, mi)))
    return jax.pure_callback(_np_ti2b, jax.ShapeDtypeStruct(e.shape, jnp.float64), e, s, sn, mi)


class patched_jax:
    def __enter__(self):
        self.saved = (SJ.Ti, SJ.Ti2b)
        SJ.Ti, SJ.Ti2b = ti_cb, ti2b_cb
        jax.clear_caches()

    def __exit__(self, *a):
        SJ.Ti, SJ.Ti2b = self.saved
        jax.clear_caches()


def entry_vs_real(date='nov26', it0=33312, nsteps=1, ff=L.FF, log=print):
    """Our ice entering ADVSI (surface_pre -> surface_post, no ADVSI applied, state carried from our own previous step through ADVSI as in
    surface_loop_advsi.run_free) versus the real ADVSI entry dump."""
    st = L.load_statics(date, ff)
    st['ctx'] = L.make_ocean_ctx(date, ff)
    S = L.init_surface_state(date, ff, st=st)
    adv = LA.init_adv(date, ff)
    advdir = f"{ff}/advsi_dumps/{date}"
    advgeo = LA.advsi_geo_from_dump(date, it0, ff, adv_dir=advdir)
    rows = []
    for k in range(nsteps):
        it = it0 + k
        inp = L.replay_inputs(date, it, ff)
        S1, mid = L.surface_pre(S, st, inp)
        ausi, avsi = LA.usi_vsi_from_ffy(date, it, ff)
        rsisave = S1['ice']['rsi'].copy()
        S2, post = L.surface_post(S1, st, inp, mid)
        d, _ = A.read_dump(f"{advdir}/ffadv_in_{it}.bin")
        oc = st['geo']['is_ocean'].copy()
        oc[1:, 0] = False; oc[1:, -1] = False
        row = dict(step=k, itime=it)
        for kk in ('rsi', 'msi', 'snowi', 'hsi', 'ssi'):
            a, b = S2['ice'][kk], d[kk]
            diff = np.abs(a - b)
            m = oc[..., None] if diff.ndim == 3 else oc
            diff = np.where(m, diff, 0.0)
            cells = (diff > 0).any(axis=-1) if diff.ndim == 3 else diff > 0
            row[kk] = dict(ncells=int(cells.sum()), maxabs=float(diff.max()), scale=float(np.abs(np.where(m, b, 0.0)).max()),
                           nbitwise_elems=int(np.sum(np.where(m, a == b, True) & m)) if False else int((np.where(m, a, 0.0) == np.where(m, b, 0.0)).sum()))
        rows.append(row)
        log(f"step {k} it={it}: " + "  ".join(f"{kk}: {row[kk]['ncells']} cells, max {row[kk]['maxabs']:.3e} (scale {row[kk]['scale']:.2e})" for kk in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')))
        # carry our state through ADVSI (as in the D166 free loop), using the R1-replayed inputs
        ice2, adv, _ = LA.advsi_on_state(S2['ice'], adv, rsisave, ausi, avsi, st['geo'], advgeo)
        S = dict(S2, ice=ice2)
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("== float64 Ti/Ti2b (as D166)")
    base = entry_vs_real(nsteps=n)
    print("== binary128 Ti/Ti2b (callback)")
    with patched_jax():
        quad = entry_vs_real(nsteps=n)
