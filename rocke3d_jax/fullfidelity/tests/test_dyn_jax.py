"""Tests for the JAX dynamics stages (D139-D141): dyn_jax_fft/advecv/pgf/filter/pointwise.py and dyn_step_jax.py against the
numpy chain (dyn_step.py, numpy-pow mode) and the real-Fortran dumps.  Skipped if the ff_data dumps are absent.

Bounds (scale-relative = max|a-b| / max|b|): the stage target is 1e-12; measured JAX-vs-numpy on this CPU with the
FMA-free XLA flag (dyn_jax_env) is 0 (bitwise) for every stage, so the tests use 1e-13 for stages and 1e-12 for the end
state of the step (measured: see Reports ledger D140).  Mutation checks: dropping a stage or perturbing a constant must
break the agreement by orders of magnitude.
"""
import dyn_jax_env  # noqa: F401  (must precede the first jax import)
import copy

import numpy as np
import pytest

import dyn_step as ds
import dyn_step_compare as dc
import dyn_step_jax as dj
import dyn_jax_compare as jc
import dyn_jax_fft as jf
import dyn_jax_advecv as jav
import dyn_fft72_ff as f72
import dyn_advecv_compare as vc
import dyn_aflux_compare as cm
import dyn_pgf_compare as pc
import dyn_pgf_ff as fp
import dyn_advecv_ff as fv

HAVE = ds.available('nov26')
NEED = pytest.mark.skipif(not HAVE, reason="ff_data ffd_state_* dumps not present on this host")
DATE, IT0 = 'nov26', 33312
F = lambda x: np.ascontiguousarray(x, dtype=np.float64)
STAGE_TOL = 1e-13
END_TOL = 1e-12
_C = {}


def ctx_kit():
    if 'k' not in _C:
        ctx = ds.load_ctx(DATE, imf_pow=False)
        _C['k'] = (ctx, dj.Kit(ctx))
    return _C['k']


def test_xla_fma_free():
    assert dj.dyn_jax_env.fma_free(), "XLA contracts a*b+c into FMA: JAX stages would differ from numpy at ~1e-13"


def test_fft_roundtrip_bitwise_vs_numpy():
    import jax
    import jax.numpy as jnp
    x = np.random.default_rng(3).standard_normal((72, 200))
    A0, B0 = f72.fft_rows(x)
    A1, B1 = jax.jit(jf.fft_rows_jax)(jnp.asarray(x))
    y0 = f72.ffti_rows(A0, B0)
    y1 = jax.jit(jf.ffti_rows_jax)(jnp.asarray(A0), jnp.asarray(B0))
    for a, b in ((A0, A1), (B0, B1), (y0, y1)):
        assert dc.stats(np.asarray(b), a)['rel'] <= 1e-15
    # mutation: a perturbed input must be detected
    assert dc.stats(np.asarray(jax.jit(jf.fft_rows_jax)(jnp.asarray(x * (1 + 1e-9))[:, :])[0]), A0)['rel'] > 1e-12


@pytest.fixture(scope='module')
def stage_check():
    if not HAVE:
        pytest.skip("no dumps")
    ctx, kit = ctx_kit()
    s1 = ds.load_state(ds.state_path(DATE, IT0, 1))
    ck = jc.StageCheck(DATE, IT0, kit)
    ds.dyn_step(s1, ctx, itime=IT0, hook=ck)
    return ck


@NEED
def test_every_converted_stage_matches_numpy(stage_check):
    kinds_seen = {k.split('.')[0] for k in stage_check.jn}
    assert kinds_seen == set(dj.JAX_KINDS)
    bad = {k: s['rel'] for k, s in stage_check.jn.items() if s['rel'] > STAGE_TOL}
    assert not bad, bad
    # a stage output that is all-zero would make the comparison vacuous
    assert all(s['n'] > 0 for s in stage_check.jn.values())


@NEED
def test_jax_advecv_pgf_vs_real_dump_as_good_as_numpy(stage_check):
    for k, s in stage_check.jr.items():
        n = stage_check.nr[k]
        assert s['rel'] <= max(1.5 * n['rel'], 1e-14) and s['rel'] <= 1e-12, (k, s['rel'], n['rel'])


@NEED
def test_advecv_pgf_direct_on_dumped_inputs_and_mutations():
    ctx, kit = ctx_kit()
    g = ctx.g
    i = vc.load_advecv_in(cm.fname(DATE, IT0, 3, 'advecv', ds.FF_DEFAULT) + "_in.bin")
    args = [F(i[k]) for k in ('u', 'v', 'mmean', 'mbefor', 'ut', 'vt', 'mafter', 'mu', 'mv', 'mw', 'spa')]
    un, vn = fv.advecv(float(i['dt1']), *[i[k] for k in ('u', 'v', 'mmean', 'mbefor', 'ut', 'vt', 'mafter', 'mu', 'mv', 'mw',
                                                         'spa')], g)
    uj, vj = jav.advecv_jax(float(i['dt1']), *args, kit.adv_geo)
    assert dc.stats(np.asarray(uj), un)['rel'] <= STAGE_TOL and dc.stats(np.asarray(vj), vn)['rel'] <= STAGE_TOL
    geo = dict(kit.adv_geo); geo['dxv'] = geo['dxv'] * 1.001                   # mutation: perturbed geometry
    um, _ = jav.advecv_jax(float(i['dt1']), *args, geo)
    assert dc.stats(np.asarray(um), un)['rel'] > 1e-8
    p = pc.load_pgf_in(cm.fname(DATE, IT0, 3, 'pgf', ds.FF_DEFAULT) + "_in.bin")
    rn = fp.pgf(p['dt1'], p['mam'], p['ut'], p['vt'], p['mafter'], p['s0'], p['sz'], p['dut'], p['dvt'], g, tab=ctx.tab)
    pa = [F(p[k]) for k in ('mam', 'ut', 'vt', 'mafter', 's0', 'sz', 'dut', 'dvt')]
    rj = kit.pgf(float(p['dt1']), *pa, kit.pgf_geo)
    for k in ('gz', 'adm', 'pgfu', 'dut', 'dvt', 'ut', 'vt'):
        assert dc.stats(np.asarray(rj[k]), rn[k])['rel'] <= STAGE_TOL, k
    geo = dict(kit.pgf_geo); geo['kapa'] = geo['kapa'] * 1.0001                 # mutation: wrong exponent
    rm = kit.pgf(float(p['dt1']), *pa, geo)
    assert dc.stats(np.asarray(rm['gz']), rn['gz'])['rel'] > 1e-6


@NEED
def test_end_state_step0_matches_numpy_and_real_and_mutations():
    ctx, kit = ctx_kit()
    r = jc.end_state(DATE, IT0, ctx, kit)
    for k, s in r['jn'].items():
        assert s['rel'] <= END_TOL, (k, s)
    for k, s in r['jr'].items():                                # no worse than the numpy chain against the real state
        assert s['rel'] <= max(2 * r['nr'][k]['rel'], 1e-14) and s['rel'] <= END_TOL, (k, s, r['nr'][k])
    # mutation 1: dropping the SDRAG stages from the JAX plan changes U,V far beyond the agreement
    s1 = ds.load_state(ds.state_path(DATE, IT0, 1))
    plan = ds.step_plan(nstep=(IT0 - ds.ITIMEI) * 4)
    wn = ds.dyn_step(s1, ctx, itime=IT0, plan=plan)
    wm = dj.dyn_step_jax(s1, ctx, kit, plan=[st for st in plan if st.kind != 'sdrag'])
    assert dc.stats(wm['U'], wn['U'])['rel'] > 1e-7
    # mutation 2: a swapped pair of dependent stages (pgf before advecv in the first pass) also breaks it
    idx = [n for n, st in enumerate(plan) if st.kind == 'pgf'][0]
    pl2 = list(plan); pl2[idx - 1], pl2[idx] = pl2[idx], pl2[idx - 1]
    wm2 = dj.dyn_step_jax(s1, ctx, kit, plan=pl2)
    assert dc.stats(wm2['U'], wn['U'])['rel'] > 1e-9 or dc.stats(wm2['V'], wn['V'])['rel'] > 1e-9
