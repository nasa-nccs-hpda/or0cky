"""Does stconv_jax.reduce_fig (frexp / ldexp / exp2) behave the same on this device as in numpy? (small, no data needed)

usage: python gpu_reduce_fig_probe.py        (run on the GPU node, then compare with a CPU run)
Prints mismatch and non-finite counts for exponent(), reduce_fig() and the primitives frexp, ldexp, exp2 against numpy references, on structured and random values at
the scales of the strait moments. Nonzero counts or NaN on the GPU would explain the NaN of the strait arrays (job 58801352); zero on both rules this function out.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import stconv_jax as SC   # noqa: E402

print('backend', jax.default_backend(), 'devices', jax.devices())
rng = np.random.default_rng(0)
vals = np.concatenate([np.zeros(4), np.array([1e-300, 5e-324, 1.0, 0.5, 0.999999999999, 1e13, 1e14, 3.7e17, 1e30, 1e-20]),
                       rng.normal(size=4000) * 10.0 ** rng.integers(-30, 31, size=4000)])
vals = np.concatenate([vals, -vals])


def np_exponent(x):
    _, e = np.frexp(x)
    return np.where(x == 0.0, 0, e)


def np_reduce_fig(nsig, rx):
    nint = np.sign(rx) * np.floor(np.abs(rx) * np.exp2(-nsig.astype(np.float64)) + 0.5)
    red = np.ldexp(nint, nsig)
    return np.where(nsig + 30 > np_exponent(rx), red, rx)


def report(name, got, want):
    got, want = np.asarray(got, float), np.asarray(want, float)
    nf = int((~np.isfinite(got)).sum())
    mism = int((~((got == want) | (np.isnan(got) & np.isnan(want)))).sum())
    print(f'{name:28s} non-finite {nf:5d}   mismatches vs numpy {mism:5d}  of {got.size}')


x = jnp.asarray(vals)
m, e = jnp.frexp(x)
mn, en = np.frexp(vals)
report('frexp mantissa', m, mn)
report('frexp exponent', e, en)
report('exponent()', SC.exponent(x), np_exponent(vals))
nsig = np.asarray(np_exponent(vals) - 1 - 42, dtype=np.int32)
report('exp2(-nsig)', jnp.exp2(-jnp.asarray(nsig).astype(jnp.float64)), np.exp2(-nsig.astype(np.float64)))
nint = np.sign(vals) * np.floor(np.abs(vals) * np.exp2(-nsig.astype(np.float64)) + 0.5)
report('ldexp(nint, nsig)', jnp.ldexp(jnp.asarray(nint), jnp.asarray(nsig)), np.ldexp(nint, nsig))
report('reduce_fig', SC.reduce_fig(jnp.asarray(nsig), x), np_reduce_fig(nsig, vals))
nsig4 = nsig + 4
report('reduce_fig (+4, salt)', SC.reduce_fig(jnp.asarray(nsig4), x), np_reduce_fig(nsig4, vals))
