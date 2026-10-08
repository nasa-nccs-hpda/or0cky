"""D190 helpers shared by the comparison scripts: reference npz access, host/device conversion, category comparison (jax_harness.field_category)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import jax_harness as H  # noqa: E402

DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}


def sync(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)
    return x


def group(ref, prefix):
    """{name: array} of the reference entries 'prefix/name' (one level; deeper levels keep their slashes)."""
    n = len(prefix)
    return {k[n:]: ref[k] for k in ref if k.startswith(prefix)}


def tree(ref, prefix):
    """Nested dict from 'a/b/c' keys below prefix."""
    out = {}
    for k, v in group(ref, prefix).items():
        d = out
        parts = k.split('/')
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def to_dev(x):
    if isinstance(x, dict):
        return {k: to_dev(v) for k, v in x.items()}
    return jnp.asarray(x)


def to_np(x):
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    return np.asarray(x)


def cat(a, b, mask=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mask is not None:
        m = mask if a.ndim == mask.ndim else mask[..., None]
        a, b = np.where(m, a, 0.0), np.where(m, b, 0.0)
    r = H.field_category(a, b)
    return r['cat'], r['max_abs'], r['rel'], r['n_diff'], r['n']


def compare(name_prefix, got, ref, mask=None, out=None, quiet=False, skip=()):
    """Compare two dicts (flat) key by key.  Returns {key: (cat, max_abs, rel, ndiff, n)} and prints the non-A ones."""
    res = {} if out is None else out
    for k in sorted(ref):
        if k in skip or k not in got:
            continue
        g, r = np.asarray(got[k]), np.asarray(ref[k])
        if r.dtype.kind not in 'fiub' or g.shape != r.shape:
            if g.shape != r.shape:
                res[name_prefix + k] = ('D', float('inf'), float('inf'), -1, r.size)
            continue
        res[name_prefix + k] = cat(g.astype(float), r.astype(float), mask)
    if not quiet:
        cnt = {}
        for kk, v in res.items():
            if kk.startswith(name_prefix):
                cnt[v[0]] = cnt.get(v[0], 0) + 1
        print(f'{name_prefix:30s} {cnt}')
        for kk, v in res.items():
            if kk.startswith(name_prefix) and v[0] != 'A':
                print(f'    {kk:50s} {v[0]} max_abs {v[1]:.3e} rel {v[2]:.3e} ndiff {v[3]}/{v[4]}')
    return res
