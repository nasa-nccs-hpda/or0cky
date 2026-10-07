"""D186: execution counters that do not depend on how a kernel is called.

`install()` (call it BEFORE any module that defines `@jax.jit` functions is imported) replaces `jax.jit` by a thin wrapper whose callable counts one
EXECUTION per call made outside a trace (calls made while another function is being traced are not executions and are not counted), and reads the
number of eager single-primitive dispatches from the cache statistics of jax._src.dispatch.xla_primitive_callable.  `snapshot()` returns the totals;
`stage(name)` attributes the counts of a block to a name.  The counters see jit/primitive executions only: host<->device copies are counted by
jax_harness.Counters.instrument_transfers (limits documented there).  Pure Python bookkeeping; results are never touched.
"""
import contextlib
import functools
import threading
import time

_state = dict(installed=False, jit_calls=0, by_name={}, stack=[])
_lock = threading.Lock()


def _has_tracer(args, kwargs):
    import jax
    leaves = jax.tree_util.tree_leaves((args, kwargs))
    return any(isinstance(x, jax.core.Tracer) for x in leaves)


class _CountedJit:
    def __init__(self, fn):
        self._fn = fn
        functools.update_wrapper(self, getattr(fn, '__wrapped__', fn), updated=())

    def __call__(self, *a, **k):
        if not _has_tracer(a, k):
            with _lock:
                _state['jit_calls'] += 1
                nm = _state['stack'][-1] if _state['stack'] else '(outside)'
                _state['by_name'][nm] = _state['by_name'].get(nm, 0) + 1
        return self._fn(*a, **k)

    def __getattr__(self, item):
        return getattr(self._fn, item)


def install():
    import jax
    if _state['installed']:
        return
    orig = jax.jit

    def jit(fun=None, *a, **k):
        if fun is None:                                   # jax.jit(static_argnames=...) used as a decorator factory
            return lambda f: _CountedJit(orig(f, *a, **k))
        return _CountedJit(orig(fun, *a, **k))
    jit.__wrapped__ = orig
    jax.jit = jit
    _state['orig_jit'] = orig
    _state['installed'] = True


def _prim_calls():
    try:
        from jax._src import dispatch
        ci = dispatch.xla_primitive_callable.cache_info()
        return ci.hits + ci.misses
    except Exception:
        return None


def snapshot():
    with _lock:
        return dict(jit_calls=_state['jit_calls'], by_name=dict(_state['by_name']), eager_primitive_calls=_prim_calls())


def reset():
    with _lock:
        _state['jit_calls'] = 0
        _state['by_name'] = {}
        _state['base_prim'] = _prim_calls()


@contextlib.contextmanager
def stage(name):
    """Attribute jit executions inside the block to `name`; yields a dict filled at exit with jit_calls, eager primitive calls and seconds."""
    res = {}
    with _lock:
        _state['stack'].append(name)
        j0 = _state['by_name'].get(name, 0)
    p0, t0 = _prim_calls(), time.perf_counter()
    try:
        yield res
    finally:
        res['seconds'] = time.perf_counter() - t0
        p1 = _prim_calls()
        res['eager_primitive_calls'] = None if p0 is None or p1 is None else p1 - p0
        with _lock:
            _state['stack'].pop()
            res['jit_calls'] = _state['by_name'].get(name, 0) - j0
