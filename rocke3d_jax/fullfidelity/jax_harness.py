"""D184 (stage S0 of Reports/JAX_COVERAGE_MATRIX.md section 6): harness for every later result of the JAX-driven coupled step.

Nothing in this file computes physics.  It provides, for ACCEPTANCE_CRITERIA.md sections 1, 2, 3, 6 and 8:

 (1) provenance_header()          host, core affinity and thread env, XLA_FLAGS, JAX version/backend/devices, compile-cache check
                                  (raises HarnessError if a persistent cache is configured), git head and number of modified tracked
                                  files, libimf presence, python/numpy versions, timestamp.
 (2) Counters                     per named stage: jit calls, host<->device transfers (calls, bytes), host callbacks (calls, bytes,
                                  seconds), compiles; usable as context managers and decorators and as wrappers of existing (also
                                  jitted) functions.  The wrapped function's return value is passed through untouched.
 (3) RecordedInputRegistry        stages read recorded inputs only through it; a read of a name that was not declared raises
                                  UndeclaredRecordedInput; emits the list for the result file (ACCEPTANCE 1.5).
 (4) RadiationCallbackLog /       the required sentence (ACCEPTANCE 1.4) plus calls, bytes, seconds, sync points, and the libimf
     LibimfCallbackLog            sentence of ACCEPTANCE section 8.1.
 (5) StageRegistry                stage name, implementation kind (FORT / REC / NP / EJ / JJ of the matrix), measured time share;
                                  generates the "non-JAX stages" list (ACCEPTANCE 1.3).
 (6) the C1 REFERENCE             run_reference(): the NumPy chained step in libm mode (atm_step_fast.run_chain, ctx imf=False), step 0,
                                  end state saved to ff_data/ref_libm/ (outside git); determinism_check(); and
                                  compare_to_reference(): per field A/B/C/D categories of ACCEPTANCE section 3 and the
                                  named-exception-columns rule.

Command line (run under  taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_harness.py ...):
    header                              print the provenance header (fails if a compile cache is set)
    ref DATE TAG [OUTDIR]               run the libm-mode reference, step 0 of DATE (nov26|dec01|jan01); TAG names the run (1, 2)
    determinism DATE [OUTDIR]           compare run 1 and run 2 of the reference bitwise
    compare CAND.npz DATE [TAG]         compare a candidate end state (npz of field arrays) with the reference

Limits are stated where they arise; the transfer counters in particular count only what passes through the public entry points listed
in Counters.instrument_transfers (see its docstring).  Measured shares of time are inclusive of nothing: StageRegistry uses exclusive time.
"""
import contextlib
import datetime
import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_PRELOADED = 'jax' in sys.modules      # was jax imported before this module (then XLA_FLAGS set later may be too late)

SENTENCE_RADIATION = 'radiation computed by the original Fortran (hybrid component)'
SENTENCE_RADIATION_REPLAY = ('radiation replayed from the real record (not computed; no Fortran callback in this result)')
SENTENCE_LIBIMF = "libimf math functions provided by a host callback to the original build's runtime"
SENTENCE_LIBM = 'libm mode: the Intel libimf runtime is NOT used in this result (numpy/glibc pow and exp)'
REQUIRED_XLA_FLAGS = ('--xla_cpu_max_isa=AVX', '--xla_disable_hlo_passes=algsimp')
CACHE_ENV_VARS = ('JAX_COMPILATION_CACHE_DIR', 'CLOUDS_JAX_CACHE')
FF_DEFAULT = os.environ.get('FF_DATA', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data')
REF_DIR = os.path.join(FF_DEFAULT, 'ref_libm')
DATES = [('nov26', 33312), ('dec01', 33552), ('jan01', 17520)]
IM, JM = 72, 46
KINDS = ('FORT', 'REC', 'NP', 'EJ', 'JJ')


class HarnessError(RuntimeError):
    pass


class UndeclaredRecordedInput(HarnessError):
    pass


def _nbytes(obj, _seen=None):
    """Bytes held by numpy arrays / jax arrays inside obj (dicts, lists, tuples are walked)."""
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if hasattr(obj, 'nbytes') and hasattr(obj, 'shape') and not isinstance(obj, (dict, list, tuple)):
        try:
            return int(obj.nbytes)
        except Exception:
            return 0
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


# ======================================================================================= (1) provenance header
def _git(*args):
    try:
        return subprocess.run(['git', '-C', HERE] + list(args), capture_output=True, text=True, timeout=60).stdout
    except Exception as e:      # pragma: no cover
        return f'unavailable: {e}'


def check_no_compile_cache():
    """Raise HarnessError if any persistent JAX compile cache is configured (env or jax.config).  Returns the list of what was checked."""
    checked, bad = [], []
    for v in CACHE_ENV_VARS:
        checked.append(v)
        if os.environ.get(v):
            bad.append(f'env {v}={os.environ[v]}')
    if 'jax' in sys.modules:
        import jax
        for opt in ('jax_compilation_cache_dir',):
            checked.append(f'jax.config.{opt}')
            try:
                val = getattr(jax.config, opt, None)
            except Exception:
                val = None
            if val:
                bad.append(f'jax.config.{opt}={val}')
    if bad:
        raise HarnessError('persistent compile cache is set (not allowed for validated runs, ACCEPTANCE 2): ' + '; '.join(bad))
    return checked


def provenance_header(extra=None, require_xla_flags=False, check_cache=True):
    """Provenance of one run, as a JSON-able dict.  Raises HarnessError if a compile cache is set (check_cache) or, when
    require_xla_flags is True, if XLA_FLAGS lacks the two documented flags."""
    h = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
         'timestamp_local': datetime.datetime.now().astimezone().isoformat(timespec='seconds')}
    h['host'] = dict(hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                     cpu_count_visible=os.cpu_count())
    aff = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    h['core_affinity'] = dict(cores=aff, n_cores=len(aff) if aff else None, pid=os.getpid())
    thread_vars = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'XLA_FLAGS',
                   'TF_NUM_INTRAOP_THREADS', 'XLA_PYTHON_CLIENT_PREALLOCATE')
    h['thread_env'] = {k: os.environ.get(k) for k in thread_vars}
    xf = os.environ.get('XLA_FLAGS', '')
    flags_ok = all(f in xf for f in REQUIRED_XLA_FLAGS)
    h['xla_flags'] = dict(XLA_FLAGS=xf, documented_flags_present=flags_ok, required=list(REQUIRED_XLA_FLAGS),
                          jax_imported_before_harness=_JAX_PRELOADED)
    if require_xla_flags and not flags_ok:
        raise HarnessError(f'XLA_FLAGS lacks {REQUIRED_XLA_FLAGS}: {xf!r}')
    try:
        import jax
        devs = jax.devices()
        h['jax'] = dict(version=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in devs],
                        x64_enabled=bool(jax.config.jax_enable_x64))
        try:
            import jaxlib
            h['jax']['jaxlib_version'] = jaxlib.__version__
        except Exception:
            pass
    except Exception as e:
        h['jax'] = dict(error=repr(e))
    if check_cache:
        h['compile_cache'] = dict(checked=check_no_compile_cache(), cache_set=False)
    else:
        h['compile_cache'] = dict(checked=[], cache_set=None, note='check skipped by caller')
    head = _git('rev-parse', 'HEAD').strip()
    mod = [ln for ln in _git('status', '--porcelain', '--untracked-files=no').splitlines() if ln.strip()]
    untr = [ln for ln in _git('status', '--porcelain', '--untracked-files=normal').splitlines() if ln.startswith('??')]
    h['git'] = dict(head=head, branch=_git('rev-parse', '--abbrev-ref', 'HEAD').strip(), modified_tracked_files=len(mod),
                    modified_tracked_list=[ln[3:] for ln in mod][:50], untracked_entries=len(untr))
    try:
        sys.path.insert(0, HERE)
        import intel_libm_ff
        h['libimf'] = dict(available=bool(intel_libm_ff.available()), search_dir=os.environ.get('INTEL_LIBIMF_DIR', intel_libm_ff._DEFAULT))
    except Exception as e:
        h['libimf'] = dict(available=False, error=repr(e))
    h['versions'] = dict(python=sys.version.split()[0], python_full=sys.version.replace('\n', ' '), numpy=np.__version__,
                         executable=sys.executable)
    h['argv'] = list(sys.argv)
    h['sentences'] = dict(radiation=SENTENCE_RADIATION, libimf=SENTENCE_LIBIMF)
    if extra:
        h['extra'] = extra
    return h


HEADER_REQUIRED_KEYS = ('timestamp_utc', 'host', 'core_affinity', 'thread_env', 'xla_flags', 'jax', 'compile_cache', 'git', 'libimf',
                        'versions')


def header_text(h):
    """Human-readable lines of a header."""
    j, g = h['jax'], h['git']
    lines = [f"timestamp {h['timestamp_utc']} (UTC)", f"host {h['host']['hostname']} ({h['host']['platform']})",
             f"core affinity {h['core_affinity']['cores']} ({h['core_affinity']['n_cores']} cores), pid {h['core_affinity']['pid']}",
             'thread env ' + ' '.join(f'{k}={v}' for k, v in h['thread_env'].items() if v is not None and k != 'XLA_FLAGS'),
             f"XLA_FLAGS {h['xla_flags']['XLA_FLAGS']!r} (documented flags present: {h['xla_flags']['documented_flags_present']})",
             f"jax {j.get('version')} backend {j.get('backend')} devices {j.get('devices')}",
             f"compile cache: none set (checked {', '.join(h['compile_cache']['checked'])})",
             f"git head {g['head']} branch {g['branch']}, {g['modified_tracked_files']} modified tracked files, "
             f"{g['untracked_entries']} untracked entries",
             f"libimf available: {h['libimf']['available']}",
             f"python {h['versions']['python']} numpy {h['versions']['numpy']}"]
    return '\n'.join(lines)


def write_header(path, extra=None, **kw):
    h = provenance_header(extra=extra, **kw)
    with open(path, 'w') as f:
        json.dump(h, f, indent=1, default=str)
    return h


# ======================================================================================= (2) counters
_MONITOR_INSTALLED = False
_ACTIVE_COUNTERS = []


def _install_compile_listener():
    global _MONITOR_INSTALLED
    if _MONITOR_INSTALLED:
        return
    try:
        import jax.monitoring as jm

        def cb(event, duration, **kw):
            if event == '/jax/core/compile/backend_compile_duration':
                for c in list(_ACTIVE_COUNTERS):
                    c._add_compile(duration)
        jm.register_event_duration_secs_listener(cb)
        _MONITOR_INSTALLED = True
    except Exception:       # pragma: no cover
        pass


_STAT_FIELDS = ('entries', 'jit_calls', 'h2d_calls', 'h2d_bytes', 'd2h_calls', 'd2h_bytes', 'cb_calls', 'cb_bytes', 'cb_seconds',
                'compiles', 'compile_seconds', 'wall_seconds', 'self_seconds')


class _StageCtx:
    """Context manager and decorator for one named stage of a Counters object."""

    def __init__(self, counters, name):
        self.c, self.name = counters, name

    def __enter__(self):
        self.c._push(self.name)
        return self

    def __exit__(self, *exc):
        self.c._pop()
        return False

    def __call__(self, fn):
        c, name = self.c, self.name

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with _StageCtx(c, name):
                return fn(*a, **k)
        return wrapper


class _CountedFn:
    """Transparent wrapper around a callable (also a jitted one): counts one jit call per call and the host arrays in its arguments as
    host->device transfers; returns exactly what the wrapped function returns."""

    def __init__(self, counters, fn, name):
        self._c, self._fn, self._name = counters, fn, name
        functools.update_wrapper(self, fn, updated=())

    def __call__(self, *a, **k):
        c = self._c
        with c._attr(self._name):
            c._add('jit_calls', 1)
            n, nb = _host_array_leaves((a, k))
            if n:
                c._add('h2d_calls', n)
                c._add('h2d_bytes', nb)
            return self._fn(*a, **k)

    def __getattr__(self, item):
        return getattr(self._fn, item)


def _host_array_leaves(tree):
    """(number, bytes) of numpy arrays among the leaves of a nested args structure (these are copied to the device by a jit call)."""
    n = nb = 0
    stack = [tree]
    while stack:
        x = stack.pop()
        if isinstance(x, np.ndarray):
            n += 1
            nb += x.nbytes
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return n, nb


class Counters:
    """Per-stage counters.  Events go to the innermost active stage (a stack), or to '(outside)'.

    with C.stage('dyn'): ...        @C.stage('dyn') def f(...)        g = C.count_jit(jitted_fn, 'dyn')        C.host_callback(fn, 'rad')
    instrument_transfers() (context manager) additionally counts transfers made through the public entry points
    jax.device_put, jax.device_get, jnp.asarray / jnp.array of a numpy array (host->device) and calls of ArrayImpl.__array__
    (device->host; this is the path numpy takes for arrays on an accelerator).  NOT counted: transfers JAX makes internally without
    these entry points (e.g. numpy arrays passed to a jitted function that is not wrapped by count_jit), device-to-device copies, and,
    MEASURED LIMIT on the CPU backend (jax 0.5.3), np.asarray(jax_array), which numpy serves through the buffer protocol without calling
    __array__ (probe in the tests: not counted).  On the CPU backend there is no physical transfer in that case; on a GPU it goes through
    __array__ and is counted (not tested here: no GPU on this node).  Use Counters.to_host / jax.device_get for explicit,
    always-counted device->host copies."""

    def __init__(self):
        self._st = {}
        self._tl = threading.local()
        self._lock = threading.RLock()
        self.t_created = time.perf_counter()

    # -- stack
    def _stack(self):
        if not hasattr(self._tl, 'stack'):
            self._tl.stack = []
        return self._tl.stack

    def _rec(self, name):
        with self._lock:
            if name not in self._st:
                self._st[name] = {k: 0 if not k.endswith('seconds') else 0.0 for k in _STAT_FIELDS}
            return self._st[name]

    def _cur(self):
        s = self._stack()
        return s[-1][0] if s else '(outside)'

    def _push(self, name):
        self._stack().append([name, time.perf_counter(), 0.0])
        self._rec(name)['entries'] += 1

    def _pop(self):
        name, t0, child = self._stack().pop()
        dt = time.perf_counter() - t0
        r = self._rec(name)
        with self._lock:
            r['wall_seconds'] += dt
            r['self_seconds'] += dt - child
        if self._stack():
            self._stack()[-1][2] += dt

    @contextlib.contextmanager
    def _attr(self, name):
        """Attribute events to `name` if given (without timing), else to the current stage."""
        if name is None:
            yield
            return
        st = self._stack()
        st.append([name, None, 0.0])
        try:
            yield
        finally:
            st.pop()

    def _add(self, field, v):
        with self._lock:
            self._rec(self._cur())[field] += v

    def _add_compile(self, dur):
        self._add('compiles', 1)
        self._add('compile_seconds', float(dur))

    # -- public
    def stage(self, name):
        return _StageCtx(self, name)

    def count_jit(self, fn, name=None):
        return _CountedFn(self, fn, name)

    def to_device(self, x, name=None):
        import jax
        with self._attr(name):
            self._add('h2d_calls', 1)
            self._add('h2d_bytes', _nbytes(x))
            return _ORIG.get('device_put', jax.device_put)(x)

    def to_host(self, x, name=None):
        import jax
        with self._attr(name):
            self._add('d2h_calls', 1)
            self._add('d2h_bytes', _nbytes(x))
            return _ORIG.get('device_get', jax.device_get)(x)

    def host_callback(self, fn=None, name=None):
        """Wrap a host function that a jitted program calls (e.g. a jax.pure_callback target) or a host stage called from JAX code:
        counts calls, bytes (numpy arrays among arguments and results) and seconds.  Usable as decorator (with or without name)."""
        def deco(f):
            nm = name or getattr(f, '__name__', 'callback')

            @functools.wraps(f)
            def wrapper(*a, **k):
                t0 = time.perf_counter()
                res = f(*a, **k)
                dt = time.perf_counter() - t0
                with self._attr(nm):
                    self._add('cb_calls', 1)
                    self._add('cb_bytes', _nbytes((a, k)) + _nbytes(res))
                    self._add('cb_seconds', dt)
                return res
            return wrapper
        return deco(fn) if callable(fn) else deco

    @contextlib.contextmanager
    def instrument_transfers(self):
        import jax
        import jax.numpy as jnp
        from jax._src import array as jarray
        tl = threading.local()
        orig_dp, orig_dg, orig_arr = jax.device_put, jax.device_get, jarray.ArrayImpl.__array__
        orig_jasarray, orig_jarray = jnp.asarray, jnp.array
        _ORIG['device_put'], _ORIG['device_get'] = orig_dp, orig_dg
        me = self

        def guarded(kind, fn, nb_of):
            @functools.wraps(fn)
            def w(*a, **k):
                if getattr(tl, 'busy', False):
                    return fn(*a, **k)
                tl.busy = True
                try:
                    out = fn(*a, **k)
                    nb = nb_of(a, out)
                    if nb is not None:
                        me._add(f'{kind}_calls', 1)
                        me._add(f'{kind}_bytes', nb)
                    return out
                finally:
                    tl.busy = False
            return w

        def h2d_bytes(a, out):
            return _nbytes(a[0]) if a and isinstance(a[0], (np.ndarray, dict, list, tuple)) and _nbytes(a[0]) else None

        jax.device_put = guarded('h2d', orig_dp, lambda a, out: _nbytes(a[0]) if a else 0)
        jax.device_get = guarded('d2h', orig_dg, lambda a, out: _nbytes(out))
        jnp.asarray = guarded('h2d', orig_jasarray, h2d_bytes)
        jnp.array = guarded('h2d', orig_jarray, h2d_bytes)
        jarray.ArrayImpl.__array__ = guarded('d2h', orig_arr, lambda a, out: int(getattr(out, 'nbytes', 0)))
        try:
            yield self
        finally:
            jax.device_put, jax.device_get, jarray.ArrayImpl.__array__ = orig_dp, orig_dg, orig_arr
            jnp.asarray, jnp.array = orig_jasarray, orig_jarray

    def listen_compiles(self):
        """Count XLA backend compilations (jax.monitoring) into the active stage while this object is registered."""
        _install_compile_listener()
        if self not in _ACTIVE_COUNTERS:
            _ACTIVE_COUNTERS.append(self)
        return self

    def stop_listening(self):
        if self in _ACTIVE_COUNTERS:
            _ACTIVE_COUNTERS.remove(self)

    def reset(self):
        with self._lock:
            self._st.clear()

    def snapshot(self):
        with self._lock:
            return {k: dict(v) for k, v in self._st.items()}

    def totals(self):
        t = {k: 0 for k in _STAT_FIELDS}
        for v in self.snapshot().values():
            for k in _STAT_FIELDS:
                t[k] += v[k]
        return t

    def report_text(self):
        rows = ['stage | entries | jit calls | h2d calls/bytes | d2h calls/bytes | callbacks calls/bytes/s | compiles | self s']
        for n, v in sorted(self.snapshot().items()):
            rows.append(f"{n} | {v['entries']} | {v['jit_calls']} | {v['h2d_calls']}/{v['h2d_bytes']} | {v['d2h_calls']}/{v['d2h_bytes']} | "
                        f"{v['cb_calls']}/{v['cb_bytes']}/{v['cb_seconds']:.4f} | {v['compiles']} | {v['self_seconds']:.4f}")
        return '\n'.join(rows)


_ORIG = {}


# ======================================================================================= (3) recorded-input registry
class RecordedInputRegistry:
    """Declare name, source file, size; read only through read().  An undeclared name raises UndeclaredRecordedInput.

    role 'input'            a recorded input the step consumes instead of computing it (counted in the ACCEPTANCE 1.5 list)
    role 'comparison_only'  real-model data read only to compare against (not an input of the step; listed separately)"""

    def __init__(self):
        self._decl = {}
        self._reads = {}
        self.strict = True
        self.undeclared_attempts = []

    def declare(self, name, source_file, size_bytes=None, description='', role='input', origin=''):
        if role not in ('input', 'comparison_only'):
            raise ValueError(role)
        self._decl[name] = dict(name=name, source_file=source_file, declared_size_bytes=size_bytes, description=description, role=role,
                                origin=origin)
        return self

    def declared(self, name):
        return name in self._decl

    def read(self, name, loader=None, stage=None, nbytes=None):
        """Mark a read of `name` (and run loader() if given, returning its value).  Fails if `name` was not declared."""
        if name not in self._decl:
            self.undeclared_attempts.append(name)
            raise UndeclaredRecordedInput(f'recorded input {name!r} was read but is not declared (ACCEPTANCE 1.5); declared: '
                                          f'{sorted(self._decl)}')
        t0 = time.perf_counter()
        val = loader() if loader is not None else None
        dt = time.perf_counter() - t0
        nb = nbytes if nbytes is not None else _nbytes(val)
        r = self._reads.setdefault(name, dict(read_count=0, bytes_read=0, seconds=0.0, stages=set()))
        r['read_count'] += 1
        r['bytes_read'] += int(nb)
        r['last_bytes'] = int(nb)
        r['seconds'] += dt
        if stage:
            r['stages'].add(stage)
        return val

    def unread(self):
        return sorted(n for n in self._decl if n not in self._reads)

    def emit(self):
        """The list for the result file: one dict per declared item (read or not) plus a summary."""
        items = []
        for n, d in self._decl.items():
            r = self._reads.get(n)
            e = dict(d)
            e['read_count'] = r['read_count'] if r else 0
            e['bytes_read'] = r['bytes_read'] if r else 0
            e['size_bytes'] = d['declared_size_bytes'] if d['declared_size_bytes'] is not None else (r['last_bytes'] if r else None)
            e['stages'] = sorted(r['stages']) if r else []
            items.append(e)
        used = [e for e in items if e['read_count'] and e['role'] == 'input']
        summ = dict(n_declared=len(items), n_inputs_declared=sum(e['role'] == 'input' for e in items), n_inputs_read=len(used),
                    n_comparison_only_read=sum(1 for e in items if e['read_count'] and e['role'] == 'comparison_only'),
                    bytes_inputs_read_total=int(sum(e['bytes_read'] for e in used)),
                    undeclared_read_attempts=list(self.undeclared_attempts))
        summ['sentence'] = (f"{summ['n_inputs_read']} recorded inputs read of {summ['n_inputs_declared']} declared, "
                            f"{summ['bytes_inputs_read_total']} bytes read in total; every one is listed with source file and size")
        return dict(summary=summ, items=items)

    def write_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.emit(), f, indent=1, default=str)

    def guard_real(self, real_cls, stage=None):
        """Subclass of an atm_step.Real-like class whose _get() goes through this registry (names = Real's cache keys)."""
        reg = self

        class GuardedReal(real_cls):
            def _get(self, key, fn):
                if key in self._c:
                    return self._c[key]
                self._c[key] = reg.read(key, fn, stage=stage)
                return self._c[key]
        GuardedReal.__name__ = 'Guarded' + real_cls.__name__
        return GuardedReal

    def guard_function(self, name, fn, stage=None, bytes_of=None):
        """Wrap fn so each call is a read of `name` (e.g. atm_step.surface_records)."""
        reg = self

        @functools.wraps(fn)
        def w(*a, **k):
            return reg.read(name, lambda: fn(*a, **k), stage=stage)
        return w


def declare_d174_inputs(reg, date, itime=None, ff=FF_DEFAULT):
    """Declarations for the record keys of atm_step.Real and the SURFACE record files, from the D174 inventory and
    jax_atm_step.RecordBoundary.recorded_inputs() (D180).  Source files are patterns (<it> = step number).  The list of D174 that is
    NOT reached by the atmosphere chained step (Ent exports, ocean/ice/lake restart items, MMST, ADVSI geometry, ...) is declared by
    declare_d174_surface_items() for the full coupled step."""
    d = f'{ff}/{date}'
    reg.declare('sitea', f'{d}/ffa_step_<it>_a.bin', description='hidden/PBL start state, PDSIG, PEK (D180: step 0 only used for start)',
                origin='D174 "First-step atmosphere state"; D180 sitea')
    reg.declare('siter', f'{d}/ffa_step_<it>_r.bin', description='SRHR/TRHR/COSZ1 and the radiation-step state (radiation output recorded)',
                origin='D180 siter; D174 "Radiation itself"')
    reg.declare('s1', f'{d}/ffd_state_<it>_s1.bin', description='start state of the step (step 0 only)', origin='D180 s1')
    reg.declare('ci', f'{d}/ffc_cse_in_<it>.bin', description='non-dynamic CONDSE entry fields (cloud/precip carry, ground fields, tuning)',
                origin='D174 "CONDSE entry non-state fields"; D180 ci')
    reg.declare('co', f'{d}/ffc_cse_out_<it>.bin', description='CONDSE exit; read for RADIA cloud masking on radiation steps / stage references',
                origin='D180 co')
    reg.declare('sited', f'{d}/ffa_step_<it>_d.bin', role='comparison_only', description='real dissip-stage state (reference only)')
    reg.declare('sitee', f'{d}/ffa_step_<it>_e.bin', role='comparison_only', description='real end-of-step state (reference only)')
    reg.declare('s3', f'{d}/ffd_state_<it>_s3.bin', role='comparison_only', description='real state after dynamics (stage reference)')
    reg.declare('s4', f'{d}/ffd_state_<it>_s4.bin', role='comparison_only', description='real dynamics exports (stage reference; dpdx.. used by isolated stages)')
    reg.declare('ps', f'{d}/post_surface_<it>', role='comparison_only', description='real post-surface state (stage reference)')
    reg.declare('fi', f'{d}/ffd_filt_<it>_in.bin', role='comparison_only', description='real filter input (stage reference)')
    reg.declare('fo', f'{d}/ffd_filt_<it>_out.bin', role='comparison_only', description='real filter output (stage reference)')
    reg.declare('surface_records', f'{d}/ffp_<it>.bin, ffs_<it>.bin, ffl_<it>.bin, ffg_<it>.bin, fft_<it>.bin',
                description='SURFACE tile (ffp/ffs), land-ice (ffl), land forcing + Ent exports (ffg), recorded GHY outputs (fft) templates',
                origin='D174 table rows 1-16; D180 section 3')
    reg.declare('ctx_static', f'{d}/<restart>, geometry and constants files of make_ctx(date)',
                description='restart-derived static data, dynamics geometry, CONDSE/filter constants for the date',
                origin='D180 "CONDSE module constants and geometry (make_cfg)"')
    return reg


def declare_d174_surface_items(reg, date, ff=FF_DEFAULT):
    """The remaining D174 inventory (inputs of the full coupled step beyond the atmosphere chain); declared, not read by the S0 reference."""
    d = f'{ff}/{date}'
    items = [
        ('ent_exports', 'ffg cols 3-6 and per-iteration Ent exports (ent=record)', 'D171/D174'),
        ('trup_in_rad', 'ffs col 80 / ffl col 14 (TRUP_in_rad of land)', 'D174'),
        ('mmst', 'init_STRAITS MMST start state', 'D174'),
        ('advsi_geometry', '5 static ADVSI geometry vectors', 'D166/D174'),
        ('irrigation_demand', 'ffg cols 147,148 (reconstructed from the recorded actual flux)', 'D174'),
        ('radiation_packet_surface_side', '20-field surface side of the radiation packet (until drv_radpacket is accepted)', 'D174/D176'),
        ('mdrya_dh2o_ca', 'MDRYA / dH2O / Ca constants per date', 'D174/D178'),
        ('seeds0', 'SEEDS[0] of the first step (later seeds computed by drv_rng)', 'D174 section 2'),
        ('ag2og_ig2og', 'AG2OG/IG2OG items of the replay ocean chain', 'D174'),
        ('straits_start', 'straits start state', 'D174'),
    ]
    for n, desc, org in items:
        reg.declare(n, f'{d}/(see D174 inventory)', description=desc, origin=org)
    return reg


# ======================================================================================= (4) radiation / libimf callback logs
class RadiationCallbackLog:
    """Log of the radiation callback (ACCEPTANCE 1.4, JAX_COVERAGE_MATRIX 5.3).  mode 'fortran' = served by the real RADIA through the
    persistent radiation server; 'replay' = recorded radiation (then the sentence says so and does NOT say computed)."""

    def __init__(self, mode='fortran'):
        if mode not in ('fortran', 'replay'):
            raise ValueError(mode)
        self.mode = mode
        self.calls = []

    @property
    def sentence(self):
        return SENTENCE_RADIATION if self.mode == 'fortran' else SENTENCE_RADIATION_REPLAY

    def record_call(self, itime, inputs=None, outputs=None, wall_seconds=0.0, server_seconds=None, sync_points=1, seed=None, bytes_in=None,
                    bytes_out=None):
        """inputs/outputs: dicts name -> array (field lists, shapes, dtypes and bytes are derived) ."""
        inputs, outputs = inputs or {}, outputs or {}
        self.calls.append(dict(itime=itime, seed=seed,
                               input_fields={k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in inputs.items()},
                               output_fields={k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in outputs.items()},
                               bytes_in=int(bytes_in if bytes_in is not None else _nbytes(inputs)),
                               bytes_out=int(bytes_out if bytes_out is not None else _nbytes(outputs)),
                               wall_seconds=float(wall_seconds), server_seconds=None if server_seconds is None else float(server_seconds),
                               sync_points=int(sync_points)))

    def wrap(self, fn, itime_of=None):
        """Wrap a function (packet dict -> outputs dict) so each call is logged (timing, bytes, one sync point)."""
        log = self

        @functools.wraps(fn)
        def w(packet, *a, **k):
            t0 = time.perf_counter()
            out = fn(packet, *a, **k)
            log.record_call(itime_of(packet) if itime_of else None, packet if isinstance(packet, dict) else {}, out if isinstance(out, dict) else {},
                            wall_seconds=time.perf_counter() - t0)
            return out
        return w

    def summary(self):
        c = self.calls
        return dict(mode=self.mode, sentence=self.sentence, n_calls=len(c), bytes_in=sum(x['bytes_in'] for x in c),
                    bytes_out=sum(x['bytes_out'] for x in c), wall_seconds=sum(x['wall_seconds'] for x in c),
                    server_seconds=sum(x['server_seconds'] or 0.0 for x in c), sync_points=sum(x['sync_points'] for x in c),
                    SOCRATES_modified=False, itimes=[x['itime'] for x in c])

    def text(self):
        s = self.summary()
        return (f"{s['sentence']}: {s['n_calls']} call(s), {s['bytes_in']} bytes sent, {s['bytes_out']} bytes received, "
                f"{s['wall_seconds']:.3f} s wall ({s['server_seconds']:.3f} s server), {s['sync_points']} host synchronisation point(s)")


class LibimfCallbackLog:
    """Log of the libimf host callback (ACCEPTANCE section 8.1): call count, bytes, seconds.  active=False -> libm-mode sentence."""

    def __init__(self, active=False):
        self.active = active
        self.calls = 0
        self.elements = 0
        self.seconds = 0.0
        self.bytes = 0

    def add(self, calls=1, elements=0, seconds=0.0, nbytes=0):
        self.calls += calls
        self.elements += elements
        self.seconds += seconds
        self.bytes += nbytes

    def wrap(self, fn):
        log = self

        @functools.wraps(fn)
        def w(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            log.add(1, int(np.size(r)), time.perf_counter() - t0, _nbytes((a, k)) + _nbytes(r))
            return r
        return w

    @property
    def sentence(self):
        return SENTENCE_LIBIMF if self.active else SENTENCE_LIBM

    def text(self):
        if not self.active:
            return self.sentence
        return f"{self.sentence}: {self.calls} call(s), {self.elements} element(s), {self.bytes} bytes, {self.seconds:.3f} s"


def result_preamble(rad_log, libimf_log, header=None):
    """The lines every result file starts with (and the sentences every table/plot of it carries)."""
    lines = [rad_log.sentence, libimf_log.sentence, '']
    if header:
        lines += [header_text(header), '']
    lines += [rad_log.text(), libimf_log.text()]
    return '\n'.join(lines)


# ======================================================================================= (5) stage registry
class StageRegistry:
    """Stages of the step with implementation kind (FORT/REC/NP/EJ/JJ, or a '/'-joined mix such as 'EJ/NP') and measured exclusive time.
    A timed stage must be registered first (nothing is omitted); non_jax_list() lists every stage that is not purely JJ."""

    def __init__(self):
        self._st = {}
        self._order = []
        self._tl = threading.local()

    def register(self, name, kind, description=''):
        parts = kind.split('/')
        bad = [p for p in parts if p not in KINDS]
        if bad:
            raise ValueError(f'unknown implementation kind {bad} (allowed {KINDS})')
        self._st[name] = dict(name=name, kind=kind, description=description, seconds=0.0, calls=0)
        self._order.append(name)
        return self

    def _stack(self):
        if not hasattr(self._tl, 's'):
            self._tl.s = []
        return self._tl.s

    @contextlib.contextmanager
    def time(self, name):
        if name not in self._st:
            raise HarnessError(f'stage {name!r} is not registered (nothing may be omitted from the non-JAX list)')
        st = self._stack()
        st.append([time.perf_counter(), 0.0])
        try:
            yield
        finally:
            t0, child = st.pop()
            dt = time.perf_counter() - t0
            self._st[name]['seconds'] += dt - child
            self._st[name]['calls'] += 1
            if st:
                st[-1][1] += dt

    def timed(self, name):
        def deco(fn):
            @functools.wraps(fn)
            def w(*a, **k):
                with self.time(name):
                    return fn(*a, **k)
            return w
        return deco

    def add_seconds(self, name, seconds, calls=1):
        """Add time measured elsewhere (e.g. the stage timings of an existing driver)."""
        if name not in self._st:
            raise HarnessError(f'stage {name!r} is not registered')
        self._st[name]['seconds'] += float(seconds)
        self._st[name]['calls'] += calls

    def table(self):
        tot = sum(s['seconds'] for s in self._st.values())
        return [dict(name=n, kind=self._st[n]['kind'], seconds=self._st[n]['seconds'], calls=self._st[n]['calls'],
                     share=(self._st[n]['seconds'] / tot if tot > 0 and self._st[n]['calls'] > 0 else None), description=self._st[n]['description'])
                for n in self._order]

    def non_jax_list(self):
        return [r for r in self.table() if any(p != 'JJ' for p in r['kind'].split('/'))]

    def non_jax_text(self):
        out = ['Non-JAX stages (every stage whose implementation kind is not purely JJ; FORT=original Fortran, REC=recorded input, '
               'NP=NumPy/Python, EJ=eager JAX, JJ=jitted JAX):']
        for r in self.non_jax_list():
            sh = 'not measured' if r['share'] is None else f"{100 * r['share']:.1f}% of measured step time ({r['seconds']:.3f} s, {r['calls']} calls)"
            out.append(f"  {r['name']} [{r['kind']}] {sh}. {r['description']}")
        return '\n'.join(out)


# ======================================================================================= (6) comparison with the reference
CATEGORY_TEXT = {'A': 'bitwise equal', 'B': 'max |diff| <= 1e-12 of scale', 'C': 'max |diff| <= 1e-6 of scale', 'D': 'worse than 1e-6 of scale'}
GATE_FIELDS = ['U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'TMOM', 'QMOM', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP']


def _ij_axes(shape):
    ai = [k for k, n in enumerate(shape) if n == IM]
    aj = [k for k, n in enumerate(shape) if n == JM]
    if not ai or not aj:
        return None
    return ai[0], aj[0]


def field_category(cand, ref, bound=1e-12):
    """Per-field difference statistics and category (ACCEPTANCE section 3): scale = max |ref|; rel = max|diff| / scale;
    A bitwise (NaN positions equal), B rel <= bound (1e-12), C rel <= 1e-6, D worse.  Also the horizontal columns (i, j; 1-based, as in the
    ledger) where |diff| > bound * scale."""
    a, b = np.asarray(cand, float), np.asarray(ref, float)
    if a.shape != b.shape:
        return dict(cat='D', shape_mismatch=[list(a.shape), list(b.shape)], rel=np.inf, max_abs=np.inf, scale=None, n_diff=None,
                    n_cols_over=None, cols_over=[], max_rel_over=np.inf)
    nn = np.isnan(a) & np.isnan(b)
    d = np.where(nn, 0.0, np.abs(a - b))
    scale = float(np.max(np.abs(np.where(np.isnan(b), 0.0, b)))) if b.size else 0.0
    mx = float(np.nanmax(d)) if d.size else 0.0
    neq = int(((a != b) & ~nn).sum())
    rel = mx / scale if scale > 0 else (0.0 if mx == 0 else np.inf)
    cat = 'A' if (mx == 0 and neq == 0) else ('B' if rel <= bound else ('C' if rel <= 1e-6 else 'D'))
    cols, ncols, worst = [], 0, None
    if mx > 0:
        worst = tuple(int(x) + 1 for x in np.unravel_index(np.nanargmax(d), d.shape))
        over = d > bound * scale
        ax = _ij_axes(d.shape)
        if ax is not None and over.any():
            other = tuple(k for k in range(d.ndim) if k not in ax)
            m = over.any(axis=other) if other else over
            if ax[0] > ax[1]:
                m = m.T
            ij = np.argwhere(m)
            ncols = len(ij)
            cols = [(int(i) + 1, int(j) + 1) for i, j in ij[:50]]
        elif over.any():
            ncols = int(over.any())
    return dict(cat=cat, max_abs=mx, scale=scale, rel=rel, n_diff=neq, n=int(a.size), worst=worst, n_cols_over=ncols, cols_over=cols,
                max_rel_over=rel)


def compare_end_state(cand, ref, fields=None, bound=1e-12, max_exception_columns=10, exception_rel=1e-9):
    """Compare a candidate end state (dict name -> array) with a reference per field (default: the gate fields of atm_step_compare, plus
    every other common array reported without affecting the verdict).  Verdict by ACCEPTANCE section 3 (categories fixed here, not
    tunable per run):
      MET                                   every gate field A or B
      MET with named exception columns      fields beyond B confined to <= 10 horizontal columns each and <= 1e-9 of scale (listed)
      PARTLY MET                            every gate field A, B or C (the first failing stage must be named by the caller)
      NOT MET                               otherwise
    A gate field missing from the candidate counts as D (never silently skipped)."""
    gate = list(fields) if fields is not None else [f for f in GATE_FIELDS]
    per, missing = {}, []
    for f in gate:
        if f not in cand or f not in ref:
            missing.append(f)
            per[f] = dict(cat='D', rel=np.inf, max_abs=np.inf, missing=True, n_cols_over=None, cols_over=[])
        else:
            per[f] = field_category(cand[f], ref[f], bound)
    others = {}
    if fields is None:
        for f in sorted(set(cand) & set(ref)):
            if f not in per:
                try:
                    others[f] = field_category(cand[f], ref[f], bound)
                except Exception as e:      # non-numeric entries
                    others[f] = dict(cat='?', error=repr(e))
    cats = {}
    for v in per.values():
        cats[v['cat']] = cats.get(v['cat'], 0) + 1
    beyond = {f: v for f, v in per.items() if v['cat'] not in 'AB'}
    if not beyond:
        verdict = 'MET'
    elif all(v.get('n_cols_over') is not None and v['n_cols_over'] <= max_exception_columns and v['rel'] <= exception_rel
             for v in beyond.values()):
        verdict = 'MET with named exception columns'
    elif all(v['cat'] in 'ABC' for v in per.values()):
        verdict = 'PARTLY MET (all gate fields <= 1e-6 of scale; first failing stage to be named by the caller)'
    else:
        verdict = 'NOT MET (gate fields beyond 1e-6 of scale: ' + ','.join(f for f, v in per.items() if v['cat'] == 'D') + ')'
    return dict(verdict=verdict, categories=cats, gate_fields=per, other_fields=others, missing_gate_fields=missing,
                exception_columns={f: dict(cols=v['cols_over'][:max_exception_columns], n_cols=v['n_cols_over'], rel=v['rel'])
                                   for f, v in beyond.items()},
                rule=dict(bound_B=bound, bound_C=1e-6, max_exception_columns=max_exception_columns, exception_rel=exception_rel,
                          definitions=CATEGORY_TEXT))


def comparison_text(res):
    lines = [f"verdict: {res['verdict']}", f"categories over gate fields: {res['categories']}"]
    for f, v in res['gate_fields'].items():
        if v.get('missing'):
            lines.append(f'  {f:8s} MISSING in candidate or reference -> D')
        else:
            lines.append(f"  {f:8s} {v['cat']}  max|diff| {v['max_abs']:.3e}  rel {v['rel']:.3e}  n_diff {v['n_diff']}  columns>1e-12 {v['n_cols_over']}")
    for f, e in res['exception_columns'].items():
        lines.append(f"  beyond B: {f}: {e['n_cols']} column(s), rel {e['rel']:.2e}, first columns (i,j 1-based) {e['cols'][:10]}")
    return '\n'.join(lines)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ref_path(date, tag, outdir=None):
    return os.path.join(outdir or REF_DIR, f'{date}_step0_run{tag}.npz')


def compare_to_reference(cand, date, tag='1', outdir=None, **kw):
    """cand: dict of arrays or a path to an npz.  Reference: ff_data/ref_libm/<date>_step0_run<tag>.npz (C1 reference, libm mode)."""
    if isinstance(cand, (str, os.PathLike)):
        cand = load_npz(cand)
    return compare_end_state(cand, load_npz(ref_path(date, tag, outdir)), **kw)


# ----------------------------------------------------------------------------------- the reference run
def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def end_state_arrays(S):
    """Numeric arrays of a chained-step state dict (keys not starting with '_'); returns (arrays, skipped_names)."""
    out, skipped = {}, []
    for k, v in S.items():
        if k.startswith('_'):
            continue
        if isinstance(v, (np.ndarray, np.generic, int, float, bool)):
            a = np.asarray(v)
            if a.dtype != object:
                out[k] = a
                continue
        skipped.append(k)
    return out, skipped


def run_reference(date, tag, outdir=None, steps=1):
    """C1 reference: NumPy chained step in libm mode (atm_step_fast.run_chain with ctx imf=False, land_mode='recorded'), `steps` step(s)
    from the real restart state of `date` (step 0 by default).  The end state of the LAST step is saved to
    <outdir>/<date>_step<steps-1>_run<tag>.npz (default ff_data/ref_libm; outside git) with a header json and a digest json.
    Every recorded record read passes through a RecordedInputRegistry (fails on an undeclared read).  The process must already be pinned
    (taskset) and have OMP_NUM_THREADS=1; clouds_jax_env must have been imported before jax (done by the CLI)."""
    import clouds_jax_env  # noqa: F401  (XLA flags before jax)
    outdir = outdir or REF_DIR
    os.makedirs(outdir, exist_ok=True)
    hdr = provenance_header(extra=dict(task='D184 C1 reference, libm mode', date=date, tag=str(tag), steps=steps,
                                       ctx_imf=False, land_mode='recorded', chain='atm_step_fast.run_chain'), require_xla_flags=True)
    import atm_step as A
    import atm_step_fast as F
    it0 = dict(DATES)[date]
    reg = RecordedInputRegistry()
    declare_d174_inputs(reg, date)
    real_orig, surf_orig, ctx_orig = A.Real, A.surface_records, A.make_ctx
    A.Real = reg.guard_real(real_orig, stage='ref_libm')
    A.surface_records = reg.guard_function('surface_records', surf_orig, stage='ref_libm',
                                           bytes_of=None)
    A.make_ctx = reg.guard_function('ctx_static', ctx_orig, stage='ref_libm')
    sr = StageRegistry()
    holder = {}
    try:
        ctx = A.make_ctx(date, imf=False)

        def cb(k, it, R, sn, S, tm):
            if k == steps - 1:
                holder['arrays'], holder['skipped'] = end_state_arrays(S)
                holder['timing'] = {a: b for a, b in tm.items() if a.startswith('stage_')}
        t0 = time.perf_counter()
        tl = F.run_chain(date, it0, steps, ctx, land_mode='recorded', on_step=cb)
        wall = time.perf_counter() - t0
    finally:
        A.Real, A.surface_records, A.make_ctx = real_orig, surf_orig, ctx_orig
    arrays = holder['arrays']
    base = f'{date}_step{steps - 1}_run{tag}'
    np.savez(os.path.join(outdir, base + '.npz'), **arrays)
    dig = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=_digest(v)) for k, v in arrays.items()}
    json.dump(dig, open(os.path.join(outdir, base + '.digest.json'), 'w'), indent=0)
    for k, v in holder['timing'].items():
        sr.register(k, 'NP', 'timing of the NumPy chained step as measured by atm_step.run_step')
        sr.add_seconds(k, v)
    meta = dict(header=hdr, date=date, itime=it0 + steps - 1, tag=str(tag), wall_seconds=wall, stage_timing=holder['timing'],
                n_arrays=len(arrays), skipped_non_array=holder['skipped'], recorded_inputs=reg.emit(),
                statement='C1 reference (ACCEPTANCE section 2): NumPy chained atmosphere step, libm mode (imf=False), land recorded; '
                          'not libimf, not real-Fortran. ' + SENTENCE_LIBM)
    json.dump(meta, open(os.path.join(outdir, base + '.meta.json'), 'w'), indent=1, default=str)
    return os.path.join(outdir, base + '.npz'), meta


def determinism_check(date, outdir=None, steps=1, tags=('1', '2')):
    """Compare run tags[0] with tags[1] of the reference bitwise (same field set, same shapes/dtypes, array_equal with NaN positions
    equal, and byte-identical content).  Writes <outdir>/determinism_<date>.json and returns the result."""
    outdir = outdir or REF_DIR
    pa = os.path.join(outdir, f'{date}_step{steps - 1}_run{tags[0]}.npz')
    pb = os.path.join(outdir, f'{date}_step{steps - 1}_run{tags[1]}.npz')
    a, b = load_npz(pa), load_npz(pb)
    res = dict(date=date, files=[pa, pb], n_fields=[len(a), len(b)], same_field_set=sorted(a) == sorted(b), unequal=[])
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        if not same:
            st = field_category(x, y)
            res['unequal'].append(dict(field=k, cat=st['cat'], max_abs=st['max_abs'], n_diff=st['n_diff']))
    res['bitwise_equal'] = bool(res['same_field_set'] and not res['unequal'])
    json.dump(res, open(os.path.join(outdir, f'determinism_{date}.json'), 'w'), indent=1)
    return res


# ======================================================================================= command line
def _main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == 'header':
        import clouds_jax_env  # noqa: F401
        print(header_text(provenance_header()))
        return 0
    if cmd == 'ref':
        date, tag = argv[1], argv[2]
        outdir = argv[3] if len(argv) > 3 else None
        path, meta = run_reference(date, tag, outdir)
        print(path, f"wall {meta['wall_seconds']:.1f}s", f"{meta['n_arrays']} arrays", meta['recorded_inputs']['summary']['sentence'])
        return 0
    if cmd == 'determinism':
        r = determinism_check(argv[1], argv[2] if len(argv) > 2 else None)
        print(json.dumps({k: r[k] for k in ('date', 'same_field_set', 'bitwise_equal', 'n_fields')}), 'unequal:', r['unequal'])
        return 0 if r['bitwise_equal'] else 1
    if cmd == 'compare':
        res = compare_to_reference(argv[1], argv[2], tag=argv[3] if len(argv) > 3 else '1')
        print(comparison_text(res))
        return 0
    print('unknown command', cmd)
    return 2


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
