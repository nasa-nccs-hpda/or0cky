"""D184 tests of jax_harness (S0).  Quick unit tests; the reference-vs-reference test reads saved files under ff_data/ref_libm and
skips when they are absent.  Run: taskset -c 0-1 env OMP_NUM_THREADS=1 python -m pytest tests/test_jax_harness.py -q  (from fullfidelity/)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import jax_harness as H  # noqa: E402


# ----------------------------------------------------------------------------- header
def test_header_complete(monkeypatch):
    for v in H.CACHE_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    h = H.provenance_header()
    for k in H.HEADER_REQUIRED_KEYS:
        assert k in h, k
    assert h['core_affinity']['cores'] and h['core_affinity']['n_cores'] >= 1
    assert 'OMP_NUM_THREADS' in h['thread_env'] and 'XLA_FLAGS' in h['thread_env']
    assert h['jax']['version'] and h['jax']['backend'] and h['jax']['devices']
    assert h['compile_cache']['cache_set'] is False
    # a clean `git archive` export (used for the batch regression) has no .git: the header then has an empty head
    import subprocess
    in_repo = subprocess.run(['git', 'rev-parse', '--git-dir'], capture_output=True, cwd=os.path.dirname(os.path.abspath(__file__))).returncode == 0
    assert (len(h['git']['head']) == 40 if in_repo else h['git']['head'] in ('', None)) and isinstance(h['git']['modified_tracked_files'], int)
    assert isinstance(h['libimf']['available'], bool)
    assert h['versions']['python'] and h['versions']['numpy'] == np.__version__
    assert h['timestamp_utc']
    assert H.SENTENCE_RADIATION in json.dumps(h['sentences'])
    txt = H.header_text(h)
    assert 'git head' in txt and 'libimf' in txt


def test_header_fails_when_cache_set(monkeypatch):
    monkeypatch.setenv('JAX_COMPILATION_CACHE_DIR', '/nonexistent/cache')
    with pytest.raises(H.HarnessError):
        H.provenance_header()
    monkeypatch.delenv('JAX_COMPILATION_CACHE_DIR')
    monkeypatch.setenv('CLOUDS_JAX_CACHE', '1')
    with pytest.raises(H.HarnessError):
        H.check_no_compile_cache()


# ----------------------------------------------------------------------------- counters on a toy jit
def test_counters_toy_jit_and_results_unchanged():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    @jax.jit
    def f(x, y):
        return jnp.sin(x) * y + 1.0

    x = np.linspace(0, 1, 1000)
    y = np.linspace(1, 2, 1000)
    ref = np.asarray(f(x, y))
    C = H.Counters().listen_compiles()
    try:
        g = C.count_jit(f, 'toy')
        with C.stage('toy'):
            for _ in range(3):
                r = g(x, y)
        np.testing.assert_array_equal(np.asarray(r), ref)          # result unchanged, bitwise
        s = C.snapshot()['toy']
        assert s['jit_calls'] == 3 and s['entries'] == 1
        assert s['h2d_calls'] == 6 and s['h2d_bytes'] == 6 * x.nbytes
        # decorator form around a plain function and around a jitted one
        d = C.stage('deco')(lambda a: f(a, a))
        np.testing.assert_array_equal(np.asarray(d(x)), np.asarray(f(x, x)))
        assert C.snapshot()['deco']['entries'] == 1

        # explicit and patched transfers
        dp0, da0 = jax.device_put, jnp.asarray
        with C.instrument_transfers():
            with C.stage('xfer'):
                a = jax.device_put(x)                      # h2d 8000 B
                b = jnp.asarray(y)                         # h2d 8000 B
                host = jax.device_get(a)                   # d2h 8000 B (not double counted via __array__)
                host2 = b.__array__()                      # d2h 8000 B (np.asarray(b) is served by the buffer protocol on CPU: not counted)
        s = C.snapshot()['xfer']
        assert (s['h2d_calls'], s['h2d_bytes']) == (2, 2 * x.nbytes), s
        assert (s['d2h_calls'], s['d2h_bytes']) == (2, 2 * x.nbytes), s
        np.testing.assert_array_equal(host, x)
        np.testing.assert_array_equal(host2, y)
        assert jax.device_put is dp0 and jnp.asarray is da0           # patched entries restored

        # a fresh jit compiles once (compile listener), then not
        @jax.jit
        def h(z):
            return z * 3.0 + 1.0
        with C.stage('compile'):
            h(x).block_until_ready()
            h(x).block_until_ready()
        assert C.snapshot()['compile']['compiles'] >= 1
        n1 = C.snapshot()['compile']['compiles']
        with C.stage('compile'):
            h(x).block_until_ready()
        assert C.snapshot()['compile']['compiles'] == n1
    finally:
        C.stop_listening()
    assert 'toy' in C.report_text()


def test_host_callback_counted_and_nested_stages():
    C = H.Counters()

    @C.host_callback(name='cb')
    def cb(a):
        return a * 2.0

    with C.stage('outer'):
        with C.stage('inner'):
            out = cb(np.ones(10))
    s = C.snapshot()
    assert np.array_equal(out, 2 * np.ones(10))
    assert s['cb']['cb_calls'] == 1 and s['cb']['cb_bytes'] == 160 and s['cb']['cb_seconds'] >= 0
    assert s['outer']['wall_seconds'] >= s['inner']['wall_seconds'] >= 0
    assert s['outer']['self_seconds'] <= s['outer']['wall_seconds']


# ----------------------------------------------------------------------------- recorded-input registry
def test_registry_fails_on_undeclared_read():
    reg = H.RecordedInputRegistry()
    reg.declare('ci', '/x/ffc_cse_in_<it>.bin', size_bytes=100, description='d')
    a = reg.read('ci', lambda: np.zeros(5), stage='condse')
    assert a.shape == (5,)
    with pytest.raises(H.UndeclaredRecordedInput):
        reg.read('mystery', lambda: np.zeros(3))
    e = reg.emit()
    assert e['summary']['undeclared_read_attempts'] == ['mystery']
    assert e['summary']['n_inputs_read'] == 1 and e['items'][0]['bytes_read'] == 40 and e['items'][0]['stages'] == ['condse']
    assert 'recorded inputs read' in e['summary']['sentence']


def test_registry_guards_real_like_object_and_d174_list():
    class FakeReal:
        def __init__(self):
            self._c = {}

        def _get(self, key, fn):
            if key not in self._c:
                self._c[key] = fn()
            return self._c[key]

        def good(self):
            return self._get('ci', lambda: {'a': np.zeros(4)})

        def bad(self):
            return self._get('not_declared', lambda: np.zeros(2))

    reg = H.declare_d174_inputs(H.RecordedInputRegistry(), 'nov26')
    G = reg.guard_real(FakeReal)
    r = G()
    assert r.good()['a'].shape == (4,)
    with pytest.raises(H.UndeclaredRecordedInput):
        r.bad()
    # every record key of atm_step.Real is declared
    for k in ('sitea', 'siter', 'sited', 'sitee', 's1', 's3', 's4', 'ci', 'co', 'ps', 'fi', 'fo', 'surface_records', 'ctx_static'):
        assert reg.declared(k), k
    H.declare_d174_surface_items(reg, 'nov26')
    out = reg.emit()
    assert out['summary']['n_inputs_declared'] > 10
    assert all(it['source_file'] for it in out['items'])
    fn = reg.guard_function('surface_records', lambda R: {'pa': np.zeros(3)})
    assert fn(None)['pa'].shape == (3,)
    assert reg.emit()['summary']['n_inputs_read'] == 2


# ----------------------------------------------------------------------------- radiation / libimf
def test_radiation_log_sentences():
    L = H.RadiationCallbackLog('fortran')
    L.record_call(33312, {'t': np.zeros((4, 5))}, {'srhr': np.zeros((3,))}, wall_seconds=1.5, server_seconds=1.2, seed=7)
    L.record_call(33317, {'t': np.zeros((4, 5))}, {'srhr': np.zeros((3,))}, wall_seconds=0.5)
    s = L.summary()
    assert s['sentence'] == 'radiation computed by the original Fortran (hybrid component)'
    assert s['n_calls'] == 2 and s['bytes_in'] == 320 and s['bytes_out'] == 48 and s['sync_points'] == 2 and abs(s['wall_seconds'] - 2.0) < 1e-12
    assert H.SENTENCE_RADIATION in L.text()
    assert H.RadiationCallbackLog('replay').sentence != H.SENTENCE_RADIATION          # replay never says "computed"
    lib = H.LibimfCallbackLog(active=True)
    f = lib.wrap(lambda x: x ** 2)
    f(np.arange(4.0))
    assert lib.calls == 1 and lib.elements == 4
    assert lib.sentence == "libimf math functions provided by a host callback to the original build's runtime"
    assert H.LibimfCallbackLog(False).sentence == H.SENTENCE_LIBM
    pre = H.result_preamble(L, lib)
    assert pre.startswith(H.SENTENCE_RADIATION)


# ----------------------------------------------------------------------------- stage registry
def test_stage_registry_non_jax_list():
    import time
    R = H.StageRegistry()
    R.register('dyn', 'EJ/NP', 'x').register('condse', 'NP', 'y').register('radia', 'FORT', 'z').register('phase1', 'JJ', 'w')
    R.register('untimed', 'REC', 'never timed')
    with pytest.raises(ValueError):
        R.register('bad', 'GPU')
    with R.time('dyn'):
        time.sleep(0.02)
        with R.time('phase1'):
            time.sleep(0.02)
    with R.time('condse'):
        time.sleep(0.01)
    with pytest.raises(H.HarnessError):
        with R.time('unregistered'):
            pass
    names = [r['name'] for r in R.non_jax_list()]
    assert names == ['dyn', 'condse', 'radia', 'untimed']            # JJ-only stage excluded, unmeasured stages NOT omitted
    t = {r['name']: r for r in R.table()}
    assert abs(sum(r['share'] for r in t.values() if r['share'] is not None) - 1.0) < 1e-12
    assert t['dyn']['seconds'] < 0.03 + 0.01 and t['phase1']['seconds'] >= 0.02                      # exclusive time
    assert 'not measured' in R.non_jax_text() and 'FORT' in R.non_jax_text()


# ----------------------------------------------------------------------------- categories A/B/C/D and exception columns
def _rnd(shape=(40, 72, 46), seed=1):
    return np.random.default_rng(seed).uniform(1.0, 2.0, shape)


def test_categories_and_verdicts():
    ref = {f: _rnd(seed=i) for i, f in enumerate(H.GATE_FIELDS)}
    cand = {k: v.copy() for k, v in ref.items()}
    r = H.compare_end_state(cand, ref)
    assert r['verdict'] == 'MET' and r['categories'] == {'A': len(H.GATE_FIELDS)}
    # B: one element at 1e-13 of scale
    cand['T'][3, 5, 7] += 1e-13 * np.abs(ref['T']).max()
    r = H.compare_end_state(cand, ref)
    assert r['gate_fields']['T']['cat'] == 'B' and r['verdict'] == 'MET'
    # exceptions: 3 columns at 1e-10 of scale -> C? no: 1e-10 > 1e-12 so beyond B but category C; <= 10 columns and <= 1e-9 -> named exceptions
    for (i, j) in [(1, 2), (10, 11), (20, 21)]:
        cand['U'][4, i, j] += 1e-10 * np.abs(ref['U']).max()
    r = H.compare_end_state(cand, ref)
    assert r['gate_fields']['U']['cat'] == 'C' and r['gate_fields']['U']['n_cols_over'] == 3
    assert r['verdict'] == 'MET with named exception columns'
    assert sorted(r['exception_columns']['U']['cols']) == [(2, 3), (11, 12), (21, 22)]       # 1-based (i, j)
    # 11 columns -> no longer within the rule -> PARTLY MET (all <= 1e-6)
    for i in range(11):
        cand['V'][2, i, 0] += 1e-10 * np.abs(ref['V']).max()
    r = H.compare_end_state(cand, ref)
    assert r['gate_fields']['V']['n_cols_over'] == 11 and r['verdict'].startswith('PARTLY MET')
    # one column at 5e-9 (> 1e-9 exception bound) -> also partly
    cand2 = {k: v.copy() for k, v in ref.items()}
    cand2['Q'][0, 0, 0] += 5e-9 * np.abs(ref['Q']).max()
    assert H.compare_end_state(cand2, ref)['verdict'].startswith('PARTLY MET')
    # D: 1e-3 of scale -> NOT MET
    cand3 = {k: v.copy() for k, v in ref.items()}
    cand3['QCL'][0, 0, 0] += 1e-3 * np.abs(ref['QCL']).max()
    r = H.compare_end_state(cand3, ref)
    assert r['gate_fields']['QCL']['cat'] == 'D' and r['verdict'].startswith('NOT MET')
    # missing gate field counts as D, never skipped
    cand4 = {k: v for k, v in ref.items() if k != 'MA'}
    r = H.compare_end_state(cand4, ref)
    assert r['missing_gate_fields'] == ['MA'] and r['verdict'].startswith('NOT MET')
    assert 'verdict' in H.comparison_text(r)


def test_category_boundaries_exact():
    ref = np.ones((72, 46)) * 4.0               # scale 4
    for delta, cat in [(0.0, 'A'), (3.9e-12, 'B'), (4.1e-12, 'C'), (3.9e-6, 'C'), (4.1e-6, 'D')]:
        c = ref.copy()
        c[0, 0] += delta
        assert H.field_category(c, ref)['cat'] == cat, delta


def test_field_category_agrees_with_atm_step_field_stats():
    try:
        import atm_step as A
    except Exception as e:      # pragma: no cover
        pytest.skip(f'atm_step not importable: {e}')
    ref = _rnd((72, 46, 5), 3)
    for scale in (0.0, 1e-14, 1e-11, 1e-9, 1e-7, 1e-3):
        c = ref.copy()
        c[2, 3, 1] += scale * ref.max()
        c[9, 9, 2] += scale * ref.max() * 0.5
        a, b = A.field_stats(c, ref), H.field_category(c, ref)
        assert a['cat'] == b['cat'] and a['max_abs'] == b['max_abs'] and a['scale'] == b['scale']
        assert a['cols_over'] == b['n_cols_over'] and a['n_diff'] == b['n_diff']


# ----------------------------------------------------------------------------- reference vs reference (S0 gate), needs the saved runs
@pytest.mark.parametrize('date', [d for d, _ in H.DATES])
def test_reference_runs_bitwise_equal(date):
    p1, p2 = H.ref_path(date, '1'), H.ref_path(date, '2')
    if not (os.path.exists(p1) and os.path.exists(p2)):
        pytest.skip('reference runs not present under ff_data/ref_libm')
    r = H.determinism_check(date)
    assert r['same_field_set'] and r['unequal'] == [] and r['bitwise_equal']
    res = H.compare_to_reference(p2, date, tag='1')
    assert res['verdict'] == 'MET' and set(res['categories']) == {'A'}
    assert not res['missing_gate_fields']
    # non-vacuity: a perturbed candidate must be caught
    c = H.load_npz(p2)
    c['T'] = c['T'].copy()
    c['T'].flat[123] = np.nextafter(c['T'].flat[123], np.inf)
    res = H.compare_end_state(c, H.load_npz(p1))
    assert res['gate_fields']['T']['cat'] == 'B' and res['gate_fields']['T']['n_diff'] == 1
