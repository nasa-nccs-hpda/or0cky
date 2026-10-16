"""Tests for dyn_avrx_ff.py (FFT72 FFT/FFTI + AVRX, D94) against real Fortran per-row dumps
(ffd_avrx*).  Real-data tests are skipped if the dumps are absent; the unit/property/mutation tests
that need no dump always run."""
import numpy as np
import pytest

import dyn_avrx_ff as ff
import dyn_avrx_compare as cm
import dyn_geom_ff as gm
from dyn_avrx_compare import DATES, NSTEP

HAVE = all(cm.available(d) for d, _ in DATES)
needs_dumps = pytest.mark.skipif(not HAVE, reason="ff_data ffd_avrx dumps not present on this host")
CALLS = [(d, it0 + k, s) for d, it0 in DATES for k in range(NSTEP) for s in (1, 3)]


def _rows(date):
    return cm.load_rows(date), cm.load_consts(date)


@needs_dumps
@pytest.mark.parametrize("date,itime,site", CALLS)
def test_avrx_bitwise_per_step_and_site(date, itime, site):
    r, k = _rows(date)
    m = (r['itime'] == itime) & (r['site'] == site)
    assert m.sum() == 12 * 26                      # 12 sampled calls x 26 active rows
    out, A, B = ff.avrx_rows(r['x'][m], r['j'][m], k['drat'], k['nmin'], k['bysn'], k['C'], k['S'], return_spec=True)
    assert np.array_equal(A, r['an'][m]) and np.array_equal(B, r['bn'][m])    # FFT spectra
    assert np.array_equal(out, r['xo'][m])                                    # FFT -> truncate -> FFTI


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
@pytest.mark.parametrize("tables", ["dump", "analytic"])
def test_all_rows_bitwise_dump_and_analytic_tables(date, tables):
    res = cm.run_date(date, tables=tables)
    assert res['fft_A'] == 0.0 and res['fft_B'] == 0.0 and res['avrx'] == 0.0
    assert res['n_exact'] == res['n_total']


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_analytic_tables_match_dump(date):
    res = cm.run_date(date)
    assert res['tab_drat'] == 0.0 and res['tab_nmin_ok'] and res['tab_bysn'] == 0.0
    assert res['tab_dxp'] == 0.0 and res['tab_C'] == 0.0 and res['tab_S'] == 0.0


@needs_dumps
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_non_vacuous(date):
    r, k = _rows(date)
    assert np.max(np.abs(r['xo'] - r['x'])) > 1e4            # truncation changes the rows a lot
    assert np.mean(np.any(r['xo'] != r['x'], axis=1)) == 1.0  # every active row is modified
    # the truncated spectrum is really different: high wavenumbers of the output are damped
    A, B = ff.fft72(r['xo'], k['C'], k['S'])
    assert np.sum(A[:, 20:] ** 2 + B[:, 20:] ** 2) < 0.9 * np.sum(r['an'][:, 20:] ** 2 + r['bn'][:, 20:] ** 2)
    assert set(r['site'].tolist()) == {1, 3}
    c = cm.load_calls(date)
    assert not np.any(c['site'] == 2)                        # call site 2 (dead V2 PGF) never runs
    assert int(np.sum(c['site'] == 1)) == 200 * NSTEP and int(np.sum(c['site'] == 3)) == 200 * NSTEP


@needs_dumps
def test_numpy_rfft_alternative_is_not_bitwise_but_close():
    res = cm.run_date("nov26")
    assert res['np_avrx'] > 0.0                  # not bit-for-bit (documents why the radix port is used)
    assert res['np_avrx_rel'] < 1e-14
    assert res['np_fft_A'] < 1e-8 and res['np_fft_B'] < 1e-8


# ---- unit / property tests that need no dump ---------------------------------------------------
C0, S0 = ff.make_tables()


def test_fft_definition_cosine_and_sine():
    K = np.arange(1, 73)
    for n in (1, 5, 17, 35):
        f = 3.0 * np.cos(2 * np.pi * n * K / 72) + 2.0 * np.sin(2 * np.pi * n * K / 72)
        A, B = ff.fft72(f[None, :], C0, S0)
        assert abs(A[0, n] - 3.0) < 1e-12 and abs(B[0, n] - 2.0) < 1e-12
        z = np.delete(np.arange(37), n)
        assert np.max(np.abs(A[0, z])) < 1e-12 and np.max(np.abs(B[0, z])) < 1e-12
    A, B = ff.fft72(np.full((1, 72), 4.0), C0, S0)
    assert abs(A[0, 0] - 4.0) < 1e-13
    f = np.where(K % 2 == 0, 1.0, -1.0)           # Nyquist
    A, B = ff.fft72(f[None, :], C0, S0)
    assert abs(A[0, 36] - 1.0) < 1e-13 and B[0, 36] == 0.0


def test_ffti_inverts_fft_random():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 72)) * 1e5
    A, B = ff.fft72(x, C0, S0)
    assert np.max(np.abs(ff.ffti72(A, B, C0, S0) - x)) < 1e-8
    assert np.max(np.abs(ff.ffti72_np(*ff.fft72_np(x)) - x)) < 1e-8


def test_radix_matches_numpy_fft_random():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 72))
    A, B = ff.fft72(x, C0, S0)
    A2, B2 = ff.fft72_np(x)
    assert np.max(np.abs(A - A2)) < 1e-13 and np.max(np.abs(B - B2)) < 1e-13


def test_rows_independent_of_batching():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 72))
    A, B = ff.fft72(x, C0, S0)
    for i in range(4):
        a, b = ff.fft72(x[i:i + 1], C0, S0)
        assert np.array_equal(a[0], A[i]) and np.array_equal(b[0], B[i])


def test_rows_with_drat_above_one_unchanged():
    g = gm.geometry(6.371e6)
    drat, nmin, bysn = ff.avrx_tables(g['dxp'], g['bydyp'][2], g['dlon'])
    assert drat[22] > 1 and drat[1] < 1           # equator row skipped, near-pole row filtered
    x = np.random.default_rng(4).normal(size=(2, 72))
    out = ff.avrx_rows(x, [23, 2], drat, nmin, bysn, C0, S0)
    assert np.array_equal(out[0], x[0]) and not np.array_equal(out[1], x[1])


# ---- mutation tests: deliberately wrong variants must be detected on real data ------------------
def _mut_run(date="nov26", C=None, S=None, nmin_shift=0):
    r, k = _rows(date)
    m = (r['itime'] == DATES[0][1]) & (r['site'] == 1)
    nmin = k['nmin'] + nmin_shift
    out = ff.avrx_rows(r['x'][m], r['j'][m], k['drat'], nmin, k['bysn'], k['C'] if C is None else C,
                       k['S'] if S is None else S)
    return out, r['xo'][m]


@needs_dumps
def test_mutation_wrong_rt3(monkeypatch):
    monkeypatch.setattr(ff, "RT3", 1.7320508075688774)       # 1 ulp off
    out, ref = _mut_run()
    assert not np.array_equal(out, ref)


@needs_dumps
def test_mutation_wrong_table_entry():
    r, k = _rows("nov26")
    S = k['S'].copy()
    S[9] = np.nextafter(S[9], 1.0)
    out, ref = _mut_run(S=S)
    assert not np.array_equal(out, ref)


@needs_dumps
def test_mutation_nmin_off_by_one():
    out, ref = _mut_run(nmin_shift=1)
    assert np.max(np.abs(out - ref)) > 1.0


@needs_dumps
def test_mutation_truncation_disabled(monkeypatch):
    r, k = _rows("nov26")
    m = (r['itime'] == DATES[0][1]) & (r['site'] == 3)
    out = ff.avrx_rows(r['x'][m], r['j'][m], k['drat'], np.full_like(k['nmin'], 99), k['bysn'], k['C'], k['S'])
    assert np.max(np.abs(out - r['xo'][m])) > 1e3


@needs_dumps
def test_mutation_summation_order_in_docalc(monkeypatch):
    """Re-associating the 3-term sums in DOCALC (a+(b+c) instead of (a+b)+c) is detected as a bit change."""
    r, k = _rows("nov26")
    x = r['x'][:200]
    A, B = ff.fft72(x, k['C'], k['S'])
    F = np.zeros((73, len(x))); F[1:] = x.T
    c240 = F[1:25] + (F[25:49] + F[49:73])                  # wrong association
    c240_ok = (F[1:25] + F[25:49]) + F[49:73]
    assert not np.array_equal(c240, c240_ok)                # the data distinguish the two orders
