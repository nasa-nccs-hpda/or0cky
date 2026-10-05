"""Tests for dyn_pgf_ff.py (ATMDYN.f live PGF, D97) against real-Fortran per-call dumps (ffd_pgf_*,
3 dates x 6 steps x 5 passes = 90 calls).  Bitwise needs the Intel libimf pow (intel_libm_ff.py); without it
the numpy-pow mode is checked to a tolerance.  Skipped if the dumps are absent."""
import copy

import numpy as np
import pytest

import dyn_aflux_ff as fa
import dyn_aflux_compare as cm
import dyn_pgf_ff as fp
import dyn_pgf_compare as pc
import intel_libm_ff
from dyn_aflux_ff import IM, JM, LM

HAVE = all(pc.available(d) for d, _ in cm.DATES) and all(cm.available(d) for d, _ in cm.DATES)
pytestmark = pytest.mark.skipif(not HAVE, reason="ff_data ffd_pgf/ffd_aflux dumps not present on this host")
NO_IMF = not intel_libm_ff.available()

CALLS = cm.calls()
_G = {}


def geom(date):
    if date not in _G:
        g = cm.load_geom_date(date)
        _G[date] = (g, fa.avrx_tables(g))
    return _G[date]


@pytest.mark.skipif(NO_IMF, reason="Intel libimf not available")
@pytest.mark.parametrize("date,itime,pas", CALLS)
def test_pgf_bitwise_with_libimf_pow(date, itime, pas):
    g, tab = geom(date)
    i, o, r = pc.run_pgf(date, itime, pas, g, tab, imf_pow=True)
    d = pc.pgf_compare(i, o, r)
    for k, v in d.items():
        assert v == 0.0, (k, v)


@pytest.mark.parametrize("date,itime,pas", CALLS[::3])
def test_pgf_numpy_pow_tolerance(date, itime, pas):
    """numpy pow differs from libimf by 1 ulp in rare cells; the effect stays below 2e-12 of the field scale."""
    g, tab = geom(date)
    i, o, r = pc.run_pgf(date, itime, pas, g, tab, imf_pow=False)
    for k in ('gz', 'adm', 'pgfu', 'dut', 'dvt', 'ut', 'vt'):
        sc = np.max(np.abs(o[k]))
        assert np.max(np.abs(r[k] - o[k])) <= 2e-12 * sc, k


def test_phi_equals_gz_and_row1_untouched():
    g, tab = geom("nov26")
    i, o, r = pc.run_pgf("nov26", 33312, 1, g, tab)
    assert np.array_equal(o['phi'], o['gz'])
    assert np.array_equal(o['dut'][:, 0, :], i['dut'][:, 0, :]) and np.array_equal(o['ut'][:, 0, :], i['ut'][:, 0, :])


# ---------------------------------------------------------------- non-vacuity
@pytest.mark.parametrize("date,itime,pas", CALLS[::6])
def test_non_vacuous(date, itime, pas):
    g, tab = geom(date)
    i, o, r = pc.run_pgf(date, itime, pas, g, tab)
    assert np.max(np.abs(o['ut'] - i['ut'])) > 1e-6 and np.max(np.abs(o['vt'] - i['vt'])) > 1e-6
    assert np.max(np.abs(o['dut'])) > 1e3 and np.max(np.abs(o['dvt'])) > 1e3
    # AVRX acts on the near-polar rows of PGFU, and not on the others
    dd = np.abs(o['pgfu'] - o['pgfu0'])
    assert dd[:, 1:14, :].max() > 0 and dd[:, 32:45, :].max() > 0 and dd[:, 14:32, :].max() == 0
    # real input DUT,DVT are zero at entry in every call (ADVECV zeroes them on exit)
    assert np.max(np.abs(i['dut'])) == 0.0 and np.max(np.abs(i['dvt'])) == 0.0


def test_polar_rows_polefix_changes_result():
    g, tab = geom("dec01")
    i, o, r = pc.run_pgf("dec01", 33552, 2, g, tab)
    r0 = fp.pgf(i['dt1'], i['mam'], i['ut'], i['vt'], i['mafter'], i['s0'], i['sz'], i['dut'], i['dvt'], g,
                tab=tab, polefix=False)
    assert np.max(np.abs(r0['dut'][:, 1, :] - o['dut'][:, 1, :])) > 1.0
    assert np.max(np.abs(r0['dvt'][:, JM - 1, :] - o['dvt'][:, JM - 1, :])) > 1.0


def test_unexercised_nonzero_entry_dut_not_validated():
    """The real calls never enter PGF with nonzero DUT/DVT; the port accepts them (cell-IM update order
    follows the Fortran I loop) but that path has no real reference.  This test only pins the behaviour."""
    g, tab = geom("nov26")
    i, o, r = pc.run_pgf("nov26", 33312, 1, g, tab)
    d1 = np.ones((IM, JM, LM))
    r1 = fp.pgf(i['dt1'], i['mam'], i['ut'], i['vt'], i['mafter'], i['s0'], i['sz'], d1, d1, g, tab=tab)
    sl = slice(2, JM - 2)                    # away from the polar rows (ACOR2 also scales the added 1.0)
    assert np.allclose(r1['dvt'][:, sl, :] - 1.0, r['dvt'][:, sl, :], rtol=1e-9, atol=1e-3)


# ---------------------------------------------------------------- mutations (numpy-pow mode; tolerance 1e-9 of scale)
def _base(date="nov26", itime=33312, pas=3):
    g, tab = geom(date)
    i, o, r = pc.run_pgf(date, itime, pas, g, tab)
    return g, tab, i, o


def _run(i, g, tab, **kw):
    return fp.pgf(i['dt1'], i['mam'], i['ut'], i['vt'], i['mafter'], i['s0'], i['sz'], i['dut'], i['dvt'], g,
                  tab=tab, **kw)


def _far(r, o, key, tol=1e-9):
    return np.max(np.abs(r[key] - o[key])) > tol * np.max(np.abs(o[key]))


def test_mutation_no_avrx():
    g, tab, i, o = _base()
    assert _far(_run(i, g, tab, avrx=False), o, 'dut')


def test_mutation_wrong_dt():
    g, tab, i, o = _base()
    i2 = dict(i); i2['dt1'] = i['dt1'] * 1.001
    r = _run(i2, g, tab)
    assert _far(r, o, 'dut') and _far(r, o, 'dvt')


def test_mutation_acor_swap():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['acor'], g2['acor2'] = 1.0, 1.0
    r = _run(i, g2, tab)
    assert _far(r, o, 'dut') or _far(r, o, 'dvt')


def test_mutation_wrong_kapa_reciprocal():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['bykapap1'] = g['bykapap2']
    assert _far(_run(i, g2, tab), o, 'gz')


def test_mutation_gz_uses_wrong_layer_term():
    g, tab, i, o = _base()
    g2 = copy.deepcopy(g); g2['zatmo'] = g['zatmo'] * 1.01
    assert _far(_run(i, g2, tab), o, 'gz', 1e-6)


def test_mutation_no_polar_copy():
    """Without the I=1 copy of polar columns the pole rows of GZ would be taken from the per-column
    integral, which is identical here (poles have identical columns); the real distinguishing check is that
    the dump's polar rows are I-independent."""
    g, tab, i, o = _base()
    assert np.all(o['gz'][:, 0, :] == o['gz'][0, 0, :][None, :])
    assert np.all(o['adm'][:, JM - 1, :] == o['adm'][0, JM - 1, :][None, :])


def test_mutation_dvt_sign():
    g, tab, i, o = _base()
    r = _run(i, g, tab)
    r['dvt'] = -r['dvt']
    assert _far(r, o, 'dvt')
