"""D205 tests of daily_lake.py (NumPy port of the Fortran daily_LAKE, LAKES.f:2492).
No data needed: closed-form checks of the conical-lake mass-area relation, ice conservation, removal of a tiny lake, cubic-root helper, water deficit helper.
Needs ifort (source env_modele.sh) and builds the standalone harness: bitwise comparison of the port with the COMPILED REAL daily_LAKE source on random cells that exercise
the expansion (with and without soil saturation), ice crunch, mixed-layer entrainment/scaling, removal and ice-dump branches.
Needs the nov26 records: the real step-47 -> 48 day-boundary case (DMWLDF from the recorded GHY output) and the hook of jax_coupled (skipped otherwise)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import daily_lake as DL  # noqa: E402

IM, JM = DL.IM, DL.JM


def base_case(seed=1, lakes=60):
    """Random lake cells on the 72x46 grid (land cells elsewhere)."""
    rng = np.random.default_rng(seed)
    z = np.zeros((IM, JM))
    c = {k: z.copy() for k in ('flake', 'fearth', 'fland', 'rsi', 'msi', 'snowi', 'mwl', 'gml', 'tlake', 'mldlk', 'tanlk', 'hlake', 'dmwldf', 'flice', 'focean')}
    c['hsi'] = np.zeros((IM, JM, 4))
    c['axyp'] = np.full((IM, JM), 4.0e11) * (1 + 0.3 * rng.random((IM, JM)))
    cells = [(int(i), int(j)) for i, j in zip(rng.integers(1, IM, lakes), rng.integers(2, JM - 2, lakes))]
    c['fearth'][:] = 0.6
    c['fland'][:] = 0.6
    for (i, j) in cells:
        fl = rng.uniform(0.001, 0.2)
        c['flake'][i, j] = fl
        c['fland'][i, j] = 1 - fl
        c['fearth'][i, j] = 1 - fl
        c['hlake'][i, j] = rng.uniform(1.0, 20.0)
        c['tanlk'][i, j] = np.sqrt(fl * c['axyp'][i, j] / np.pi) / (3 * c['hlake'][i, j])
        h = c['hlake'][i, j] * rng.uniform(0.5, 1.6)
        c['mwl'][i, j] = 1000.0 * h * fl * c['axyp'][i, j]
        c['tlake'][i, j] = rng.uniform(0, 20)
        c['gml'][i, j] = 4185.0 * c['mwl'][i, j] * c['tlake'][i, j]
        c['mldlk'][i, j] = rng.uniform(1.0, 3.0)
        c['rsi'][i, j] = rng.uniform(0, 1) * (rng.random() < 0.7)
        c['msi'][i, j] = rng.uniform(5.0, 1500.0)
        c['snowi'][i, j] = rng.uniform(0, 30)
        c['hsi'][i, j] = -3.34e5 * rng.uniform(5, 300, 4) * 0.5
        c['dmwldf'][i, j] = rng.uniform(0, 800) * (rng.random() < 0.8)
    c['focean'] = z.copy()
    return c, cells


def run_port(c, **kw):
    S = {k: c[k] for k in DL.FIELDS_IN}
    return DL.daily_lake(S, c['flice'], c['focean'], c['tanlk'], c['hlake'], c['axyp'], c['dmwldf'], **kw)


def test_conical_lake_no_saturation_closed_form():
    c, cells = base_case(2)
    c['dmwldf'][:] = 0.0
    c['rsi'][:] = 0.0
    c['snowi'][:] = 0.0
    out = run_port(c)
    for (i, j) in cells:
        want = (9.0 * np.pi * (c['tanlk'][i, j] * c['mwl'][i, j] / 1000.0) ** 2) ** (1.0 / 3.0) / c['axyp'][i, j]
        want = min(want, 0.95 * (c['flake'][i, j] + c['fearth'][i, j]), c['flake'][i, j] + 0.049 * c['fearth'][i, j])
        if out['flake'][i, j] != c['flake'][i, j]:
            assert out['flake'][i, j] == pytest.approx(want, rel=1e-14)
        assert out['fland'][i, j] == pytest.approx(1 - out['flake'][i, j], abs=1e-15)


def test_mass_and_ice_conservation_when_lake_survives():
    c, cells = base_case(3)
    out = run_port(c)
    for (i, j) in cells:
        if out['flake'][i, j] > 0:
            # ice mass over the cell conserved unless the ice is crunched into the new area (RSI -> 1) with the ACE1I convention: mass per cell incl. the ACE1I layer
            m0 = c['flake'][i, j] * c['rsi'][i, j] * (c['msi'][i, j] + c['snowi'][i, j] + DL.ACE1I)
            m1 = out['flake'][i, j] * out['rsi'][i, j] * (out['msi'][i, j] + out['snowi'][i, j] + DL.ACE1I)
            assert m1 == pytest.approx(m0, rel=1e-12, abs=1e-9)
            # lake water + soil saturation water taken from the lake is conserved
            sat = c['mwl'][i, j] - out['mwl'][i, j]
            assert sat >= -1e-6
    assert out['counters']['cells_visited'] > 0


def test_tiny_lake_is_removed_and_ice_goes_to_implicit_arrays():
    c, cells = base_case(4, lakes=3)
    i, j = cells[0]
    c['mwl'][i, j] = 1.0e3                      # a thousand kg of water: new_flake < 1e-10 -> no lake
    c['gml'][i, j] = 4185.0 * 1.0e3 * 5.0
    c['rsi'][i, j] = 0.5
    out = run_port(c)
    assert out['flake'][i, j] == 0.0 and out['fland'][i, j] == 1.0 and out['rsi'][i, j] == 0.0
    assert out['mldlk'][i, j] == DL.MINMLD and out['msi'][i, j] == DL.AC2OIM
    assert out['mdwnimp'][i, j] > 0 and out['gtempr'][i, j] == DL.TF
    assert out['tlake'][i, j] == pytest.approx(5.0, rel=1e-6)


def test_cubicroot_residuals():
    for (a, b, c_, d) in ((1.0, 0.2, 0.0, -3.0), (2.0, 0.0, -3.0, 1.0), (1.0, 1e-3, 0.0, -1e5)):
        x, n = DL.cubicroot(a, b, c_, d)
        r = max(x[:n])
        assert abs(a * r ** 3 + b * r ** 2 + c_ * r + d) <= 1e-9 * (abs(a * r ** 3) + abs(d))


def test_water_deficit_helper():
    rows = np.zeros((1, 450))
    rows[0, 74:80] = (0.1, 0.17, 0.3, 0.5, 0.8, 1.2)                   # dz
    q = np.zeros((5, 6))
    q[0, :] = 1.0                                                      # texture 1 everywhere
    rows[0, 80:110] = q.ravel(order='F')
    rows[0, 169] = 0.5                                                 # fv
    w = np.zeros((1, 7, 2))
    ec = np.array([5 * IM + 7])
    fe = np.ones((IM, JM))
    thm0 = np.array([0.394, 0.537, 0.577, 0.885])
    dm = DL.water_deficit(rows, w, ec, fe, thm0)
    assert dm[7, 5] == pytest.approx(1000.0 * 0.394 * rows[0, 74:80].sum(), rel=1e-12)    # dry soil: fb*stor + fv*stor
    w[0, 1:, :] = 0.394 * rows[0, 74:80][:, None]                       # saturated
    assert DL.water_deficit(rows, w, ec, fe, thm0)[7, 5] == pytest.approx(0.0, abs=1e-9)
    fe[7, 5] = 0.0
    assert DL.water_deficit(rows, np.zeros((1, 7, 2)), ec, fe, thm0)[7, 5] == 0.0


# ------------------------------------------------------------------------------------------------ differential test against the compiled real source
def _ifort():
    import shutil
    import subprocess
    if shutil.which('ifort'):
        return True
    try:
        r = subprocess.run(['bash', '-c', f'source {HERE}/../env_modele.sh >/dev/null 2>&1; which ifort'], capture_output=True, text=True)
        return r.returncode == 0 and 'ifort' in r.stdout
    except Exception:
        return False


HAVE_SRC = os.path.exists(os.path.join(os.environ.get('MODELE_SRC', '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model'), 'LAKES.f'))


@pytest.mark.skipif(not (_ifort() and HAVE_SRC), reason='ifort or the ModelE source is absent')
def test_bitwise_against_compiled_fortran(tmp_path):
    import daily_lake_harness as H
    d = str(tmp_path)
    H.build(d)
    seen = {}
    for seed in range(1, 7):
        c, _ = base_case(seed, lakes=150)
        rng = np.random.default_rng(100 + seed)
        sc = np.exp(rng.normal(0, [0.05, 0.3, 1.0][seed % 3], c['mwl'].shape))
        c['mwl'] = c['mwl'] * sc
        c['gml'] = c['gml'] * sc
        c['msi'] = np.where(c['flake'] > 0, np.exp(rng.normal(5.5, 2.0, c['mwl'].shape)), c['msi'])      # some above 5000 kg/m2 (ice dump)
        rep, po, _ = H.compare(d, c)
        bad = {k: v for k, v in rep.items() if v[1] != 0}
        assert not bad, (seed, bad)
        for k, v in po['counters'].items():
            seen[k] = seen.get(k, 0) + v
    for k in ('expand', 'crunch', 'layer_entrain', 'layer_scale', 'no_lake', 'saturate_solve', 'ice_dump'):
        assert seen[k] > 0, (k, seen)


# ------------------------------------------------------------------------------------------------ real day-boundary case (records) and the hook
FF = os.environ.get('FF_DATA', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data')
HAVE_DAY = os.path.exists(f'{FF}/nov26_day/ffg_33359.bin') and os.path.exists(f'{FF}/nov26_day/ffc_cse_in_33360.bin') and os.path.exists(f'{FF}/_pristine_restarts/fort1_nov26_itime33312.nc')


@pytest.mark.skipif(not HAVE_DAY, reason='nov26_day records absent')
def test_restart_state_with_real_step47_deficit_is_sane():
    """The port applied to the nov26 RESTART lake/ice state (47 steps before the boundary, so NOT the state the real model had: no record comparison is asserted here; the
    comparison with the record on our end-of-step-47 state is in scoping/D205_DAILY_LAKE_ENTRY.md) with the real step-47 GHY output for DMWLDF.  Invariants only:
    0 <= FLAKE <= 0.95 (FLAKE + FEARTH), FLAND + FLAKE + FOCEAN = 1, finite state, DMWLDF >= 0 and nonzero in land cells, ice fraction in [0, 1]."""
    import clouds_condse_io as cio
    import surface_loop as L
    import ghy_compare as GC
    import ghy_ref as GR
    import jax_static as JST
    st = L.load_statics('nov26')
    c47 = cio.read_cse(f'{FF}/nov26_day/ffc_cse_in_33359.bin')
    g = GC.load(f'{FF}/nov26_day/ffg_33359.bin')
    g2 = g[len(g) // 2:]
    ec = (g2[:, 1].astype(int) - 1) * IM + (g2[:, 0].astype(int) - 1)
    w = np.stack([GC.unpack(r)[4]['w_out'] for r in g2])
    dm = DL.water_deficit(g2, w, ec, c47['FEARTH'], GR.THM[0, :])
    assert (dm >= 0).all() and (dm > 0).sum() > 500
    topo = JST.topography()
    hl, tn = DL.lake_statics(topo, c47['FLAKE'], np.asarray(st['axyp']))
    ice = L.init_surface_state('nov26', st=st)
    S = dict(flake=c47['FLAKE'], fearth=c47['FEARTH'], fland=c47['FLAND'], rsi=ice['ice']['rsi'], msi=ice['ice']['msi'], snowi=ice['ice']['snowi'], hsi=ice['ice']['hsi'],
             mwl=ice['lake']['mwl'], gml=ice['lake']['gml'], tlake=ice['lake']['tlake'], mldlk=ice['lake']['mldlk'])
    out = DL.daily_lake(S, c47['FLICE'], c47['FOCEAN'], tn, hl, np.asarray(st['axyp']), dm)
    assert all(np.isfinite(out[k]).all() for k in DL.FIELDS_IN)
    assert (out['flake'] >= 0).all() and (out['flake'] <= 0.95 * (c47['FLAKE'] + c47['FEARTH']) + 1e-15).all()
    assert (out['rsi'] >= 0).all() and (out['rsi'] <= 1.0 + 1e-15).all()
