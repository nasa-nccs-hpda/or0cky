"""D149-D151 tests: one-model-day open-loop run and the noise-floor overlay.  Skipped when ff_data/nov26_day is absent.
Run (from the repo root or fullfidelity/):  PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_atm_day.py    (about 4-5 min)
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_condse_io as cio
import atm_day_report as RP

FF = cio.FF_DEFAULT
DAY = f"{FF}/nov26_day"
IT0 = 33312


def _have(patterns):
    return all(glob.glob(p) for p in patterns)


HAVE_E = _have([f"{DAY}/ffa_step_{IT0 + k}_e.bin" for k in (0, 47, 53)])
HAVE_PT = _have([f"{DAY}/ffpt_{m}_{IT0}.bin" for m in ('ctrl', 'p1', 'p3', 'p5')])
HAVE_STEPS = _have([f"{DAY}/{n}_{IT0 + k}.bin" for k in range(6) for n in ('ffa_step', )] if False else []) and all(
    os.path.exists(f"{DAY}/{n}_{IT0 + k}.bin") for k in range(6) for n in ('ffp', 'ffs', 'ffl', 'ffg', 'fft', 'ffc_cse_in', 'ffc_cse_out')) \
    and all(os.path.exists(f"{DAY}/ffa_step_{IT0 + k}_{s}.bin") for k in range(6) for s in 'arde')
needs_day = pytest.mark.skipif(not HAVE_E, reason="ff_data/nov26_day not present")
needs_pt = pytest.mark.skipif(not (HAVE_E and HAVE_PT), reason="ffpt member dumps not present")
needs_steps = pytest.mark.skipif(not HAVE_STEPS, reason="per-step chain inputs of nov26_day not present")


# ----------------------------------------------------------------------------------------------- statistics (no dumps needed)
def test_valid_mask_counts():
    assert RP.valid_mask('T').sum() == 44 * 72 + 2          # pole rows carry one distinct cell
    assert RP.valid_mask('U').sum() == 45 * 72              # B grid row 1 undefined


def test_diff_metrics_known_offset():
    a = np.random.default_rng(0).standard_normal((72, 46, 40))
    assert RP.diff_metrics(a, a, 'T')['rms'] == 0.0
    d = RP.diff_metrics(a + 1e-3, a, 'T')
    assert abs(d['rms'] - 1e-3) < 1e-15 and abs(d['zrms'] - 1e-3) < 1e-15 and abs(d['maxabs'] - 1e-3) < 1e-15
    w = np.ones_like(a)
    assert abs(RP.diff_metrics(a + 1e-3, a, 'T', w)['gmean'] - 1e-3) < 1e-15
    # a difference only in an undefined cell (polar row i>1, B-grid row 1) must be invisible
    b = a.copy(); b[5, 0, :] += 1.0
    assert RP.diff_metrics(b, a, 'T')['rms'] == 0.0
    assert RP.diff_metrics(b, a, 'U')['rms'] == 0.0


def test_classification_and_overlay():
    assert RP.classify(0.9, 0.7) == 'within' and RP.classify(0.3, 0.2) == 'below'
    assert RP.classify(1.5, 1.4) == 'near' and RP.classify(2.5, 2.4) == 'beyond'
    res = dict(it0=0, nsteps=3, ours=[{'T': {'rms': v}} for v in (1e-3, 6e-3, 3e-2)],
               mem={'a': [{'T': {'rms': v}} for v in (2e-3, 4e-3, 8e-3)], 'b': [{'T': {'rms': v}} for v in (1e-3, 2e-3, 4e-3)]})
    rows, first = RP.overlay(res, ['a', 'b'], 'rms', fields=('T',))
    assert [r['T']['cls'] for r in rows] == ['within', 'near', 'beyond']
    assert first['T'] == dict(near=1, beyond=2)


# ----------------------------------------------------------------------------------------------- dumps of the real day run
@needs_pt
def test_ffpt_control_equals_ffa_step_e():
    """the small D151 end-state hook writes exactly what the D128 site 'e' writes (same place in atm_phase2)."""
    for k in (0, 17, 53):
        e = cio.read_cse(f"{DAY}/ffa_step_{IT0 + k}_e.bin")
        c = cio.read_cse(f"{DAY}/ffpt_ctrl_{IT0 + k}.bin")
        for f in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'P'):
            assert np.array_equal(e[f], c[f]), (k, f)


@needs_day
def test_new_day_dumps_reproduce_original_dumps():
    """the widened (54-step) run reproduces the existing 6-step dumps bit for bit (same restart, same binary stack)."""
    for it in (IT0, IT0 + 5):
        for s in 'are':
            a = cio.read_cse(f"{FF}/nov26/ffa_step_{it}_{s}.bin")
            b = cio.read_cse(f"{DAY}/ffa_step_{it}_{s}.bin")
            assert all(np.array_equal(a[k], b[k]) for k in a if k in b and k != 'HDR'), (it, s)


@needs_pt
def test_perturbation_is_one_ulp_at_one_point():
    """member p3 = +1 ulp in Q at one point: one step later only a handful of cells differ and by ~1e-16 (the perturbation, not noise)."""
    c = cio.read_cse(f"{DAY}/ffpt_ctrl_{IT0}.bin")
    p = cio.read_cse(f"{DAY}/ffpt_p3_{IT0}.bin")
    assert int((p['Q'] != c['Q']).sum()) <= 10
    assert float(np.abs(p['Q'] - c['Q']).max()) < 1e-15
    # but the same perturbation has grown to macroscopic size after the day
    c2 = cio.read_cse(f"{DAY}/ffpt_ctrl_{IT0 + 47}.bin")
    p2 = cio.read_cse(f"{DAY}/ffpt_p3_{IT0 + 47}.bin")
    assert RP.diff_metrics(p2['T'], c2['T'], 'T')['rms'] > 1e-3


@needs_day
def test_day_boundary_inside_window():
    """the day boundary (step 33360) changes exactly MA/PEDN/PMID/PK/PDSIG/PEK/P (DAILY_ATMDYN) and Q (DAILY_ch4ox) of the atmosphere state."""
    e = cio.read_cse(f"{DAY}/ffa_step_33359_e.bin")
    a = cio.read_cse(f"{DAY}/ffa_step_33360_a.bin")
    changed = sorted(k for k in e if k in a and k != 'HDR' and not np.array_equal(e[k], a[k]))
    assert changed == sorted(['MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'PEK', 'P', 'Q'])


@needs_day
def test_daily_replay_on_real_state():
    import atm_step as A
    import atm_day_open_loop as D
    ctx = A.make_ctx('nov26', imf=True)
    mdrya, _, _ = D.daily_mdrya(FF, 'nov26')
    r = D.daily_replay_check(FF, 'nov26_day', ctx, mdrya, 33360)
    for k in ('MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P', 'PEK'):
        assert r['fields'][k]['n_diff'] == 0, k          # DAILY_ATMDYN + MAtoPMB bitwise on the real end-of-day state
    assert r['fields']['Q']['max_abs'] < 1e-18            # recorded ch4ox mass increment (recovered from the same dumps: structural check)
    assert r['info']['ch4ox_dm_i_spread_rel'] < 1e-10     # the increment depends on (j,l) only


# ----------------------------------------------------------------------------------------------- open-loop replay
@needs_steps
def test_open_loop_first_six_steps():
    """6 steps (radiation steps 0 and 5, frozen heating rates in between): step 0 at rounding level, radiation feed identical to the real
    own record every step, all fields finite, divergence grows."""
    import atm_day_open_loop as D
    res = D.run_day(nsteps=6, save=False, tag='test', log=lambda *a: None)
    rows = res['rows']
    assert [r['radiation_step'] for r in rows] == [True, False, False, False, False, True]
    assert all(r['rad_feed_vs_own_r'] == 0.0 for r in rows)
    r0 = rows[0]['stats']
    assert r0['T']['rms'] < 1e-9 and r0['U']['rms'] < 1e-9 and r0['Q']['rms'] < 1e-12 and r0['P']['rms'] < 1e-9
    assert all(np.isfinite(r['stats'][f]['rms']) for r in rows for f in ('T', 'Q', 'U', 'V', 'P'))
    assert rows[5]['stats']['T']['rms'] > rows[0]['stats']['T']['rms']
    assert all(r['wall'] < 600 for r in rows)


@pytest.mark.skipif(not glob.glob(f"{DAY}/ours_*/run.json"), reason="no saved open-loop run")
def test_saved_run_consistency():
    runs = sorted(glob.glob(f"{DAY}/ours_*/run.json"))
    for p in runs:
        d = json.load(open(p))
        assert d['nsteps'] >= 48 and len(d['rows']) == d['nsteps'], p
        d0 = os.path.dirname(p)
        assert all(os.path.exists(f"{d0}/step_{IT0 + k}.npz") for k in range(d['nsteps'])), p
        # day-boundary step carries the recorded daily update
        k = 33360 - IT0
        if d['nsteps'] > k:
            assert d['rows'][k]['daily'] is not None, p
        if p.endswith("ours_imf_np/run.json"):
            # numpy dynamics + libimf pow is the rounding-level path (step 0 T rms 7e-14 K); the JAX-dynamics path differs
            # at 8e-5 K at step 0 (an equally valid rounding-level member, D150) so the strict check is for the numpy run only
            assert d['rows'][0]['stats']['T']['rms'] < 1e-9
        else:
            assert d['rows'][0]['stats']['T']['rms'] < 1e-3


@pytest.mark.skipif(not (HAVE_E and HAVE_PT and glob.glob(f"{DAY}/ours_*/run.json")), reason="inputs for the overlay not present")
def test_overlay_runs_on_saved_data():
    import dyn_glue_io as gio
    p = sorted(glob.glob(f"{DAY}/ours_*/run.json"))[0]
    axyp = gio.load_g('nov26', FF)['axyp']
    res = RP.curves(FF, 'nov26_day', os.path.dirname(p), ['p1', 'p3', 'p5'], IT0, 3, axyp)
    rows, first = RP.overlay(res, ['p1', 'p3', 'p5'])
    assert len(rows) == 3 and 'T' in rows[0]
    assert all(np.isfinite(rows[0][f]['hi']) for f in ('T', 'Q', 'U', 'V'))
