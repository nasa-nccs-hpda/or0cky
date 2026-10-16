"""D192 tests of multiday_score.py.  Synthetic tests need no data; the validation tests re-score the existing D157 (free radiation), D150/D157
open loop, D171 (Ent computed) and D171 ghyrec days and compare with the PUBLISHED numbers of the ledger (exact counts, ratios to the 2 decimals
published).  They are skipped when ff_data/nov26_day is absent.
Run:  PYTHONPATH=fullfidelity taskset -c 3-5 python -m pytest fullfidelity/tests/test_multiday_score.py   (about 2 min)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_condse_io as cio
import atm_day_report as RP
import multiday_score as MS

FF = cio.FF_DEFAULT
DAY = f"{FF}/nov26_day"
HAVE = all(os.path.exists(f"{DAY}/{n}") for n in ('ffa_step_33312_e.bin', 'ffa_step_33365_e.bin', 'ffpt_p1_33312.bin', 'ffpt_p5_33365.bin',
                                                 'ours_free_np/step_33312.npz', 'ours_d171_ent/step_33312.npz'))
needs_day = pytest.mark.skipif(not HAVE, reason='ff_data/nov26_day not present')


# ------------------------------------------------------------------------------------------------ synthetic (no data)
def _fake_res(ours_rms, floor_lo, floor_hi, field='T'):
    n = len(ours_rms)
    mem = {m: [{field: {'rms': (v := (floor_lo if i == 0 else floor_hi)), 'zrms': v, 'gmean': v}} for _ in range(n)] for i, m in enumerate(('p1', 'p2'))}
    ours = [None if o is None else {field: {'rms': o, 'zrms': o, 'gmean': o}} for o in ours_rms]
    return dict(it0=0, nsteps=n, ours=ours, mem=mem)


def test_classification_thresholds_are_the_published_ones():
    assert RP.classify(1.0, 1.0) == 'within' and RP.classify(1.0000001, 1.0) == 'near'
    assert RP.classify(2.0, 5.0) == 'near' and RP.classify(2.0000001, 5.0) == 'beyond'
    assert RP.classify(0.4, 0.3) == 'below'


def test_summarize_synthetic_ratios_and_verdict():
    res = _fake_res([0.4, 1.0, 1.5, 1.2, 2.5, 1.0], 1.0, 1.0)  # ratios: .4 1 1.5 1.2 2.5 1
    s = MS.summarize(res, ['p1', 'p2'], fields=('T',))
    f = s['per_field']['T']
    assert f['counts'] == {'below': 1, 'within': 2, 'near': 2, 'beyond': 1}
    assert f['n_within_incl_below'] == 3
    assert f['near_steps'] == [2, 3] and f['beyond_steps'] == [4]
    assert f['worst_ratio_all'] == 2.5 and f['worst_step_all'] == 4 and f['worst_ratio_k3'] == 2.5
    assert not s['day_all_within'] and not s['day_never_beyond_2x'] and not s['day_never_beyond_2x_k3']
    assert 'criterion not met' in s['verdict']
    assert s['short_window'] and MS.LABEL_SHORT_WINDOW in s['verdict']


def test_summarize_beyond_only_before_k3_and_all_within():
    s = MS.summarize(_fake_res([3.0, 1, 1, 1], 1.0, 1.0), ['p1', 'p2'], fields=('T',))
    assert not s['day_never_beyond_2x'] and s['day_never_beyond_2x_k3'] and s['per_field']['T']['beyond_only_before_k3']
    s = MS.summarize(_fake_res([.9, 1, .8, .7], 1.0, 1.0), ['p1', 'p2'], fields=('T',))
    assert s['day_all_within'] and s['day_never_beyond_2x'] and 'within at every' in s['verdict']


def test_missing_steps_are_skipped_and_label_short_window():
    s = MS.summarize(_fake_res([.5, None, 1.0], 1.0, 1.0), ['p1', 'p2'], fields=('T',))
    assert s['steps_scored'] == [0, 2] and s['short_window']
    assert MS.NSTEPS_DAY == 54 and MS.RADIATION_SENTENCE in MS.report_markdown(dict(score=s))


def test_chain_adapter_accepts_filter_arrays_flat_npz_and_jax_like(tmp_path):
    T = np.ones((72, 46, 40))
    a0 = {'filter': {'T': T, 'Q': T * 2, 'KEA_NEW': 1.0}}
    p = MS.chain_states([a0, {'filter/T': T * 3, 'dyn/T': T * 9}])
    assert set(p(0)) == {'T', 'Q'} and p(0)['T'].dtype == np.float64
    assert p(1)['T'][0, 0, 0] == 3 and p(2) is None
    f = tmp_path / 'nov26_a.npz'
    np.savez(f, **{'filter/T': T * 5})
    assert MS.chain_states(str(tmp_path / 'nov26_*.npz'))(0)['T'][0, 0, 0] == 5


# ------------------------------------------------------------------------------------------------ validation vs the published tables
@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp('md') / 'members.json')


def _score(tag, cache):
    return MS.score_dir(f"{DAY}/ours_{tag}", cache=cache)['score']


def _wc(s):
    """whole-column rms counts as the ledger quotes them: (within incl. below, near, beyond)"""
    return {f: (v['n_within_incl_below'], v['counts'].get('near', 0), v['counts'].get('beyond', 0)) for f, v in s['per_field'].items()}


def _r(s):
    return {f: round(v['worst_ratio_k3'], 2) for f, v in s['per_field'].items()}


# D157 (free radiation; ours_free_np), open loop in brackets = D150/D151 day (ours_imf_np), D171 table (ours_d171_ent, ours_d171_ghyrec)
PUB = {
    'free_np': dict(counts=dict(T=(54, 0, 0), U=(54, 0, 0), V=(54, 0, 0), Q=(49, 5, 0), P=(52, 2, 0), QCL=(43, 11, 0), QCI=(44, 10, 0)),
                    ratio=dict(T=.96, U=.95, V=.96, Q=1.13, P=1.00, QCL=1.44, QCI=1.95)),
    'imf_np': dict(counts=dict(T=(54, 0, 0), U=(54, 0, 0), V=(54, 0, 0), Q=(39, 15, 0), P=(52, 2, 0), QCL=(50, 4, 0), QCI=(51, 3, 0)),
                   ratio=dict(T=.92, U=.90, V=.98, Q=1.09, P=1.05, QCL=1.09, QCI=1.11)),
    'd171_ent': dict(counts=dict(T=(54, 0, 0), U=(54, 0, 0), V=(54, 0, 0), Q=(53, 1, 0), P=(54, 0, 0), QCL=(52, 1, 1), QCI=(41, 13, 0)),
                     ratio=dict(T=.85, U=.84, V=.87, Q=1.00, P=.92, QCL=1.00, QCI=1.62)),
    'd171_ghyrec': dict(counts=dict(T=(54, 0, 0), U=(53, 1, 0), V=(53, 1, 0), Q=(38, 16, 0), P=(53, 1, 0), QCL=(50, 3, 1), QCI=(52, 2, 0)),
                        ratio=dict(T=.94, U=.97, V=.93, Q=1.05, P=1.02, QCL=1.08, QCI=1.06)),
}


@needs_day
@pytest.mark.parametrize('tag', list(PUB))
def test_reproduces_published_whole_column_tables(tag, cache):
    s = _score(tag, cache)
    assert _wc(s) == PUB[tag]['counts']
    assert _r(s) == PUB[tag]['ratio']
    assert s['n_steps_scored'] == 54 and not s['short_window']
    assert s['day_all_within'] is False and s['day_never_beyond_2x_k3'] is True


@needs_day
def test_d157_published_detail_free_day(cache):
    s = _score('free_np', cache)
    pf = s['per_field']
    assert (pf['QCL']['worst_step_k3'], pf['QCI']['worst_step_k3']) == (34, 47)
    assert s['day_never_beyond_2x'] is True
    z = {k: v for k, v in s['secondary'].items() if k.endswith('.zrms')}
    for f, n in dict(T=14, Q=9, V=9, P=16, QCL=10, QCI=8, U=0).items():
        assert z[f + '.zrms']['counts'].get('near', 0) == n
    for f, r in dict(T=1.18, Q=1.15, V=1.17, P=1.50, QCL=1.53, QCI=1.98).items():
        assert round(z[f + '.zrms']['max_ratio'], 2) == r
    g = {k[:-len('.absgmean')]: v for k, v in s['secondary'].items() if k.endswith('.absgmean')}
    assert [g[f]['counts'].get('beyond', 0) for f in ('QCL', 'QCI', 'P')] == [1, 2, 2]
    assert [g[f]['counts'].get('near', 0) for f in ('T', 'Q', 'V')] == [7, 6, 12]
    assert [round(g[f]['max_ratio'], 2) for f in ('QCL', 'QCI', 'P', 'T', 'V')] == [2.50, 2.44, 2.47, 1.74, 1.71]


@needs_day
def test_d171_ent_published_detail(cache):
    s = _score('d171_ent', cache)
    pf = s['per_field']
    assert pf['QCL']['beyond_steps'] == [1] and pf['QCL']['worst_ratio_all'] > 2  # the published 'beyond' is at a step < 3
    assert pf['QCL']['beyond_only_before_k3'] and s['day_never_beyond_2x'] is False and s['day_never_beyond_2x_k3'] is True
    assert round(s['secondary']['T.absgmean']['max_ratio'], 1) == 3.2


@needs_day
def test_short_window_adapter_equals_dir_scoring(cache):
    """6-step window through the d191_chain adapter (arrays['filter'] layout) equals scoring the same steps from the npz directory."""
    n = 6
    seq = [{'filter': RP.load_ours(f"{DAY}/ours_d171_ent", MS.IT0_NOV26 + k)} for k in range(n)]
    a = MS.score_sequence(MS.chain_states(seq), MS.IT0_NOV26, n, cache=cache)['score']
    b = MS.score_sequence(MS.dir_states(f"{DAY}/ours_d171_ent", MS.IT0_NOV26, n), MS.IT0_NOV26, n, cache=cache)['score']
    assert a['short_window'] and MS.LABEL_SHORT_WINDOW in a['verdict']
    assert a['steps_scored'] == list(range(n))
    assert a['per_field'] == b['per_field']
    # first six steps are also what the 54-step score says for k < 6
    full = _score('d171_ent', cache)
    for f in MS.SCORED:
        assert {k: v for k, v in full['per_field'][f]['cls'].items() if int(k) < n} == a['per_field'][f]['cls']
