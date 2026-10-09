"""D208 tests of daily_land_fractions.py (NumPy port of GHY_DRV.f update_land_fractions, the land side of the daily lake update).
No data needed: conservation of water and heat (shrink: exact redistribution; expansion: the water taken from the lake is what the underwater fraction gains), snow fraction rules,
guards.  Needs ifort: bitwise comparison with the COMPILED REAL subroutine on random cells (shrink and expansion, storage-limited and unlimited layers, fb or fv = 0).
Needs the nov26 records: the real step-47 -> 48 transfer (w, ht of the bare and vegetated fractions, the underwater fraction and fr_snow reproduced exactly from the real
fractions of the CONDSE entries and the DMWLDF of the real step-47 water)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import daily_land_fractions as DF  # noqa: E402
import ghy_ref as GR  # noqa: E402

THM0 = GR.THM[0, :]


def rand_case(n=200, seed=1):
    rng = np.random.default_rng(seed)
    dz = np.tile(np.array([0.1, 0.17, 0.3, 0.5, 0.9, 1.5]), (n, 1)) * (1 + 0.1 * rng.random((n, 6)))
    q = rng.random((n, 5, 6)) + 0.05
    q = q / q.sum(1, keepdims=True)
    ws = np.einsum('nmk,m,nk->nk', q[:, :4, :], THM0, dz)                    # storage per layer, (n, 6)
    w = np.zeros((n, 7, 3))
    ht = np.zeros((n, 7, 3))
    for ib in range(3):
        w[:, 1:, ib] = ws * rng.uniform(0.3, 1.0, (n, 6))
        ht[:, 1:, ib] = rng.normal(2e7, 3e6, (n, 6)) * (1 + 0.3 * rng.random((n, 6)))
    w[:, 0, 1] = rng.uniform(0, 1e-3, n)
    ht[:, 0, 1] = rng.normal(1e3, 5e2, n)
    w[:, 1:, 2] = np.where(rng.random((n, 1)) < 0.5, ws, w[:, 1:, 2])        # saturated underwater soil in half of the cells
    sv = rng.uniform(0.001, 0.6, n)
    flake = np.clip(sv + rng.normal(0, 0.02, n), 1e-4, 0.9)
    flake = np.where(rng.random(n) < 0.15, sv, flake)                         # some unchanged cells
    fearth = 1.0 - flake - rng.uniform(0, 0.1, n)
    fv = np.where(rng.random(n) < 0.15, 0.0, np.where(rng.random(n) < 0.15, 1.0, rng.random(n)))
    c = dict(w=w, ht=ht, fr_snow=rng.random((n, 2)) * 0.6, dz=dz, q=q, fv=fv, fearth=fearth, flake=flake, svflake=sv,
             focean=np.where(rng.random(n) < 0.05, 1.0, 0.0), dmwldf=rng.uniform(0, 600, n) * (rng.random(n) < 0.8), byaxyp=1.0 / rng.uniform(2e11, 6e11, n), thm0=THM0)
    c['dgml'] = c['dmwldf'] * rng.uniform(0, 4e5, n)
    return c


def totals(S, out):
    """Water (m * area fraction) and heat summed over the fractions and layers, before and after (shrink conserves exactly; expansion adds the lake water)."""
    res = []
    for w, ht, fe_old, fe_new in ((S['w'], S['ht'], None, None), (out['w'], out['ht'], None, None)):
        res.append((w, ht))
    return res


def test_shrink_conserves_water_and_heat():
    c = rand_case(300, 3)
    S = {k: (np.asarray(v, dtype=np.float64) if k != 'thm0' else v) for k, v in c.items() if k != 'thm0'}
    S['fv'] = DF.fv_from_ent(c['fv'])
    out, info = DF.update_land_fractions(S, THM0)
    sh = info['shrunk']
    assert sh.sum() > 10
    for r in np.nonzero(sh)[0]:
        sv, fl, fe = S['svflake'][r], S['flake'][r], S['fearth'][r]
        dfrac = min(sv - fl, fe)
        fv = S['fv'][r]
        fb = 1.0 - fv
        for arr_in, arr_out in ((S['w'][r], out['w'][r]), (S['ht'][r], out['ht'][r])):
            # layers 1..6: lake*svflake + soil*(fearth - dfrac)  ==  lake*flake_new_share unchanged lake + soil*fearth  (the lake column keeps its values)
            before = arr_in[1:, 2] * sv + (arr_in[1:, 0] * fb + arr_in[1:, 1] * fv) * (fe - dfrac)
            after = arr_out[1:, 2] * fl + (arr_out[1:, 0] * fb + arr_out[1:, 1] * fv) * fe
            assert np.allclose(before, after, rtol=1e-12, atol=1e-9 * np.abs(before).max()), r
        assert np.array_equal(out['w'][r, :, 2], S['w'][r, :, 2])                  # the underwater fraction is unchanged when the lake shrinks
        assert np.all(out['fr_snow'][r] <= S['fr_snow'][r] + 1e-15)


def test_expansion_water_balance_and_snow_rules():
    c = rand_case(300, 4)
    S = {k: np.asarray(v, dtype=np.float64) for k, v in c.items() if k != 'thm0'}
    S['fv'] = DF.fv_from_ent(c['fv'])
    S['focean'] = np.zeros_like(S['flake'])
    out, info = DF.update_land_fractions(S, THM0)
    ex = info['expanded']
    assert ex.sum() > 10
    for r in np.nonzero(ex)[0]:
        sv, fl = S['svflake'][r], S['flake'][r]
        dfrac = fl - sv
        fv = S['fv'][r]
        fb = 1.0 - fv
        gain = ((out['w'][r, 1:, 2] * fl) - (S['w'][r, 1:, 2] * sv)).sum() + dfrac * fv * S['w'][r, 0, 1] * 0   # canopy dumped into layer 1 is inside the sum
        soil = dfrac * ((fb * S['w'][r, 1:, 0] + fv * S['w'][r, 1:, 1]).sum() + fv * S['w'][r, 0, 1])
        lake = gain - soil                                                          # water taken from the lake
        assert lake <= S['dmwldf'][r] * dfrac / DF.RHOW + 1e-12                    # never more than the lake's share
        assert np.array_equal(out['w'][r, :, :2], S['w'][r, :, :2])                # the soil fractions are not changed by an expansion
        if fb <= 0:
            assert out['fr_snow'][r, 0] <= 0.95 + 1e-15
        if fv <= 0:
            assert out['fr_snow'][r, 1] <= 0.95 + 1e-15
    nochange = (S['svflake'] == S['flake']) | (S['focean'] >= 1.0)
    assert np.array_equal(out['w'][nochange], S['w'][nochange]) and np.array_equal(out['ht'][nochange], S['ht'][nochange])


def test_guards():
    c = rand_case(5, 5)
    S = {k: np.asarray(v, dtype=np.float64) for k, v in c.items() if k != 'thm0'}
    S['fv'] = DF.fv_from_ent(c['fv'])
    S['focean'][:] = 0.0
    S['svflake'][:] = 0.1
    S['flake'][:] = 0.2
    S['fearth'][:] = 0.0                                                            # expansion with no land: Fortran stop_model
    with pytest.raises(RuntimeError):
        DF.update_land_fractions(S, THM0)


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


HAVE_SRC = os.path.exists(os.path.join(os.environ.get('MODELE_SRC', '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model'), 'GHY_DRV.f'))


@pytest.mark.skipif(not (_ifort() and HAVE_SRC), reason='ifort or the ModelE source is absent')
def test_bitwise_against_compiled_fortran(tmp_path):
    import land_fractions_harness as H
    d = str(tmp_path)
    H.build(d)
    nsh = nex = 0
    for seed in range(1, 6):
        c = rand_case(400, seed)
        rep, po, fo, info = H.compare(d, c)
        bad = {k: v for k, v in rep.items() if v[1] != 0}
        assert not bad, (seed, bad)
        nsh += info['n_shrunk']
        nex += info['n_expanded']
    assert nsh > 100 and nex > 100


FF = os.environ.get('FF_DATA', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data')
HAVE_DAY = all(os.path.exists(f'{FF}/nov26_day/{f}') for f in ('ffg_33359.bin', 'ffg_33360.bin', 'ffc_cse_in_33359.bin', 'ffc_cse_in_33360.bin'))


@pytest.mark.skipif(not HAVE_DAY, reason='nov26_day records absent')
def test_real_step47_to_48_transfer_exact():
    """The REAL boundary: before = real GHY output of step 47 (second substep), after = real GHY entry of step 48 (first substep).  With the real fractions (FLAKE/FEARTH of
    the CONDSE entries of steps 47 and 48) and the DMWLDF of the real step-47 water, w and ht of the bare and vegetated fractions, w of the underwater fraction and fr_snow
    are reproduced EXACTLY (0.0 difference) in all 753 land cells (293 shrunk + 339 expanded rows).  ht of the underwater fraction needs the heat of the lake water (lake side):
    it is checked at 1e-6 relative on the rows without lake heat input."""
    import clouds_condse_io as cio
    import daily_lake as DL
    import ghy_compare as GC
    FFd = f'{FF}/nov26_day'
    a, b = GC.load(f'{FFd}/ffg_33359.bin'), GC.load(f'{FFd}/ffg_33360.bin')
    n = len(a) // 2
    a2, b1 = a[n:], b[:n]
    ii, jj = a2[:, 0].astype(int) - 1, a2[:, 1].astype(int) - 1
    R3 = lambda r, s, e: np.stack([x[s - 1:e].reshape(7, 3, order='F') for x in r])      # noqa: E731
    wb, hb = R3(a2, 181, 201), R3(a2, 202, 222)
    wb[:, :, 2], hb[:, :, 2] = R3(a2, 9, 29)[:, :, 2], R3(a2, 30, 50)[:, :, 2]
    wa, ha = R3(b1, 9, 29), R3(b1, 30, 50)
    c47, c48 = cio.read_cse(f'{FFd}/ffc_cse_in_33359.bin'), cio.read_cse(f'{FFd}/ffc_cse_in_33360.bin')
    dz, q, fv = DF.rows_from_ffg(a2)
    wout = np.stack([wb[r] for r in range(n)])
    dm = DL.water_deficit(a2, wout, jj * 72 + ii, c47['FEARTH'], THM0)[ii, jj]
    S = dict(w=wb, ht=hb, fr_snow=a2[:, 242:244], dz=dz, q=q, fv=fv, fearth=c48['FEARTH'][ii, jj], flake=c48['FLAKE'][ii, jj], svflake=c47['FLAKE'][ii, jj],
             focean=c47['FOCEAN'][ii, jj], dmwldf=dm, dgml=np.zeros(n), byaxyp=np.ones(n))
    out, info = DF.update_land_fractions(S, THM0)
    assert info['n_shrunk'] == 293 and info['n_expanded'] == 339
    assert np.array_equal(out['w'], wa)
    assert np.array_equal(out['ht'][:, :, :2], ha[:, :, :2])
    assert np.array_equal(out['fr_snow'], b1[:, 70:72])
    # underwater heat: exact where the lake heat input is zero (no lake water used) -- all other rows are lake-side inputs
    no_lake = info['expanded'] & (info['dw_lake'] <= 0)
    assert no_lake.sum() == 0 or np.allclose(out['ht'][no_lake, :, 2], ha[no_lake, :, 2], rtol=1e-6)
