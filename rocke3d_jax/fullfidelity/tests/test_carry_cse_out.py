"""D221: ROCKE_CARRY_CSE_OUT chains the eight CONDSE outputs that the next entry record repeats; default is unchanged."""
import os
import subprocess
import sys

import numpy as np
import pytest

KEYS = "PREC EPREC PRECSS DDM1 DDML DDMS TDN1 QDN1".split()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _carry(flag):
    env = dict(os.environ, ROCKE_CARRY_CSE_OUT=flag, JAX_PLATFORMS='cpu')
    code = "import atm_step as A; print(' '.join(A.CARRY_KEYS))"
    out = subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env, capture_output=True, text=True, check=True).stdout.split()
    return set(out)


def test_default_off_and_flag_on():
    off, on = _carry('0'), _carry('1')
    assert not (set(KEYS) & off)
    assert set(KEYS) <= on and off <= on and len(on) == len(off) + len(KEYS)


def test_record_repeats_exit_bitwise():
    import clouds_condse_io as cio
    d = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day2'
    f = [f'{d}/ffc_cse_out_33312.bin', f'{d}/ffc_cse_in_33313.bin']
    if not all(os.path.exists(p) for p in f):
        pytest.skip('records not available')
    co, ci = cio.read_cse(f[0]), cio.read_cse(f[1])
    for k in KEYS:
        assert np.array_equal(np.asarray(co[k]), np.asarray(ci[k])), k


def test_rndss_vectorised_equals_randu_stream():
    import clouds_condse_ff as cf
    import drv_rng
    imaxj = [1] + [3 + (j % 5) for j in range(1, 45)] + [1]
    for ix0 in (0, 12345, 0x80000001, 0xFFFFFFFF, -588724193):
        ref, seed = cf.randu_stream(ix0, imaxj, 5)
        got = drv_rng.rndss(ix0, imaxj, 5)
        assert np.array_equal(ref, got), ix0


def test_rndss_matches_record():
    import clouds_condse_io as cio
    import clouds_condse_ff as cf
    import drv_rng
    d = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day2'
    p = f'{d}/ffc_cse_in_33330.bin'
    if not os.path.exists(p):
        pytest.skip('records not available')
    ci = cio.read_cse(p)
    cfg_geom = cio.load_geom('nov26')
    got = drv_rng.rndss(int(round(float(ci['SEEDS'][0]))), cfg_geom['IMAXJ'], cf.LMCLD_RUN)
    rec = np.asarray(ci['RNDSS'])
    mask = np.zeros(rec.shape, bool)               # outside i < IMAXJ(j), L <= LMCLD the record holds uninitialised Fortran memory (1e40)
    for j in range(46):
        mask[:, :cf.LMCLD_RUN, :int(cfg_geom['IMAXJ'][j]), j] = True
    assert np.array_equal(got[mask], rec[mask])
