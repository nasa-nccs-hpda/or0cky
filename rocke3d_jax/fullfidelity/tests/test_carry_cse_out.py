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
