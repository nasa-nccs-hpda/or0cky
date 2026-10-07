"""D175: parallel drop-in variants reproduce the existing stages BITWISE (2 worker processes each).  Skipped when data are absent."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DAY = f"{FF}/nov26_day"
HAVE = all(os.path.exists(p) for p in (f"{FF}/_pristine_restarts/fort1_nov26_itime33312.nc", f"{DAY}/ffg_33312.bin", f"{DAY}/ffg_33313.bin",
                                      f"{DAY}/ffp_33312.bin", f"{DAY}/ffp_33313.bin"))
need_data = pytest.mark.skipif(not HAVE, reason="nov26_day dumps / restart not available")


@need_data
def test_land_ent_par_bitwise():
    import land_chain_ent as LE
    import land_ent_par as LP
    import ghy_compare as GC
    import pbl_compare as PC
    e1 = LE.EntLand(LE.RESTART['nov26_day']); e2 = LP.ParEntLand(LE.RESTART['nov26_day'], 2)
    d1 = d2 = None
    try:
        for it in (33312, 33313):
            p = PC.load(f"{DAY}/ffp_{it}.bin"); g = GC.load(f"{DAY}/ffg_{it}.bin"); n = len(p) // 2; ng = len(g) // 2
            e1.begin_step(it); e2.begin_step(it)
            for s in (0, 1):
                ps = p[s * n:(s + 1) * n]; p4 = ps[ps[:, 2] == 4]; gg = g[s * ng:(s + 1) * ng]
                q1 = np.full(len(gg), 0.004); trup = np.full(len(gg), 300.0)
                o1 = LE.land_substep_ent(p4, gg, q1, trup, 900.0, d1, e1)
                o2 = LP.land_substep_ent_par(p4, gg, q1, trup, 900.0, d2, e2)
                for k in o1['ghy']:
                    assert np.array_equal(o1['ghy'][k], o2['ghy'][k]), k
                for k in o1['patch']:
                    assert np.array_equal(np.asarray(o1['patch'][k]), np.asarray(o2['patch'][k])), k
                d1, d2 = o1['dyn_next'], o2['dyn_next']
            e1.end_step(); e2.end_step()
    finally:
        e2.close()


@need_data
def test_condse_par_bitwise():
    import clouds_condse_batch as cb
    import clouds_condse_par as cp
    import clouds_condse_ff as cf
    import clouds_condse_io as cio
    import intel_libm_ff
    backend = 'imf' if intel_libm_ff.available() else 'numpy'
    try:
        it0 = dict(cio.DATES)['nov26']
        inp, _ = cio.load_step('nov26', it0 + 1)
    except Exception:
        pytest.skip("CONDSE step dump not available")
    cf.set_backend(backend)
    cfg = cf.make_cfg('nov26')
    X1, _ = cb.condse_step_batch(inp, cfg, ms={})
    X2, _ = cp.condse_step_batch_par(inp, cfg, ms={}, nproc=2, backend=backend)
    cp.close_pools()
    for k in X1:
        assert np.array_equal(np.asarray(X1[k]), np.asarray(X2[k]), equal_nan=True), k
