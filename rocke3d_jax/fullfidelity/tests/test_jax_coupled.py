"""D191 quick tests of the pieces of the assembled step that are new (jax_tpl_state; jax_phase2 has its own test file).  Real nov26 step-0 data; own process, >= 2 cores:
  taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python -m pytest tests/test_jax_coupled.py -q     (from fullfidelity/; about 30 s)
The whole step is validated by d191_run.py (C1/C2), not here."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import clouds_jax_env  # noqa: E402,F401

pytestmark = pytest.mark.skipif(not os.path.exists('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26/ffa_step_33312_a.bin'),
                                reason='real step-0 data absent')


def test_device_template_state_rewrite_is_bitwise_numpy_apply_state():
    import d191_check_tpl as T
    assert T.main('nov26') == 0


def test_mutation_template_rewrite_changes_result():
    """the rewrite must change the recorded template (non-vacuity) and a perturbed surface state must change the device result."""
    import jax.numpy as jnp
    import atm_step as A
    import surface_loop as L
    import jax_surface as JS
    import jax_posttile as PT
    import jax_tpl_state as TS
    import d190_util as U
    date, it0 = 'nov26', 33312
    st = L.load_statics(date)
    st['ctx'] = L.make_ocean_ctx(date)
    SS = L.init_surface_state(date, st=st)
    R = A.Real(date, it0)
    S = A._native(A.real_state_at(R, 'surface', None))
    rec0 = A.surface_records(R)
    tpl, host = JS.build_template(rec0)
    K, _ = PT.make_static_all(st, date, it0)
    apply = TS.make_apply_state(host, K)
    ice, melt = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], st['geo'])
    inp = dict(prec=np.asarray(S['PREC']), eprec=np.asarray(S['EPREC']), irrig_act=np.zeros((72, 46)))
    S1, mid = L.surface_pre(SS, st, inp, melt_done=(ice, melt))
    sd = U.to_dev({g: {k: v for k, v in S1[g].items() if isinstance(v, np.ndarray)} for g in ('ocean', 'ice', 'lake', 'li', 'atm')})
    ag = U.to_dev(mid['ag'])
    args = (jnp.asarray(S['PEDN'][0]), jnp.asarray(S['PREC']), jnp.asarray(S['EPREC']), jnp.asarray(S['PRECSS']), None, None)
    a = apply(tpl, sd, ag, *args)
    sd2 = dict(sd, atm=dict(sd['atm'], sss=sd['atm']['sss'] * (1 + 1e-9)))
    b = apply(tpl, sd2, ag, *args)
    assert not np.array_equal(np.asarray(a['ns'][0]['tw']), np.asarray(tpl['ns'][0]['tw']))
    assert not np.array_equal(np.asarray(a['ns'][0]['tw']), np.asarray(b['ns'][0]['tw']))
