"""D191: check of jax_tpl_state.apply_state against the NumPy path.  For one date at step 0:
  NumPy:  rec = surface_loop.apply_state_to_records(rec0, S1, mid, ...) + ffg columns 143-145 from PREC/EPREC/PRECSS, then jax_surface.build_template(rec)
  device: tpl0 = jax_surface.build_template(rec0) (recorded template, unmodified), then jax_tpl_state.apply_state(tpl0, S1, ...) on the device
compared leaf by leaf, BITWISE, over the VALID slots (invalid slots hold dummy rows by construction).
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 python d191_check_tpl.py DATE
"""
import clouds_jax_env  # noqa: F401
import sys
import copy
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import atm_step as A
import surface_loop as L
import surface_loop_v2 as V2
import jax_surface as JS
import jax_posttile as PT
import jax_tpl_state as TS
import d190_util as U


def leaves(tree, prefix=''):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(leaves(v, f'{prefix}{k}/'))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(leaves(v, f'{prefix}{i}/'))
    else:
        out[prefix[:-1]] = np.asarray(tree)
    return out


def main(date):
    it0 = dict(A.DATES)[date]
    st = L.load_statics(date)
    st['ctx'] = L.make_ocean_ctx(date)
    SS = L.init_surface_state(date, st=st)
    R = A.Real(date, it0)
    S = A._native(A.real_state_at(R, 'surface', None))
    rec0 = A.surface_records(R)
    g1 = rec0['g1']
    irrig = np.zeros((72, 46))
    irrig[g1[:, 0].astype(int) - 1, g1[:, 1].astype(int) - 1] = g1[:, 147]
    inp = dict(prec=np.asarray(S['PREC']), eprec=np.asarray(S['EPREC']), irrig_act=irrig * st['fearth'])
    ice, melt = L.melt_si(SS['ice'], SS['atm']['gtemp'], SS['atm']['sss'], SS['atm']['mlhc'], st['geo'])
    S1, mid = L.surface_pre(SS, st, inp, melt_done=(ice, melt))
    ps = np.asarray(S['PEDN'][0])
    rec, rep = L.apply_state_to_records(rec0, S1, mid, st, ps)
    for key in ('g1', 'g2'):
        g = rec[key]
        i, j = L._rows_ij(g)
        g[:, 143] = np.asarray(S['PREC'])[i, j] / (L.DTSRC * L.RHOW)
        g[:, 144] = np.asarray(S['EPREC'])[i, j] / L.DTSRC
        g[:, 145] = np.asarray(S['PRECSS'])[i, j] / (L.DTSRC * L.RHOW)
    print('tile report', rep)
    tpl_np, host_np = JS.build_template(rec)
    tpl0, host0 = JS.build_template(rec0)
    K, Kb = PT.make_static_all(st, date, it0)
    apply = TS.make_apply_state(host0, K)
    S1d = U.to_dev({g: {k: v for k, v in S1[g].items() if isinstance(v, np.ndarray)} for g in ('ocean', 'ice', 'lake', 'li', 'atm')})
    agd = U.to_dev(mid['ag'])
    tpl_dev = apply(tpl0, S1d, agd, jnp.asarray(ps), jnp.asarray(S['PREC']), jnp.asarray(S['EPREC']), jnp.asarray(S['PRECSS']), None, None)
    la, lb = leaves(tpl_np), leaves(tpl_dev)
    nbad, ntot = 0, 0
    wv = [host0['wvalid'][0], host0['wvalid'][1]]
    Nw = host0['Nw']
    for k in sorted(la):
        a, b = la[k], lb.get(k)
        if b is None:
            print('MISSING in device', k)
            nbad += 1
            continue
        if a.shape != b.shape:
            print('SHAPE', k, a.shape, b.shape)
            nbad += 1
            continue
        if k.split('/')[-1] in ('pw', 'tw'):
            ns = int(k.split('/')[1])
            m = np.concatenate([wv[ns][0], wv[ns][1]])
            a, b = a[m], b[m]
        ntot += 1
        eq = a.tobytes() == b.tobytes() if a.dtype == b.dtype else False
        if not eq:
            nbad += 1
            d = np.abs(a.astype(float) - b.astype(float))
            print('NOT EQUAL', k, a.dtype, b.dtype, 'max abs', np.nanmax(d) if d.size else 0, 'n', int((d > 0).sum()))
    print(f'{date}: {ntot} leaves compared, {nbad} not bitwise equal')
    # non-vacuity: the unmodified recorded template differs from the NumPy-modified one
    l0 = leaves(tpl0)
    nd = sum(1 for k in la if k in l0 and la[k].shape == l0[k].shape and la[k].tobytes() != l0[k].tobytes())
    print('leaves where the recorded template differs from the state-rewritten one (non-vacuity):', nd)
    return nbad


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1]) else 0)
