"""D202 driver: d201_day.py (fix switch) plus a capture of the full land-stage INPUTS and outputs of every step >= --capfrom:
rows1/rows2 (PBL record rows of the land slots after the atmosphere overrides, incl. ddml/gusti/tkv/utop/vtop/...), the GHY batch (static, dyn0, forcing, edts, nsub, ...)
of both substeps, trup, ma1_land, q1 and the land results r1/r2.  New file only.
    taskset -c 0-2,6-7 env OMP_NUM_THREADS=1 TMPDIR=S D189_SHIM_DIR=S python d202_day.py OUT --fix 0|1 --cap DIR --capfrom 20 --nsteps N --c1dir DUMMY --nit-strict 0"""
import sys
import numpy as np
fix_arg = None
capfrom = 20
if '--capfrom' in sys.argv:
    k = sys.argv.index('--capfrom'); capfrom = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
cap = sys.argv[sys.argv.index('--cap') + 1] if '--cap' in sys.argv else None
gustifix = 0
if '--gustifix' in sys.argv:
    k = sys.argv.index('--gustifix'); gustifix = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
gdtmfix = 0
if '--gdtmfix' in sys.argv:
    k = sys.argv.index('--gdtmfix'); gdtmfix = int(sys.argv[k + 1]); del sys.argv[k:k + 2]
import d201_day as D1          # parses --fix/--cap/--nit-strict, installs the elhx switch and its allarr_of
import jax_surface as JS
import jax_coupled as C
import d187_common as K
import os

_orig_make_stage = JS.make_stage
_N = [0]


def _make_stage(host):
    stage = _orig_make_stage(host)

    def wrapped(Sd, tpl_all, timers=None):
        out, aux = stage(Sd, tpl_all, timers)
        k = _N[0]; _N[0] += 1
        if cap and k >= capfrom:
            os.makedirs(cap, exist_ok=True)
            d = {}
            d.update(K.flatten(C.tree_np(dict(rows1=aux['rows1']['pe'], rows2=aux['rows2']['pe'])), 'rows/'))
            d.update(K.flatten(C.tree_np(dict(g0=tpl_all['ghy'][0], g1=tpl_all['ghy'][1])), 'ghy/'))
            d.update(K.flatten(C.tree_np(dict(trup=tpl_all['trup'], ma1=tpl_all['ma1_land'])), 'misc/'))
            d.update(K.flatten(C.tree_np(dict(r1=aux['r1']['land'], r2=aux['r2']['land'])), 'res/'))
            d['misc/q1_ns1'] = np.asarray(Sd['Q']).transpose(1, 0, 2)[host['ej'], host['ei'], 0]     # atm_layout: (JM,IM,L)
            d['misc/q1_ns2'] = np.asarray(aux['ex1']['Q'])[host['ej'], host['ei'], 0]
            d['misc/q1'] = d['misc/q1_ns1']
            d['host/ej'] = np.asarray(host['ej']); d['host/ei'] = np.asarray(host['ei'])
            np.savez(os.path.join(cap, f'in_{k}.npz'), **d)
        return out, aux
    wrapped.units = stage.units
    return wrapped


JS.make_stage = _make_stage
if gustifix or gdtmfix:
    import d202_fix
    if gustifix:
        d202_fix.install_gusti()
    if gdtmfix:
        d202_fix.install(width=40)
if __name__ == '__main__':
    D1.C.Coupled = D1.functools.partial(D1.C.Coupled, nit_strict=bool(D1.strict))
    D1.D.main()
