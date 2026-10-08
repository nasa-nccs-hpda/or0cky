"""D188 (stage S4 step one): the NumPy SURFACE stage (atm_step.stage_surface, land_mode='ghy') run from the REAL entry state of the stage
(atm_step.real_state_at(R,'surface')), saved for C1.  Run: taskset -c 3-5 env OMP_NUM_THREADS=1 python d188_ref_numpy.py DATE OUT.npz
Saves the exit state S fields and the per-substep internals (tile / PBL / land-ice / composite / land outputs) of both substeps."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clouds_jax_env  # noqa
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import atm_step as A

DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}
EXIT = ('T', 'Q', 'U', 'V', 'UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'TSAVG', 'QSAVG',
        'USTARPBL', 'LMONINPBL', 'TMOM', 'QMOM')


def flat(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            flat(prefix + k + '.', v, out)
        elif v is None:
            continue
        else:
            a = np.asarray(v)
            if a.dtype.kind in 'fiub':
                out[prefix + k] = a


def run(date, repeats=2):
    it = DATE_IT0[date]
    R = A.Real(date, it)
    times = []
    for rep in range(repeats):
        S = A.real_state_at(R, 'surface', None)
        A._native(S)
        t0 = time.perf_counter()
        A.stage_surface(S, R, None, land_mode='ghy')
        times.append(time.perf_counter() - t0)
    out = {}
    for k in EXIT:
        out['S.' + k] = np.asarray(S[k])
    sd = S['_surface']
    for r in ('r1', 'r2'):
        x = sd[r]
        for part in ('tile', 'pbl', 'li', 'pbl_li', 'comp'):
            flat(f'{r}.{part}.', x[part], out)
        out[f'{r}.ftype'] = np.asarray(x['ftype'])
        out[f'{r}.blk'] = np.asarray(x['blk'])
        flat(f'{r}.land.patch.', x['land']['patch'], out)
        flat(f'{r}.land.pbl.', x['land']['pbl'], out)
        flat(f'{r}.land.ghy.', x['land']['ghy'], out)
    for e in ('ex1', 'ex2'):
        flat(e + '.', {k: v for k, v in sd[e].items() if k != 'm'}, out)
    out['times'] = np.array(times)
    return out


if __name__ == '__main__':
    date, path = sys.argv[1], sys.argv[2]
    o = run(date)
    np.savez(path, **o)
    print('times', o['times'], 'keys', len(o))
