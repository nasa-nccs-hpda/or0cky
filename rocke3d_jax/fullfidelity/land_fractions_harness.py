"""D208: differential test of daily_land_fractions.py against the REAL Fortran update_land_fractions (GHY_DRV.f:4367-4645 and get_fb_fv 4881-4896 extracted verbatim,
compiled with `ifort -O2 -ftz -convert big_endian -assume protect_parens -fp-model strict -fpp`, stubs in instrumentation/land_fractions_standalone_stubs.f90).
    python land_fractions_harness.py build DIR        # compile (sources env_modele.sh; ifort 19.1.3)
    python land_fractions_harness.py run DIR CASE.npz # write in.bin, run ./drv, compare bitwise with daily_land_fractions
CASE.npz keys: w ht (N,7,3), fr_snow (N,2), dz (N,6), q (N,5,6), fv (N,) [Ent fv, raw], fearth flake svflake focean dmwldf dgml byaxyp (N,), thm0 (4,)."""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import daily_land_fractions as DF  # noqa: E402

MODEL = os.environ.get('MODELE_SRC', '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model')


def build(d):
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(MODEL, 'GHY_DRV.f')).read().split('\n')
    open(os.path.join(d, 'land_fractions_ext.f'), 'w').write('\n'.join(src[4366:4645] + src[4880:4896]) + '\n')   # update_land_fractions 4367-4645, get_fb_fv 4881-4896
    ins = os.path.join(HERE, 'instrumentation')
    F = '-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict'
    cmd = (f'source {HERE}/env_modele.sh >/dev/null 2>&1; cd {d}; '
           f'ifort {F} -c {ins}/land_fractions_standalone_stubs.f90 && ifort {F} -fpp -c land_fractions_ext.f && '
           f'ifort {F} -c {ins}/land_fractions_standalone_drv.f90 && ifort {F} *.o -o drv')
    subprocess.run(['bash', '-c', cmd], check=True)


def write_in(d, c):
    n = np.asarray(c['w']).shape[0]
    f = lambda a: np.asarray(a, dtype='>f8').ravel(order='F').tobytes()    # noqa: E731
    parts = [f(np.array([float(n)])), f(c['thm0'])]
    parts.append(f(np.transpose(c['w'], (1, 2, 0))))                       # w_ij(0:6,3,n)
    parts.append(f(np.transpose(c['ht'], (1, 2, 0))))
    parts.append(f(np.transpose(c['fr_snow'], (1, 0))))                    # fr_snow_ij(2,n)
    parts.append(f(np.transpose(c['dz'], (0, 1))))                         # dz_ij(n,1,6): (n,6) Fortran order
    parts.append(f(np.transpose(c['q'], (0, 1, 2))))                       # q_ij(n,1,5,6): (n,5,6)
    for k in ('fv', 'fearth', 'flake', 'svflake', 'focean', 'dmwldf', 'dgml', 'byaxyp'):
        parts.append(f(c[k]))
    open(os.path.join(d, 'in.bin'), 'wb').write(b''.join(parts))


def read_out(d, n):
    buf = np.fromfile(os.path.join(d, 'out.bin'), dtype='>f8')
    nw = 7 * 3 * n
    w = buf[:nw].reshape((7, 3, n), order='F')
    ht = buf[nw:2 * nw].reshape((7, 3, n), order='F')
    fs = buf[2 * nw:].reshape((2, n), order='F')
    assert buf.size == 2 * nw + 2 * n
    return dict(w=np.transpose(w, (2, 0, 1)), ht=np.transpose(ht, (2, 0, 1)), fr_snow=fs.T)


def case_to_S(c):
    S = {k: np.asarray(c[k], dtype=np.float64) for k in ('w', 'ht', 'fr_snow', 'dz', 'q', 'fearth', 'flake', 'svflake', 'focean', 'dmwldf', 'dgml', 'byaxyp')}
    S['fv'] = DF.fv_from_ent(c['fv'])
    return S


def compare(d, c):
    n = np.asarray(c['w']).shape[0]
    write_in(d, c)
    subprocess.run(['./drv'], cwd=d, check=True, capture_output=True)
    fo = read_out(d, n)
    po, info = DF.update_land_fractions(case_to_S(c), c['thm0'])
    rep = {k: (float(np.abs(po[k] - fo[k]).max()), int((po[k] != fo[k]).sum()), int(po[k].size)) for k in ('w', 'ht', 'fr_snow')}
    return rep, po, fo, info


if __name__ == '__main__':
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == 'build':
        build(d)
    else:
        rep, _, _, info = compare(d, dict(np.load(sys.argv[3])))
        print(f"rows shrunk {info['n_shrunk']} expanded {info['n_expanded']}")
        for k, (mx, nd, n) in rep.items():
            print(f'{k:8s} max|port-fortran| {mx:.3e}  differing elements {nd} / {n}')
