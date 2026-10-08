"""Synthetic stress cases for the AADVQ0 / AADVQ branches that the real 6-step windows never reach (D103-D106),
validated against a STANDALONE ifort build of the REAL QUS3D.f (instrumentation/qus3d_standalone_stubs.f90 + _drv.f90;
only the MPI/domain-decomposition/ATM_COM/GEOM modules are replaced by serial stubs).  THE DATA HERE ARE SYNTHETIC
(real recorded tracer/mass state of dec01 step 33552 with hand-designed flux patterns), NOT model output: this
validates the port's branch logic against the real Fortran source, not the physical behaviour of the model.

Cases (all start from the dec01 33552 mass-unit state RM/RMOM/MB; fluxes are zero except where stated):
  real        the recorded real fluxes of that call (sanity: must reproduce the recorded dump)
  zero        all fluxes zero
  ncyc2       cell (30,20,l=10): zonal outflow 0.9 mass, refilled from below by a vertical flux 0.9 mass:
              horizontal mass ratio fails at ncyc=1, passes at ncyc=2 (hand-derived: ncyc == 2)
  ncycxy2     cell (30,20,l=15): meridional outflow 0.9 mass with zonal inflow 0.3 mass: y-only mass ratio
              (< mrat_limy) fails at ncycxy=1 but the combined horizontal ratio is fine (hand-derived: ncyc == 1,
              ncycxy(15) == 2 and 1 elsewhere)
  nstepx4     row j=25, level 12: uniform eastward flux of 3.7 x mean mass (mass unchanged, Courant 3.7): XSTEP nstep >= 4
  combo       the three patterns above together on top of 0.5 x the real fluxes
  (a hand-built z-extra case was dropped: with ncyc=1 the clipped flux leaves a layer at exactly zero mass and the
   real ZSTEP then never converges -- the Fortran only terminates through int32 wrap-around; z-extra is covered by the
   real stress dumps ffd_qdynS8_*, 57 columns, bitwise)
  real_x3y3z4 real fluxes x(3,3,4): ncyc=3
  err         zonal outflow 0.9 mass with no vertical refill: AADVQ0 stops with 'ncyc>ncmax' (stop_model)
Usage:  python3 dyn_qus3d_harness.py --gen DIR     (writes DIR/in.bin; run the standalone `drv` in DIR; then)
        python3 dyn_qus3d_harness.py --compare DIR
"""
import os
import subprocess
import sys
import numpy as np
import dyn_aadvq_ff as aq
import dyn_qdynam_io as io
from dyn_qdynam_io import IM, JM, LM

FF_H = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data") + "/qus3d_harness"
BASE = ("dec01", 33552)
NAMES = ['real', 'zero', 'ncyc2', 'ncycxy2', 'nstepx4', 'combo', 'real_x3y3z4']
ERR_NAMES = ['err']


def base_state(ff=io.FF_DEFAULT):
    d, it = BASE
    i = io.load_in(io.fname(d, it, 'in', ff))
    ain = io.load_ain(io.fname(d, it, 'ain', ff))
    g = io.load_geom(io.gname(d, ff))
    return i, ain, g


def patterns(mb, mu, mv, mw, which):
    mu = mu.copy(); mv = mv.copy(); mw = mw.copy()
    if 'ncyc2' in which:
        mu[29, 19, 9] = 0.9 * mb[29, 19, 9]; mw[29, 19, 9] = 0.9 * mb[29, 19, 9]
    if 'ncycxy2' in which:
        mv[29, 20, 14] = 0.9 * mb[29, 19, 14]; mu[28, 19, 14] = 0.3 * mb[29, 19, 14]
    if 'nstepx4' in which:
        mu[:, 24, 11] = 3.7 * float(np.mean(mb[:, 24, 11]))
    if 'zextra' in which:
        mw[39, 14, 7] = 1.5 * mb[39, 14, 7]
    if 'err' in which:
        mu[29, 19, 9] = 0.9 * mb[29, 19, 9]
    return mu, mv, mw


def make_case(name, i, ain):
    mb = i['mb']
    z = np.zeros((IM, JM, LM))
    if name == 'real':
        mu, mv, mw = i['mu'], i['mv'], i['mw']
    elif name == 'zero':
        mu, mv, mw = z, z, z
    elif name in ('ncyc2', 'ncycxy2', 'nstepx4', 'zextra', 'err'):
        mu, mv, mw = patterns(mb, z, z, z, [name])
    elif name == 'combo':
        mu, mv, mw = patterns(mb, 0.5 * i['mu'], 0.5 * i['mv'], 0.5 * i['mw'], ['ncyc2', 'ncycxy2', 'nstepx4'])
    elif name == 'real_x3y3z4':
        mu, mv, mw = i['mu'] * 3, i['mv'] * 3, i['mw'] * 4
    else:
        raise KeyError(name)
    return dict(name=name, rm=ain['rm'], rmom=ain['rmom'], mb=mb, mu=mu, mv=mv, mw=mw)


def write_in(path, cases, imaxj):
    with open(path, 'wb') as f:
        f.write(np.array([len(cases)], dtype='>f8').tobytes())
        for k, c in enumerate(cases):
            f.write(np.array([k + 1], dtype='>f8').tobytes())
            for key in ('rm', 'rmom', 'mb', 'mu', 'mv', 'mw'):
                f.write(np.asarray(c[key], dtype='>f8').tobytes(order='F'))
            f.write(np.asarray(imaxj, dtype='>f8').tobytes())


def gen(dirpath, names, ff=io.FF_DEFAULT):
    i, ain, g = base_state(ff)
    os.makedirs(dirpath, exist_ok=True)
    write_in(os.path.join(dirpath, 'in.bin'), [make_case(n, i, ain) for n in names], g['imaxj'])


def run_port(c, g, stats=None):
    q0 = aq.aadvq0(c['mu'], c['mv'], c['mw'], c['mb'], g['imaxj'], g['byim_geom'], stats)
    out = aq.aadvq(c['rm'], c['rmom'], c['mb'], q0, g['imaxj'], g['byim_qus'], stats)
    return q0, out


def compare_case(c, g, dirpath, k, stats=None):
    """Port vs recorded standalone-Fortran outputs q0_<k>.bin / out_<k>.bin.  -> (dict of diffs, q0, out)"""
    q0, out = run_port(c, g, stats)
    fq = io.load_q0(os.path.join(dirpath, f'q0_{k}.bin'))
    fo = io.load_out(os.path.join(dirpath, f'out_{k}.bin'))
    from dyn_qdynam_compare import q0_compare, maxdiff
    d = {'q0_' + a: b for a, b in q0_compare(q0, fq).items()}
    d['rm'] = maxdiff(out['rm'], fo['rm']); d['rmom'] = maxdiff(out['rmom'], fo['rmom']); d['mma'] = maxdiff(out['mma'], fo['mma'])
    for key in ('sbf', 'sbm', 'sfbm', 'scf', 'scm', 'sfcm', 'scf3d'):
        d[key] = maxdiff(out[key], fo[key])
    d['n_exact'] = int(np.sum(out['rm'] == fo['rm']) + np.sum(out['rmom'] == fo['rmom']))
    d['n_total'] = int(fo['rm'].size + fo['rmom'].size)
    return d, q0, out


def available(ff=FF_H):
    return all(os.path.exists(os.path.join(ff, f'out_{k + 1}.bin')) for k in range(len(NAMES))) and \
        os.path.exists(os.path.join(ff, 'in.bin'))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == '--gen':
        gen(sys.argv[2], NAMES if (len(sys.argv) < 4 or sys.argv[3] != 'err') else ERR_NAMES)
    elif len(sys.argv) >= 3 and sys.argv[1] == '--compare':
        i, ain, g = base_state()
        stats = {}
        for k, n in enumerate(NAMES, 1):
            c = make_case(n, i, ain)
            d, q0, out = compare_case(c, g, sys.argv[2], k, stats)
            bad = {a: b for a, b in d.items() if isinstance(b, (int, float)) and a not in ('n_exact', 'n_total') and b != 0}
            print(n, 'ALL ZERO' if not bad else bad, 'exact', d['n_exact'], '/', d['n_total'],
                  '| ncyc', q0['ncyc'], 'ncycxy max', int(q0['ncycxy'].max()), 'nstepx max', int(q0['nstepx'].max()),
                  'z_extra cols', int((q0['nstepz_extra'] > 0).sum()), flush=True)
        print('branch statistics:', dict(sorted(stats.items())))
    else:
        print(__doc__)
