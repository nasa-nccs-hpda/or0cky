"""Speed track (branch speed/step-time): host-compiled MSTCNV for the JAX run.  Flag `ROCKE_MSTCNV_HOST=1` (default off).

The numba port (clouds_mstcnv_nb, bitwise equal to the NumPy port on all 3,168 columns of the captured nov26 step-1
inputs) cannot be imported in the JAX environment (graphcast-env: NumPy 2.2.4, numba 0.58.1 refuses it), so it runs in
a persistent worker process under a numba-capable interpreter (`ROCKE_MSTCNV_PY`, default the DEV-Kernel env:
numba 0.60 + NumPy 2.0.2).  Inputs and outputs travel through one memory-mapped file in /dev/shm (no pickling); the
control channel is the worker's stdin/stdout.

Device side only sees: R (device arrays of CondseDevice.setup) -> numpy -> shared file -> worker -> shared file ->
jnp arrays shaped like clouds_mstcnv_jax._post's output dict.  The returned negative-cloud flag is a jnp bool.
"""
import atexit
import os
import subprocess
import sys

import numpy as np

LM = 40
NMOM = 9
DEFAULT_PY = os.environ.get('ROCKE_MSTCNV_PY', '/home/gtamkin/.conda/envs/DEV-Kernel/bin/python')
IN_L = 'pl plk airm byam etal tl tvl sm qm qcll qcil sdl wturb gzl'.split()
IN_M = 'smom qmom'.split()
IN_K = 'um vm u0 v0'.split()
OUT_SC = ('ierr lerr lmcmin lmcmax prcpmc cldslwij clddepij airxl prheat wmsum wmctwp wmclwp fmc1 mccont '
          'negcover').split()
INT_SC = ('ierr', 'lerr', 'lmcmin', 'lmcmax', 'mccont')
OUT_L = ('tl sm qm fssl cldmcl taumcl svlatl svlat1 svwmxl csizel condpt vsubl tpsav mcflx dgdsm dgdeep dgshlw '
         'dphase dphadeep dphashlw dtotw dqcond dgdqm dqmtotal dqmshlw dqmdeep dqctotal dqcshlw dqcdeep ddmflx '
         'tdnl qdnl u00l qlmc qimc cnvmmrl condmmr').split()
TUNE_NAMES = ('entrainment_cont1 entrainment_cont2 radiusl_multiplier radiusi_multiplier u00a u00b wmu_multiplier '
              'rwcldox rimax mc_fddrt mc_entr_mass_lim_plume mc_new_ddrft_thetav mc_revp_abv_cldbase').split()


def layout(N, nks):
    """[(name, dtype, shape)] of the shared file, inputs then outputs."""
    f8, i8 = np.float64, np.int64
    L = [('pearth', f8, (N,)), ('pland', f8, (N,)), ('dcl', i8, (N,)), ('ple', f8, (LM + 1, N)),
         ('ra', f8, (nks, N)), ('scal', f8, (5,)), ('tune', f8, (13,))]
    L += [(n, f8, (LM, N)) for n in IN_L]
    L += [(n, f8, (NMOM, LM, N)) for n in IN_M]
    L += [(n, f8, (nks, LM, N)) for n in IN_K]
    L += [('o_sc', f8, (15, N)), ('o_l', f8, (37, N, LM)), ('o_l1', f8, (2, N, LM + 1)),
          ('o_m', f8, (2, N, NMOM, LM)), ('o_k', f8, (2, N, nks, LM))]
    return L


def views(path, N, nks, create):
    spec = layout(N, nks)
    offs, o = {}, 0
    for n, dt, sh in spec:
        offs[n] = o
        o += (int(np.prod(sh)) * np.dtype(dt).itemsize + 63) // 64 * 64
    mm = np.memmap(path, dtype=np.uint8, mode='w+' if create else 'r+', shape=(o,))
    out = {}
    for n, dt, sh in spec:
        cnt = int(np.prod(sh))
        out[n] = np.frombuffer(mm, dtype=dt, count=cnt, offset=offs[n]).reshape(sh)
    return mm, out


class MstcnvHost:
    def __init__(self, N, nks=4, py=DEFAULT_PY, threads=None):
        self.N, self.nks = N, nks
        self.path = f'/dev/shm/rocke_mstcnv_{os.getpid()}_{N}.bin'
        self.mm, self.v = views(self.path, N, nks, True)
        env = dict(os.environ)
        # the worker interpreter must not inherit the JAX environment's site-packages
        for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
            env.pop(k, None)
        env['PYTHONNOUSERSITE'] = '1'
        threads = threads or int(os.environ.get('MSTCNV_HOST_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', 0))) or None
        if threads:
            env['NUMBA_NUM_THREADS'] = str(threads)
        here = os.path.dirname(os.path.abspath(__file__))
        self.p = subprocess.Popen([py, os.path.join(here, 'mstcnv_host.py'), '--worker', self.path, str(N), str(nks),
                                   here], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
        self._expect(b'ready')
        atexit.register(self.close)

    def _expect(self, tag):
        line = self.p.stdout.readline().strip()
        if line != tag:
            raise RuntimeError(f'mstcnv host worker: expected {tag!r}, got {line!r}')

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write(b'quit\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
        except Exception:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def run(self, R, cfg):
        """R: dict of arrays (device or host) as CondseDevice.setup returns; cfg: CondseDevice.cfg.  -> dict of numpy."""
        v = self.v
        for n in ('pearth', 'pland', 'dcl', 'ple', 'ra') + tuple(IN_L) + tuple(IN_M) + tuple(IN_K):
            np.copyto(v[n], np.asarray(R[n]))
        c = cfg['tune']
        v['tune'][:] = [float(c[k]) for k in TUNE_NAMES]
        v['scal'][:] = [float(cfg['lmcm']), float(cfg['xmass']), float(cfg['bydtsrc']), float(cfg['dtsrc']),
                        float(cfg['bybr'])]
        self._go()
        return v

    def _go(self):
        self.p.stdin.write(b'go\n')
        self.p.stdin.flush()
        self._expect(b'done')


_COL = {}


def column_mstcnv(r, c, ck=None, br=None, mut=None):
    """Drop-in for clouds_mstcnv_ff.mstcnv_column (no checkpoints / mutations): one column through a one-column
    worker (N=1, one thread).  Used by the pole columns of the host CONDSE path (`ROCKE_MSTCNV_HOST=1`)."""
    assert ck is None and not mut
    ra = np.asarray(r['ra'], dtype=np.float64)
    h = _COL.get(ra.shape[0])
    if h is None:
        h = _COL[ra.shape[0]] = MstcnvHost(1, nks=ra.shape[0], threads=1)
    v = h.v
    v['pearth'][0], v['pland'][0], v['dcl'][0] = float(r['pearth']), float(r['pland']), int(r['dcl'])
    v['ple'][:, 0] = np.asarray(r['ple'], dtype=np.float64)
    v['ra'][:, 0] = ra
    for n in IN_L:
        v[n][:, 0] = np.asarray(r[n], dtype=np.float64)
    for n in IN_M:
        v[n][:, :, 0] = np.asarray(r[n], dtype=np.float64)
    for n in IN_K:
        v[n][:, :, 0] = np.asarray(r[n], dtype=np.float64)
    v['tune'][:] = [float(c[k]) for k in TUNE_NAMES]
    v['scal'][:] = [float(r['lmcm']), float(r['xmass']), float(r['bydtsrc']), float(r['dtsrc']), float(r['bybr'])]
    h._go()
    o = {}
    for i, n in enumerate(OUT_SC[:14]):
        x = float(v['o_sc'][i, 0])
        o[n] = int(x) if n in INT_SC else x
    if v['o_sc'][14, 0]:
        raise RuntimeError('MSTCNV: negative cloud cover')
    for i, n in enumerate(OUT_L):
        o[n] = np.array(v['o_l'][i, 0])
    o['lhp'], o['precnvl'] = np.array(v['o_l1'][0, 0]), np.array(v['o_l1'][1, 0])
    o['smom'], o['qmom'] = np.array(v['o_m'][0, 0]), np.array(v['o_m'][1, 0])
    o['um'], o['vm'] = np.array(v['o_k'][0, 0]), np.array(v['o_k'][1, 0])
    return o


def worker(path, N, nks, here):
    sys.path.insert(0, here)
    import clouds_mstcnv_nb as nb
    mm, v = views(path, N, nks, False)
    # warm-up on the current (zero) inputs would divide by zero in every column; compile on a benign column set
    out = sys.stdout
    out.write('ready\n')
    out.flush()
    first = True
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == 'quit':
            break
        if cmd == 'go':
            nb.mstcnv_batch(v['pearth'], v['pland'], v['dcl'], v['pl'], v['ple'], v['plk'], v['airm'], v['byam'],
                            v['etal'], v['tl'], v['tvl'], v['sm'], v['qm'], v['qcll'], v['qcil'], v['sdl'],
                            v['wturb'], v['gzl'], v['smom'], v['qmom'], v['ra'], v['um'], v['vm'], v['u0'], v['v0'],
                            v['scal'], v['tune'], v['o_sc'], v['o_l'], v['o_l1'], v['o_m'], v['o_k'])
            out.write('done\n')
            out.flush()


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--worker':
    worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
