"""GPU assembled step with the radiation COMPUTED by the real Fortran persistent server (not replayed): the cost of a real radiation cycle on the device. (D214)

usage: python -u gpu_server_step_run.py [NSTEPS]        (default 16: steps 0-15 include the radiation steps 0, 5, 10 and 15 of NRAD = 5; fewer than 48: no day boundary hooks)
Same machinery as d210_day.py (server mode of Coupled: packet from OUR state, ordered io_callback to the persistent server, held fluxes on the other steps), run on the
default JAX backend (the GPU).  libm mode, hybrid.  Prints per step: wall seconds, radiation step or not, non-finite counts; and the server request seconds.  The server child
is pinned to the last two usable CPUs.  No comparison with recorded packets (see D210/D212 for the CPU day).
Labels: radiation computed by the original Fortran (hybrid component, unmodified RADIA/SOCRATES in the persistent server); everything else as gpu_step_run.py.
"""
import json
import os
import sys
import time

_A = list(sys.argv)
sys.argv = [_A[0], 'nov26', os.environ.get('GPU_IMPORT_SCRATCH', '/explore/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/gpu_runs/_import_scratch')]
import gpu_step_run as G   # noqa: E402  (backend flags; parses argv at import)
sys.argv = _A
import numpy as np   # noqa: E402
import jax   # noqa: E402

import atm_step as A   # noqa: E402
import radiation_server as rs   # noqa: E402
import radiation_server_persist as RP   # noqa: E402

C = G.C
NSTEPS = int(_A[1]) if len(_A) > 1 else 16


class DayReal(A.Real):
    """records of every step from ff_data/nov26_day (the 54-step record set), as d193_day.py"""

    def __init__(self, date, itime, ff=A.FF):
        super().__init__('nov26_day', itime, ff)


A.Real = DayReal            # patched before Coupled / Phase1 read their records (the day boundary hooks of d193_day are NOT installed: run fewer than 48 steps)
RAD = dict(wrap=None, lv=None, calls=[])


class Wrap:
    def __init__(self, srv):
        self.srv = srv
        self.binary, self.restart, self.rundir = srv.binary, getattr(srv, 'restart', None), srv.rundir

    def request(self, state, itime, seed=None):
        t0 = time.perf_counter()
        out = self.srv.request(state, itime, seed)
        RAD['calls'].append(dict(itime=int(itime), server_s=float(out['_server_s']), wall_s=float(out['_wall_s']), request_s=time.perf_counter() - t0))
        return out


def _child():
    import ctypes
    import signal
    try:
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:
        pass
    try:
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(cpus[-2:]))
    except Exception:
        pass


class CoupledSrv(C.Coupled):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        RP._pdeathsig = _child
        srv = RP.PersistentServer('nov26').start()
        RAD['wrap'] = Wrap(srv)
        RAD['lv'] = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{self.it0}_in.bin')
        ph = self.ph
        ph.rad_mode, ph.server, ph.live_packet = 'server', RAD['wrap'], RAD['lv']
        for att in ('_handoff', '_hstep'):
            if hasattr(ph, att):
                delattr(ph, att)

    def step(self, k, state, *a, **kw):
        it = self.it0 + k
        if A.is_radiation_step(it):
            self.ph.live_packet = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it}_in.bin')
        kw['server'], kw['lv'] = RAD['wrap'], RAD['lv']
        return super().step(k, state, *a, **kw)


t_all = time.perf_counter()
cp = CoupledSrv('nov26')
print(f'setup {cp.setup_seconds:.1f} s (server started); backend {jax.default_backend()}; usable cpus {len(os.sched_getaffinity(0))}', flush=True)
sr = G.build_registry()
state, rec, dev = cp.initial_state()
rows = []
for k in range(NSTEPS):
    if k > 0:
        rec = cp.ph.load_records(cp.it0 + k, first=False)
        dev = cp.ph.to_device(rec, first=False)
    n0 = len(RAD['calls'])
    out, state, arrays, sr = G.run_step(cp, k, state, rec, dev, sr, False)
    ea = G.end_arrays(arrays)
    nf = {kk: int((~np.isfinite(ea[kk])).sum()) for kk in ('filter/T', 'filter/U', 'filter/V', 'filter/Q', 'filter/P') if kk in ea}
    rad = len(RAD['calls']) > n0
    rows.append(dict(k=k, wall=out['step_seconds'], radiation_step=rad, request_s=(RAD['calls'][-1]['request_s'] if rad else 0.0), nonfinite=sum(nf.values())))
    print(f'[step {k}] {"RADIATION step" if rad else "held step     "} wall {out["step_seconds"]:8.2f} s   server request {(RAD["calls"][-1]["request_s"] if rad else 0.0):6.2f} s   non-finite {sum(nf.values())}', flush=True)
steady = [r for r in rows[2:] ]
nrad = [r['wall'] for r in steady if r['radiation_step']]
held = [r['wall'] for r in steady if not r['radiation_step']]
print('\nsummary (steps >= 2): held steps mean %.2f s (n=%d); radiation steps mean %s s (n=%d)' % (np.mean(held) if held else float('nan'), len(held), ('%.2f' % np.mean(nrad)) if nrad else 'n/a', len(nrad)))
if nrad and held:
    print('cycle average over NRAD=5 steps (4 held + 1 radiation): %.2f s per step' % ((4 * np.mean(held) + np.mean(nrad)) / 5))
print('LABEL: radiation computed by the original Fortran (hybrid component; real RADIA/SOCRATES in the persistent server); libm mode; hybrid step; V100/A100 timing, not bitwise to the CPU')
print('server calls:', json.dumps(RAD['calls']))
