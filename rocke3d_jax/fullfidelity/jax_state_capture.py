"""D181 helper: run the existing NumPy chained driver (model_driver.ModelDriver, closed surface, record Ent, libimf) for N steps from the nov26 restart and save,
per step k, the dict-world state BEFORE the step (state_<k>.pkl = ModelDriver.state_dict(), with S filled in at k = 0) and the sea-ice fraction at the SURFACE
call of the step (s1_<k>.pkl: ice rsi after MELT_SI + PRECIP_SI, the value that decides which tiles exist).  Used by tests/test_jax_state.py (JAX_STATE_DIR).
Usage: OMP_NUM_THREADS=1 taskset -c 0-1 python jax_state_capture.py OUTDIR [NSTEPS=6]    (first step ~270 s on 2 cores, then 25-50 s per step)"""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402


def main(out, n=6):
    import model_driver as M
    import atm_step as A
    os.makedirs(out, exist_ok=True)
    d = M.ModelDriver(M.RecordProvider(), surface='closed', ent='record', rng='chain', daily='recorded', imf=True)
    for k in range(n):
        sd = d.state_dict()
        if sd['S'] is None:
            sd['S'] = M._np(A._native(A.init_state(d.provider.real(d.itime))))
        pickle.dump(sd, open(f'{out}/state_{k}.pkl', 'wb'), protocol=4)
        t = time.time()
        d.step()
        S1 = d.loop.last['S1']
        pickle.dump(dict(rsi=np.array(S1['ice']['rsi']), ice=M._np(S1['ice']), lake=M._np(S1['lake'])), open(f'{out}/s1_{k}.pkl', 'wb'), protocol=4)
        print('step', k, round(time.time() - t, 1), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
