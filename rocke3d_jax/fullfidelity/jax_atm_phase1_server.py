"""D186: ONE variant of phase 1 (nov26 step 33312, a radiation step) with the REAL persistent Fortran radiation server through the D185 hand-off.
Run: RADSRVP_SCRATCH=<dir with mE2/model/P2SAoM40.bin> taskset -c 0-1 env OMP_NUM_THREADS=1 python jax_atm_phase1_server.py OUTDIR RUNDIR REFDIR
Radiation computed by the original Fortran (hybrid component).  The packet is built from OUR device state (T,Q after our CONDSE, PK, PMID, PDSIG, PEDN, MA, LTROPO
from our TROP, the 19 cloud arrays from our CONDSE); RQT, KLIQ and the 21 surface fields are COPIED from the live recorded packet rsv_n26_33312_in.bin (declared
recorded input, D185 practice); SNOAGE from our CONDSE.  Compared with (a) the replay-mode result of the same program (isolates server vs record), (b) the recorded
server output rsv_n26_33312_out.bin (SRHR, TRHR, COSZ1: differ where our libm-mode CONDSE state differs from the real one)."""
import clouds_jax_env_fast  # noqa: F401
import json
import os
import sys
import time

import numpy as np

import jax_atm_phase1 as P1
import jax_harness as H
import jax
import jax.numpy as jnp
import radiation_server as rs
import radiation_server_persist as RP
import jax_radiation as JR


def main(outdir, rundir, refdir):
    os.makedirs(outdir, exist_ok=True)
    date, it0 = 'nov26', 33312
    lv = rs.read_packet(f"{rs.FF}/nov26_day/rsv_n26_{it0}_in.bin")
    rv = rs.read_packet(f"{rs.FF}/nov26_day/rsv_n26_{it0}_out.bin")
    res = {}
    with RP.PersistentServer("nov26", rundir=rundir) as srv:
        res['server_start_s'] = srv.start_wall
        ph = P1.Phase1(date, rad='server', server=srv, live_packet=lv)
        ph.reg.read('live_radiation_packet', lambda: lv, stage='record_load')
        rec = ph.load_records(it0, first=True)
        dev = ph.to_device(rec, first=True)
        hold = ph.new_hold(dev['rad'])
        # RQT / KLIQ of the carry come from the live packet (hold carry), SNOAGE from our CONDSE is taken inside the packet via hold: set from live
        for k in ('RQT', 'KLIQ'):
            hold[k] = jnp.asarray(lv[k])
        hold['SNOAGE'] = jnp.asarray(lv['SNOAGE'])
        t0 = time.perf_counter()
        r = ph.step(it0, dev, dev['S'], {}, P1.CS.ms_zero(), hold, dev['ice'], timed=False, rec=rec)
        jax.block_until_ready((r['S'], r['X']))
        res['wall_s'] = time.perf_counter() - t0
        log = ph._handoff.log
        res['radiation_log'] = log.totals()
        res['sentence'] = log.sentence()
        out = ph._handoff.last_out
    ref = H.load_npz(os.path.join(refdir, f'{date}_p1_1.npz'))
    S = {k: np.asarray(v) for k, v in r['S'].items() if hasattr(v, 'shape')}
    cmp_rec = {k: H.field_category(np.asarray(out[k]), rv[k]) for k in ('SRHR', 'TRHR', 'COSZ1')}
    res['server_out_vs_recorded_server_out'] = {k: {kk: v[kk] for kk in ('cat', 'max_abs', 'rel', 'n_diff')} for k, v in cmp_rec.items()}
    cm = r['cloud_masked']
    res['server_CLDSS_CLDMC_vs_device_masking'] = {k: H.field_category(np.asarray(out[k]), np.asarray(cm[k])) for k in ('CLDSS', 'CLDMC')}
    res['server_CLDSS_CLDMC_vs_device_masking'] = {k: {kk: v[kk] for kk in ('cat', 'max_abs', 'n_diff')} for k, v in res['server_CLDSS_CLDMC_vs_device_masking'].items()}
    # T after RADIA with the server's SRHR/TRHR vs the NumPy-chain radia stage (recorded SRHR/TRHR): the difference is what the server (acting on our state) adds
    res['T_after_radia_vs_numpy_libm_reference_radia_T'] = {kk: v for kk, v in H.field_category(S['T'], ref['radia/T']).items() if kk in ('cat', 'max_abs', 'rel', 'n_diff')}
    res['Q_after_radia_vs_reference'] = {kk: v for kk, v in H.field_category(S['Q'], ref['radia/Q']).items() if kk in ('cat', 'max_abs', 'rel', 'n_diff')}
    # where do the differences sit?  columns (i,j) with |SRHR diff| or |TRHR diff| > 1e-6 of scale, versus the columns where OUR CONDSE state differs from the REAL CONDSE exit record
    import atm_step as A
    cout = A.Real(date, it0).cse_out
    def cols(a, b, rel):
        d = np.abs(np.asarray(a) - np.asarray(b))
        sc = np.abs(b).max()
        m = d > rel * sc
        ax = tuple(i for i, n in enumerate(d.shape) if n not in (72, 46))
        m = m.any(axis=ax) if ax else m
        return m if m.shape == (72, 46) else m.T
    rad_cols = cols(out['SRHR'].transpose(1, 2, 0), rv['SRHR'].transpose(1, 2, 0), 1e-6) | cols(out['TRHR'].transpose(1, 2, 0), rv['TRHR'].transpose(1, 2, 0), 1e-6)
    cloud_cols = np.zeros((72, 46), bool)
    for k in ('CLDSS', 'CLDMC', 'TAUSS', 'TAUMC'):
        cloud_cols |= (np.asarray(r['X'][k]) != cout[k]).any(axis=0)
    t_cols = (np.asarray(r['X']['T']) != cout['T']).any(axis=2)
    res['where'] = dict(n_columns_srhr_trhr_over_1em6_of_scale=int(rad_cols.sum()), n_columns_our_cloud_arrays_differ_from_real_condse_exit=int(cloud_cols.sum()),
                        n_columns_our_T_differs_from_real_condse_exit=int(t_cols.sum()), n_radiation_columns_inside_cloud_flip_columns=int((rad_cols & cloud_cols).sum()),
                        n_radiation_columns_outside=int((rad_cols & ~cloud_cols).sum()))
    res['note'] = ('differences are expected: the server acts on OUR state (libm-mode CONDSE) while the reference uses the real recorded SRHR/TRHR; '
                   'server also resets negative Q (Q after RADIA is the server Q)')
    json.dump(res, open(os.path.join(outdir, 'nov26_phase1_server.json'), 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main(*sys.argv[1:4])
