"""D137: chained ocean step with ODIFF ported (replaces the recorded UO/VO/VONP of ocean_step.stage_odiff).
ocean_step.py is not modified; this module re-uses its stages and swaps only the 'odiff' stage."""
import numpy as np
import ocean_step as O
import ocean_odiff as D


def stage_odiff_ported(s, fx, ctx):
    """ODIFF every 6th step (mod(itime,6)==0, OCNDYN2.f:416-420) computed by ocean_odiff.odiff from the live state.
    DH comes from the state ('dh3d', set by the second ODHORZ0 in stage_post). No recorded boundary is read."""
    s = O.copy_state(s)
    if fx.get('itime', 1) % 6 == 0:
        if 'dh3d' not in s:
            s['dh3d'] = O.run_odhorz0(s, ctx)['dh3d']
        uo, vo, vonp = D.odiff(s['mo'], s['uo'], s['vo'], s['dh3d'], ctx['lmu'], ctx['lmv'])
        s['uo'], s['vo'], s['vonp'] = uo, vo, vonp
    return s


STAGES_ODIFF = [(n, (stage_odiff_ported if n == 'odiff' else f), t) for (n, f, t) in O.STAGES]


def ocean_step_odiff(state, fx, ctx, trace=None):
    """ocean_step.ocean_step with the ported ODIFF; fx needs only 'itime' and the recorded fluxes."""
    return O.ocean_step(state, fx, ctx, trace=trace, stages=STAGES_ODIFF)
