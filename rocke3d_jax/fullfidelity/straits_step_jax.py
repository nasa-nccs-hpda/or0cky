"""One straits step in batched JAX (Stage 2, D73): STPGF -> STADV -> STCONV -> STBDRA, in the
order OCNDYN2.f:483-488 calls them. Each routine is validated separately (stpgf_compare,
stadv_compare, stconv_compare, stdrag check); this module chains them and compares the whole step.
The seawater EOS is the OFTAB table (eos_jax). The end-point (ME) arrays evolve in STPGF/STADV."""
import numpy as np

from straits_jax import stpgf_jax, stadv_seq, stbdra_jax, parse_straits_nml
from stconv_jax import stconv_jax
from eos_jax import load_vgsp

GRAV = 9.80665


def straits_step(vgsp, ze, geo, dxypo, state, me, tabs):
    """state: dict of (N, LMO+1) arrays: must, mmst, g0, gx, gz, s0, sx, sz, plus lmst (N,),
    lmme (N,2), oprese (N,2), hoceane (N,2), distpg, wist, dist (N,), dts. me: dict of
    (N,2,LMO+1) arrays: moe, g0me, gxme, gyme, gzme, s0me, sxme, syme, szme. Returns (state, me)."""
    dxyp = dxypo[geo['jst']]
    dts = state['dts']
    # 1. STPGF (MUST only)
    must = np.asarray(stpgf_jax(vgsp, GRAV, dts, state['lmst'], state['lmme'], state['oprese'],
                                state['hoceane'], me['moe'], me['g0me'], me['gzme'], me['s0me'],
                                me['szme'], dxyp, state['must'], state['distpg'], state['wist']))
    # 2. STADV (moments, end points; shared-cell copy)
    me, mst = stadv_seq(dts, state['lmst'], state['mmst'], must, dxyp[:, 0], dxyp[:, 1],
                        geo['xst'], geo['yst'], geo['ist'], geo['jst'], me,
                        {'g0': state['g0'], 'gx': state['gx'], 'gz': state['gz'],
                         's0': state['s0'], 'sx': state['sx'], 'sz': state['sz']})
    st = dict(state)
    st.update({'must': must, 'g0': mst['g0'], 'gx': mst['gx'], 'gz': mst['gz'], 's0': mst['s0'],
               'sx': mst['sx'], 'sz': mst['sz']})
    # 3. STCONV (half-box convection and diffusion)
    res = stconv_jax(ze, GRAV, dts, state['nmst'], state['lmst'], state['mmst'], state['dist'],
                     state['wist'], geo['jst'], state['sinpo'], vgsp, st['must'], st['g0'], st['gx'],
                     st['gz'], st['s0'], st['sx'], st['sz'], tabs)
    st.update({'must': np.asarray(res['must']), 'g0': np.asarray(res['g0mst']),
               'gx': np.asarray(res['gxmst']), 'gz': np.asarray(res['gzmst']),
               's0': np.asarray(res['s0mst']), 'sx': np.asarray(res['sxmst']),
               'sz': np.asarray(res['szmst'])})
    # 4. STBDRA (drag)
    must2, gx2, sx2 = stbdra_jax(dts, state['nmst'], state['lmst'], st['must'], state['mmst'],
                                 state['wist'], state['dist'], st['gx'], st['sx'])
    st.update({'must': np.asarray(must2), 'gx': np.asarray(gx2), 'sx': np.asarray(sx2)})
    return st, me
