"""End-to-end straits step check (D73): entry state (ffz_pgf_in + ffz_stadv_in moments, ffz_me_in
end-point arrays) through straits_step.straits_step, compared with the recorded post-drag state
(ffz_stdrag) and end-point arrays (ffz_me_out)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stconv_compare as C  # noqa: E402
import stadv_compare as A  # noqa: E402
import stpgf_compare as P  # noqa: E402
from straits_jax import parse_straits_nml  # noqa: E402
from straits_step_jax import straits_step  # noqa: E402
from eos_jax import load_vgsp  # noqa: E402
from odhorz_ff import geomo_dyn_arrays  # noqa: E402
from kppmix_compare import load_kppmix_records  # noqa: E402

TOL = 1e-9


def run(d, it, vgsp, dxypo):
    geo = parse_straits_nml(f'{d}/OSTRAITS')
    sa = C.load_in(f'{d}/ffz_stadv_in_{it}.bin')
    pg = P.load_pgf(f'{d}/ffz_pgf_in_{it}.bin')
    me_in = A.load_me(f'{d}/ffz_me_in_{it}.bin')
    me_out = A.load_me(f'{d}/ffz_me_out_{it}.bin')
    ref = C.load_in(f'{d}/ffz_stdrag_{it}.bin')
    ze = np.asarray(load_kppmix_records(f'{d}/ffz_kppmix_{it}.bin')[0]['ze'], dtype=np.float64)
    tabs = C.tabs_and_grid(ze)
    state = dict(must=pg['must'], mmst=sa['mmst'], g0=sa['g0'], gx=sa['gx'], gz=sa['gz'],
                 s0=sa['s0'], sx=sa['sx'], sz=sa['sz'], lmst=sa['lmst'], lmme=pg['lmme'],
                 oprese=pg['oprese'], hoceane=pg['hoceane'], distpg=pg['distpg'], wist=sa['wist'],
                 dist=sa['dist'], dts=sa['dts'], nmst=12, sinpo=sa['sinpo'])
    me = {k: me_in[k] for k in me_in}
    st, me_new = straits_step(vgsp, ze, geo, dxypo, state, me, tabs)
    err = {}
    for k, rk in [('must', 'must'), ('g0', 'g0'), ('gx', 'gx'), ('gz', 'gz'), ('s0', 's0'),
                  ('sx', 'sx'), ('sz', 'sz')]:
        w = 0.0
        for n in range(12):
            m = sa['lmst'][n]
            sc = max(np.max(np.abs(ref[rk][n, 1:m + 1])), 1e-300)
            w = max(w, float(np.max(np.abs(st[k][n, 1:m + 1] - ref[rk][n, 1:m + 1])) / sc))
        err[k] = w
    for name in ['moe', 'g0me', 'gxme', 'gyme', 'gzme', 's0me', 'sxme', 'syme', 'szme']:
        w = 0.0
        for n in range(12):
            m = sa['lmst'][n]
            ref_a = me_out[name][n, :, 1:m + 1]
            sc = max(np.max(np.abs(ref_a)), 1e-300)
            w = max(w, float(np.max(np.abs(me_new[name][n][:, 1:m + 1] - ref_a)) / sc))
        err[name] = w
    return err


if __name__ == '__main__':
    vgsp = load_vgsp()
    worst = {}
    for d in sys.argv[1:]:
        dxypo = geomo_dyn_arrays()[-1]
        for f in sorted(glob.glob(f'{d}/ffz_stdrag_*.bin')):
            it = f.split('_')[-1][:-4]
            e = run(d, it, vgsp, dxypo)
            for k, v in e.items():
                worst[k] = max(worst.get(k, 0.0), v)
    print('STRAITS STEP worst rel error', {k: f'{v:.1e}' for k, v in worst.items()},
          'PASS' if max(worst.values()) <= TOL else 'FAIL', 'tol', TOL)
