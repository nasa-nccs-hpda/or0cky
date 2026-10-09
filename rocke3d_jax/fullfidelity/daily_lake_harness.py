"""D205: differential test of daily_lake.py against the REAL Fortran daily_LAKE (LAKES.f:2492-3010 extracted verbatim, compiled with the production flags
`ifort -O2 -ftz -convert big_endian -assume protect_parens -fp-model strict`, stub modules in instrumentation/daily_lake_standalone_stubs.f90).
    python daily_lake_harness.py build DIR              # compile (needs `source env_modele.sh`; ifort 19.1.3)
    python daily_lake_harness.py makecase DIR OURS.npz  # DIR/case.npz from the nov26 step-47 records and a saved end-of-step-47 state (make_case)
    python daily_lake_harness.py run DIR CASE.npz       # write in.bin from CASE, run ./drv, compare with daily_lake.daily_lake (bitwise)
CASE.npz keys: flake fearth fland rsi msi snowi mwl gml tlake mldlk (IM,JM), hsi (IM,JM,4), tanlk hlake dmwldf flice focean axyp (IM,JM).
Returns/prints, per output field, max |port - Fortran| and the number of differing cells (bitwise)."""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import daily_lake as DL  # noqa: E402

MODEL = os.environ.get('MODELE_SRC', '/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/model')
IM, JM = DL.IM, DL.JM
IN_ORDER = ('flake', 'fearth', 'fland', 'rsi', 'msi', 'snowi', 'mwl', 'gml', 'tlake', 'mldlk', 'hsi', 'tanlk', 'hlake', 'dmwldf', 'flice', 'focean', 'axyp')
OUT_ORDER = ('flake', 'fearth', 'fland', 'rsi', 'msi', 'snowi', 'mwl', 'gml', 'tlake', 'mldlk', 'hsi', 'dmwldf', 'dgml', 'svflake', 'dlake', 'glake',
             'gtemp', 'gtempr', 'mlhc', 'mdwnimp', 'edwnimp')


def build(d):
    os.makedirs(d, exist_ok=True)
    ext = os.path.join(d, 'daily_lake_ext.f')
    src = open(os.path.join(MODEL, 'LAKES.f')).read().split('\n')
    open(ext, 'w').write('\n'.join(src[2491:3010] + src[4058:4083]) + '\n')       # daily_LAKE (2492-3010) and newton (4059-4083), verbatim
    ins = os.path.join(HERE, 'instrumentation')
    F = '-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict'
    cmd = (f'source {HERE}/env_modele.sh >/dev/null 2>&1; cd {d}; '
           f'ifort {F} -c {MODEL}/shared/MathematicalConstants.F90 && ifort {F} -c {MODEL}/shared/CubicEquation_mod.F90 && '
           f'ifort {F} -c {ins}/daily_lake_standalone_stubs.f90 && ifort {F} -fpp -c daily_lake_ext.f && '
           f'ifort {F} -c {ins}/daily_lake_standalone_drv.f90 && ifort {F} *.o -o drv')
    subprocess.run(['bash', '-c', cmd], check=True)


def write_in(d, c):
    parts = []
    for k in IN_ORDER:
        a = np.asarray(c[k], dtype='>f8')
        if k == 'hsi':
            a = np.transpose(a, (2, 0, 1))                                   # Fortran HSI(4, im, jm)
        parts.append(a.ravel(order='F').tobytes())
    open(os.path.join(d, 'in.bin'), 'wb').write(b''.join(parts))


def read_out(d):
    buf = np.fromfile(os.path.join(d, 'out.bin'), dtype='>f8')
    out, off = {}, 0
    for k in OUT_ORDER:
        n = IM * JM * (4 if k == 'hsi' else 1)
        a = buf[off:off + n]
        off += n
        out[k] = np.transpose(a.reshape((4, IM, JM), order='F'), (1, 2, 0)) if k == 'hsi' else a.reshape((IM, JM), order='F')
    assert off == buf.size
    return out


def run(d, c):
    write_in(d, c)
    subprocess.run(['./drv'], cwd=d, check=True, capture_output=True)
    return read_out(d)


def make_case(ours_npz, ff='/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data', it_before=33359, daydir='nov26_day', w_from='ffg'):
    """The nov26 day-boundary case: surface fractions and statics from the recorded CONDSE entry of step 47 (ffc_cse_in_<it_before>), FLAKE0/HLAKE from the TOPO file,
    HLAKE clamp as LAKES.f:690, AXYP = DXYPO, DMWLDF from the REAL GHY output w_out of the second substep of step 47 (ffg_<it_before>), ice and lake state from
    `ours_npz` (a saved end-of-step state with surf/ice/{rsi,msi,snowi,hsi} and surf/lake/{mwl,gml,tlake,mldlk})."""
    import clouds_condse_io as cio
    import ghy_compare as GC
    import ghy_ref as R
    import jax_static as JST
    import ocean_step as O
    d = f'{ff}/{daydir}'
    c47 = cio.read_cse(f'{d}/ffc_cse_in_{it_before}.bin')
    z = np.load(ours_npz)
    topo = JST.topography()
    axyp = np.repeat(O._DXYPO[None, :], IM, axis=0)
    hl, tn = DL.lake_statics(topo, c47['FLAKE'], axyp)
    g = GC.load(f'{d}/ffg_{it_before}.bin')
    g2 = g[len(g) // 2:]
    ecells = (g2[:, 1].astype(int) - 1) * IM + (g2[:, 0].astype(int) - 1)
    w_out = np.stack([GC.unpack(r)[4]['w_out'] for r in g2])
    dm = DL.water_deficit(g2, w_out, ecells, c47['FEARTH'], R.THM[0, :])
    return dict(flake=c47['FLAKE'], fearth=c47['FEARTH'], fland=c47['FLAND'], rsi=z['surf/ice/rsi'], msi=z['surf/ice/msi'], snowi=z['surf/ice/snowi'],
                hsi=z['surf/ice/hsi'], mwl=z['surf/lake/mwl'], gml=z['surf/lake/gml'], tlake=z['surf/lake/tlake'], mldlk=z['surf/lake/mldlk'],
                tanlk=tn, hlake=hl, dmwldf=dm, flice=c47['FLICE'], focean=c47['FOCEAN'], axyp=axyp)


def compare(d, c):
    fo = run(d, c)
    S = {k: c[k] for k in DL.FIELDS_IN}
    valid = np.ones((IM, JM), bool)
    valid[1:, 0] = False
    valid[1:, JM - 1] = False
    po = DL.daily_lake(S, c['flice'], c['focean'], c['tanlk'], c['hlake'], c['axyp'], c['dmwldf'], valid=valid)
    rep = {}
    for k in OUT_ORDER:
        a, b = np.asarray(po[k]), np.asarray(fo[k])
        if k in ('gtemp', 'gtempr', 'mlhc'):                       # NaN in the port where not set; -1e300 sentinel in the harness
            m = np.isfinite(a)
            assert np.array_equal(m, b > -1e299), k
            a, b = a[m], b[m]
        elif k in ('dlake', 'glake', 'svflake'):                   # the harness keeps its -1e300 sentinel outside the IMAXJ domain (pole rows i > 1)
            a, b = a[valid], b[valid]
        diff = a != b
        rep[k] = (float(np.abs(a - b).max()) if a.size else 0.0, int(diff.sum()), int(a.size))
    return rep, po, fo


if __name__ == '__main__':
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == 'build':
        build(d)
    elif cmd == 'makecase':                                  # makecase DIR OURS.npz -> DIR/case.npz
        np.savez(os.path.join(d, 'case.npz'), **make_case(sys.argv[3]))
    else:
        c = dict(np.load(sys.argv[3]))
        rep, _, _ = compare(d, c)
        for k, (mx, nd, n) in rep.items():
            print(f'{k:8s} max|port-fortran| {mx:.3e}  differing elements {nd} / {n}')
