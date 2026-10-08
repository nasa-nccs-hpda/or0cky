"""Validate dyn_adv1d_ff.py (adv1d incl. the qlimit/limitq path, all three direction permutations) against a
STANDALONE ifort build of the real QUSDEF.f adv1d + limitq (NOT model dumps).

Why: in the AADVT dump windows (D99) qlimit is always .false. (so limitq never runs) and every Courant nstep
is 1; the qlimit path is only live in CLOUDS2 (moist convection).  This harness drives the real routines with
seeded random lines that deliberately hit every limitq branch, so the qlimit port is validated against real
Fortran compiled with the model flags, but only on synthetic inputs.

Build (scratch copy of the model sources, ifort flags of the model; see build_and_run.md D99/D100 section):
  { sed -n 1,221p QUSDEF.f; sed -n 636,758p QUSDEF.f; } > qus1d_ext.f     # module QUSDEF + adv1d + limitq
  ifort -O2 -ftz -convert big_endian -assume protect_parens -fp-model strict -c qus1d_ext.f
  ifort ... instrumentation/qus1d_standalone_drv.f90 qus1d_ext.o -o drv
Data: ff_data/qus1d_harness/{in,out}.bin (written by `python3 dyn_adv1d_compare.py --gen <dir>` + running drv there).
Layout: in.bin = [nb,0] then per line [qlimit,dirtype(0 x,1 y,2 z)] s(72) smom(9,72) mass(72) dm(72);
out.bin = per line [ierr,nerr] s smom mass f fmom (the adv1d results), big-endian float64.
"""
import os
import sys
import numpy as np
from dyn_adv1d_ff import adv1d, XDIR, YDIR, ZDIR

NX = 72
FF_DEFAULT = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
HARNESS_DIR = FF_DEFAULT + "/qus1d_harness"
DIRS = (XDIR, YDIR, ZDIR)


def gen_cases(nb=3000, seed=20261005):
    rng = np.random.default_rng(seed)
    cases = []
    for b in range(nb):
        ql = int(rng.integers(0, 2))
        dt = int(rng.integers(0, 3))
        kind = int(rng.integers(0, 4))
        mass = rng.uniform(0.5, 1.5, NX)
        if kind == 0:       # smooth positive, small Courant
            s = mass * rng.uniform(1.0, 2.0, NX)
            sm = rng.normal(0, 0.2, (9, NX)) * s
            c = rng.uniform(-0.45, 0.45, NX)
        elif kind == 1:     # strong gradients, divergent flow (hits both-edge outflow branches), small tracer
            s = mass * rng.uniform(0.0, 1.0, NX) ** 3
            sm = rng.normal(0, 1.5, (9, NX)) * s
            c = rng.uniform(-0.9, 0.9, NX)
        elif kind == 2:     # near-zero tracer with large moments, fluxes up to |a|>1 (ierr 1/2 branches)
            s = mass * rng.uniform(0.0, 0.05, NX)
            sm = rng.normal(0, 2.0, (9, NX)) * mass * 0.05
            c = rng.uniform(-1.4, 1.4, NX)
        else:               # coherent shear-like flow, mass nearly zero in places
            s = mass * rng.uniform(0.2, 1.0, NX)
            sm = rng.normal(0, 1.0, (9, NX)) * s
            mass = np.where(rng.random(NX) < 0.03, 0.0, mass)
            c = np.sin(np.linspace(0, 2 * np.pi * rng.integers(1, 5), NX) + rng.uniform(0, 6)) * rng.uniform(0.2, 1.1)
        dm = c * mass
        cases.append((ql, dt, s, sm, mass, dm))
    return cases


def write_in(path, cases):
    with open(path, 'wb') as fh:
        np.array([len(cases), 0.], dtype='>f8').tofile(fh)
        for ql, dt, s, sm, mass, dm in cases:
            np.array([ql, dt], dtype='>f8').tofile(fh)
            for a in (s, sm, mass, dm):          # Fortran order: smom(nmom,nx) -> moment index fastest
                np.asarray(a, dtype='>f8').ravel(order='F').tofile(fh)


def read_out(path, nb):
    raw = np.fromfile(path, dtype='>f8')
    per = 2 + NX + 9 * NX + NX + NX + 9 * NX
    assert raw.size == nb * per, (raw.size, nb * per)
    out = []
    for b in range(nb):
        r = raw[b * per:(b + 1) * per]
        o = 2
        d = dict(ierr=int(r[0]), nerr=int(r[1]))
        d['s'] = r[o:o + NX]; o += NX
        d['smom'] = r[o:o + 9 * NX].reshape((9, NX), order='F'); o += 9 * NX
        d['mass'] = r[o:o + NX]; o += NX
        d['f'] = r[o:o + NX]; o += NX
        d['fmom'] = r[o:o + 9 * NX].reshape((9, NX), order='F')
        out.append(d)
    return out


def load_cases(dirpath=HARNESS_DIR):
    raw = np.fromfile(dirpath + "/in.bin", dtype='>f8')
    nb = int(raw[0]); o = 2; cases = []
    for b in range(nb):
        ql, dt = int(raw[o]), int(raw[o + 1]); o += 2
        s = raw[o:o + NX].copy(); o += NX
        sm = raw[o:o + 9 * NX].reshape((9, NX), order='F').copy(); o += 9 * NX
        mass = raw[o:o + NX].copy(); o += NX
        dm = raw[o:o + NX].copy(); o += NX
        cases.append((ql, dt, s, sm, mass, dm))
    return cases


def available(dirpath=HARNESS_DIR):
    return os.path.exists(dirpath + "/in.bin") and os.path.exists(dirpath + "/out.bin")


def run_case(case, stats=None, **kw):
    ql, dt, s, sm, mass, dm = case
    s, sm, mass = s.copy()[None], sm.copy()[:, None, :], mass.copy()[None]
    f, fm, ierr, nerr = adv1d(s, sm, mass, dm[None], bool(ql), DIRS[dt], stats, **kw)
    return dict(ierr=ierr, nerr=nerr, s=s[0], smom=sm[:, 0, :], mass=mass[0], f=f[0], fmom=None if fm is None else fm[:, 0, :])


def compare(case, out):
    r = run_case(case)
    if out['ierr'] == 2:     # Fortran returns early: only ierr/nerr are meaningful
        return dict(ierr_ok=r['ierr'] == 2 and r['nerr'] == out['nerr'], early=True)
    d = dict(ierr_ok=(r['ierr'] == out['ierr'] and r['nerr'] == out['nerr']), early=False)
    d['s'] = float(np.max(np.abs(r['s'] - out['s'])))
    d['smom'] = float(np.max(np.abs(r['smom'] - out['smom'])))
    d['mass'] = float(np.max(np.abs(r['mass'] - out['mass'])))
    d['f'] = float(np.max(np.abs(r['f'] - out['f'])))
    d['fmom'] = float(np.max(np.abs(r['fmom'] - out['fmom'])))
    return d


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--gen":
        os.makedirs(sys.argv[2], exist_ok=True)
        write_in(sys.argv[2] + "/in.bin", gen_cases())
        print("wrote", sys.argv[2] + "/in.bin")
        sys.exit(0)
    cases = load_cases(); outs = read_out(HARNESS_DIR + "/out.bin", len(cases))
    stats = {}
    worst = {}
    nbad = 0
    for c, o in zip(cases, outs):
        run_case(c, stats)
        d = compare(c, o)
        nbad += (not d['ierr_ok'])
        for k, v in d.items():
            if isinstance(v, float):
                worst[k] = max(worst.get(k, 0.), v)
    print("cases", len(cases), "ierr/nerr mismatches", nbad, "worst abs diff", worst)
    print("ierr histogram (Fortran):", {i: sum(o['ierr'] == i for o in outs) for i in (0, 1, 2)})
    print("limitq branch counts (qlimit lines):", dict(sorted(stats.items())))
