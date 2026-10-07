"""D171: characterise the first-step ulp residual of D169 (Ent exports not bitwise in the first step after the restart).

Teacher mode (GHY driven by the recorded exports, Ent alongside, as `ent_ghy_compare`), nov26 ffg_33312 (restart step) and the next
files.  For every export that is not bitwise equal to the record the probe stores call (sub-step, cell), sub-iteration, field,
the relative difference and the GHY-side Ent inputs of that iteration (tcan, Qf, w/ws of the layers, fice, fw, ts, ch, vs, cosz,
vis_rad) plus the Ent cell's own state, so that the pattern can be examined.  `variants` re-runs the offending calls with
one input changed to test hypotheses (Qf, Ca, Sacclim rounding...).

Usage (fullfidelity/):  python ent_first_step_probe.py collect <tag> <nfiles> <out.json>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent_ff as E
import ent_ghy_compare as C
import ghy_compare as GC
import ghy_ref as G


def collect(tag="nov26", nfiles=2, dump=None):
    import glob
    cells = E.load_restart_cells(C.RESTART[tag])
    paths = sorted(glob.glob(f"{C.FF}/{tag}/ffg_[0-9]*.bin"))[:nfiles]
    rows = []
    stats = []
    for ip, p in enumerate(paths):
        rec = GC.load(p)
        ng = len(rec) // 2
        for k, r in enumerate(rec):
            key = (int(r[0]), int(r[1]))
            cell = cells.get(key)
            if cell is None:
                continue
            sub = k // ng
            o = C.run_cell(r, cell, "teacher")
            fv = r[169]
            for it, comp in enumerate(o["comp"], 1):
                for f in C.FIELDS:
                    got, ref = np.atleast_1d(comp[f][0]), np.atleast_1d(comp[f][1])
                    if not np.array_equal(got, ref):
                        d = np.abs(got - ref)
                        rows.append(dict(file=ip, sub=sub, cell=key, nit=it, field=f, rel=float((d / np.maximum(np.abs(ref), 1e-300)).max()),
                                         got=got.tolist(), ref=ref.tolist(), fv=float(fv), Qf0=float(r[7]), cosz1=float(r[4]), vis=float(r[5]),
                                         Ca=float(r[3]), ffnit=int(o["refs"]["ffnit"])))
            stats.append((ip, sub, key, o["nit"]))
    if dump:
        json.dump(dict(rows=rows, ncalls=len(stats)), open(dump, "w"))
    return rows, stats


NAMES = ["ts", "tcan", "Qf", "pres", "Ca", "ch", "vs", "vis", "dvis", "cosz", "fw"] + [f"w{k}" for k in range(6)] + \
        [f"ws{k}" for k in range(6)] + [f"fice{k}" for k in range(6)]


def _cohorts(c):
    return [co for p in c.patches for co in p.cohorts]


STATE_MODS = {
    "airtemp_10d": lambda c, d: setattr(c, "airtemp_10d", float(np.nextafter(c.airtemp_10d, d))),
    "par_10d": lambda c, d: setattr(c, "par_10d", float(np.nextafter(c.par_10d, d))),
    "daylength1": lambda c, d: c.daylength.__setitem__(1, float(np.nextafter(c.daylength[1], d))),
    "Sacclim": lambda c, d: [setattr(o, "Sacclim", float(np.nextafter(o.Sacclim, d))) for o in _cohorts(c)],
    "lai": lambda c, d: [setattr(o, "lai", float(np.nextafter(o.lai, d))) for o in _cohorts(c)],
    "fracroot": lambda c, d: [setattr(o, "fracroot", np.nextafter(o.fracroot, d)) for o in _cohorts(c)],
    "albedo": lambda c, d: [setattr(p, "albedo", np.nextafter(p.albedo, d)) for p in c.patches],
}


def _perturb(args, name, sgn):
    """args = (ts, tcan, Qf, pres, Ca, ch, vs, vis, dvis, cosz, fw, w_veg, ws_veg, fice_veg); return a copy with one input moved 1 ulp."""
    a = list(args)
    a[11], a[12], a[13] = (np.array(a[11], float), np.array(a[12], float), np.array(a[13], float))
    tgt = 1.0 if sgn > 0 else -1.0
    if name in NAMES[:11]:
        i = NAMES.index(name)
        a[i] = float(np.nextafter(a[i], np.inf * tgt))
    else:
        grp = {"w": 11, "ws": 12, "fice": 13}
        for g in ("fice", "ws", "w"):
            if name.startswith(g):
                k = int(name[len(g):])
                a[grp[g]][k] = np.nextafter(a[grp[g]][k], np.inf * tgt)
                break
    return a


def ulp_variants(tag="nov26", ip=0, dump=None):
    """Hypothesis test: for every call of file `ip` whose FIRST sub-iteration exports are not bitwise equal to the record, re-run that first
    Ent call from the saved cell state with ONE forcing input moved by +-1 ulp and report which single-input variants make all 7
    exports bitwise equal to the record (also whether doing nothing, i.e. the unperturbed re-run, reproduces the mismatch)."""
    import copy
    import glob
    cells = E.load_restart_cells(C.RESTART[tag])
    p = sorted(glob.glob(f"{C.FF}/{tag}/ffg_[0-9]*.bin"))[ip]
    rec = GC.load(p)
    cap = dict(on=False, args=None, cell0=None, dts=None, upd=None)
    orig_sf, orig_run = E.set_forcings, E.ent_run

    def sf(cell, *a, **k):
        if cap["on"] and cap["args"] is None:
            cap["args"] = a
        return orig_sf(cell, *a, **k)

    def run(cell, dts, upd, *a, **k):
        if cap["on"] and cap["cell0"] is None:
            cap["cell0"] = copy.deepcopy(cell)
            cap["dts"], cap["upd"] = dts, upd
        return orig_run(cell, dts, upd, *a, **k)
    E.set_forcings, E.ent_run = sf, run
    bad = []
    try:
        for k, r in enumerate(rec):
            key = (int(r[0]), int(r[1]))
            cell = cells.get(key)
            if cell is None:
                continue
            cap.update(on=True, args=None, cell0=None)
            o = C.run_cell(r, cell, "teacher")
            cap["on"] = False
            if not o["comp"]:
                continue
            c0 = o["comp"][0]
            if all(np.array_equal(np.atleast_1d(c0[f][0]), np.atleast_1d(c0[f][1])) for f in C.FIELDS):
                continue
            bad.append((k, key, {f: c0[f][1] for f in C.FIELDS}, cap["cell0"], cap["args"], cap["dts"], cap["upd"]))
    finally:
        E.set_forcings, E.ent_run = orig_sf, orig_run

    def trial(cell0, args, dts, upd, ref, mod=None):
        c = copy.deepcopy(cell0)
        if mod is not None:
            mod(c)
        orig_sf(c, *args)
        orig_run(c, dts, upd)
        ex = E.get_exports(c)
        return all(np.array_equal(np.atleast_1d(ex[f]), np.atleast_1d(ref[f])) for f in C.FIELDS)
    summary = {}
    for (k, key, ref, cell0, args, dts, upd) in bad:
        fixed = []
        base = trial(cell0, args, dts, upd, ref)
        for nm in NAMES:
            for sgn in (+1, -1):
                try:
                    if trial(cell0, _perturb(args, nm, sgn), dts, upd, ref):
                        fixed.append(f"{nm}{'+' if sgn > 0 else '-'}")
                except Exception:
                    pass
        for nm, mod in STATE_MODS.items():
            for sgn in (+1, -1):
                try:
                    if trial(cell0, args, dts, upd, ref, lambda c, mod=mod, sgn=sgn: mod(c, np.inf * sgn)):
                        fixed.append(f"state:{nm}{'+' if sgn > 0 else '-'}")
                except Exception:
                    pass
        summary[k] = dict(cell=key, baseline_bitwise=base, fixed_by=fixed)
    if dump:
        json.dump(summary, open(dump, "w"))
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "ulp":
        sm = ulp_variants(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        print(len(sm), "calls with a non-bitwise first iteration")
    elif sys.argv[1] == "collect":
        rows, stats = collect(sys.argv[2], int(sys.argv[3]), sys.argv[4])
        print(len(rows), "non-bitwise (field, iteration) entries in", len(stats), "calls")
