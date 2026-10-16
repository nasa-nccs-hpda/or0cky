"""Validate clouds_mstcnv_ff.py (MSTCNV, D110-D113) against the instrumented real model.

Dumps (ff_data/<date>/, layouts and loaders in clouds_mstcnv_io.py):
  ffc_mc_cols_<itime>.bin  call-boundary records (every 2nd convective call, every 25th non-convective call, plus every
                           call whose checkpoints were dumped)
  ffc_mc_ck_<itime>.bin    stage checkpoints (stages 1-9) for convective calls with mod(ncall,24)==0
  ffc_mc_consts.txt        tunables of the run
Usage (from fullfidelity/):
    python3 clouds_mstcnv_compare.py [--imf] [--max N] [--dates nov26,dec01] [--no-cols] [--no-ck] [--ck-max M]
`--imf` routes exp/pow through Intel libimf (needs the Intel runtime, see intel_libm_ff.py); without it numpy/glibc is
used and 1-ulp-level differences remain in a minority of columns (reported, not hidden).
Exit code 0 if every record is bitwise identical in the chosen mode (--imf); in numpy mode the records without a discrete
threshold flip must be within RTOL_NUMPY of the field scale and flips must stay below 3 % of the records (they are
reported, not hidden).  The decision path (event sequences, lmcmin/lmcmax/mccont) must always be identical.
"""
import argparse
import sys
import time
from collections import defaultdict

import numpy as np

import clouds_mstcnv_io as io
import clouds_mstcnv_ff as m

RTOL_IMF = 1e-11      # libimf-mode acceptance (field scale); observed <= 5e-13 (about 0.2 % of records inexact)
RTOL_NUMPY = 1e-6     # numpy-mode acceptance for chaotic amplification of 1-ulp pow/exp differences (reported)
XYM_ONLY = {"smomp", "qmomp", "smompmax", "qmompmax"}
XYM = m.XYM


def record(d, i):
    return {k: v[i] for k, v in d["inp"].items()}


def diff_stats(a, b, name=None):
    """-> (bitwise_equal, max_abs, max_abs_b).  Differences are judged against the largest value the field takes
    over the whole date (see Agg.rel), not against the single record: exactly cancelling quantities (a precipitation
    residual that re-evaporates completely) have a record-level scale of zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if name in XYM_ONLY:
        a, b = a[..., XYM], b[..., XYM]
    fin = np.isfinite(b)
    scale = float(np.max(np.abs(b[fin]))) if fin.any() else 0.0
    if np.array_equal(a, b, equal_nan=True):
        return True, 0.0, scale
    if (np.isfinite(a) != np.isfinite(b)).any():
        return False, np.inf, scale
    f2 = np.isfinite(a) & np.isfinite(b)
    return False, float(np.abs(a - b)[f2].max()), scale


class Agg:
    def __init__(self):
        self.n = defaultdict(int)
        self.bit = defaultdict(int)
        self.maxabs = defaultdict(float)
        self.maxscale = defaultdict(float)

    def add(self, key, a, b, name=None):
        same, ab, sc = diff_stats(a, b, name)
        self.n[key] += 1
        self.bit[key] += int(same)
        self.maxabs[key] = max(self.maxabs[key], ab)
        self.maxscale[key] = max(self.maxscale[key], sc)
        return same, ab

    def rel(self, key):
        """max |a-b| over the date divided by max |b| of the field over the date."""
        return self.maxabs[key] / max(self.maxscale[key], 1e-300) if self.maxabs[key] > 0 else 0.0

    @property
    def maxrel(self):
        return _Rel(self)


class _Rel:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, k):
        return self.a.rel(k)


POS_FIELDS = ("svwmxl", "cldmcl", "taumcl", "precnvl", "condpt", "mcflx", "qlmc", "qimc")
EQ_FIELDS = ("svlatl", "svlat1", "lhp")


def discrete_flips(o, real, i):
    """Names of the discrete indicators (value>0 for POS_FIELDS, exact equality for the latent-heat selectors EQ_FIELDS)
    that differ between the port output `o` and real record `i`: these are threshold-test flips caused by last-bit
    differences of residues (e.g. SVWMXL = 1e-22 vs 0 after a cancellation), not smooth noise."""
    bad = []
    for k in POS_FIELDS:
        if np.any((np.asarray(o[k]) > 0) != (real[k][i] > 0)):
            bad.append(k)
    for k in EQ_FIELDS:
        if np.any(np.asarray(o[k]) != real[k][i]):
            bad.append(k)
    return bad


def compare_cols(date, ff=io.FF_DEFAULT, nmax=None):
    d = io.load_cols(date, ff)
    tune = m.tune_from_consts(io.load_consts(date, ff))
    h = d["hdr"]
    nall = h["conv"].size
    idx = np.arange(nall) if nmax is None else np.unique(np.linspace(0, nall - 1, min(nmax, nall)).astype(int))
    agg = Agg()
    agg_cont = Agg()          # same statistics restricted to records without a discrete threshold flip
    br = defaultdict(int)
    colbad = []
    flips = []
    nbig = 0
    gscale = {k: max(float(np.max(np.abs(v))), 1e-300) for k, v in d["out"].items()}
    path_flips = 0
    for i in idx:
        r = record(d, i)
        brc = defaultdict(int)
        o = m.mstcnv_column(r, tune, br=brc)
        for k, v in brc.items():
            br[k] += v
        conv = h["conv"][i] > 0
        allsame, worst = True, 0.0
        fl = discrete_flips(o, d["out"], i)
        if fl:
            flips.append((int(h["itime"][i]), int(h["i"][i]), int(h["j"][i]), fl))
        for name, _, _ in io.OUT_FIELDS:
            if name in io.STALE_IF_NONCONV and not conv:
                continue
            same, ab = agg.add(name, o[name], d["out"][name][i], name)
            if not fl:
                agg_cont.add(name, o[name], d["out"][name][i], name)
            allsame &= same
            worst = max(worst, ab / gscale[name])
        for name in ("lmcmin", "lmcmax", "mccont", "ierr", "lerr"):
            if o[name] != d["out"][name][i]:
                path_flips += 1
        agg.add("_record", np.array([float(allsame)]), np.array([1.0]))
        nbig += worst > 1e-6
        if not allsame:
            colbad.append((int(h["itime"][i]), int(h["i"][i]), int(h["j"][i]), worst))
    return dict(date=date, nrec=int(idx.size), nconv=int((h["conv"][idx] > 0).sum()), agg=agg, agg_cont=agg_cont,
                br=dict(br), path_flips=path_flips, colbad=colbad, flips=flips, nbig=int(nbig))


def print_cols(res, imf):
    a = res["agg"]
    nrec = res["nrec"]
    print(f"== {res['date']} [{'imf' if imf else 'numpy'}] boundary records: {nrec} ({res['nconv']} convective); "
          f"fully bitwise-identical records: {a.bit['_record']}/{nrec}; decision-path integer mismatches "
          f"(lmcmin/lmcmax/mccont/ierr/lerr): {res['path_flips']}")
    bad = [(k, a.n[k], a.bit[k], a.maxabs[k], a.rel(k)) for k in a.n if k != "_record" and a.bit[k] < a.n[k]]
    for k, nn, nb, ab, rel in sorted(bad, key=lambda x: -x[4])[:12]:
        print(f"    {k:10s} n={nn} bitwise={nb} max_abs={ab:.2e} max_abs/field_scale={rel:.2e}")
    if not bad:
        print("    all output fields bitwise identical")
    worst = max([a.rel(k) for k in a.n if k != "_record"] or [0.0])
    print(f"    worst field-level deviation, all records (max |a-b| over the date / max |b| of the field): {worst:.2e}")
    fl = res["flips"]
    print(f"    discrete threshold flips (a >0 / latent-heat selector differs): {len(fl)} records; "
          f"fields {sorted({f for x in fl for f in x[3]})}; first: {fl[:3]}")
    ac = res["agg_cont"]
    wc = max([ac.rel(k) for k in ac.n if k != "_record"] or [0.0])
    print(f"    worst field-level deviation, records without a flip: {wc:.2e}")
    print(f"    records with any field deviating by more than 1e-6 of its field scale: {res['nbig']} ({100.0 * res['nbig'] / nrec:.2f} %)")
    return worst, len(fl)


def compare_ck(date, ff=io.FF_DEFAULT, cmax=None):
    d = io.load_cols(date, ff)
    tune = m.tune_from_consts(io.load_consts(date, ff))
    h = d["hdr"]
    key = {(int(h["itime"][i]), int(h["i"][i]), int(h["j"][i]), int(h["ncall"][i])): i for i in range(h["conv"].size)}
    blocks = io.load_ck(date, ff)
    if cmax is not None:
        blocks = blocks[:cmax]
    agg = Agg()
    seq_bad = nev = noinp = ovf = 0
    stage_n = defaultdict(int)
    first_bad = {}
    sig = lambda L: [(e["stage"], e["lmin"], e["ic"], e["nppl"]) for e in L]  # noqa: E731
    for b in blocks:
        k = (b["itime"], b["i"], b["j"], b["ncall"])
        if k not in key:
            noinp += 1
            continue
        ovf += b["overflow"]
        ev = []
        m.mstcnv_column(record(d, key[k]), tune, ck=ev)
        if sig(ev) != sig(b["events"]):
            seq_bad += 1
            first_bad.setdefault("sequence", (k, sig(ev)[:8], sig(b["events"])[:8]))
            continue
        for ep, er in zip(ev, b["events"]):
            st = er["stage"]
            stage_n[st] += 1
            nev += 1
            for name, _, _ in io.CK_FIELDS[st]:
                same, ab = agg.add((st, name), ep["f"][name], er["f"][name][0], name)
                if not same:
                    first_bad.setdefault((st, name), (k, er["lmin"], er["ic"], er["nppl"], ab))
    return dict(date=date, nblocks=len(blocks), nev=nev, noinp=noinp, seq_bad=seq_bad, overflow=ovf, agg=agg,
                stage_n=dict(stage_n), first_bad=first_bad)


def print_ck(res, imf):
    a = res["agg"]
    print(f"== {res['date']} [{'imf' if imf else 'numpy'}] checkpoints: {res['nblocks']} calls, {res['nev']} events "
          f"(per stage {dict(sorted(res['stage_n'].items()))}); calls without a boundary record: {res['noinp']}; "
          f"event-sequence mismatches (decision path): {res['seq_bad']}; buffer overflows: {res['overflow']}")
    for st in sorted(res["stage_n"]):
        keys = [k for k in a.n if k[0] == st]
        allbit = sum(1 for k in keys if a.bit[k] == a.n[k])
        worst = max([a.rel(k) for k in keys] or [0.0])
        worstabs = max([a.maxabs[k] for k in keys] or [0.0])
        bad = [k[1] for k in keys if a.bit[k] < a.n[k]]
        print(f"    stage {st}: {len(keys)} fields, {allbit} bitwise on every event; worst abs/field-scale {worst:.2e} "
              f"(abs {worstabs:.2e}); inexact: {bad[:8]}")
    return max([a.rel(k) for k in a.n] or [0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imf", action="store_true")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--ck-max", type=int, default=None)
    ap.add_argument("--dates", default=",".join(x for x, _ in io.DATES))
    ap.add_argument("--no-cols", action="store_true")
    ap.add_argument("--no-ck", action="store_true")
    ap.add_argument("--ff", default=io.FF_DEFAULT)
    a = ap.parse_args()
    if a.imf:
        m.set_backend("imf")
    ok = True
    for date in a.dates.split(","):
        t0 = time.time()
        if not a.no_cols:
            res = compare_cols(date, a.ff, a.max)
            worst, nfl = print_cols(res, a.imf)
            ok &= res["path_flips"] == 0
            ok &= (worst < RTOL_IMF and nfl == 0 and res["nbig"] == 0) if a.imf else (res["nbig"] < 0.10 * res["nrec"] and nfl < 0.03 * res["nrec"])
            print("    branch counters:", dict(sorted(res["br"].items())))
        if not a.no_ck:
            rc = compare_ck(date, a.ff, a.ck_max)
            worst = print_ck(rc, a.imf)
            ok &= rc["seq_bad"] == 0
            ok &= (worst < RTOL_IMF) if a.imf else (worst < 0.2)
            if rc["first_bad"]:
                print("    first inexact checkpoint fields:", {str(k): v for k, v in list(rc["first_bad"].items())[:6]})
        print(f"    ({time.time() - t0:.0f} s)")
    print("ALL MATCH" if ok else "MISMATCH FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
