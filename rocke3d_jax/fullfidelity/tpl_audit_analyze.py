"""D220: column-level audit of the SURFACE template (see tpl_audit_capture.py).  python tpl_audit_analyze.py CAPTURE_DIR [--steps 2,20,40]
For every column of the tile row families (ffs water rows `tw`, ffp rows `pw`/`pl`/`pe`, ffl rows `tl`):
  written   = the value changes between tpl_before and tpl_after (the device-side rewrite of the step writes it from OUR state), on rows that exist;
  varies    = the RECORDED value differs between recorded steps (it depends on the real trajectory or the date), compared on the same rows;
  => class: WRITTEN, STATIC (not written, identical in all recorded steps), RECORDED-DYNAMIC (not written, varies: the step still takes it from the real run).
Radiation columns are written by drv_radcols_dev (ROCKE_RADCOLS=1), NOT by the apply stage this capture ran: they are listed separately as 'RAD'."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atm_step as A
import jax_surface as JS
import surface_tile_ff as ST
import landice_tile_ff as LI

CAP = sys.argv[1]
STEPS = [int(x) for x in (sys.argv[sys.argv.index('--steps') + 1] if '--steps' in sys.argv else '2,20,40').split(',')]
DAY = 'nov26_day'
RAD_COLS = dict(tw=[15, 24, 80], pw=[17, 20], pl=[17, 20], pe=[17, 20], tl=[8, 12, 14])
PBL_NAMES = {5: "dtsurf", 6: "zs1", 7: "tgv", 8: "tkv", 9: "qg_sat", 10: "qg_aver", 11: "hemi", 12: "tr4", 13: "evap_max", 14: "fr_sat", 15: "uocean", 16: "vocean", 17: "psurf",
             18: "trhr0", 19: "tg", 20: "elhx", 21: "qsol", 22: "sss_loc", 23: "ocean", 24: "ddml", 25: "gusti", 26: "tdns", 27: "qdns", 28: "snow", 29: "dskin_in", 30: "dbl",
             31: "khs_in", 32: "ug", 33: "vg", 34: "cm", 35: "ch", 36: "cq", 37: "coriol", 38: "utop", 39: "vtop", 40: "qtop", 41: "ztop", 42: "mdf", 43: "dpdxr", 44: "dpdyr",
             45: "dpdxr0", 46: "dpdyr0", 47: "dtdt_gcm", 50: "z0m", 3: "itype"}
names = dict(tw={v: k for k, v in ST.IN.items()}, tl={v: k for k, v in LI.IN.items()},
             pw={k - 1: v for k, v in PBL_NAMES.items()}, pl={k - 1: v for k, v in PBL_NAMES.items()}, pe={k - 1: v for k, v in PBL_NAMES.items()})


def nm(fam, c):
    d = names.get(fam, {})
    if c in d:
        return d[c]
    if fam in ('pw', 'pl', 'pe'):
        if 49 <= c < 57: return f'u[{c - 49}]'
        if 57 <= c < 65: return f'v[{c - 57}]'
        if 65 <= c < 73: return f't[{c - 65}]'
        if 73 <= c < 81: return f'q[{c - 73}]'
        if 81 <= c < 88: return f'e[{c - 81}]'
    return f'col{c}'


def key(ns, fam):
    return f"['ns'][{ns}]['{fam}']"


b = np.load(os.path.join(CAP, 'tpl_before.npz'))
a = np.load(os.path.join(CAP, 'tpl_after.npz'))
recs = {}
for k in STEPS:
    R = A.Real(DAY, 33312 + k)
    tpl, host = JS.build_template(A.surface_records(R))
    import jax
    recs[k] = ({jax.tree_util.keystr(pp): np.asarray(v) for pp, v in jax.tree_util.tree_flatten_with_path(tpl)[0]}, host)

rows = []
summary = {}
for fam in ('tw', 'pw', 'pl', 'pe', 'tl'):
    for ns in (0, 1):
        kk = key(ns, fam)
        x, y = b[kk], a[kk]
        ncol = x.shape[1]
        valid = np.ones(x.shape[0], bool)
        if fam in ('tw', 'pw'):                       # rows of a slot that exists
            wv = b[f"['ns'][{ns}]['wvalid']"]
            Nw = wv.shape[1]
            valid = np.concatenate([wv[0], wv[1]])
        written = np.array([not np.array_equal(x[valid, c], y[valid, c]) for c in range(ncol)])
        varies = np.zeros(ncol, bool)
        base = recs[STEPS[0]][0][kk]
        for k in STEPS[1:]:
            o = recs[k][0][kk]
            if o.shape != base.shape:
                continue
            vv = valid
            varies |= np.array([not np.array_equal(base[vv, c], o[vv, c]) for c in range(ncol)])
        for c in range(ncol):
            rows.append((fam, ns, c, nm(fam, c), bool(written[c]), bool(varies[c])))
out = []
out.append('| family | columns | WRITTEN by the step | RAD (computed with ROCKE_RADCOLS=1) | STATIC | RECORDED-DYNAMIC |')
out.append('|---|---|---|---|---|---|')
dyn_lists = {}
for fam in ('tw', 'pw', 'pl', 'pe', 'tl'):
    r = [x for x in rows if x[0] == fam]
    cols = sorted({x[2] for x in r})
    W, S, D, RD = [], [], [], []
    for c in cols:
        rr = [x for x in r if x[2] == c]
        w = any(x[4] for x in rr)
        v = any(x[5] for x in rr)
        if c in RAD_COLS[fam]:
            RD.append(c)
        elif w:
            W.append(c)
        elif v:
            D.append(c)
        else:
            S.append(c)
    out.append(f'| {fam} | {len(cols)} | {len(W)} | {len(RD)} | {len(S)} | {len(D)} |')
    dyn_lists[fam] = D
out.append('')
for fam, D in dyn_lists.items():
    out.append(f'RECORDED-DYNAMIC {fam}: ' + (', '.join(f'{c}:{nm(fam, c)}' for c in D) or 'none'))
print('\n'.join(out))
open(os.path.join(CAP, 'audit_table.md'), 'w').write('\n'.join(out))
