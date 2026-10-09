"""D204: validate ghy_jax.advnc_gdtm (the production state-derived GHY schedule) on REAL records, all land cells of a nov26_day step/substep.
(a) computed nit == recorded ffnit per cell; (b) outputs vs the recorded-schedule ghy_jax.advnc, categories A bitwise / B <=1e-12 / C <=1e-6 / D worse (scaled by max(1,|ref|));
(c) outputs vs the real Fortran values in the record (tbcs tsns ashg alhg aevap); (d) optional bitwise check vs the D202 scan prototype; (e) NumPy ghy_ref_nit (use_recorded_dts=False) on a cell sample.
    python d204_validate.py OUT.json STEP[:NS] ... [--np N]"""
import sys, json
import numpy as np
import clouds_jax_env  # noqa
import jax, jax.numpy as jnp
import ghy_compare as GC
import ghy_advnc_test as AT
import jax_surface as JS
import ghy_jax as J
import d202_fix as F

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
argv = sys.argv[1:]
outf = argv.pop(0)
nnp = 0
if '--np' in argv:
    k = argv.index('--np'); nnp = int(argv[k + 1]); del argv[k:k + 2]
KEYS = ('w', 'ht', 'nsn', 'dzsn', 'wsn', 'hsn', 'fr_snow', 'tp', 'fice', 'tbcs', 'tsns', 'ashg', 'alhg', 'aevap', 'aruns', 'arunu', 'aeruns', 'aerunu', 'ae0', 'abetad', 'evap_max_ij', 'fr_sat_ij')


def cat(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b, equal_nan=True):
        return 'A'
    d = np.nanmax(np.abs(a - b) / np.maximum(1.0, np.abs(b)))
    return 'B' if d <= 1e-12 else ('C' if d <= 1e-6 else 'D')


res = []
jit_rec = jax.jit(J.advnc, static_argnames=('max_substeps',))
jit_new = jax.jit(lambda *a, max_substeps: J.advnc_sched(*a, max_substeps=max_substeps, mode='computed'), static_argnames=('max_substeps',))
jit_scan = jax.jit(F.advnc_gdtm, static_argnames=('max_substeps',))
for spec in argv:
    k, _, ns = spec.partition(':')
    k = int(k); nss = [int(ns)] if ns else [0, 1]
    g = GC.load(f'{FF}/ffg_{33312+k}.bin'); n = len(g) // 2
    for ns in nss:
        rec = g[ns * n:(ns + 1) * n]
        gb = JS._ghy_batch(rec, AT)
        ms = gb['edts'].shape[1]
        args = lambda: ({a: jnp.asarray(v) for a, v in gb['static'].items()}, {a: jnp.asarray(v) for a, v in gb['dyn0'].items()}, {a: jnp.asarray(v) for a, v in gb['forcing'].items()},
                        jnp.asarray(gb['edts']), jnp.asarray(gb['ecnc']), jnp.asarray(gb['ebet']), jnp.asarray(gb['elai']), jnp.asarray(gb['nsub']), jnp.asarray(gb['dt']), jnp.asarray(gb['snowm']))
        base = {a: np.asarray(v) for a, v in jit_rec(*args(), max_substeps=ms).items()}
        new = {a: np.asarray(v) for a, v in jit_new(*args(), max_substeps=ms).items()}
        scan = {a: np.asarray(v) for a, v in jit_scan(*args(), max_substeps=ms).items()}
        ffnit = np.round(rec[:, 289]).astype(int)
        nit = new['nit_gdtm']
        same = nit == ffnit
        r = dict(step=k, ns=ns + 1, ncell=int(n), nit_equal=int(same.sum()), nit_diff=[(int(i), int(ffnit[i]), int(nit[i])) for i in np.where(~same)[0]][:30],
                 exhausted=int(new['nit_exhausted'].sum()), ffnit_hist={int(a): int(b) for a, b in zip(*np.unique(ffnit, return_counts=True))},
                 n_4plus=int((ffnit >= 4).sum()), n_4=int((ffnit == 4).sum()), n_ge2=int((ffnit >= 2).sum()),
                 while_vs_scan_bitwise=all(np.array_equal(new[a], scan[a], equal_nan=True) for a in KEYS))
        r['cat_vs_recorded'] = {a: cat(new[a], base[a]) for a in KEYS}
        r['maxrel_vs_recorded'] = {a: float(np.nanmax(np.abs(new[a] - base[a]) / np.maximum(1.0, np.abs(base[a])))) for a in KEYS}
        cnt = {}
        for a in KEYS:
            v = np.asarray(new[a], float); b = np.asarray(base[a], float)
            ax = tuple(range(1, v.ndim))
            if v.ndim == 1:
                dd = np.abs(v - b) / np.maximum(1.0, np.abs(b)); eq = v == b
            else:
                dd = (np.abs(v - b) / np.maximum(1.0, np.abs(b))).max(axis=ax); eq = (v == b).all(axis=ax)
            cnt[a] = dict(A=int(eq.sum()), B=int((~eq & (dd <= 1e-12)).sum()), C=int((~eq & (dd > 1e-12) & (dd <= 1e-6)).sum()), D=int((~eq & (dd > 1e-6)).sum()))
        r['cell_counts_vs_recorded'] = cnt
        r['real'] = {}
        for a, col in (('tbcs', 245), ('tsns', 246), ('ashg', 247), ('alhg', 248), ('aevap', 249)):
            rr = rec[:, col]
            r['real'][a] = dict(max_abs_recorded_sched=float(np.nanmax(np.abs(base[a] - rr))), max_abs_computed=float(np.nanmax(np.abs(new[a] - rr))))
        if nnp:
            import ghy_ref_nit as N
            idx = np.unique(np.concatenate([np.where(ffnit >= 3)[0][:nnp // 2], np.random.RandomState(k).choice(n, nnp // 2, replace=False)]))
            dn = []
            for i in idx:
                col, refs, info = N.run_cell_full(rec[i], dt=900.0, use_recorded_dts=False)
                dn.append(dict(i=int(i), nit_np=info['nit'], nit_jax=int(nit[i]), ffnit=int(ffnit[i]),
                               d_ashg=abs(col.ashg - float(new['ashg'][i])), d_aevap=abs(col.aevap - float(new['aevap'][i])), d_tsns=abs(col.tsns - float(new['tsns'][i]))))
            r['numpy_ref'] = dict(ncell=len(dn), nit_equal_numpy=sum(d['nit_np'] == d['nit_jax'] for d in dn), nit_equal_ffnit=sum(d['nit_np'] == d['ffnit'] for d in dn),
                                  max_d_ashg=max(d['d_ashg'] for d in dn), max_d_aevap=max(d['d_aevap'] for d in dn), max_d_tsns=max(d['d_tsns'] for d in dn))
        res.append(r)
        print(json.dumps({a: r[a] for a in ('step', 'ns', 'ncell', 'nit_equal', 'nit_diff', 'exhausted', 'n_4plus', 'while_vs_scan_bitwise')}), flush=True)
        print('  cats', {a: r['cat_vs_recorded'][a] for a in KEYS}, flush=True)
        json.dump(res, open(outf, 'w'), indent=1)
