"""D208 validation (a)+(b1): the transfer on the REAL step-47 -> step-48 data.
BEFORE state = real GHY output of step 47 (ffg_33359, second substep, w_out/ht_out/fr_snow_out; underwater fraction = slot 3 of the same record, unchanged since step 0).
AFTER  state = real GHY entry of step 48 (ffg_33360, first substep).  Lake inputs (flake, svflake, fearth, DMWLDF, DGML) = the hook arrays of a run (OUR daily_LAKE).
Prints: port vs compiled Fortran (bitwise) on this case, and port/Fortran vs the REAL after state, per field, with and without the transfer.
usage: python val_real.py HOOK.npz DIR_with_drv OUT.json"""
import json
import sys

import numpy as np

sys.path.insert(0, '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax/fullfidelity')
import clouds_condse_io as cio
import daily_land_fractions as DF
import ghy_compare as GC
import ghy_ref as R
import land_fractions_harness as H
import ocean_step as O

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
IM, JM = 72, 46
hook, d, outp = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else 'ours'      # ours | realw | realall
realw = mode in ('realw', 'realall')      # DMWLDF recomputed from the REAL w_out of step 47 (DGML rescaled: DGML/DMWLDF is independent of DMWLDF, LAKES.f:2706-2712)
z = np.load(hook)
a = GC.load(f'{FF}/ffg_33359.bin')
b = GC.load(f'{FF}/ffg_33360.bin')
n = len(a) // 2
a2, b1 = a[n:], b[:n]
assert np.array_equal(a2[:, :2], b1[:, :2])
ii, jj = a2[:, 0].astype(int) - 1, a2[:, 1].astype(int) - 1
R3 = lambda r, s, e: np.stack([x[s - 1:e].reshape(7, 3, order='F') for x in r])      # noqa: E731
w_before = R3(a2, 181, 201)
ht_before = R3(a2, 202, 222)
w_before[:, :, 2] = R3(a2, 9, 29)[:, :, 2]
ht_before[:, :, 2] = R3(a2, 30, 50)[:, :, 2]
fs_before = a2[:, 242:244].copy()
w_after, ht_after = R3(b1, 9, 29), R3(b1, 30, 50)
fs_after = b1[:, 70:72].copy()
c47 = cio.read_cse(f'{FF}/ffc_cse_in_33359.bin')
axyp = np.repeat(O._DXYPO[None, :], IM, axis=0)
dz, q, _ = DF.rows_from_ffg(a2)
case = dict(w=w_before, ht=ht_before, fr_snow=fs_before, dz=dz, q=q, fv=a2[:, 169], thm0=R.THM[0, :],
            fearth=z['fearth'][ii, jj], flake=z['flake'][ii, jj], svflake=z['svflake'][ii, jj], focean=c47['FOCEAN'][ii, jj],
            dmwldf=z['dmwldf'][ii, jj], dgml=z['dgml'][ii, jj], byaxyp=1.0 / axyp[ii, jj])
if realw:
    import daily_lake as DL
    ecells = jj * IM + ii
    w_out_real = np.stack([GC.unpack(r)[4]['w_out'] for r in a2])
    dm_real = DL.water_deficit(a2, w_out_real, ecells, c47['FEARTH'], R.THM[0, :])
    old = case['dmwldf'].copy()
    by = case['byaxyp']
    dfr_ours = case['flake'] - case['svflake']
    with np.errstate(all='ignore'):
        hpm3 = np.where(old > 0, case['dgml'] * by / (old * dfr_ours / 1000.0), 0.0)        # heat per m3 of added lake water (ours: GML*RHOW/MWL)
    newdm = np.where(old > 0, dm_real[ii, jj], 0.0)
    if mode == 'realall':        # also the REAL fractions: svflake = FLAKE of the CONDSE entry of step 47, flake/fearth = those of step 48 (the real daily_LAKE result)
        c48 = cio.read_cse(f'{FF}/ffc_cse_in_33360.bin')
        case['svflake'], case['flake'], case['fearth'] = c47['FLAKE'][ii, jj], c48['FLAKE'][ii, jj], c48['FEARTH'][ii, jj]
    case['dmwldf'] = newdm
    hpm3 = np.where(np.isfinite(hpm3), hpm3, 0.0)         # cells where OUR dfrac is 0 but the real lake expanded: heat per m3 unknown -> 0 (reported as lake-side input)
    case['dgml'] = hpm3 * newdm * (case['flake'] - case['svflake']) / 1000.0 / by
    HPM3 = hpm3
rep, po, fo, info = H.compare(d, case)
res = dict(n_rows=int(n), n_shrunk=info['n_shrunk'], n_expanded=info['n_expanded'], port_vs_fortran={k: v for k, v in rep.items()}, new_cell_rows=info['new_cell_rows'].tolist())
print('rows', n, 'shrunk', info['n_shrunk'], 'expanded', info['n_expanded'], 'new-cell rows', info['new_cell_rows'].tolist())
print('PORT vs COMPILED FORTRAN (bitwise):', rep)
changed_real = {}
for nm, bef, aft, new in (('w', w_before, w_after, po['w']), ('ht', ht_before, ht_after, po['ht'])):
    for ib in range(3):
        db = np.abs(aft[:, :, ib] - bef[:, :, ib]).max(1)               # real change of the transfer per cell
        dn = np.abs(new[:, :, ib] - bef[:, :, ib]).max(1)               # our change
        e_with = np.abs(new[:, :, ib] - aft[:, :, ib]).max(1)
        e_without = np.abs(bef[:, :, ib] - aft[:, :, ib]).max(1)
        ch = db > 0
        scale = np.maximum(np.abs(aft[:, :, ib]).max(1), 1e-300)
        key = f'{nm}{ib + 1}'
        r_ = dict(real_changed_cells=int(ch.sum()), port_changed_cells=int((dn > 0).sum()), changed_both=int(((dn > 0) & ch).sum()), port_only=int(((dn > 0) & ~ch).sum()),
                  real_only=int((~(dn > 0) & ch).sum()), max_abs_err_with=float(e_with.max()), max_abs_err_without=float(e_without.max()),
                  max_real_change=float(db.max()),
                  cells_err_with_gt_1em9_rel=int((e_with / scale > 1e-9).sum()), cells_err_without_gt_1em9_rel=int((e_without / scale > 1e-9).sum()),
                  cells_nearer=int((e_with < e_without).sum()), cells_farther=int((e_with > e_without).sum()), cells_equal=int((e_with == e_without).sum()),
                  median_rel_err_changed_with=float(np.median((e_with / np.maximum(db, 1e-300))[ch])) if ch.any() else None,
                  median_rel_err_changed_without=float(np.median((e_without / np.maximum(db, 1e-300))[ch])) if ch.any() else None)
        changed_real[key] = r_
        print(key, r_)
for nm, bef, aft, new in (('fr_snow', fs_before, fs_after, po['fr_snow']),):
    for ib in range(2):
        db = np.abs(aft[:, ib] - bef[:, ib]); dn = np.abs(new[:, ib] - bef[:, ib]); ew = np.abs(new[:, ib] - aft[:, ib]); eo = np.abs(bef[:, ib] - aft[:, ib])
        r_ = dict(real_changed=int((db > 0).sum()), port_changed=int((dn > 0).sum()), both=int(((db > 0) & (dn > 0)).sum()), max_err_with=float(ew.max()), max_err_without=float(eo.max()),
                  nearer=int((ew < eo).sum()), farther=int((ew > eo).sum()), equal=int((ew == eo).sum()))
        changed_real[f'fr_snow{ib + 1}'] = r_
        print(f'fr_snow{ib + 1}', r_)
res['vs_real'] = changed_real
if realw:
    # implied real heat per m3 of lake water in the expanded cells, from the real after state (all layers), against ours (GML*RHOW/MWL of the hook)
    sv_, fl_ = case['svflake'], case['flake']
    ex = info['expanded']
    imp, ours_, wl = [], [], []
    for r in np.nonzero(ex)[0]:
        df = fl_[r] - sv_[r]
        fv_ = DF.fv_from_ent(a2[r:r + 1, 169])[0]
        dht_t = (ht_after[r, 1:, 2] * fl_[r] - ht_before[r, 1:, 2] * sv_[r]).sum() - df * ((1 - fv_) * ht_before[r, 1:, 0] + fv_ * ht_before[r, 1:, 1]).sum() - df * fv_ * ht_before[r, 0, 1]
        dw_t = (w_after[r, 1:, 2] * fl_[r] - w_before[r, 1:, 2] * sv_[r]).sum() - df * ((1 - fv_) * w_before[r, 1:, 0] + fv_ * w_before[r, 1:, 1]).sum() - df * fv_ * w_before[r, 0, 1]
        if dw_t > 1e-12:
            imp.append(dht_t / dw_t); ours_.append(HPM3[r]); wl.append(dw_t)
    imp, ours_ = np.array(imp), np.array(ours_)
    ok = ours_ != 0
    rel = np.abs(ours_[ok] - imp[ok]) / np.abs(imp[ok])
    res['heat_per_m3'] = dict(n=int(ok.sum()), median_rel_diff=float(np.median(rel)), p90=float(np.percentile(rel, 90)), max=float(rel.max()))
    print('implied real vs ours heat per m3 of lake water (expanded cells with lake water):', res['heat_per_m3'])
# worst cells (with transfer) in w1/ht3
for key, arr_new, arr_aft in (('w3', po['w'][:, :, 2], w_after[:, :, 2]), ('ht3', po['ht'][:, :, 2], ht_after[:, :, 2]), ('w1', po['w'][:, :, 0], w_after[:, :, 0])):
    e = np.abs(arr_new - arr_aft).max(1)
    o = np.argsort(-e)[:5]
    print('worst', key, [(int(ii[k]) + 1, int(jj[k]) + 1, float(e[k]), float(np.abs(arr_aft[k]).max())) for k in o])
    res[f'worst_{key}'] = [(int(ii[k]) + 1, int(jj[k]) + 1, float(e[k]), float(np.abs(arr_aft[k]).max())) for k in o]
json.dump(res, open(outp, 'w'), indent=1)
np.savez(outp.replace('.json', '.npz'), ii=ii, jj=jj, w_before=w_before, ht_before=ht_before, w_after=w_after, ht_after=ht_after, w_port=po['w'], ht_port=po['ht'],
         fs_before=fs_before, fs_after=fs_after, fs_port=po['fr_snow'], shrunk=info['shrunk'], expanded=info['expanded'])
