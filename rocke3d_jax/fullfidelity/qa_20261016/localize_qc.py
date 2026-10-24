"""Localize the JAX - Fortran difference of cloud liquid (QCL) and cloud ice (QCI) at steps 0-10 of the nov26 day (review package)."""
import sys, os, json
import numpy as np

PKG = os.environ['REVIEW_PKG']   # required: the unpacked review package directory (see ENVIRONMENT.md)
sys.path.insert(0, PKG + '/scoring_tool')
import atm_day_report as RP

FF, DAY, IT0 = PKG + '/fortran_reference', 'nov26_day2', 33312
NAMES = ['ctrl', 'p1', 'p2', 'p3', 'p4', 'p5']
ours_dir = PKG + '/jax_run/ours_d193'
IM, JM, LM = 72, 46, 40
lon = -180 + 2.5 + 5.0 * np.arange(IM)
lat = -90 + 4.0 * np.arange(JM)
out = {}
for k in range(0, 11):
    it = IT0 + k
    runs = [RP.load_member(FF, DAY, nm, it) for nm in NAMES]
    jax = RP.load_ours(ours_dir, it)
    for f in ('QCL', 'QCI'):
        m = RP.valid_mask(f)
        m3 = m[:, :, None] & np.ones((1, 1, LM), bool)
        A = np.stack([r[f] for r in runs])            # (6,IM,JM,LM)
        J = jax[f]
        C = A[0]
        d = np.where(m3, J - C, 0.0)
        d2 = d ** 2
        tot = d2.sum()
        flat = np.sort(d2.ravel())[::-1]
        top20 = flat[:20].sum() / tot if tot > 0 else float('nan')
        top100 = flat[:100].sum() / tot if tot > 0 else float('nan')
        # zero / non-zero mismatches against the control, for JAX and for each member
        nz = lambda x: (x > 0) & m3
        jax_flip = int((nz(J) ^ nz(C)).sum())
        mem_flip = [int((nz(A[i]) ^ nz(C)).sum()) for i in range(1, 6)]
        n_cloud = int(nz(C).sum())
        # layer share of squared error
        lay = d2.sum(axis=(0, 1)) / tot if tot > 0 else np.zeros(LM)
        top_l = np.argsort(lay)[::-1][:3]
        # worst cells
        idx = np.argsort(d2.ravel())[::-1][:5]
        cells = []
        for q in idx:
            i, j, l = np.unravel_index(q, d2.shape)
            rng = (float(A[:, i, j, l].min()), float(A[:, i, j, l].max()))
            cells.append(dict(lon=float(lon[i]), lat=float(lat[j]), layer=int(l + 1), ctrl=float(C[i, j, l]), jax=float(J[i, j, l]),
                              member_min=rng[0], member_max=rng[1], diff=float(d[i, j, l])))
        out[f'{k}_{f}'] = dict(rms=float(np.sqrt(d2[m3].mean())), top20_share=float(top20), top100_share=float(top100),
                               n_cloud_cells=n_cloud, jax_flips=jax_flip, member_flips=mem_flip,
                               top_layers=[(int(l + 1), float(lay[l])) for l in top_l], cells=cells)
json.dump(out, open(sys.argv[1] + '/localize_qc.json', 'w'), indent=1)
for f in ('QCL', 'QCI'):
    print(f'== {f}: step | rms | share of sq. error in top 20 / top 100 cells | cloudy cells | zero<->nonzero flips JAX vs members(min-max) | top layers (share)')
    for k in range(0, 11):
        o = out[f'{k}_{f}']
        print(f'{k:2d} | {o["rms"]:.2e} | {o["top20_share"]:.2f} / {o["top100_share"]:.2f} | {o["n_cloud_cells"]:5d} | {o["jax_flips"]:3d} vs {min(o["member_flips"])}-{max(o["member_flips"])} | '
              + ', '.join(f'L{l} {s:.2f}' for l, s in o['top_layers']))
for f in ('QCL', 'QCI'):
    for k in (3, 4):
        print(f'-- worst cells, {f}, step {k}')
        for c in out[f'{k}_{f}']['cells']:
            print('   ', {a: (round(b, 3) if a in ('lon', 'lat') else (f'{b:.3e}' if isinstance(b, float) else b)) for a, b in c.items()})
