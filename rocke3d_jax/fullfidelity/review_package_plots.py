"""Quick-look plots for the SME: JAX run vs Fortran reference, nov26 day (54 steps). Reads the review package only."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG = os.environ.get('REVIEW_PKG') or sys.exit('REVIEW_PKG must name the review package directory')
# usage: REVIEW_PKG=<package dir> python review_package_plots.py <outdir>   (numpy, matplotlib; about 1.5 minutes)
sys.path.insert(0, PKG + '/scoring_tool')
import atm_day_report as RP

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
FF, DAY, IT0, NS = PKG + '/fortran_reference', 'nov26_day2', 33312, 54
IM, JM = 72, 46
lon = -180 + 2.5 + 5.0 * np.arange(IM)      # 5 deg x 4 deg grid (cell centres)
lat = -90 + 4.0 * np.arange(JM)
ours_dir = PKG + '/jax_run/ours_d193'


def mk(name):
    m = RP.valid_mask(name).astype(float)
    m[m == 0] = np.nan
    return m


def real(it):
    return RP.load_real_e(FF, DAY, it, fields=('T', 'U', 'V', 'Q', 'QCL', 'QCI', 'P'))


def ours(it):
    return RP.load_ours(ours_dir, it)


last = IT0 + NS - 1
R, O = real(last), ours(last)
R0, O0 = real(IT0), ours(IT0)

for k in ('T', 'U', 'V', 'Q', 'QCI', 'P'):
    r = R[k]
    mm = mk(k)
    rr = r * (mm if r.ndim == 2 else mm[:, :, None])
    print(k, r.shape, 'fortran min/max', np.nanmin(rr), np.nanmax(rr))

# ---------------------------------------------------------------- Fig 1: maps at the last step
def map_fig(level, fields, fname, title):
    fig, ax = plt.subplots(len(fields), 3, figsize=(15, 3.1 * len(fields)), constrained_layout=True)
    for i, f in enumerate(fields):
        m = mk(f)
        a = R[f][:, :, level] * m
        b = O[f][:, :, level] * m
        d = b - a
        lo, hi = np.nanpercentile(np.r_[a.ravel(), b.ravel()], [1, 99])
        dm = np.nanmax(np.abs(d)) or 1.0
        for j, (z, t, kw) in enumerate([(a, 'Fortran', dict(vmin=lo, vmax=hi, cmap='viridis')),
                                         (b, 'JAX', dict(vmin=lo, vmax=hi, cmap='viridis')),
                                         (d, 'JAX - Fortran', dict(vmin=-dm, vmax=dm, cmap='RdBu_r'))]):
            im = ax[i, j].pcolormesh(lon, lat, z.T, shading='auto', **kw)
            ax[i, j].set_title(f'{f}: {t}', fontsize=10)
            ax[i, j].set_xlabel('lon'); ax[i, j].set_ylabel('lat')
            fig.colorbar(im, ax=ax[i, j], shrink=0.85)
    fig.suptitle(title, fontsize=12)
    fig.savefig(f'{OUT}/{fname}', dpi=110)
    plt.close(fig)


map_fig(0, ['T', 'U', 'Q'], '1_maps_lowest_layer_step53.png',
        f'Lowest model layer, end of step 53 (itime {last}), nov26 day: Fortran vs JAX')
map_fig(12, ['T', 'U', 'QCI'], '2_maps_layer13_step53.png',
        f'Model layer 13 (counted from the surface), end of step 53 (itime {last}): Fortran vs JAX')

# ---------------------------------------------------------------- Fig 3: zonal-mean cross sections
def zmean(x, name):
    m = RP.valid_mask(name)
    out = np.full((JM, x.shape[2]), np.nan)
    for j in range(JM):
        if m[:, j].any():
            out[j] = x[m[:, j], j].mean(axis=0)
    return out


fig, ax = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
for i, f in enumerate(['T', 'U', 'Q']):
    a, b = zmean(R[f], f), zmean(O[f], f)
    d = b - a
    lo, hi = np.nanpercentile(np.r_[a.ravel(), b.ravel()], [1, 99])
    dm = np.nanmax(np.abs(d)) or 1.0
    for j, (z, t, kw) in enumerate([(a, 'Fortran', dict(vmin=lo, vmax=hi, cmap='viridis')),
                                     (b, 'JAX', dict(vmin=lo, vmax=hi, cmap='viridis')),
                                     (d, 'JAX - Fortran', dict(vmin=-dm, vmax=dm, cmap='RdBu_r'))]):
        im = ax[i, j].pcolormesh(lat, np.arange(1, z.shape[1] + 1), z.T, shading='auto', **kw)
        ax[i, j].set_title(f'zonal mean {f}: {t}', fontsize=10)
        ax[i, j].set_xlabel('lat'); ax[i, j].set_ylabel('model layer (1 = lowest)')
        fig.colorbar(im, ax=ax[i, j], shrink=0.85)
fig.suptitle(f'Zonal means, end of step 53 (itime {last}): Fortran vs JAX', fontsize=12)
fig.savefig(f'{OUT}/3_zonal_mean_step53.png', dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: difference over the 54 steps vs Fortran's own noise
def rms(x, y, name):
    m = RP.valid_mask(name)
    m3 = m[:, :, None] & np.ones((1, 1, x.shape[2]), bool)
    d = (x - y)[m3]
    return float(np.sqrt(np.mean(d ** 2)))


FIELDS = ['T', 'U', 'V', 'Q', 'QCI']
jax_rms = {f: [] for f in FIELDS}
mem_rms = {f: [[] for _ in range(5)] for f in FIELDS}
ctrl_vs_real = {f: [] for f in FIELDS}
for k in range(NS):
    it = IT0 + k
    r, o = real(it), ours(it)
    c = RP.load_member(FF, DAY, 'ctrl', it)
    for f in FIELDS:
        jax_rms[f].append(rms(o[f], r[f], f))
        ctrl_vs_real[f].append(rms(c[f], r[f], f))
    for mi in range(5):
        p = RP.load_member(FF, DAY, f'p{mi + 1}', it)
        for f in FIELDS:
            mem_rms[f][mi].append(rms(p[f], c[f], f))
print('max rms(ctrl - real) over fields:', {f: max(ctrl_vs_real[f]) for f in FIELDS})

fig, ax = plt.subplots(1, 5, figsize=(20, 3.8), constrained_layout=True)
steps = np.arange(NS)
for i, f in enumerate(FIELDS):
    mem = np.array(mem_rms[f])
    ax[i].fill_between(steps, mem.min(axis=0), mem.max(axis=0), color='0.75', label='5 Fortran members (1-ulp start nudge) vs Fortran control')
    ax[i].plot(steps, jax_rms[f], 'r-', lw=1.8, label='JAX vs Fortran')
    ax[i].set_yscale('log'); ax[i].set_title(f'{f}: rms difference per step', fontsize=10)
    ax[i].set_xlabel('step (0-53)')
    if i == 0:
        ax[i].set_ylabel('rms over valid cells and layers'); ax[i].legend(fontsize=7, loc='lower right')
fig.suptitle('Size of the JAX-Fortran difference compared with the Fortran model\'s own sensitivity to a 1-ulp start nudge', fontsize=12)
fig.savefig(f'{OUT}/4_difference_vs_fortran_noise.png', dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: start vs end, JAX only vs Fortran only (sanity: does the day evolve plausibly)
fig, ax = plt.subplots(2, 4, figsize=(18, 6), constrained_layout=True)
for i, (st, tag) in enumerate([(R0, 'start'), (R, 'end')]):
    pass
for j, f in enumerate(['T', 'U', 'Q', 'QCI']):
    m = mk(f)
    for i, (S1, S2, tag) in enumerate([(R0, O0, 'step 0'), (R, O, 'step 53')]):
        a = zmean(S1[f], f); b = zmean(S2[f], f)
        lv = [0, 8, 16]
        for L in lv:
            ax[i, j].plot(lat, a[:, L], '-', label=f'Fortran layer {L + 1}')
            ax[i, j].plot(lat, b[:, L], '--', label=f'JAX layer {L + 1}')
        ax[i, j].set_title(f'zonal mean {f}, {tag}', fontsize=10)
        ax[i, j].set_xlabel('lat')
        if i == 0 and j == 0:
            ax[i, j].legend(fontsize=6)
fig.suptitle('Zonal-mean profiles at the start and end of the day: Fortran (solid) vs JAX (dashed)', fontsize=12)
fig.savefig(f'{OUT}/5_zonal_profiles_start_end.png', dpi=110)
plt.close(fig)
print('done')
