"""Spread versus RMSE of the ensemble mean for the six Fortran runs of the review package (control + five one-ulp members),
perfect-model (leave-one-out) setting, plus the JAX run placed against the same ensemble. Unweighted rms over valid cells and layers, as in the scoring tool."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKG = os.environ.get('REVIEW_PKG', '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/rocke3d_review_package_20261016')
sys.path.insert(0, PKG + '/scoring_tool')
import atm_day_report as RP

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
FF, DAY, IT0, NS = PKG + '/fortran_reference', 'nov26_day2', 33312, 54
FIELDS = ['T', 'U', 'V', 'Q', 'QCL', 'QCI']
NAMES = ['ctrl', 'p1', 'p2', 'p3', 'p4', 'p5']
n = len(NAMES)
ours_dir = PKG + '/jax_run/ours_d193'


def sel(x, name):
    m = RP.valid_mask(name)
    m3 = m[:, :, None] & np.ones((1, 1, x.shape[-1]), bool)
    return x[..., m3] if False else x[m3]


def vals(x, name):
    m = RP.valid_mask(name)
    m3 = m[:, :, None] & np.ones((1, 1, x.shape[2]), bool)
    return x[m3]


res = {f: {k: [] for k in ('spread6', 'proj_spread', 'rmse_loo', 'spread_loo', 'jax_rms_to_mean', 'rms_mean_vs_ctrl')} for f in FIELDS}
for k in range(NS):
    it = IT0 + k
    runs = [RP.load_member(FF, DAY, nm, it) for nm in NAMES]
    jax = RP.load_ours(ours_dir, it)
    for f in FIELDS:
        A = np.stack([vals(r[f], f) for r in runs])          # (6, N)
        J = vals(jax[f], f)
        mean6 = A.mean(axis=0)
        spread6 = float(np.sqrt(np.mean(A.var(axis=0, ddof=1))))
        proj = max(float(np.sqrt(np.mean((A[i] - A[0]) ** 2))) for i in range(1, n))
        # leave-one-out perfect model: truth = run j, ensemble = the other five
        se, sp = [], []
        for j in range(n):
            others = np.delete(A, j, axis=0)
            m = others.mean(axis=0)
            se.append(np.mean((m - A[j]) ** 2))
            sp.append(np.mean(others.var(axis=0, ddof=1)))
        res[f]['spread6'].append(spread6)
        res[f]['proj_spread'].append(proj)
        res[f]['rmse_loo'].append(float(np.sqrt(np.mean(se))))
        res[f]['spread_loo'].append(float(np.sqrt(np.mean(sp))))
        res[f]['jax_rms_to_mean'].append(float(np.sqrt(np.mean((J - mean6) ** 2))))
        res[f]['rms_mean_vs_ctrl'].append(float(np.sqrt(np.mean((mean6 - A[0]) ** 2))))
json.dump(res, open(OUT + '/spread_rmse.json', 'w'))

exp_loo = np.sqrt(1 + 1 / 5)
exp_jax = np.sqrt(1 + 1 / 6)
print(f'expected RMSE/spread for an exchangeable ensemble: leave-one-out {exp_loo:.3f}; extra member vs mean of 6 {exp_jax:.3f}')
print('steps >= 3: median [min, max] of ratio per field')
print(f'{"field":5s} {"Fortran RMSE/spread (LOO)":>34s} {"JAX rms to mean / spread6":>30s} {"project spread / spread6":>26s}')
for f in FIELDS:
    r = res[f]
    a = np.array(r['rmse_loo'][3:]) / np.array(r['spread_loo'][3:])
    b = np.array(r['jax_rms_to_mean'][3:]) / np.array(r['spread6'][3:])
    c = np.array(r['proj_spread'][3:]) / np.array(r['spread6'][3:])
    fm = lambda v: f'{np.median(v):.2f} [{v.min():.2f}, {v.max():.2f}]'
    print(f'{f:5s} {fm(a):>34s} {fm(b):>30s} {fm(c):>26s}')

fig, ax = plt.subplots(1, 6, figsize=(24, 3.6), constrained_layout=True)
st = np.arange(NS)
for i, f in enumerate(FIELDS):
    r = res[f]
    ax[i].plot(st, np.array(r['rmse_loo']) / np.array(r['spread_loo']), 'k-', label='Fortran: RMSE of ens. mean / spread (leave-one-out)')
    ax[i].plot(st, np.array(r['jax_rms_to_mean']) / np.array(r['spread6']), 'r-', label='JAX: rms to Fortran ens. mean / spread')
    ax[i].axhline(exp_loo, color='0.5', ls='--', lw=1, label=f'expected {exp_loo:.2f} if exchangeable')
    ax[i].set_ylim(0, 6); ax[i].set_title(f); ax[i].set_xlabel('step')
    if i == 0:
        ax[i].legend(fontsize=6)
fig.suptitle('Spread versus RMSE: Fortran six-run ensemble (control + five one-ulp members) and the JAX run against it', fontsize=11)
fig.savefig(OUT + '/spread_vs_rmse.png', dpi=110)
print('done')
