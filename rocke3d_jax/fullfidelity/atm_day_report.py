"""D150/D151: statistics, divergence curves and the noise-floor overlay for the one-model-day open-loop run.

Inputs (all under ff_data/nov26_day/ by default):
  real, unperturbed run:   ffa_step_<it>_e.bin  (end of step, D128 hooks; full prognostic state, MA for weights)
  our open-loop run:       ours_<tag>/step_<it>.npz  (written by atm_day_open_loop.py: T Q U V QCL QCI P at the end of every step)
  real perturbed members:  ffpt_<member>_<it>.bin    (D151 hook ffpt_dump: U V T Q QCL QCI P at the end of every step; member 'ctrl' is the
                           unperturbed control of the same binary, p1..p5 are the one-ulp perturbed members)

Metrics per step k and field X in (T, Q, U, V, QCL, QCI) and for P (surface pressure):
  rms      : RMS of (A - B) over the valid cells of all layers (scalars: i <= IMAXJ(j) so only one cell on the pole rows; U,V: B grid,
             rows j >= 2 (1-based) because row 1 is undefined)
  rms_l1/5/21 : the same on layers 1, 5 and 21
  zrms     : RMS over (j,l) of the zonal mean (over valid i) of the difference
  gmean    : mass-weighted global mean of (A - B) (weights AXYP(j) * MA(l,i,j) of the real step-end state)
  maxabs   : max |A - B|
The noise floor N(k) of a metric is the [min, max] over the real members of rms(member_k - control_k).  Verdict per step and metric:
ratio = ours / max_members; 'within' ratio <= 1, 'near' <= 2, 'beyond' > 2 (thresholds fixed here in advance, see RADIATION_AND_F2_PLAN section 3.2
for the proposed acceptance band [0.5 N, 2 N]); 'below' when ours < 0.5 * min_members.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clouds_condse_io as cio

IM, JM, LM = 72, 46, 40
FF = cio.FF_DEFAULT
DAYDIR = 'nov26_day'
FIELDS = ('T', 'Q', 'U', 'V', 'QCL', 'QCI')
LEVELS = (0, 4, 20)
VERDICT_ORDER = ('below', 'within', 'near', 'beyond')


def imaxj_default():
    im = np.full(JM, IM, int)
    im[0] = im[JM - 1] = 1
    return im


def valid_mask(name, imaxj=None):
    """(IM,JM) bool mask of defined cells."""
    imaxj = imaxj_default() if imaxj is None else imaxj
    m = np.zeros((IM, JM), bool)
    if name in ('U', 'V'):
        m[:, 1:] = True
    else:
        for j in range(JM):
            m[:imaxj[j], j] = True
    return m


def load_real_e(ff, daydir, it, fields=FIELDS + ('P', 'MA')):
    """Real end-of-step state from ffa_step_<it>_e.bin (reads the whole file; ~0.3 s)."""
    d = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it}_e.bin")
    return {k: d[k] for k in fields if k in d}


def load_member(ff, daydir, member, it):
    d = cio.read_cse(f"{ff}/{daydir}/ffpt_{member}_{it}.bin")
    return {k: d[k] for k in d}


def load_ours(ours_dir, it):
    z = np.load(f"{ours_dir}/step_{it}.npz")
    return {k: z[k] for k in z.files}


def weights(ma, axyp):
    """mass weights (IM,JM,LM) from MA (LM,IM,JM) and AXYP (JM,) or (IM,JM)."""
    w = np.transpose(ma, (1, 2, 0))
    ax = np.asarray(axyp, float)
    ax = ax[None, :, None] if ax.ndim == 1 else ax[:, :, None]
    return w * ax


def diff_metrics(a, b, name, w=None):
    """Statistics of a - b for one field (3D (IM,JM,LM), or 2D (IM,JM) for P)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = valid_mask(name)
    out = {}
    if d.ndim == 2:
        dv = d[m]
        out['rms'] = float(np.sqrt(np.mean(dv ** 2)))
        out['maxabs'] = float(np.abs(dv).max())
        z = np.array([d[m[:, j], j].mean() for j in range(JM) if m[:, j].any()])
        out['zrms'] = float(np.sqrt(np.mean(z ** 2)))
        if w is not None:
            w2 = w.sum(axis=2) if w.ndim == 3 else w
            out['gmean'] = float((d * w2)[m].sum() / w2[m].sum())
        return out
    m3 = m[:, :, None] & np.ones((1, 1, d.shape[2]), bool)
    dv = d[m3]
    out['rms'] = float(np.sqrt(np.mean(dv ** 2)))
    out['maxabs'] = float(np.abs(dv).max())
    for l in LEVELS:
        dl = d[:, :, l][m]
        out[f'rms_l{l + 1}'] = float(np.sqrt(np.mean(dl ** 2)))
    zm = np.zeros((JM, d.shape[2]))
    for j in range(JM):
        if m[:, j].any():
            zm[j] = d[m[:, j], j].mean(axis=0)
    jj = np.array([m[:, j].any() for j in range(JM)])
    out['zrms'] = float(np.sqrt(np.mean(zm[jj] ** 2)))
    if w is not None:
        out['gmean'] = float((d * w)[m3].sum() / w[m3].sum())
    return out


def state_metrics(A, B, w=None, fields=FIELDS + ('P',)):
    return {f: diff_metrics(A[f], B[f], f, w) for f in fields if f in A and f in B}


def field_scale(B, fields=FIELDS):
    """global std of the real field over valid cells and layers (context for the absolute RMS values)."""
    out = {}
    for f in fields:
        if f in B:
            m = valid_mask(f)[:, :, None] & np.ones((1, 1, B[f].shape[2]), bool)
            out[f] = float(np.std(B[f][m]))
    return out


# ------------------------------------------------------------------------------------------------ curves
def curves(ff, daydir, ours_dir, members, it0, nsteps, axyp, progress=None):
    """Per-step metrics: ours vs real, and every member vs control (the unperturbed real run).  Returns dict
    ours[k][field][metric], mem[name][k][field][metric], scale[k][field]."""
    res = dict(it0=it0, nsteps=nsteps, ours=[], mem={m: [] for m in members}, scale=[], gmean_real=[])
    for k in range(nsteps):
        it = it0 + k
        real = load_real_e(ff, daydir, it)
        w = weights(real['MA'], axyp)
        res['scale'].append(field_scale(real))
        res['gmean_real'].append({f: float((real[f] * w)[valid_mask(f)[:, :, None] & np.ones((1, 1, LM), bool)].sum()
                                           / w[valid_mask(f)[:, :, None] & np.ones((1, 1, LM), bool)].sum()) for f in ('T', 'Q')})
        if ours_dir and os.path.exists(f"{ours_dir}/step_{it}.npz"):
            res['ours'].append(state_metrics(load_ours(ours_dir, it), real, w))
        else:
            res['ours'].append(None)
        for m in members:
            p = f"{ff}/{daydir}/ffpt_{m}_{it}.bin"
            res['mem'][m].append(state_metrics(load_member(ff, daydir, m, it), real, w) if os.path.exists(p) else None)
        if progress:
            progress(k)
    return res


def _val(d, metric):
    """metric value of a metrics dict; 'absgmean' is |global mean difference|."""
    return abs(d['gmean']) if metric == 'absgmean' else d[metric]


def envelope(res, members, field, metric, k):
    v = [_val(res['mem'][m][k][field], metric) for m in members if res['mem'][m][k] is not None and field in res['mem'][m][k]
         and (metric in ('absgmean',) or metric in res['mem'][m][k][field])]
    return (min(v), max(v)) if v else (np.nan, np.nan)


def classify(ratio_max, ratio_min):
    if not np.isfinite(ratio_max):
        return 'n/a'
    if ratio_max <= 1.0:
        return 'below' if ratio_min < 0.5 else 'within'
    return 'near' if ratio_max <= 2.0 else 'beyond'


def overlay(res, members, metric='rms', fields=FIELDS + ('P',)):
    """Per step and field: ours, member min/max, ratio ours/max, class.  Returns rows[k][field] and first step beyond/near per field."""
    rows, first = [], {f: dict(near=None, beyond=None) for f in fields}
    for k in range(res['nsteps']):
        r = {}
        for f in fields:
            o = res['ours'][k]
            if o is None or f not in o:
                continue
            lo, hi = envelope(res, members, f, metric, k)
            if metric != 'absgmean' and metric not in o[f]:
                continue
            ours = _val(o[f], metric)
            rmax = ours / hi if hi > 0 else (np.inf if ours > 0 else 0.0)
            rmin = ours / lo if lo > 0 else (np.inf if ours > 0 else 0.0)
            c = classify(rmax, rmin)
            r[f] = dict(ours=ours, lo=lo, hi=hi, ratio=rmax, cls=c)
            if c in ('near', 'beyond') and first[f]['near'] is None:
                first[f]['near'] = k
            if c == 'beyond' and first[f]['beyond'] is None:
                first[f]['beyond'] = k
        rows.append(r)
    return rows, first


def summarize_classes(rows, fields=FIELDS + ('P',)):
    out = {}
    for f in fields:
        c = {}
        for r in rows:
            if f in r:
                c[r[f]['cls']] = c.get(r[f]['cls'], 0) + 1
        out[f] = c
    return out


def class_counts(res, members, nsteps=None, fields=FIELDS + ('P',),
                 metrics=('rms', 'rms_l1', 'rms_l5', 'rms_l21', 'zrms', 'absgmean')):
    """for every (field, metric): counts of within/near/beyond/below over steps 0..nsteps-1 and the largest ratio ours/max-member (steps >= 3)."""
    n = nsteps or res['nsteps']
    out = {}
    for mt in metrics:
        rows, first = overlay(res, members, mt, fields)
        for f in fields:
            c, rmax, rarg = {}, 0.0, None
            for k, r in enumerate(rows[:n]):
                if f in r:
                    c[r[f]['cls']] = c.get(r[f]['cls'], 0) + 1
                    if k >= 3 and np.isfinite(r[f]['ratio']) and r[f]['ratio'] > rmax:
                        rmax, rarg = r[f]['ratio'], k
            if c:
                out[f"{f}.{mt}"] = dict(counts=c, max_ratio=rmax, at_step=rarg, first_near=first[f]['near'], first_beyond=first[f]['beyond'])
    return out


def selected_steps_table(res, members, steps=(0, 1, 2, 5, 11, 23, 35, 47, 53), fields=('T', 'Q', 'U', 'V', 'P', 'QCL', 'QCI'),
                         metrics=('rms', 'zrms', 'absgmean')):
    lines = ['| field.metric | ' + ' | '.join(f"k={k}" for k in steps if k < res['nsteps']) + ' |', '|---|' + '---|' * len([k for k in steps if k < res['nsteps']])]
    for mt in metrics:
        rows, _ = overlay(res, members, mt, fields)
        for f in fields:
            cells = []
            for k in steps:
                if k >= res['nsteps']:
                    continue
                r = rows[k].get(f)
                cells.append('-' if r is None else f"{r['ours']:.1e} [{r['lo']:.1e}..{r['hi']:.1e}]")
            lines.append(f"| {f}.{mt} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def growth_table(res, members, fields=('T', 'Q', 'U', 'V'), k0=3):
    """e-folding rate per step of rms(difference) from step k0 on: ours and each member."""
    out = {}
    for f in fields:
        row = {'ours': growth_exponent([o[f]['rms'] for o in res['ours'] if o], k0)}
        for m in members:
            row[m] = growth_exponent([o[f]['rms'] for o in res['mem'][m] if o], k0)
        out[f] = row
    return out


def cross_curve(ours_a, ours_b, ff, daydir, it0, nsteps, axyp):
    """rms etc. of (ours_b - ours_a) per step (two different implementations/rounding paths of OUR port against each other, e.g. numpy vs JAX
    dynamics); the real MA of each step provides the weights."""
    out = []
    for k in range(nsteps):
        it = it0 + k
        w = weights(load_real_e(ff, daydir, it, fields=('MA',))['MA'], axyp)
        out.append(state_metrics(load_ours(ours_b, it), load_ours(ours_a, it), w))
    return out


def growth_exponent(series, k0=5):
    """least-squares slope of log(rms) per step over steps >= k0 (e-folding rate), NaN if not enough positive values."""
    y = np.array([s for s in series[k0:]], float)
    x = np.arange(k0, k0 + len(y))
    ok = y > 0
    if ok.sum() < 4:
        return float('nan')
    return float(np.polyfit(x[ok], np.log(y[ok]), 1)[0])


def markdown(res, members, rows, first, title='one-model-day open-loop vs real'):
    lines = [f"### {title}", '',
             'RMS of (ours - real) per step vs the real run\'s own noise floor [min..max over members of rms(member - control)]', '']
    hdr = '| step | itime | ' + ' | '.join(f"{f} ours [floor]" for f in ('T', 'Q', 'U', 'V')) + ' | P ours [floor] |'
    lines += [hdr, '|' + '---|' * (hdr.count('|') - 1)]
    for k, r in enumerate(rows):
        if not r:
            continue
        cells = []
        for f in ('T', 'Q', 'U', 'V', 'P'):
            if f in r:
                c = r[f]
                cells.append(f"{c['ours']:.2e} [{c['lo']:.1e}..{c['hi']:.1e}] {c['cls']}")
            else:
                cells.append('-')
        lines.append(f"| {k} | {res['it0'] + k} | " + ' | '.join(cells) + ' |')
    lines += ['', 'class counts (steps): ' + json.dumps(summarize_classes(rows)),
              'first step ours is "near" or worse / "beyond" the max member (0-based step index): ' + json.dumps(first)]
    return '\n'.join(lines)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def plot(res, members, path, fields=('T', 'Q', 'U', 'V'), metric='rms'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = np.arange(res['nsteps'])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, f in zip(ax.flat, list(fields) + ['P', 'QCL']):
        for m in members:
            y = [res['mem'][m][k][f][metric] if res['mem'][m][k] is not None else np.nan for k in ks]
            a.semilogy(ks, y, lw=1, alpha=.7, label=m)
        y = [res['ours'][k][f][metric] if res['ours'][k] is not None else np.nan for k in ks]
        a.semilogy(ks, y, 'k', lw=2.2, label='ours (open loop)')
        a.set_title(f"{f}: rms difference to the real run"), a.set_xlabel('step (30 min)')
        a.grid(alpha=.3)
    ax.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ff', default=FF)
    ap.add_argument('--daydir', default=DAYDIR)
    ap.add_argument('--ours', required=True, help='directory with step_<it>.npz of our run')
    ap.add_argument('--members', default='p1,p2,p3,p4,p5')
    ap.add_argument('--ours2', default=None, help='second run of our port (e.g. the JAX-dynamics run): adds its overlay and its divergence from --ours')
    ap.add_argument('--it0', type=int, default=33312)
    ap.add_argument('--nsteps', type=int, default=48)
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    ap.add_argument('--png', default=None)
    a = ap.parse_args(argv)
    import dyn_glue_io as gio
    axyp = gio.load_g('nov26', a.ff)['axyp']
    members = a.members.split(',') if a.members else []
    res = curves(a.ff, a.daydir, a.ours, members, a.it0, a.nsteps, axyp)
    rows, first = overlay(res, members)
    txt = markdown(res, members, rows, first)
    cc = class_counts(res, members, a.nsteps)
    txt += '\n\n### selected steps: ours [member floor min..max]\n\n' + selected_steps_table(res, members)
    gt = growth_table(res, members)
    txt += '\n\ne-folding rate per step of rms(difference) from step 3: ' + json.dumps({f: {k: round(v, 4) for k, v in r.items()} for f, r in gt.items()})
    txt += '\n\nclass counts per (field.metric): ' + json.dumps({k: v['counts'] for k, v in cc.items()})
    txt += '\n\nmax ratio ours/max-member (steps>=3) per (field.metric): ' + json.dumps({k: [round(v['max_ratio'], 2), v['at_step']] for k, v in cc.items()})
    print(txt)
    if a.ours2:
        res2 = curves(a.ff, a.daydir, a.ours2, members, a.it0, a.nsteps, axyp)
        rows2, first2 = overlay(res2, members)
        cc2 = class_counts(res2, members, a.nsteps)
        cross = cross_curve(a.ours, a.ours2, a.ff, a.daydir, a.it0, a.nsteps, axyp)
        txt += '\n\n### second run of our port (' + os.path.basename(a.ours2) + ') vs real\n\n' + selected_steps_table(res2, members)
        txt += '\n\nclass counts (rms): ' + json.dumps({k: v['counts'] for k, v in cc2.items() if k.endswith('.rms')})
        txt += '\n\n### divergence of the two runs of our port from each other (rms; members floor for comparison = real member vs control)\n\n| step | ' + ' | '.join(f + ' ours2-ours1 [floor]' for f in ('T', 'Q', 'U', 'V')) + ' |\n|---|---|---|---|---|\n'
        for k in (0, 1, 2, 5, 11, 23, 35, 47, min(53, a.nsteps - 1)):
            cells = []
            for f in ('T', 'Q', 'U', 'V'):
                lo, hi = envelope(res, members, f, 'rms', k)
                cells.append(f"{cross[k][f]['rms']:.1e} [{lo:.1e}..{hi:.1e}]")
            txt += f"| {k} | " + ' | '.join(cells) + ' |\n'
        res['ours2'] = res2['ours']; res['cross'] = cross
    if a.md:
        open(a.md, 'w').write(txt + '\n')
    if a.json:
        json.dump(to_jsonable(dict(res=res, rows=rows, first=first, class_counts=cc, growth=gt)), open(a.json, 'w'))
    if a.png:
        plot(res, members, a.png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
