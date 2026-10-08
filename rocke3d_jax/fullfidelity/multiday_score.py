"""D192: multi-day acceptance scorer (ACCEPTANCE_CRITERIA section 4, rung F2, criterion "for the day").

Takes a per-step end-state sequence of OUR run (T U V Q P QCL QCI at the end of every step, the layout of atm_day_open_loop's
ours_<tag>/step_<it>.npz and of the 'filter' state of jax_coupled.Coupled.step) and classifies, per field and per step, the whole-column rms of
(ours - real) against the real noise floor N(k) = [min, max] over the five real one-ulp members of rms(member_k - control_k)
(D150/D151/D157 convention, implemented by REUSING atm_day_report: load_real_e, load_member, state_metrics, overlay, classify, class_counts).
Thresholds are those of atm_day_report and are not changed here: ratio = ours / max_member; 'within' <= 1, 'near' <= 2, 'beyond' > 2,
'below' = within and ours < 0.5 * min_member (counted as within for the criterion).

Criterion for the day (ACCEPTANCE 4): (a) within at every step; (b) never beyond 2x the largest member distance.  Both are reported separately
and together with the worst ratio per field (all steps, and steps >= 3 as in D157/D171 where the floor at k < 3 is ~1e-9 and ratios are not meaningful).
The scorer reports; it does not decide to relax a criterion.

Not a claim about longer windows (those need the 8 JAN1950 members and the D172 leave-one-out tool; not implemented here).

Usage (python):
    from multiday_score import score_dir, score_sequence, report_markdown
    res = score_dir('<ff>/nov26_day/ours_d171_ent', it0=33312, nsteps=54)
CLI:
    python multiday_score.py --ours DIR [--nsteps 54] [--json OUT.json] [--md OUT.md] [--cache MEMBERS.json]
    python multiday_score.py --chain-npz '<outdir>/nov26_*.npz' ...   (short-window adapter, see chain_states)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atm_day_report as RP

IT0_NOV26 = 33312
NSTEPS_DAY = 54
MEMBERS = ('p1', 'p2', 'p3', 'p4', 'p5')
SCORED = ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI')
LABEL_SHORT_WINDOW = 'SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4'
RADIATION_SENTENCE = 'radiation computed by the original Fortran (hybrid component)'
K_MIN_RATIO = 3  # ratios of steps >= 3 are quoted as "largest ratio" in D157/D171


# ------------------------------------------------------------------------------------------------ sequence sources
def dir_states(ours_dir, it0=IT0_NOV26, nsteps=NSTEPS_DAY):
    """provider k -> state dict (or None) from ours_<tag>/step_<it>.npz."""
    def get(k):
        p = f"{ours_dir}/step_{it0 + k}.npz"
        return RP.load_ours(ours_dir, it0 + k) if os.path.exists(p) else None
    return get


def as_state(d, fields=SCORED):
    """float64 NumPy end-state dict with the scored fields (accepts NumPy or jax arrays)."""
    out = {}
    for f in fields:
        if f in d:
            out[f] = np.asarray(d[f], dtype=float)
    return out


def sequence_provider(states):
    """provider from a list/dict of per-step state dicts (index k = step since the start)."""
    if isinstance(states, dict):
        return lambda k: as_state(states[k]) if k in states else None
    return lambda k: as_state(states[k]) if k < len(states) and states[k] is not None else None


def chain_states(arrays_by_step, prefix='filter/'):
    """Adapter for d191_chain.py / d191_run.py output.  d191_chain.py itself keeps the per-step end state only inside `arrays['filter']`
    (jax_coupled.Coupled.step(..., keep=True)); the chain JSON holds the C1 comparison, not states.  This accepts either
      - a list of `arrays` dicts (arrays['filter'] = the end state S of the step: T Q U V QCL QCI P in the end-state layout), or
      - a list of flat npz dicts / paths written by d191_run.validate ('filter/T', ...), or
      - a glob string of such npz files, sorted by step number given as '..._step<k>.npz' or by name.
    Returns a provider k -> state dict."""
    if isinstance(arrays_by_step, str):
        arrays_by_step = sorted(glob.glob(arrays_by_step))
    seq = []
    for a in arrays_by_step:
        if isinstance(a, (str, os.PathLike)):
            z = np.load(a)
            a = {k: z[k] for k in z.files}
        if 'filter' in a and isinstance(a['filter'], dict):
            seq.append(as_state(a['filter']))
        else:
            seq.append(as_state({k[len(prefix):]: v for k, v in a.items() if k.startswith(prefix)}))
    return sequence_provider(seq)


# ------------------------------------------------------------------------------------------------ scoring
def _members_available(ff, daydir, it0, nsteps, members):
    return [m for m in members if os.path.exists(f"{ff}/{daydir}/ffpt_{m}_{it0}.bin")]


def member_curves(ff, daydir, it0, nsteps, members, axyp, cache=None):
    """per step: rms etc. of (member - real unperturbed state) as in atm_day_report.curves (the real step-end state, not the 'ctrl' file, is the
    reference there).  Returns mem[name][k] = state_metrics or None; optional JSON cache keyed by (daydir, it0, members)."""
    key = f"{daydir}|{it0}|{','.join(members)}"
    if cache and os.path.exists(cache):
        c = json.load(open(cache))
        if c.get('key') == key and c.get('nsteps', 0) >= nsteps:
            return {m: c['mem'][m][:nsteps] for m in members}
    mem = {m: [] for m in members}
    for k in range(nsteps):
        it = it0 + k
        real = RP.load_real_e(ff, daydir, it)
        w = RP.weights(real['MA'], axyp)
        for m in members:
            p = f"{ff}/{daydir}/ffpt_{m}_{it}.bin"
            mem[m].append(RP.state_metrics(RP.load_member(ff, daydir, m, it), real, w) if os.path.exists(p) else None)
    if cache:
        json.dump(dict(key=key, nsteps=nsteps, mem=RP.to_jsonable(mem)), open(cache, 'w'))
    return mem


def score_sequence(get_state, it0=IT0_NOV26, nsteps=NSTEPS_DAY, ff=None, daydir=RP.DAYDIR, members=MEMBERS, axyp=None, cache=None,
                   fields=SCORED, label=None):
    """Score a per-step sequence.  get_state(k) -> state dict (or None if the step is not available).  Returns the atm_day_report `res` dict
    plus the scorer summary under res['score']."""
    ff = ff or RP.FF
    if axyp is None:
        import dyn_glue_io as gio
        axyp = gio.load_g('nov26', ff)['axyp']
    members = _members_available(ff, daydir, it0, nsteps, list(members))
    mem = member_curves(ff, daydir, it0, nsteps, members, axyp, cache)
    res = dict(it0=it0, nsteps=nsteps, ours=[], mem=mem, scale=[], gmean_real=[])
    for k in range(nsteps):
        S = get_state(k)
        if S is None:
            res['ours'].append(None)
            continue
        real = RP.load_real_e(ff, daydir, it0 + k)
        w = RP.weights(real['MA'], axyp)
        res['ours'].append(RP.state_metrics(S, real, w, fields=fields))
    res['score'] = summarize(res, members, fields, label)
    return res


def score_dir(ours_dir, it0=IT0_NOV26, nsteps=NSTEPS_DAY, **kw):
    return score_sequence(dir_states(ours_dir, it0, nsteps), it0, nsteps, **kw)


def summarize(res, members, fields=SCORED, label=None):
    """Per field: class per step (whole-column rms), counts, worst ratio (all steps, steps >= 3), steps near/beyond, secondary metrics, verdicts."""
    avail = [k for k, o in enumerate(res['ours']) if o is not None]
    n_have = len(avail)
    out = dict(steps_scored=avail, n_steps_scored=n_have, members=list(members), thresholds=dict(within=1.0, near=2.0, below=0.5),
               short_window=n_have < NSTEPS_DAY, per_field={}, radiation=RADIATION_SENTENCE)
    out['label'] = label or (LABEL_SHORT_WINDOW if n_have < NSTEPS_DAY else 'one model day (54 steps)')
    rows, first = RP.overlay(res, members, 'rms', fields)
    for f in fields:
        cls, ratio, ours, lo, hi = {}, {}, {}, {}, {}
        for k in avail:
            r = rows[k].get(f)
            if r is None:
                continue
            cls[k], ratio[k], ours[k], lo[k], hi[k] = r['cls'], r['ratio'], r['ours'], r['lo'], r['hi']
        counts = {}
        for c in cls.values():
            counts[c] = counts.get(c, 0) + 1
        ks_all = [k for k in cls if np.isfinite(ratio[k])]
        ks3 = [k for k in ks_all if k >= K_MIN_RATIO]
        worst_all = max(ks_all, key=lambda k: ratio[k]) if ks_all else None
        worst3 = max(ks3, key=lambda k: ratio[k]) if ks3 else None
        n_within = counts.get('within', 0) + counts.get('below', 0)
        out['per_field'][f] = dict(
            cls={str(k): v for k, v in cls.items()}, ratio={str(k): v for k, v in ratio.items()},
            counts=counts, n_within_incl_below=n_within,
            worst_ratio_all=(ratio[worst_all] if worst_all is not None else None), worst_step_all=worst_all,
            worst_ratio_k3=(ratio[worst3] if worst3 is not None else None), worst_step_k3=worst3,
            near_steps=[k for k, v in cls.items() if v == 'near'], beyond_steps=[k for k, v in cls.items() if v == 'beyond'],
            all_within=(len(cls) > 0 and n_within == len(cls)),
            never_beyond_2x=(len(cls) > 0 and counts.get('beyond', 0) == 0),
            beyond_only_before_k3=bool(counts.get('beyond', 0)) and all(k < K_MIN_RATIO for k, v in cls.items() if v == 'beyond'))
    out['secondary'] = _secondary(res, members, fields)
    pf = out['per_field']
    out['day_all_within'] = all(v['all_within'] for v in pf.values())
    out['day_never_beyond_2x'] = all(v['never_beyond_2x'] for v in pf.values())
    out['day_never_beyond_2x_k3'] = all(v['beyond_steps'] == [] or all(k < K_MIN_RATIO for k in v['beyond_steps']) for v in pf.values())
    wr = [(v['worst_ratio_k3'], f, v['worst_step_k3']) for f, v in pf.items() if v['worst_ratio_k3'] is not None]
    out['worst_ratio_k3'] = max(wr) if wr else None
    if out['day_all_within']:
        v = 'within at every scored step in every field, none beyond 2x'
    elif out['day_never_beyond_2x']:
        v = 'never beyond 2x of the largest member, but NOT within at every step (near at some steps)'
    elif out['day_never_beyond_2x_k3']:
        v = 'beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step'
    else:
        v = 'BEYOND 2x of the largest member at one or more steps >= 3: criterion not met'
    out['verdict'] = v
    if out['short_window']:
        out['verdict'] += ' -- ' + LABEL_SHORT_WINDOW
    return out


def _secondary(res, members, fields):
    """zrms and |global mean| classes (D157 quotes them), same counting as atm_day_report.class_counts."""
    sec = {}
    avail = [k for k, o in enumerate(res['ours']) if o is not None]
    n = (max(avail) + 1) if avail else 0
    cc = RP.class_counts(res, members, n, fields, metrics=('zrms', 'absgmean'))
    for key, v in cc.items():
        sec[key] = dict(counts=v['counts'], max_ratio=v['max_ratio'], at_step=v['at_step'])
    return sec


# ------------------------------------------------------------------------------------------------ report
def report_markdown(res, title='multi-day acceptance score (ACCEPTANCE 4, criterion for the day)'):
    s = res['score']
    L = [f"### {title}", '', f"{s['label']}. Members: {', '.join(s['members'])}. Steps scored: {s['n_steps_scored']} (0-based {s['steps_scored'][:1]}..{s['steps_scored'][-1:]}). "
         f"Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).", '',
         '| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |', '|---|---|---|---|---|---|---|---|']
    for f, v in s['per_field'].items():
        c = v['counts']
        wk3 = 'n/a' if v['worst_ratio_k3'] is None else f"{v['worst_ratio_k3']:.2f}"
        wall = 'n/a' if v['worst_ratio_all'] is None else f"{v['worst_ratio_all']:.2f}"
        L.append(f"| {f} | {v['n_within_incl_below']} | {c.get('near', 0)} | {c.get('beyond', 0)} | {wk3} | {v['worst_step_k3']} | {wall} | {v['worst_step_all']} |")
    L += ['', f"Criterion (a) within at every scored step, all fields: {s['day_all_within']}.  (b) never beyond 2x of the largest member: {s['day_never_beyond_2x']}"
          f" (steps >= 3 only: {s['day_never_beyond_2x_k3']}).  Worst ratio (steps >= 3): {s['worst_ratio_k3']}.", f"Verdict text: {s['verdict']}", '',
          RADIATION_SENTENCE + '.']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ff', default=RP.FF)
    ap.add_argument('--daydir', default=RP.DAYDIR)
    ap.add_argument('--ours', default=None, help='directory with step_<it>.npz')
    ap.add_argument('--chain-npz', default=None, help="glob of d191_run.validate npz files ('filter/T' keys), one per step, short-window adapter")
    ap.add_argument('--it0', type=int, default=IT0_NOV26)
    ap.add_argument('--nsteps', type=int, default=NSTEPS_DAY)
    ap.add_argument('--cache', default=None, help='JSON cache of the member curves')
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    a = ap.parse_args(argv)
    if bool(a.ours) == bool(a.chain_npz):
        ap.error('give exactly one of --ours, --chain-npz')
    prov = dir_states(a.ours, a.it0, a.nsteps) if a.ours else chain_states(a.chain_npz)
    res = score_sequence(prov, a.it0, a.nsteps, ff=a.ff, daydir=a.daydir, cache=a.cache)
    txt = report_markdown(res)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')
    if a.json:
        json.dump(RP.to_jsonable(dict(score=res['score'], ours=res['ours'])), open(a.json, 'w'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
