"""D207: wrapper around the (unchanged, tested) multiday_score.py for the TWO-DAY record set (nov26_day2, it 33312-33419, 108 steps) and for the day-2 window
(steps 54-107).  multiday_score.py itself parameterises it0, nsteps, daydir and the member cache; what is hard-coded there is only the LABEL / short-window flag
(NSTEPS_DAY = 54) and the default daydir; this wrapper passes daydir and its own labels and re-scores sub-windows by masking the provider.  Thresholds are those of
atm_day_report (within <= 1, near <= 2, beyond > 2 times the largest member rms) and are NOT changed.  The member floor N(k) at k >= 54 is the 5 one-ulp members
(perturbed at step 0 of the window) run for the same 108 steps (ff_data/nov26_day2/ffpt_p<k>_<it>.bin): it is the continuation of the day-1 curves, not a fresh
perturbation at the day-2 start.
    python d207_score.py --ours OUTDIR/ours_d193 [--nsteps 108] [--daydir nov26_day2] [--cache members_108.json] --json out.json --md out.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiday_score as MS
import atm_day_report as RP

IT0 = MS.IT0_NOV26
WINDOWS = {'day1 (steps 0-53)': (0, 54), 'day2 (steps 54-107)': (54, 108), 'both days (steps 0-107)': (0, 108)}


def masked(provider, lo, hi):
    return lambda k: provider(k) if lo <= k < hi else None


def score_window(ours_dir, lo, hi, nsteps, ff=None, daydir='nov26_day2', cache=None, label=None):
    prov = masked(MS.dir_states(ours_dir, IT0, nsteps), lo, hi)
    res = MS.score_sequence(prov, IT0, nsteps, ff=ff, daydir=daydir, cache=cache, label=label)
    s = res['score']
    s['short_window'] = False
    s['window'] = [lo, hi]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--ff', default=RP.FF)
    ap.add_argument('--daydir', default='nov26_day2')
    ap.add_argument('--nsteps', type=int, default=108)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--md', default=None)
    a = ap.parse_args(argv)
    avail = [k for k in range(a.nsteps) if os.path.exists(f"{a.ours}/step_{IT0 + k}.npz")]
    kmax = (max(avail) + 1) if avail else 0
    out, md = {}, []
    for name, (lo, hi) in WINDOWS.items():
        hi = min(hi, kmax)
        if hi <= lo:
            continue
        res = score_window(a.ours, lo, hi, a.nsteps, ff=a.ff, daydir=a.daydir, cache=a.cache,
                           label=f'{name}; D207 wrapper over multiday_score.py, members p1-p5 run for {a.nsteps} steps, daydir {a.daydir}')
        out[name] = dict(score=res['score'], ours=res['ours'])
        md.append(MS.report_markdown(res, title=f'D207 {name}'))
    txt = '\n\n'.join(md)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')
    if a.json:
        json.dump(RP.to_jsonable(out), open(a.json, 'w'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
