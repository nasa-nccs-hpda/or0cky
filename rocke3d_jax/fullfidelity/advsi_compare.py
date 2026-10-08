"""D166: compare advsi_ff.advsi against the real ADVSI entry/exit dumps (ffadv_in_<it>.bin / ffadv_out_<it>.bin, written by
instrumentation/ICEDYN_DRV_advsi.f.patch).  Usage: python advsi_compare.py [dumpdir ...]   (default: ff_data/advsi_dumps/*)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advsi_ff as A  # noqa: E402

FF = __import__("os").environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
STATE = ('rsi', 'rsix', 'rsiy', 'rsisave', 'msi', 'snowi', 'hsi', 'ssi')
MASKED = ('msicnv', 'fwsim')   # defined only where FOCEAN > 0 and (for i > 1 at the poles) not at all: compared on processed cells


def processed_mask(foc):
    m = foc > 0
    m = m.copy()
    m[1:, 0] = False
    m[1:, A.JM - 1] = False
    return m


def compare_step(din, dout, geo_override=None):
    d, o = A.read_dump(din, dout)
    st = {k: d[k] for k in STATE}
    new, out = A.advsi(st, d['ausi'], d['avsi'], d['focean'], geo_override or d['geo'])
    res = {}
    for k in STATE:
        res[k] = float(np.max(np.abs(new[k] - o[k]))), int(np.count_nonzero(new[k] != o[k]))
    pm = processed_mask(d['focean'])
    for k in out:
        if k in MASKED:
            diff = np.where(pm, out[k] != o[k], False)
            mx = float(np.max(np.where(pm, np.abs(out[k] - o[k]), 0.0)))
        elif k == 'hsicnv':
            diff = out[k] != o[k]
            mx = float(np.max(np.abs(out[k] - o[k])))
        else:
            diff = out[k] != o[k]
            mx = float(np.max(np.abs(out[k] - o[k])))
        res[k] = mx, int(np.count_nonzero(diff))
    return res, d


def main(dirs):
    tot = 0
    bad = 0
    for dd in dirs:
        its = sorted(int(os.path.basename(p)[9:-4]) for p in glob.glob(f'{dd}/ffadv_in_*.bin'))
        nrsi = 0
        for it in its:
            res, d = compare_step(f'{dd}/ffadv_in_{it}.bin', f'{dd}/ffadv_out_{it}.bin')
            tot += 1
            worst = max(v[0] for v in res.values())
            nd = sum(v[1] for v in res.values())
            nrsi += int(np.count_nonzero((d['rsi'] > 0) & (d['focean'] > 0)))
            if nd:
                bad += 1
                print(f'{dd} it={it}: DIFFERENCES  worst abs {worst:.3e}  count {nd}', {k: v for k, v in res.items() if v[1]})
        print(f'{dd}: {len(its)} steps, steps with any difference: {bad}, mean ice cells/step {nrsi / max(len(its), 1):.0f}')
    print(f'TOTAL steps {tot}, with differences {bad}')


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(glob.glob(f'{FF}/advsi_dumps/*')))
