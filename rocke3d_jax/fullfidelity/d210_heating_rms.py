"""D210: plain rms of (server output - recorded real output) / rms(recorded) for SRHR TRHR FSF SRDN per radiation step; input OUTDIR of d210_day.py --rad server."""
import json, sys
import numpy as np
import radiation_server as rs
o = sys.argv[1]; rows = []
for it in range(33312, 33366, 5):
    a = np.load(f'{o}/server_out_{it}.npz'); r = rs.read_packet(f'{rs.FF}/nov26_day/rsv_n26_{it}_out.bin')
    row = dict(it=it)
    for k in ('SRHR', 'TRHR', 'FSF', 'SRDN', 'T'):
        d = a[k] - r[k]; row[k] = dict(rms_diff=float(np.sqrt(np.mean(d * d))), rms_ref=float(np.sqrt(np.mean(r[k] ** 2))), rel=float(np.sqrt(np.mean(d * d)) / np.sqrt(np.mean(r[k] ** 2))))
    rows.append(row)
    print(it, {k: '%.2e' % row[k]['rel'] for k in ('SRHR', 'TRHR', 'FSF', 'SRDN', 'T')})
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
