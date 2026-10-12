"""R6: the call-independent tag-0 arrays (cloud/aerosol parametrisation, counts, flags: tags 0.20-0.26, 0.31) of the first LW and SW call of a
dump -> npz (lw0_<k>, sw0_<k>).   python r6_static0.py ffsoc_dump.bin out.npz"""
import sys
import numpy as np
import r6_stream as S

out, got = {}, set()
for r in S.stream(sys.argv[1], tags={0}):
    iso = "lw" if r["iso"] == 2 else "sw"
    if r["k"] in (20, 21, 22, 23, 24, 25, 26, 31) and (iso, r["k"]) not in got:
        out["%s0_%d" % (iso, r["k"])] = r["a"]
        got.add((iso, r["k"]))
    if len({g[0] for g in got}) == 2 and all((i, k) in got for i in ("lw", "sw") for k in (20, 21, 22, 23, 24, 31)):
        break
np.savez(sys.argv[2], **out)
print(sorted(out))
