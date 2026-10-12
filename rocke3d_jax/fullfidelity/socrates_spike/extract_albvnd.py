"""One-off: ALBVND(12,4,6) of ALBEDO.f (REAL*8 PARAMETER built from default-real literals -> float32 values) into data/albvnd.npz.
usage: python extract_albvnd.py /path/to/model/ALBEDO.f   (read only)"""
import re, sys
import numpy as np
src = open(sys.argv[1]).read().splitlines()
i0 = next(i for i, l in enumerate(src) if "ALBVND(NV,4,6)" in l and "PARAMETER" in l)
vals = []
for l in src[i0 + 1:]:
    if l.startswith("C") or l.startswith("!"):
        continue
    if "RESHAPE" in l or "(/" in l and False:
        continue
    if re.match(r"\s*[\*\+\d]\s", l) is None and not l.strip().startswith(("1", "2", "3", "4", "*", "+")):
        pass
    m = re.sub(r"^.{6}", "", l)                       # drop the label/continuation columns (1-6)
    toks = re.findall(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?", m)
    vals += [float(t) for t in toks]
    if len(vals) >= 288:
        break
assert len(vals) == 288, len(vals)
a = np.array(vals, dtype=np.float32).astype(np.float64).reshape((12, 4, 6), order="F")
np.savez("data/albvnd.npz", albvnd=a, season=np.array([15.0, 105.0, 196.0, 288.0]))
print(a.shape, a[:, 0, 0])
