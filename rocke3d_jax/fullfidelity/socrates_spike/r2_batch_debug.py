"""Per-column error report of the batched radiance against tag 30 (which columns, and what is special about them).
usage: python r2_batch_debug.py DUMP SPECDIR2 ISO [max_cols]"""
import sys
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import r2_parse as R, spec_data as SD, radiance_batch as RB
from r2_e2e_check import calls
recs = R.read(sys.argv[1])
iso = int(sys.argv[3]); mx = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
spec = SD.load("%s/spec_%s.npz" % (sys.argv[2], "sw" if iso == 1 else "lw"))
cs = [c for c in calls(recs) if c["ref"] and c["iso"] == iso][:mx]
r = RB.columns_from_calls(cs)
res = RB.radiance(spec, iso, r)
err = np.zeros(len(cs)); cum_d = cum_u = None
for ib, (dn, up, dr) in enumerate(res, start=1):
    dn, up = np.asarray(dn), np.asarray(up)
    if iso == 2:
        cum_d = dn if cum_d is None else cum_d + dn
        cum_u = up if cum_u is None else cum_u + up
        dn, up = cum_d, cum_u
    for ci, c in enumerate(cs):
        if ib in c["ref"]:
            err[ci] = max(err[ci], float(np.abs(dn[ci] - c["ref"][ib][1][0]).max()), float(np.abs(up[ci] - c["ref"][ib][2][0]).max()))
order = np.argsort(-err)
print("columns", len(cs), "max err", err.max(), "n(err>1e-6)", int((err > 1e-6).sum()))
for ci in order[:12]:
    c = cs[ci]; rr = c["r"]
    print("col %5d ncall %6d err %.3e zen %.3f solar %.3f w_cloud max %.3f t_ground %.1f alb(dir,dif b1) %.3f %.3f tile %s p_top %.3f" %
          (ci, c["ncall"], err[ci], rr[15][0], rr[16][0], rr[10].max(), rr[18][0], rr[17][0, 1, 0], rr[17][0, 0, 0], rr[32][:2], rr[1][0, 0]))
good = order[-5:]
for ci in good:
    c = cs[ci]; rr = c["r"]
    print("good col %5d err %.3e zen %.3f w_cloud max %.3f" % (ci, err[ci], rr[15][0], rr[10].max()))
