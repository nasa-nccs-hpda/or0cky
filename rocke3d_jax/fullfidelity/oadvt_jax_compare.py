"""Validate oadvt_jax (D84) against oadvt_vec (D80-D81) and the real Fortran OADVT2 dumps."""
import sys
import time
import numpy as np
from oadvt_vec import oadvtx2_vec, oadvty2_vec, oadvtz2_vec, oadvt2_vec
from oadvt_jax import oadvtx2_jax, oadvty2_jax, oadvtz2_jax, oadvt2_jax
from oadvt2_compare import load_oadvt2_before, load_oadvt2_after, load_mmi, _load_smfinal
from odhorz_compare import load_lmm, load_lmv, load_lmu, FF_DEFAULT


def _worst(got, ref):
    return max(float(np.max(np.abs(np.asarray(g) - r))) / max(float(np.max(np.abs(r))), 1e-300)
               for g, r in zip(got, ref))


def main(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    before = load_oadvt2_before(f"{ff}/ffz_oadvt2_before_{itime}.bin")
    after = load_oadvt2_after(f"{ff}/ffz_oadvt2_after_{itime}.bin")
    mmi = load_mmi(date, itime)
    smu, smv = _load_smfinal(ff, itime)
    dt = before["dtdum"]
    for tag, names, ql in (("G0M", ("g0m", "gxmo", "gymo", "gzmo"), False),
                           ("S0M", ("s0m", "sxmo", "symo", "szmo"), True)):
        f = [before[n] for n in names]
        ref_x = oadvtx2_vec(*f, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm)
        rm, rx, ry, rz, ma = ref_x
        got_x = oadvtx2_jax(*f, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm)
        ref_y = oadvty2_vec(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv)
        got_y = oadvty2_jax(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv)
        ref_z = oadvtz2_vec(*ref_y, before["smw"], dt, ql, lmm)
        got_z = oadvtz2_jax(*ref_y, before["smw"], dt, ql, lmm)
        ref = oadvt2_vec(mmi, *f, dt, ql, smu, smv, before["smw"], lmu, lmv, lmm)
        t0 = time.time()
        got = oadvt2_jax(mmi, *f, dt, ql, smu, smv, before["smw"], lmu, lmv, lmm)
        t_full = time.time() - t0
        real = [after[n] for n in names]
        print(f"{date} {tag}: X {_worst(got_x, ref_x):.1e} Y {_worst(got_y, ref_y):.1e} "
              f"Z {_worst(got_z, ref_z):.1e} | full vs oadvt_vec {_worst(got, ref):.1e}, "
              f"vs real Fortran {_worst(got[1:], real):.1e} ({t_full:.2f}s, warm)")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
