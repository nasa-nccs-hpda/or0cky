"""Validate oadvt_vec (D80) against the scalar oadvty2/oadvtz2 on real OADVT2 inputs (the state after
the first X half-sweep, so the Y and Z sweeps see realistic fields), and the full oadvt2_vec against
the real Fortran 'after' dumps."""
import sys
import time
import numpy as np
from oadvt2_ff import oadvtx2, oadvty2, oadvtz2, oadvt2
from oadvt_vec import oadvtx2_vec, oadvty2_vec, oadvtz2_vec, oadvt2_vec
from oadvt2_compare import load_oadvt2_before, load_oadvt2_after, load_mmi, _load_smfinal
from odhorz_compare import load_lmm, load_lmv, load_lmu, FF_DEFAULT


def _worst(got, ref):
    return max(float(np.max(np.abs(g - r))) / max(float(np.max(np.abs(r))), 1e-300)
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
        rm, rx, ry, rz = [before[n] for n in names]
        t0 = time.time(); ref_x = oadvtx2(rm, rx, ry, rz, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm); t_sx = time.time() - t0
        t0 = time.time(); got_x = oadvtx2_vec(rm, rx, ry, rz, mmi.copy(), smu, 0.5 * dt, ql, lmu, lmm); t_vx = time.time() - t0
        print(f"{date} {tag}: X vs scalar {_worst(got_x, ref_x):.1e} ({t_sx:.2f}s -> {t_vx:.2f}s)")
        rm, rx, ry, rz, ma = ref_x
        t0 = time.time(); ref_y = oadvty2(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv); t_sy = time.time() - t0
        t0 = time.time(); got_y = oadvty2_vec(rm, rx, ry, rz, ma, smv, dt, ql, lmm, lmv); t_vy = time.time() - t0
        t0 = time.time(); ref_z = oadvtz2(*ref_y, before["smw"], dt, ql, lmm); t_sz = time.time() - t0
        t0 = time.time(); got_z = oadvtz2_vec(*got_y, before["smw"], dt, ql, lmm); t_vz = time.time() - t0
        wy = _worst(got_y, ref_y)
        wz = _worst(oadvtz2_vec(*ref_y, before["smw"], dt, ql, lmm), ref_z)
        full = oadvt2_vec(mmi, *[before[n] for n in names], dt, ql, smu, smv, before["smw"],
                          lmu, lmv, lmm)
        real = [after[n] for n in names]
        wr = _worst(full[1:], real)
        print(f"{date} {tag}: Y vs scalar {wy:.1e} ({t_sy:.2f}s -> {t_vy:.2f}s), "
              f"Z vs scalar {wz:.1e} ({t_sz:.2f}s -> {t_vz:.2f}s), "
              f"full vs real Fortran {wr:.1e}, chained Z {_worst(got_z, ref_z):.1e}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
