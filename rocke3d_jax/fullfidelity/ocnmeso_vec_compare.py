"""Validate ocnmeso_vec (D82) against the scalar ocnmeso_ff ports on real OCNMESO inputs."""
import sys
import time
import numpy as np
from ocnmeso_ff import ocnstate_derived, densgrad_vertical, get_1d_mesodiff
from ocnmeso_vec import ocnstate_derived_vec, densgrad_vertical_vec, get_1d_mesodiff_vec
from ocnmeso_compare import load_ocnstate_derived, load_densgrad
from odhorz_compare import load_lmm, FF_DEFAULT
from odhorz0_compare import load_record as load_odhorz0_record


def _worst(got, ref):
    return max(float(np.max(np.abs(g - r))) for g, r in zip(got, ref))


def main(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    osd = load_ocnstate_derived(f"{ff}/ffz_ocnstate_derived_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    dh = load_odhorz0_record(f"{ff}/ffz_odhorz0_{itime}.bin")["dh3d"]
    a1 = (osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"], osd["opress0"], lmm,
          osd["vup"], osd["vdn"])
    a2 = (lmm, dh, dg["vbar"], dg["vup"], dg["vdn"], dg["vupu"], dg["vdnu"])
    t0 = time.time(); r1 = ocnstate_derived(*a1); ts1 = time.time() - t0
    t0 = time.time(); g1 = ocnstate_derived_vec(*a1); tv1 = time.time() - t0
    t0 = time.time(); r2 = densgrad_vertical(*a2); ts2 = time.time() - t0
    t0 = time.time(); g2 = densgrad_vertical_vec(*a2); tv2 = time.time() - t0
    k = np.max(np.abs(get_1d_mesodiff_vec(lmm) - get_1d_mesodiff(lmm)))
    print(f"{date}: state_derived max abs diff {_worst(g1, r1):.1e} ({ts1:.2f}s -> {tv1:.3f}s), "
          f"densgrad_vertical {_worst(g2, r2):.1e} ({ts2:.2f}s -> {tv2:.3f}s), k3d {k:.1e}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
