"""Validate ocnmeso_jax (D86) against ocnmeso_vec (D82) on real OCNMESO inputs."""
import sys
import time
import numpy as np
import jax.numpy as jnp
from ocnmeso_vec import ocnstate_derived_vec, densgrad_vertical_vec, get_1d_mesodiff_vec
from ocnmeso_jax import ocnstate_derived_jax, densgrad_vertical_jax, get_1d_mesodiff_jax
from ocnmeso_compare import load_ocnstate_derived, load_densgrad
from odhorz_compare import load_lmm, FF_DEFAULT
from odhorz0_compare import load_record as load_odhorz0_record


def _worst(got, ref):
    return max(float(np.max(np.abs(np.asarray(g) - r))) / max(float(np.max(np.abs(r))), 1e-300)
               for g, r in zip(got, ref))


def main(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    osd = load_ocnstate_derived(f"{ff}/ffz_ocnstate_derived_{itime}.bin")
    dg = load_densgrad(f"{ff}/ffz_densgrad_{itime}.bin")
    dh = load_odhorz0_record(f"{ff}/ffz_odhorz0_{itime}.bin")["dh3d"]
    a1 = (osd["mo0"], osd["g0m0"], osd["gzm0"], osd["s0m0"], osd["szm0"], osd["opress0"], lmm,
          osd["vup"], osd["vdn"])
    a2 = (lmm, dh, dg["vbar"], dg["vup"], dg["vdn"], dg["vupu"], dg["vdnu"])
    j1 = tuple(jnp.asarray(a) for a in a1)
    j2 = tuple(jnp.asarray(a) for a in a2)
    w1 = _worst(ocnstate_derived_jax(*j1), ocnstate_derived_vec(*a1))
    w2 = _worst(densgrad_vertical_jax(*j2), densgrad_vertical_vec(*a2))
    k = float(np.max(np.abs(np.asarray(get_1d_mesodiff_jax(jnp.asarray(lmm))) - get_1d_mesodiff_vec(lmm))))
    t_np = time.time()
    for _ in range(20):
        ocnstate_derived_vec(*a1); densgrad_vertical_vec(*a2)
    t_np = (time.time() - t_np) / 20
    t_jx = time.time()
    for _ in range(20):
        ocnstate_derived_jax(*j1)[0].block_until_ready(); densgrad_vertical_jax(*j2)[0].block_until_ready()
    t_jx = (time.time() - t_jx) / 20
    print(f"{date}: state_derived rel {w1:.1e}, densgrad_vertical rel {w2:.1e}, k3d abs {k:.1e}; "
          f"warm numpy {t_np*1e3:.1f} ms -> jax {t_jx*1e3:.1f} ms (both functions)")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
