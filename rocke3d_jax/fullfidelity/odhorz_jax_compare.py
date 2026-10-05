"""Validate odhorz_jax.odhorz_jax against odhorz_vec (D78) and the real ODHORZ outputs."""
import sys
import numpy as np
from odhorz_vec import odhorz_vec
from odhorz_jax import odhorz_jax, make_layer_step
from odhorz_compare import load_records, load_hocean, load_lmu
from odhorz0_compare import load_geom, load_lmv, FF_DEFAULT


def main(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_geom(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    recs = load_records(f"{ff}/ffz_odhorz_{itime}.bin")
    step = make_layer_step(lmm, lmu, lmv)
    worst_vec = 0.0
    worst_real = 0.0
    for rec in recs:
        args = (lmm, lmu, lmv, hocean, rec["dt"], rec["moh"], rec["uoh"], rec["voh"], rec["uodh"],
                rec["vodh"], rec["opboth"], rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"],
                rec["vod0"], rec["opbot0"], rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"])
        ref = odhorz_vec(*args)
        got = odhorz_jax(*args, step=step)
        for g, r, real in zip(got, ref, [rec["mo1"], rec["uo1"], rec["vo1"], rec["uod1"], rec["vod1"], rec["opbot1"]]):
            sc = max(np.max(np.abs(r)), 1e-300)
            worst_vec = max(worst_vec, float(np.max(np.abs(g - r))) / sc)
            worst_real = max(worst_real, float(np.max(np.abs(g - real))) / max(np.max(np.abs(real)), 1e-300))
    print(date, 'records', len(recs), 'worst vs odhorz_vec', f'{worst_vec:.1e}',
          'worst vs real Fortran', f'{worst_real:.1e}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
