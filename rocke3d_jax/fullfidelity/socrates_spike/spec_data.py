"""Spectral data of the rundeck as host arrays.  Source: tag 100 records of an instrumented run (the arrays of `spectrum` as the SOCRATES
reader produced them from sp_{lw,sw}_ga7/sp_*_ga7_dsa and the *_k lookup files), saved once to npz by `python spec_data.py DUMP OUTDIR`.
Index convention: all arrays keep the Fortran axis order, 0-based on read; integers (band, gas, term indices) stay 1-based values."""
import sys, os
import numpy as np
import r2_parse as R

NAMES = {1: "k_lookup", 2: "k_lookup_sb", 3: "p_lookup", 4: "t_lookup", 5: "gf_lookup", 6: "n_gas_frac", 7: "k", 8: "w",
         9: "i_scale_fnc", 10: "i_scale_k", 11: "p_ref", 12: "t_ref", 13: "scale", 14: "l_doppler", 15: "doppler_cor",
         16: "i_band_k", 17: "index_absorb", 18: "n_band_absorb", 19: "i_scat", 20: "type_absorb", 21: "l_self_broadening",
         22: "index_sb", 30: "n_band_cont", 31: "index_cont", 32: "i_band_k_cont", 33: "k_lookup_cont", 34: "t_lookup_cont",
         35: "w_cont", 36: "index_cont_gas_1", 37: "index_cont_gas_2", 38: "i_cont_overlap_band", 39: "l_cont_major",
         40: "i_scat_cont", 41: "thermal_coeff", 42: "solar_flux_band"}


def extract(dump, outdir):
    for x in R.read(dump):
        if x["tag"] != 100:
            continue
        d = os.path.join(outdir, "spec_%s.npz" % ("sw" if x["iso"] == 1 else "lw"))
        cur = dict(np.load(d)) if os.path.exists(d) else {}
        cur[NAMES[x["k"]]] = x["a"]
        np.savez(d, **cur)


def load(path):
    return {k: v for k, v in np.load(path).items()}


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
