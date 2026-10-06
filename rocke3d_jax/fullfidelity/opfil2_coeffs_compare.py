"""D138: calc_opfil2_coeffs port vs ffo_opcoef.bin (3 dates). usage: python opfil2_coeffs_compare.py"""
import numpy as np
import ocean_chain_io as C
from ocean_opfil2_coeffs import calc_opfil2_coeffs, IMZ2

if __name__ == '__main__':
    for date in ('jan01', 'nov26', 'dec01'):
        d = C.FF_DEFAULT + '/' + date
        g = C.load_geom(d)
        ref = np.fromfile(d + '/ffo_opcoef.bin', dtype='>f8').astype(np.float64)
        v, sset = calc_opfil2_coeffs(g['lmu'])
        print(date, 'length', len(v), len(ref), 'header', v[:5], ref[:5])
        if len(v) != len(ref):
            continue
        nred, nfft, nfil, nmn, nsm = [int(x) for x in v[:5]]
        # mask: smooth entries never assigned by the Fortran (uninitialised) are excluded and counted
        m = np.ones(len(v), bool); m[5:5 + nsm] = sset
        print('  unassigned smooth entries excluded:', int((~sset).sum()), ' ref values there: min/max',
              ref[5:5 + nsm][~sset].min() if (~sset).any() else None, ref[5:5 + nsm][~sset].max() if (~sset).any() else None)
        bad = (v != ref) & m
        print('  differing entries', int(bad.sum()), 'of', int(m.sum()), ' maxabs', float(np.abs(v - ref)[m].max()))
        o = 5 + nsm + nmn + nfft + 26 + 4 * nfil + 26
        print('  integer tables identical:', bool(np.array_equal(v[:o][m[:o]], ref[:o][m[:o]])),
              ' reduco: n', nred, 'differing', int((v[o:] != ref[o:]).sum()), 'maxabs', float(np.abs(v[o:] - ref[o:]).max()) if nred else 0)
