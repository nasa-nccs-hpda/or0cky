"""OPFIL2 application (OCNDYN2.f:opfil2, Stage 2, D74): the polar zonal smoother.

The setup coefficients (calc_opfil2_coeffs: SMOOTH, NMIN, REDUCO, segment tables) are taken as
recorded inputs from the run (ffz_opfil_coef), as the EOS is. This module ports the application:
the Fourier smoother on the FFT segments (OFFT/OFFTI, real 72-point transform) and the per-basin
matrix filter (with its running index into REDUCO, including wrap-around basins).

Arrays here are (IM, JM) 0-indexed in both axes; the Fortran X(I,J) is x[I-1, J-1].
"""
import numpy as np

IM = 72
JM = 46
IMZ2 = IM // 2
HWID_MAX = 15


def offt(f):
    """OFFT (OFFT72E.f): A(n) = sum f(k)cos(-2 pi n k/IM)/KMH, B(n) = sum f(k) sin(-2 pi n k/IM)/KMH,
    k = 1..IM, n = 0..IM/2, KMH = IM/2 for 0<n<IM/2 and IM for n = 0, IM/2."""
    k = np.arange(1, IM + 1)
    n = np.arange(0, IMZ2 + 1)
    ang = -2.0 * np.pi * np.outer(n, k) / IM
    kmh = np.where((n == 0) | (n == IMZ2), IM, IM // 2).astype(np.float64)
    a = (np.cos(ang) @ f) / kmh
    b = (np.sin(ang) @ f) / kmh
    return a, b


def _forward_matrix():
    """The 72x72 map from grid values to the retained coefficients [A(0..36), B(1..35)] (B(0), B(36)
    are identically zero). Built from offt, so it is exact by construction."""
    cols = []
    for k in range(IM):
        e = np.zeros(IM)
        e[k] = 1.0
        a, b = offt(e)
        cols.append(np.concatenate([a, b[1:IMZ2]]))
    return np.array(cols).T


_FM = _forward_matrix()
_FMINV = np.linalg.inv(_FM)


def offti(a, b):
    """Inverse of offt (exact inverse of the forward matrix; the filter acts on the coefficients)."""
    v = np.concatenate([a, b[1:IMZ2]])
    return _FMINV @ v


def coef_from_dump(v):
    """Build the 1-based coefficient accessors from the flat ffz_opcoef values (see ATM_DRV
    ffdump_opcoef). Returns dict with 1-based Fortran semantics:
      smooth(n, j) -> c['smooth'](n, j), nmin(j) -> c['nmin'](j), jfft(nn), n1fft(l), n2fft(l),
      jfil(nb), i1fil(nb), i2fil(nb), indx_fil(nb), n1fil(l), n2fil(l), reduco(indx)."""
    nred, nfft, nfil, nmn, nsm = [int(round(x)) for x in v[:5]]
    o = 5
    def take(n):
        nonlocal o
        out = v[o:o + n]; o += n; return out
    sm = take(nsm)
    nmin = take(nmn)
    jfft = take(nfft); n1fft = take(13); n2fft = take(13)
    jfil = take(nfil); i1fil = take(nfil); i2fil = take(nfil)
    indx_fil = take(nfil); n1fil = take(13); n2fil = take(13)
    reduco = take(nred)
    assert o == len(v)
    js0 = 2
    smooth = sm.reshape((IMZ2, nmn), order='F')           # smooth(n, j), n = 1..IMZ2, j = js0..
    def one(a):                                           # 1-based accessor
        return lambda k: a[int(k) - 1]
    return dict(
        smooth=lambda n, j: smooth[int(n) - 1, int(j) - js0],
        nmin=lambda j: int(nmin[int(j) - js0]),
        jfft=one(jfft), n1fft=one(n1fft), n2fft=one(n2fft),
        jfil=one(jfil), i1fil=one(i1fil), i2fil=one(i2fil), indx_fil=one(indx_fil),
        n1fil=one(n1fil), n2fil=one(n2fil), reduco=reduco)


def opfil2(x, l, jmin, jmax, c):
    """OPFIL2 on x (IM, JM) (0-based numpy copy of the Fortran X(1:IM, 1:JM)), layer l (1-based),
    band jmin..jmax (1-based j). c from coef_from_dump."""
    x = np.array(x, dtype=np.float64, copy=True)
    X = lambda i, j: x[i - 1, j - 1]          # noqa: E731  (1-based read)
    red = c['reduco']
    # FFT-based smoother on this layer's segments
    for nn in range(int(c['n1fft'](l)), int(c['n2fft'](l)) + 1):
        j = int(c['jfft'](nn))
        if j < jmin or j > jmax:
            continue
        a, b = offt(x[:, j - 1])
        for n in range(c['nmin'](j), IMZ2):
            a[n] *= c['smooth'](n, j)
            b[n] *= c['smooth'](n, j)
        a[IMZ2] *= c['smooth'](IMZ2, j)
        x[:, j - 1] = offti(a, b)
    # per-basin matrix filter
    for nb in range(int(c['n1fil'](l)), int(c['n2fil'](l)) + 1):
        j = int(c['jfil'](nb))
        if j < jmin or j > jmax:
            continue
        indx = int(c['indx_fil'](nb))              # Fortran INDX: incremented before each use
        i1 = int(c['i1fil'](nb))
        i2 = int(c['i2fil'](nb))
        if i2 <= IM:
            y = {i: X(i, j) for i in range(i1, i2 + 1)}
            for i in range(i1, i2 + 1):
                reduc = 0.0
                for k in range(max(i1, i - HWID_MAX), min(i2, i + HWID_MAX) + 1):
                    indx += 1
                    reduc += red[indx - 1] * y[k]
                x[i - 1, j - 1] -= reduc
        else:
            y = {}
            for i in range(i1, IM + 1):
                y[i] = X(i, j)
            for i in range(IM + 1, i2 + 1):
                y[i] = X(i - IM, j)
            for i in range(i1, IM + 1):
                reduc = 0.0
                for k in range(max(i1, i - HWID_MAX), min(i2, i + HWID_MAX) + 1):
                    indx += 1
                    reduc += red[indx - 1] * y[k]
                x[i - 1, j - 1] -= reduc
            for i in range(1, i2 - IM + 1):
                reduc = 0.0
                for k in range(max(i1, IM + i - HWID_MAX), min(i2, IM + i + HWID_MAX) + 1):
                    indx += 1
                    reduc += red[indx - 1] * y[k]
                x[i - 1, j - 1] -= reduc
    return x
