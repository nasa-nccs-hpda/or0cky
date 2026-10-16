"""D174: the CONDSE/RADIA random-number seed chain of the real model, computed instead of recorded.
Measured on nov26_day (54 steps), dec01 and jan01 (6 steps): with the ModelE LCG ix -> ix*69069+1 (mod 2^32),
  SEEDS[1] (seed at RADIA entry) = LCG^275790(SEEDS[0])            (= 3*29 draws x 3170 columns of CONDSE), every step;
  SEEDS[0] of the next step    = LCG^275790(SEEDS[0])              on non-radiation steps,
                               = LCG^405760(SEEDS[0])              on radiation steps (RADIA adds 129970 = 41 x 3170 draws to the main stream).
Only the seed of the first step of a run is a recorded value (it is not in the restart file).
"""
M = 1 << 32
N_CONDSE, N_STEP_RAD, N_STEP_NORAD = 275790, 405760, 275790


def jump(ix, n):
    A, C, ba, bc = 1, 0, 69069, 1
    while n:
        if n & 1:
            A, C = (A * ba) % M, (C * ba + bc) % M
        ba, bc = (ba * ba) % M, (bc * ba + bc) % M
        n >>= 1
    return (A * int(ix) + C) % M


def radia_seed(s0):
    return jump(s0, N_CONDSE)


def next_seed(s0, radiation_step):
    return jump(s0, N_STEP_RAD if radiation_step else N_STEP_NORAD)


_LCG_TAB = {}


def _lcg_table(n):
    """A[k] = 69069**k, C[k] = sum_{m<k} 69069**m (mod 2**32), k = 0..n: x_k = A[k]*x_0 + C[k]."""
    import numpy as np
    t = _LCG_TAB.get('t')
    if t is None or len(t[0]) <= n:
        a = np.empty(n + 1, dtype=np.uint64)
        c = np.empty(n + 1, dtype=np.uint64)
        ak, ck = 1, 0
        for k in range(n + 1):
            a[k], c[k] = ak, ck
            ak, ck = (ak * 69069) % M, (ck * 69069 + 1) % M
        t = _LCG_TAB['t'] = (a, c)
    return t


def rndss(ix0, imaxj, lmcld, lm=40, im=72, jm=46):
    """D221: RNDSS(3,LM,IM,JM) of one CONDSE call from its entry seed SEEDS[0], vectorised by LCG jump-ahead; identical to
    clouds_condse_ff.randu_stream (draw order j, i<IMAXJ(j), L=LMCLD..1, nr=1..3; L>LMCLD stays zero).  -> float64 array."""
    import numpy as np
    imaxj = [int(x) for x in imaxj]
    n = 3 * lmcld * sum(imaxj)
    a, c = _lcg_table(n)
    x = (a[1:n + 1] * np.uint64(int(ix0) & (M - 1)) + c[1:n + 1]) & np.uint64(M - 1)       # uint64 product < 2**64
    s = x.astype(np.int64)
    s = np.where(s >= (1 << 31), s - M, s)
    v = (s & ~np.int64(0xFF)).astype(np.float64) * 2.0 ** -32
    v = np.where(s < 0, 1.0 + v, v)
    out = np.zeros((3, lm, im, jm))
    k = 0
    for j in range(jm):
        m = imaxj[j]
        blk = v[k:k + m * lmcld * 3].reshape(m, lmcld, 3)          # (i, draw order L=lmcld-1..0, nr)
        out[:, :lmcld, :m, j] = np.transpose(blk[:, ::-1, :], (2, 1, 0))
        k += m * lmcld * 3
    return out
