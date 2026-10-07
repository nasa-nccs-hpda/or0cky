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
