"""OCONV mass bookkeeping for the HBL iteration: MML(1)/BYMML(1) (D61, piece 2b).

OCNKPP.f sets the mixed-layer mass used by the GHATS flux and the density (DELTAM terms):
  pre-source (ITER=1):   MML(1)  = MO1(I,J)   * DXYPO(J)     OCNKPP.f:1816 (non-pole) and
                                                              :1742 (pole, MO1(1,JM))
  post-source (ITER>=2): MML(1)  = MO(I,J,1)  * DXYPO(J)     OCNKPP.f:1813 and :1737
  BYMML(1) = 1/MML(1) in both cases (OCNKPP.f:1814, 1817, 1738, 1743).
At ITER=2 the Fortran switches MML := MML0 (OCNKPP.f:1981-1983).

Only index 1 is live: the MML(L>1) values feed the OCN_GISS_SM branch (dead for this build).

S0ML0(1) = S0M(I,J,1) (OCNKPP.f:1865, non-pole; 1741 pole). It is an input here, since the
S0M array is not recorded by any existing dump.
"""


def mass_bookkeeping(mo1_prev, mo_cur, dxypo_j, iter_):
    """Returns (mml1, bymml1) for the current ITER. mo1_prev is MO1(I,J) (pre-step mass),
    mo_cur is MO(I,J,1) (current mass). OCNKPP.f:1813-1817 and :1963-1983."""
    if iter_ == 1:
        mml1 = mo1_prev * dxypo_j
    else:
        mml1 = mo_cur * dxypo_j
    bymml1 = 1.0 / mml1
    return mml1, bymml1
