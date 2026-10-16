"""Validate kppmix_ff.py's kppmix port against real Fortran dumps (D54).

Unlike most Stage-2 dumps (one record per itime, full (IM,JM,LMO) grid), `ffz_kppmix_*.bin` is a
per-call stream: one record per real `CALL KPPMIX`, which happens once per ocean column per
`OCONV` fixed-point iteration on HBL (up to 4 times per column per `OCEANS` call, see D54's
module docstring in kppmix_ff.py). Each record is fully self-contained (it carries its own
`zgrid`/`hwide`/`byhwide`, already ZSCALE-rescaled for that column, plus `ze`, which is the same
fixed global grid every record) so records can be validated independently of one another and of
the outer OCONV loop that produced them.
"""
import numpy as np
from odhorz0_compare import FF_DEFAULT

LMO = 13
RECLEN = 179  # doubles per record, see ATM_DRV.f's ffdump_kppmix (D54)


def load_kppmix_records(path):
    """Returns a list of dicts, one per real KPPMIX call in this itime's dump file."""
    raw = np.fromfile(path, dtype='>f8')
    assert raw.size % RECLEN == 0, (raw.size, RECLEN)
    nrec = raw.size // RECLEN
    recs = []
    for r in range(nrec):
        row = raw[r * RECLEN:(r + 1) * RECLEN]
        off = 0

        def take(n):
            nonlocal off
            v = row[off:off + n]
            off += n
            return v

        itime = take(1)[0]
        i = int(round(take(1)[0]))
        j = int(round(take(1)[0]))
        it = int(round(take(1)[0]))
        lmij = int(round(take(1)[0]))
        ze = take(LMO + 1)
        zgrid = take(LMO + 2)
        hwide = take(LMO + 2)
        byhwide = take(LMO + 2)
        shsq_flat = take(LMO)
        dvsq_flat = take(LMO)
        ustar = take(1)[0]
        bo = take(1)[0]
        bosol = take(1)[0]
        dbloc_flat = take(LMO)
        ritop_flat = take(LMO)
        akvm = take(LMO + 2)
        akvs = take(LMO + 2)
        akvg = take(LMO + 2)
        ghat_flat = take(LMO)
        hbl = take(1)[0]
        kbl = int(round(take(1)[0]))
        assert off == RECLEN

        def pad1(flat):
            a = np.zeros(LMO + 1)
            a[1:] = flat
            return a

        recs.append(dict(
            itime=itime, i=i, j=j, iter=it, lmij=lmij,
            ze=ze, zgrid=zgrid, hwide=hwide, byhwide=byhwide,
            shsq=pad1(shsq_flat), dvsq=pad1(dvsq_flat),
            ustar=ustar, bo=bo, bosol=bosol,
            dbloc=pad1(dbloc_flat), ritop=pad1(ritop_flat),
            akvm=akvm, akvs=akvs, akvg=akvg,
            ghat=pad1(ghat_flat), hbl=hbl, kbl=kbl,
        ))
    return recs
