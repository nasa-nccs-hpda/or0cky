"""D177: recorded inputs of the coupled path that can be computed from our own state (items 2 and, see other sections, 1/3/4).

Item 2 (this section): the persistent PBL state.
Source: PBL_DRV.f.  PBL() reads, per (surface type, patch, i, j), the arrays atm%uabl/vabl/tabl/qabl/eabl(1:npbl,i,j)
(PBL_DRV.f:255-259) and atm%cmgs/chgs/cqgs(i,j) (:282-284) and writes them back after ADVANC (:341-348, :397) and sets
atm%ipbl(i,j)=1 (:400).  Nothing else of the PBL persists as a *record input* column: z0m (rec 50) is a local that DFLUX overwrites for
itype 1,2 before use and is the constant ROUGHL for itype 3,4; dskin (rec 29) is an output only in pbl_ff.
loadbl (PBL_DRV.f:1098-1237, called once per SURFACE step from SURFACE.f:409): for each type, a tile whose ipbl is 0 (no PBL call in the
previous step) takes the whole state (profiles, cm/ch/cq, ustar_pbl, lmonin_pbl) from a donor type of the same cell with ipbl == 1:
    ocean <- ice, else land      ice <- ocean      land ice <- land      land <- land ice, else ocean
then every ipbl is reset to 0 (set to 1 again by each PBL call).  The restart stores the arrays and ipbl as *_ocn01/ice01/gla01/lnd01.
So at SURFACE substep 1 of step k+1 the profiles and cm/ch/cq are exactly the substep-2 outputs of step k (bitwise, measured), the
restart supplies the first step, and a tile that is new gets its donor's state.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TYPES = {1: 'ocn', 2: 'ice', 3: 'gla', 4: 'lnd'}
# record column slices (0-based): inputs of a ffp record and the matching outputs (PBL_DRV.f.patch rec(51:89) in, rec(116:154) out)
IN_PROF = {'u': slice(50, 58), 'v': slice(58, 66), 't': slice(66, 74), 'q': slice(74, 82), 'e': slice(82, 89)}
OUT_PROF = {'u': slice(115, 123), 'v': slice(123, 131), 't': slice(131, 139), 'q': slice(139, 147), 'e': slice(147, 154)}
IN_C = {'cm': 33, 'ch': 34, 'cq': 35}
OUT_C = {'cm': 94, 'ch': 95, 'cq': 96}
DONORS = {1: (2, 4), 2: (1,), 3: (4,), 4: (3, 1)}      # loadbl order of donors
_NP = {'u': 8, 'v': 8, 't': 8, 'q': 8, 'e': 7}


class PBLCarry:
    """Per-type (JM, IM) state arrays of the PBL. Usage per SURFACE step: begin_step() (= loadbl), fill(rows) to build the
    substep-1 input columns, update(rows, out) after each substep with that substep's PBL outputs."""

    def __init__(self, st):
        self.st = st          # {itype: dict(u,v,t,q,e: (JM,IM,n), cm,ch,cq: (JM,IM), ipbl: (JM,IM) int)}

    @classmethod
    def from_restart(cls, date='nov26', ff=None):
        import netCDF4 as nc
        import surface_loop as L
        ff = ff or L.FF
        R = nc.Dataset(f"{ff}/_pristine_restarts/{L.RESTART[date]}")
        st = {}
        for t, nm in TYPES.items():
            s = nm + '01'
            d = {k: np.array(R.variables[f"{k}abl_{s}"][:], dtype=np.float64)[..., :_NP[k]] for k in _NP}
            for k, v in (('cm', 'cmgs'), ('ch', 'chgs'), ('cq', 'cqgs')):
                d[k] = np.array(R.variables[f"{v}_{s}"][:], dtype=np.float64)
            d['ipbl'] = np.rint(np.array(R.variables[f"ipbl_{s}"][:])).astype(int)
            st[t] = d
        R.close()
        return cls(st)

    def copy(self):
        return PBLCarry({t: {k: v.copy() for k, v in d.items()} for t, d in self.st.items()})

    def begin_step(self):
        """loadbl: initialise absent tiles from the donor type, then clear all ipbl."""
        old = {t: d['ipbl'].copy() for t, d in self.st.items()}
        for t, donors in DONORS.items():
            need = old[t] == 0
            done = np.zeros_like(need)
            for dn in donors:
                m = need & ~done & (old[dn] == 1)
                for k in ('u', 'v', 't', 'q', 'e', 'cm', 'ch', 'cq'):
                    self.st[t][k][m] = self.st[dn][k][m]
                done |= m
        for t in self.st:
            self.st[t]['ipbl'][:] = 0
        return self

    def fill(self, rows):
        """Overwrite the persistent input columns of ffp-layout rows (N,154) (substep-1 inputs) from the carried state."""
        rows = np.array(rows, dtype=np.float64)
        i = rows[:, 0].astype(int) - 1
        j = rows[:, 1].astype(int) - 1
        it = rows[:, 2].astype(int)
        for t in TYPES:
            m = it == t
            if not m.any():
                continue
            for k, sl in IN_PROF.items():
                rows[m, sl] = self.st[t][k][j[m], i[m]]
            for k, c in IN_C.items():
                rows[m, c] = self.st[t][k][j[m], i[m]]
        return rows

    def update(self, rows, out):
        """Store the PBL outputs of one substep. rows: the substep's input rows (for i, j, itype); out: dict u,v,t,q,e (N,8|7),
        cm,ch,cq (N,) as returned by pbl_ff.advanc_batch (pbl_compare.run)."""
        i = rows[:, 0].astype(int) - 1
        j = rows[:, 1].astype(int) - 1
        it = rows[:, 2].astype(int)
        for t in TYPES:
            m = it == t
            if not m.any():
                continue
            for k in _NP:
                self.st[t][k][j[m], i[m]] = np.asarray(out[k])[m][:, :_NP[k]]
            for k in ('cm', 'ch', 'cq'):
                self.st[t][k][j[m], i[m]] = np.asarray(out[k])[m]
            self.st[t]['ipbl'][j[m], i[m]] = 1
        return self

    @staticmethod
    def out_from_records(rows):
        """The PBL outputs stored in recorded ffp rows (outputs of the real call), same layout as `update` expects."""
        d = {k: rows[:, sl] for k, sl in OUT_PROF.items()}
        d.update({k: rows[:, c] for k, c in OUT_C.items()})
        return d


PERSIST_COLS = [(k, sl) for k, sl in IN_PROF.items()] + [(k, c) for k, c in IN_C.items()]


def diff_persistent(built, rec):
    """max |built - rec| per persistent quantity, and the count of unequal elements."""
    res = {}
    for k, sl in IN_PROF.items():
        d = np.abs(built[:, sl] - rec[:, sl])
        res[k] = (float(d.max()) if d.size else 0.0, int((d != 0).sum()), int(d.size))
    for k, c in IN_C.items():
        d = np.abs(built[:, c] - rec[:, c])
        res[k] = (float(d.max()) if d.size else 0.0, int((d != 0).sum()), int(d.size))
    return res
