"""Loaders for the D107-D109 LSCOND dumps (instrumentation/CLOUDS2_lscond.f90.patch + ATM_DRV_clouds_lscond.f.patch).

Files (big-endian f8 streams, ff_data/<date>/): ffc_ls_bnd_<itime>.bin (LSCOND call boundary: entry and exit),
ffc_ls_mid_<itime>.bin (checkpoint after the main L loop, before CTEI), ffc_ls_tail_<itime>.bin (entry/exit of the
particle size / optical thickness tail block), ffc_ls_consts.txt.  Every record starts with a 6-double header
itime, kind (1 bnd, 2 mid, 3 tail), ncall, i, j, nbody; the body layouts are listed below (LM = 40, NMOM = 9; arrays are
index 0..LM-1 = layers 1..LM; "(LM+1)" arrays have index 0..LM = layers 1..LM+1).

Sampling: bnd/mid every 10th LSCOND call of the step (FFC_LS_STRIDE_B=10), tail every 4th (FFC_LS_STRIDE_T=4); a call
is taken when mod(ncall + (itime - FFD_START), stride) == 0, so every step samples a different column subset.
3170 LSCOND calls per step (72x44 interior columns + the two pole columns, which are computed for I=1 only), so 317 bnd/mid and about 792 tail records per step.  UM/VM are recorded for K=1..4 only (the pole rows J=1, JM have KMAX=72, all others 4).
"""
import glob
import numpy as np

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
LM = 40
NMOM = 9
HDR = 6

PAR = "bybr rimax rwmax rwcldox rcldlx rcldix wmui cmx u00a u00b rtemp use_vmp do_blu00 wconst scdncw scdnci bydtsrc dtsrc".split()
SCAL = "pearth dcl lmcld kmax".split()
IN_LM = ("qcll qcil svlatl svlat1 svwmxl sdl vsubl fssl ttoldl aq dpdt pl plk airm byam u00l pdsigl00 taumcl").split()
STATE_LM = ("tl ql th rh qclx qcix svlhxl cldsavl cldssl taussl tausslip csizel csizelip cldsal cldsv1 dqlsc sm qm "
            "rh1 wmpr qlss qiss sshr dctei").split()
LOC_LM = "cleara rhf rh00 er ec prep qheatl qheati qheat".split()
EXIT_SC = "prcpss hcndss wmsum ierr lerr wmerr".split()
TAIL_SC = "pearth lmcld dcl ckij hcndss bybr rimax rwmax rwcldox rcldlx rcldix use_vmp".split()
TAIL_IN_LM = ("cldssl qclx qcix svlhxl tl pl airm fssl cleara qheatl ec er prep wmpr taumcl taussl tausslip csizel "
              "csizelip cldsal cldsv1 qlss qiss").split()
TAIL_OUT_LM = "svlhxl qclx qcix cldssl taussl tausslip csizel csizelip cldsal cldsv1 qlss qiss wmpr".split()


class _Cur:
    def __init__(self, a):
        self.a, self.p = a, 0

    def take(self, n):
        v = self.a[:, self.p:self.p + n]
        self.p += n
        return v

    def scal(self, names, out, pre=""):
        for n in names:
            out[pre + n] = self.take(1)[:, 0]

    def lm(self, names, out, pre="", n=LM):
        for k in names:
            out[pre + k] = self.take(n)

    def state(self, out, pre=""):
        self.lm(STATE_LM, out, pre)
        out[pre + "lhp"] = self.take(LM + 1)
        out[pre + "prebar1"] = self.take(LM + 1)
        out[pre + "qmom"] = self.take(NMOM * LM).reshape(-1, LM, NMOM)
        out[pre + "smom"] = self.take(NMOM * LM).reshape(-1, LM, NMOM)
        out[pre + "um"] = self.take(4 * LM).reshape(-1, LM, 4)
        out[pre + "vm"] = self.take(4 * LM).reshape(-1, LM, 4)


def _read(files, nbody):
    parts = []
    for f in files:
        raw = np.fromfile(f, dtype=">f8")
        rl = HDR + nbody
        assert raw.size % rl == 0, (f, raw.size, rl)
        r = raw.reshape(-1, rl).astype(np.float64)
        assert np.all(r[:, 5] == nbody)
        parts.append(r)
    return np.concatenate(parts) if parts else None


def _files(date, kind, ff):
    return sorted(glob.glob(f"{ff}/{date}/ffc_ls_{kind}_[0-9]*.bin"))


def _hdr(r, out):
    for k, n in enumerate("itime kind ncall i j".split()):
        out[n] = r[:, k]


NB_BND = len(PAR) + len(SCAL) + len(IN_LM) * LM + 3 * LM + (LM + 1) + 4 + 2 * (len(STATE_LM) * LM + 2 * (LM + 1) +
                                                                              2 * NMOM * LM + 2 * 4 * LM) + len(EXIT_SC)
NB_MID = 2 + (len(STATE_LM) * LM + 2 * (LM + 1) + 2 * NMOM * LM + 2 * 4 * LM) + len(LOC_LM) * LM + 2 * (LM + 1)
NB_TAIL = len(TAIL_SC) + len(TAIL_IN_LM) * LM + (LM + 1) + 1 + len(TAIL_OUT_LM) * LM


def load_bnd(date, ff=FF_DEFAULT):
    """-> dict; inputs keyed by name; state arrays as 'in_<name>' (entry) and 'out_<name>' (exit); exit scalars plain."""
    fs = _files(date, "bnd", ff)
    if not fs:
        return None
    r = _read(fs, NB_BND)
    o = {}
    _hdr(r, o)
    c = _Cur(r[:, HDR:])
    c.scal(PAR, o)
    c.scal(SCAL, o)
    c.lm(IN_LM, o)
    o["rndssl"] = c.take(3 * LM).reshape(-1, LM, 3)
    o["precnvl"] = c.take(LM + 1)
    o["ra"] = c.take(4)
    c.state(o, "in_")
    c.state(o, "out_")
    c.scal(EXIT_SC, o)
    assert c.p == NB_BND
    return o


def load_mid(date, ff=FF_DEFAULT):
    fs = _files(date, "mid", ff)
    if not fs:
        return None
    r = _read(fs, NB_MID)
    o = {}
    _hdr(r, o)
    c = _Cur(r[:, HDR:])
    c.scal(["prcpss", "hcndss"], o)
    c.state(o)
    c.lm(LOC_LM, o)
    o["prebar"] = c.take(LM + 1)
    o["preice"] = c.take(LM + 1)
    assert c.p == NB_MID
    return o


def load_tail(date, ff=FF_DEFAULT):
    fs = _files(date, "tail", ff)
    if not fs:
        return None
    r = _read(fs, NB_TAIL)
    o = {}
    _hdr(r, o)
    c = _Cur(r[:, HDR:])
    c.scal(TAIL_SC, o)
    c.lm(TAIL_IN_LM, o)
    o["lhp"] = c.take(LM + 1)
    o["out_wmsum"] = c.take(1)[:, 0]
    c.lm(TAIL_OUT_LM, o, "out_")
    assert c.p == NB_TAIL
    return o


def read_consts(path):
    out = {}
    for line in open(path):
        k, v = line.split()
        out[k] = float(v) if not k.startswith("stride") else int(v)
    return out
