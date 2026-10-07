"""Loaders for the D125 CONDSE boundary dumps (instrumentation/CLOUDS2_DRV_condse.f90.patch + ATM_DRV_clouds_condse.f.patch,
helper `ffcse_dump`, units 1330-1333).

Files in ff_data/<date>/ (big-endian streams):
  ffc_cse_in_<itime>.bin    state at CONDSE entry (after recalc_agrid_uv / replicate_uv_to_agrid, before `TLS=T`)
  ffc_cse_out_<itime>.bin   state at CONDSE exit (after avg_replicated_duv_to_vgrid and recalc_agrid_uv, before RINIT(seed))
  ffc_cse_geom.bin          once: AXYP, BYAXYP, RAVJ (IM,JM), IMAXJ, KMAXJ (JM)
  ffc_cse_consts.txt        dtsrc grav bygrav rgas lhe lhs lhm tf sha bysha deltx teeny isc lmcld lmcm do_blu00 use_vmp
Record: char*16 name, int32 rank, int32 n(1:4), float64 data in Fortran column-major order; shapes returned as the Fortran
shapes, e.g. TTOLD (LM,IM,JM), TMOM (9,IM,JM,LM), UKM (4,LM,IM,JM), RNDSS (3,LM,IM,JM; layers above LMCLD are undefined).
Fields only in `in`: GZ MWS (IM,JM,LM) PMIDOLD EGCM W2GCM PEK (LM,IM,JM) DCLEV PBLHT PBLPTOP TSAVG QSAVG USAVG VSAVG TGVAVG QGAVG RSI
FEARTH FLAND FOCEAN FLICE FLAKE (IM,JM) RNDSS.  Only in `out`: TLS QLS TMC QMC UALIJ VALIJ.  Both: SEEDS (ix0, seed, ix at the hook),
TMOM QMOM SNOAGE(3,IM,JM) P_ACC PM_ACC PREC EPREC PRECSS DDM1 DDMS TDN1 QDN1 DDML AIRX LMC(2,IM,JM) TTOLD QTOLD SVLHX SVLAT RHSAV CLDSAV
CLDSAV1 FSS TAUSS TAUSSIP TAUMC CLDSS CLDMC CSIZMC CSIZSS CSIZSSIP QLSS QISS QLMC QIMC and the 15 radiation hand-off arrays
W_CLOUD FRAC_ST_WATER FRAC_ST_ICE FRAC_CNV_WATER FRAC_CNV_ICE MIX_ST_WATER MIX_ST_ICE MIX_CNV_WATER MIX_CNV_ICE DIM_ST_WATER DIM_ST_ICE
DIM_CNV_WATER DIM_CNV_ICE FRAC_AREA_ST FRAC_AREA_CNV (all (LM,IM,JM)), UKM VKM (4,LM,IM,JM), UKMSP VKMSP UKMNP VKMNP (IM,LM).
U V T Q QCL QCI (IM,JM,LM; both files) and PK PMID PDSIG (LM,IM,JM), PEDN (LM+1,IM,JM) (`in` only) are also in the new files (added in
the second build of the patch because the old ffd_<itime>_pre_condse/post_condse dumps (ffdump_reader) exist only for the first two steps of
each window); `load_step` falls back to the old dumps when the new files lack them.
"""
import glob
import os

import numpy as np

import ffdump_reader as fr

FF_DEFAULT = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
IM, JM, LM = 72, 46, 40
NSTEP = 6


def read_cse(path):
    """-> dict name -> ndarray (Fortran shape)."""
    buf = open(path, "rb").read()
    out, off = {}, 0
    while off < len(buf):
        name = buf[off:off + 16].decode().strip()
        off += 16
        rank = int(np.frombuffer(buf, ">i4", 1, off)[0])
        off += 4
        n = [int(x) for x in np.frombuffer(buf, ">i4", 4, off)]
        off += 16
        shp = tuple(n[:rank])
        cnt = int(np.prod(shp))
        out[name] = np.frombuffer(buf, ">f8", cnt, off).astype(np.float64).reshape(shp, order="F")
        off += 8 * cnt
    return out


def read_text(path):
    d = {}
    for ln in open(path):
        t = ln.split()
        if len(t) == 2:
            d[t[0]] = t[1]
    return d


def load_geom(date, ff=FF_DEFAULT):
    g = read_cse(f"{ff}/{date}/ffc_cse_geom.bin")
    g["IMAXJ"] = g["IMAXJ"].astype(int)
    g["KMAXJ"] = g["KMAXJ"].astype(int)
    return g


def load_consts(date, ff=FF_DEFAULT):
    d = read_text(f"{ff}/{date}/ffc_cse_consts.txt")
    out = {}
    for k, v in d.items():
        if k == "use_vmp":
            out[k] = v.strip().upper().startswith("T")
        elif k in ("isc", "lmcld", "lmcm", "do_blu00"):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def _interior(a, lb, shape_tail=(IM, JM)):
    """Cut the halo off an ffdump_reader array (lb = lower bounds as allocated) to the 1:IM, 1:JM interior."""
    return a


def old_state(date, itime, tag, ff=FF_DEFAULT):
    """Old ffd_ dump (tag 'pre_condse' / 'post_condse') -> dict of interior arrays: U V T Q QCL QCI (IM,JM,LM),
    PK PMID PEDN PDSIG (LM,IM,JM)."""
    d = fr.read_dump(f"{ff}/{date}/ffd_{itime}_{tag}.bin")
    out = {}
    for k in ("U", "V", "T", "Q", "QCL", "QCI"):
        a, lb = d[k], d[k + "__lb"]
        j0 = 1 - lb[1]
        out[k] = a[:IM, j0:j0 + JM, :LM].copy()
    for k in ("PK", "PMID", "PEDN", "PDSIG"):
        a, lb = d[k], d[k + "__lb"]
        j0 = 1 - lb[2]
        out[k] = a[:LM + (1 if k == "PEDN" else 0), :IM, j0:j0 + JM].copy()
    return out


def load_step(date, itime, ff=FF_DEFAULT):
    """-> (inp, ref): inp = entry arrays (new `in` dump + old pre_condse), ref = exit arrays (new `out` dump + old post_condse).
    Returns (None, None) when the dumps are missing."""
    pin, pout = f"{ff}/{date}/ffc_cse_in_{itime}.bin", f"{ff}/{date}/ffc_cse_out_{itime}.bin"
    if not (os.path.exists(pin) and os.path.exists(pout)):
        return None, None
    inp, ref = read_cse(pin), read_cse(pout)
    if "T" not in inp or "PK" not in inp:          # dumps written before U,V,T,Q,QCL,QCI,PK,PMID,PEDN,PDSIG were added: use the old ffd_ dumps
        try:
            inp.update(old_state(date, itime, "pre_condse", ff))
            ref.update(old_state(date, itime, "post_condse", ff))
        except FileNotFoundError:
            return None, None
    return inp, ref


def have_dumps(date, ff=FF_DEFAULT):
    return os.path.exists(f"{ff}/{date}/ffc_cse_geom.bin") and bool(glob.glob(f"{ff}/{date}/ffc_cse_in_[0-9]*.bin"))
