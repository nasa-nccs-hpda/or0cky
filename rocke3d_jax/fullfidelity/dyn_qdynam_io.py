"""Readers for the QDYNAM dumps (D103-D106): ff_data/<date>/ffd_qdyn_*.bin (and ffd_qdynS<f>_* stress runs).

Producer: instrumentation/ATMDYN_qdynam.f.patch + QUS3D_qdynam.f.patch + ATM_DRV_dynE.f.patch (units 1140-1149).
All files are big-endian float64 streams in Fortran order; 1 QDYNAM call per physics step, 6 steps per date.

  ffd_qdyn_geom.bin  (once)  [im,jm,lm,KG2MB,BYIM(GEOM),BYIM(QUSCOM),BYIM(DIAG_COM)] AXYP(IM,JM) IMAXJ(JM)
  ffd_qdyn_<it>_in.bin       [itime,stress_factor,0,0] Q(IM,JM,LM) QMOM(9,IM,JM,LM) MAOLD(LM,IM,JM) MUs MVs MWs MB (IM,JM,LM)
                             (taken after MB=MAOLD*KG2MB*AXYP, before AADVQ0; MUs/MVs/MWs already scaled in a stress run)
  ffd_qdyn_<it>_q0.bin       [itime,ncyc,do_z_extra,0] NCYCXY(LM) NSTEPX(JM,LM) NSTEPZ_EXTRA(IM,JM) LMINZIJ(IM,JM)
                             LMAXZIJ(IM,JM) MW_EXTRA(IM,JM,LM) (raw memory where undefined) MUs MVs MWs (IM,JM,LM) AFTER
                             AADVQ0, PV_SOUTH(IM,LM), NI_CHECKFOBS_Y(JM,LM) NI_CHECKFOBS_Z(JM,LM), n_y + n_y*(j,l,i),
                             n_z + n_z*(j,l,i) (1-based Fortran indices) of the i_checkfobs lists
  ffd_qdyn_<it>_ain.bin      [itime] RM=Q*MB (IM,JM,LM) RMOM (9,IM,JM,LM)  (mass units at AADVQ entry)
  ffd_qdyn_<it>_ck.bin       stream of records, header [stage,nc,ncxy,l] (l 1-based):
                               0 after y-checkflux (before AADVQY): RMOM(my) RMOM(myy) of level l (IM,JM each)
                               1 after AADVQY, 2 after AADVQX: RM(IM,JM) RMOM(9,IM,JM) MMA(IM,JM) of level l
                               3 after z-checkflux (before AADVQZ): RMOM(mz) RMOM(mzz) of level l
                               4 after AADVQZ at iteration l (layers l-1,l): MWDN FDN FDN0 (IM,JM) FMOMDN(9,IM,JM)
                               6 after the L loop of cycle nc: RM(IM,JM,LM) RMOM(9,..) MMA(IM,JM,LM)
                               7 after the extra z advection (only written when do_z_extra): same layout as 6
  ffd_qdyn_<it>_out.bin      [itime] RM RMOM (mass units, after AADVQ) MMA, SBF SBM SFBM SCF SCM SFCM (JM,LM), SCF3D (IM,JM,LM)
  ffd_qdyn_<it>_fin.bin      [itime] Q(IM,JM,LM) QMOM(9,IM,JM,LM)  (after QDYNAM, concentration units)
"""
import os
import numpy as np

IM, JM, LM = 72, 46, 40
FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6


def _r(raw, o, shape):
    n = int(np.prod(shape))
    return raw[o:o + n].reshape(shape, order='F').copy(), o + n


def fname(date, itime, ph, ff=FF_DEFAULT, tag="qdyn"):
    return f"{ff}/{date}/ffd_{tag}_{itime}_{ph}.bin"


def gname(date, ff=FF_DEFAULT, tag="qdyn"):
    return f"{ff}/{date}/ffd_{tag}_geom.bin"


def calls(tag="qdyn", ff=FF_DEFAULT, complete=True):
    """(date, itime) with all dump files present (stress runs may stop early)."""
    out = []
    for d, it0 in DATES:
        for k in range(NSTEP):
            it = it0 + k
            if all(os.path.exists(fname(d, it, ph, ff, tag)) for ph in (('in', 'q0', 'ain', 'ck', 'out', 'fin') if complete else ('in',))):
                out.append((d, it))
    return out


def available(ff=FF_DEFAULT):
    return len(calls("qdyn", ff)) == len(DATES) * NSTEP and all(os.path.exists(gname(d, ff)) for d, _ in DATES)


def load_geom(path):
    raw = np.fromfile(path, dtype='>f8')
    g = dict(im=int(raw[0]), jm=int(raw[1]), lm=int(raw[2]), kg2mb=raw[3], byim_geom=raw[4], byim_qus=raw[5],
             byim_diag=raw[6])
    o = 7
    g['axyp'], o = _r(raw, o, (IM, JM))
    g['imaxj'] = raw[o:o + JM].astype(int); o += JM
    assert o == raw.size
    return g


def load_in(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), stress=raw[1]); o = 4
    for k, sh in (('q', (IM, JM, LM)), ('qmom', (9, IM, JM, LM)), ('maold', (LM, IM, JM)), ('mu', (IM, JM, LM)),
                  ('mv', (IM, JM, LM)), ('mw', (IM, JM, LM)), ('mb', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_q0(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0]), ncyc=int(raw[1]), do_z_extra=bool(raw[2])); o = 4
    for k, sh in (('ncycxy', (LM,)), ('nstepx', (JM, LM)), ('nstepz_extra', (IM, JM)), ('lminzij', (IM, JM)),
                  ('lmaxzij', (IM, JM)), ('mw_extra', (IM, JM, LM)), ('mu', (IM, JM, LM)), ('mv', (IM, JM, LM)),
                  ('mw', (IM, JM, LM)), ('pv_south', (IM, LM)), ('ni_y', (JM, LM)), ('ni_z', (JM, LM))):
        d[k], o = _r(raw, o, sh)
        if k in ('ncycxy', 'nstepx', 'nstepz_extra', 'lminzij', 'lmaxzij', 'ni_y', 'ni_z'):
            d[k] = d[k].astype(int)
    for k in ('y', 'z'):
        n = int(raw[o]); o += 1
        d['list_' + k] = raw[o:o + 3 * n].reshape(n, 3).astype(int); o += 3 * n
    assert o == raw.size
    return d


def load_ain(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0])); o = 1
    for k, sh in (('rm', (IM, JM, LM)), ('rmom', (9, IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_out(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0])); o = 1
    for k, sh in (('rm', (IM, JM, LM)), ('rmom', (9, IM, JM, LM)), ('mma', (IM, JM, LM)), ('sbf', (JM, LM)),
                  ('sbm', (JM, LM)), ('sfbm', (JM, LM)), ('scf', (JM, LM)), ('scm', (JM, LM)), ('sfcm', (JM, LM)),
                  ('scf3d', (IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_fin(path):
    raw = np.fromfile(path, dtype='>f8')
    d = dict(itime=int(raw[0])); o = 1
    for k, sh in (('q', (IM, JM, LM)), ('qmom', (9, IM, JM, LM))):
        d[k], o = _r(raw, o, sh)
    assert o == raw.size
    return d


def load_ck(path):
    """-> list of records (dict stage,nc,ncxy,l + arrays); a stress run that stopped may leave a truncated last record
    (ignored)."""
    raw = np.fromfile(path, dtype='>f8')
    recs = []
    o = 0
    S = IM * JM
    while o + 4 <= raw.size:
        st, nc, ncxy, l = (int(x) for x in raw[o:o + 4])
        r = dict(stage=st, nc=nc, ncxy=ncxy, l=l)
        o += 4
        if st in (0, 3):
            shapes = [('a', (IM, JM)), ('b', (IM, JM))]
        elif st in (1, 2):
            shapes = [('rm', (IM, JM)), ('rmom', (9, IM, JM)), ('mma', (IM, JM))]
        elif st == 4:
            shapes = [('mwdn', (IM, JM)), ('fdn', (IM, JM)), ('fdn0', (IM, JM)), ('fmomdn', (9, IM, JM))]
        elif st in (6, 7):
            shapes = [('rm', (IM, JM, LM)), ('rmom', (9, IM, JM, LM)), ('mma', (IM, JM, LM))]
        else:
            raise ValueError(f"bad stage {st} at {o}")
        need = sum(int(np.prod(sh)) for _, sh in shapes)
        if o + need > raw.size:
            break
        for k, sh in shapes:
            r[k], o = _r(raw, o, sh)
        recs.append(r)
    return recs
