"""Loaders for the D114-D117 coupling-glue dumps (ffd_glue_*), written by the instrumentation in
instrumentation/ATM_DRV_dynF.f.patch (helpers ffdg_*, units 1240-1251) with hooks from ATMDYN_glue.f.patch,
ATM_UTILS_glue.f.patch, ATMDYN_COM_glue.f90.patch and the call-site patches SURFACE_glue / ATURB_glue /
CLOUDS2_DRV_glue / DIAG_glue.  All files are big-endian float64 streams in ff_data/<date>/.

Conventions: 2-D fields (IM,JM); 3-D fields (IM,JM,LM) ("ijl"); ModelE column-first arrays (LM,IM,JM) ("lij").
Records (itime always the first double):
  ffd_glue_consts.bin             once.  scalars[32] = im,jm,lm,0,grav,rgas,kapa,bykapa,psf,bygrav,sha,dtsrc,byim,
                                  mtop,pmtop,areag,lhe,lhm,0*14; JM-arrays bydxp,bydyp,kmaxj,imaxj,dxyp,dxyn,dxys,
                                  dxyv,rapvs,rapvn; IM-arrays sinip,cosip,siniv,cosiv; LM-array mfrac; (IM,JM) arrays
                                  rapj,idjj,byaxyp,axyp,zatmo; idij(K=IM,I=IM,J=JM).
  ffd_glue_trop_<itime>.bin       1 record/step: itime,ncall; T(ijl), PK(lij), PMID(lij), PTROPO(ij), LTROPO(ij)
  ffd_glue_wsave_<itime>.bin      itime,ncall; MWs(ijl), T(ijl), PK(lij), PEDN(LM+1,IM,JM), WSAVE(IM,JM,LM-1)
  ffd_glue_pgrad_<itime>.bin      itime,ncall; T1, PK1, PMID1, PEDN1, PHI1, ZATMO, DPDX, DPDY, DPDX0, DPDY0 (ij each)
  ffd_glue_kea_<itime>.bin        2 records/step: itime,ncall,site(1 ATM_DRV step-7 call, 2 DISSIP); U,V,KEA (ijl)
  ffd_glue_rg3_<itime>.bin        2 records/step: itime,ncall,site; X_in (ijl; row J=1 undefined), X_out (ijl)
  ffd_glue_dissip_<itime>.bin     itime,ncall; KEA(saved), DKE, T_in, T_out (ijl), PK(lij)
  ffd_glue_recalc_<itime>.bin     itime,ncall,site(1 SURFACE, 2 ATURB, 3/4 CLOUDS2_DRV, 5 DIAG, 0 unlabelled);
                                  U,V (ijl), UALIJ, VALIJ (lij)
  ffd_glue_efix_<itime>.bin       3 records/step, header itime,kind.  kind 1 / 2: MA(lij), U,V,T(ijl), PK(lij),
                                  MASUM(ij), Q, QCI (ijl), a, b (ij) = SEINIT,KEINIT / raw SEFINAL,KEFINAL;
                                  kind 3: dSEpKE, MMGLOB, SEFINAL, KEFINAL (combined, ij), T(ijl) after the fix.
  ffd_glue_daily_<itime>.bin      (D117) kind 1: itime,1, deltam,smass,mdryanow,mdrya, MA(lij), MASUM(ij);
                                  kind 2: itime,2, MA(lij), MASUM(ij), PEDN(LM+1,IM,JM)
  ffd_glue_daily_calls.txt        every DAILY_ATMDYN call: itime itimei end_of_day
"""
import glob
import os
import numpy as np

IM, JM, LM = 72, 46, 40
FF_DEFAULT = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6


def _ij(a):
    return a.reshape(JM, IM).T.copy()


def _ijl(a):
    return a.reshape(LM, JM, IM).transpose(2, 1, 0).copy()


def _lij(a, n=LM):
    return a.reshape(JM, IM, n).transpose(2, 1, 0).copy()


class _Rd:
    def __init__(self, path):
        self.raw = np.fromfile(path, dtype='>f8')
        self.o = 0

    def take(self, n):
        a = self.raw[self.o:self.o + n]
        assert a.size == n, (self.o, n, self.raw.size)
        self.o += n
        return a

    def eof(self):
        return self.o >= self.raw.size

    def ij(self):
        return _ij(self.take(IM * JM))

    def ijl(self, n=LM):
        return self.take(IM * JM * n).reshape(n, JM, IM).transpose(2, 1, 0).copy()

    def lij(self, n=LM):
        return _lij(self.take(n * IM * JM), n)


def path(date, name, ff=FF_DEFAULT):
    return f"{ff}/{date}/ffd_glue_{name}"


def available(date, ff=FF_DEFAULT):
    return (os.path.exists(path(date, "consts.bin", ff)) and
            len(glob.glob(path(date, "efix_*.bin", ff))) >= NSTEP)


def load_consts(p):
    raw = np.fromfile(p, dtype='>f8')
    s = raw[:32]
    names = ['im', 'jm', 'lm', 'r3', 'grav', 'rgas', 'kapa', 'bykapa', 'psf', 'bygrav', 'sha', 'dtsrc', 'byim',
             'mtop', 'pmtop', 'areag', 'lhe', 'lhm']
    g = {n: float(s[i]) for i, n in enumerate(names)}
    o = 32
    for n in ['bydxp', 'bydyp', 'kmaxj', 'imaxj', 'dxyp', 'dxyn', 'dxys', 'dxyv', 'rapvs', 'rapvn']:
        g[n] = raw[o:o + JM].copy(); o += JM
    for n in ('kmaxj', 'imaxj'):
        g[n] = g[n].astype(int)
    for n in ['sinip', 'cosip', 'siniv', 'cosiv']:
        g[n] = raw[o:o + IM].copy(); o += IM
    g['mfrac'] = raw[o:o + LM].copy(); o += LM
    for n in ['rapj', 'idjj', 'byaxyp', 'axyp', 'zatmo']:
        g[n] = _ij(raw[o:o + IM * JM]); o += IM * JM
    g['idjj'] = g['idjj'].astype(int)
    g['idij'] = raw[o:o + IM * IM * JM].reshape(JM, IM, IM).transpose(2, 1, 0).astype(int).copy()
    o += IM * IM * JM
    assert o == raw.size, (o, raw.size)
    for n in ('im', 'jm', 'lm'):
        g[n] = int(g[n])
    return g


def load_g(date, ff=FF_DEFAULT):
    return load_consts(path(date, "consts.bin", ff))


# ----------------------------------------------------------------------------- record loaders
def load_trop(p):
    r = _Rd(p); d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]))
    d['t'] = r.ijl(); d['pk'] = r.lij(); d['pmid'] = r.lij(); d['ptropo'] = r.ij()
    d['ltropo'] = r.ij().astype(int); assert r.eof(); return d


def load_wsave(p):
    r = _Rd(p); d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]))
    d['mws'] = r.ijl(); d['t'] = r.ijl(); d['pk'] = r.lij(); d['pedn'] = r.lij(LM + 1)
    d['wsave'] = r.ijl(LM - 1); assert r.eof(); return d


def load_pgrad(p):
    r = _Rd(p); d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]))
    for n in ('t1', 'pk1', 'pmid1', 'pedn1', 'phi1', 'zatmo', 'dpdx', 'dpdy', 'dpdx0', 'dpdy0'):
        d[n] = r.ij()
    assert r.eof(); return d


def load_kea(p):
    r = _Rd(p); out = []
    while not r.eof():
        d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]), site=int(r.take(1)[0]))
        d['u'] = r.ijl(); d['v'] = r.ijl(); d['kea'] = r.ijl(); out.append(d)
    return out


def load_rg3(p):
    r = _Rd(p); out = []
    while not r.eof():
        d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]), site=int(r.take(1)[0]))
        d['x_in'] = r.ijl(); d['x_out'] = r.ijl(); out.append(d)
    return out


def load_dissip(p):
    r = _Rd(p); out = []
    while not r.eof():
        d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]))
        d['kea'] = r.ijl(); d['dke'] = r.ijl(); d['t_in'] = r.ijl(); d['t_out'] = r.ijl(); d['pk'] = r.lij()
        out.append(d)
    return out


def load_recalc(p):
    r = _Rd(p); out = []
    while not r.eof():
        d = dict(itime=r.take(1)[0], ncall=int(r.take(1)[0]), site=int(r.take(1)[0]))
        d['u'] = r.ijl(); d['v'] = r.ijl(); d['ua'] = r.lij(); d['va'] = r.lij(); out.append(d)
    return out


def load_efix(p):
    r = _Rd(p); out = {}
    while not r.eof():
        itime = r.take(1)[0]; kd = int(r.take(1)[0])
        d = dict(itime=itime, kind=kd)
        if kd in (1, 2):
            d['ma'] = r.lij(); d['u'] = r.ijl(); d['v'] = r.ijl(); d['t'] = r.ijl(); d['pk'] = r.lij()
            d['masum'] = r.ij(); d['q'] = r.ijl(); d['qci'] = r.ijl(); d['a'] = r.ij(); d['b'] = r.ij()
        else:
            d['dse'] = r.take(1)[0]; d['mmg'] = r.take(1)[0]
            d['a'] = r.ij(); d['b'] = r.ij(); d['t'] = r.ijl()
        out[kd] = d
    assert set(out) == {1, 2, 3}, sorted(out)
    return out


def load_daily(p):
    r = _Rd(p); out = {}
    while not r.eof():
        itime = r.take(1)[0]; kd = int(r.take(1)[0])
        d = dict(itime=itime, kind=kd)
        if kd == 1:
            d['deltam'], d['smass'], d['mdryanow'], d['mdrya'] = r.take(4)
        d['ma'] = r.lij(); d['masum'] = r.ij()
        if kd == 2:
            d['pedn'] = r.lij(LM + 1)
        out[kd] = d
    return out


def load_daily_calls(date, ff=FF_DEFAULT):
    p = path(date, "daily_calls.txt", ff)
    if not os.path.exists(p):
        return []
    rows = []
    for ln in open(p):
        a = ln.split()
        if len(a) == 3:
            rows.append((int(a[0]), int(a[1]), a[2] == 'T'))
    return rows


def daily_files(date, ff=FF_DEFAULT):
    return sorted(glob.glob(path(date, "daily_[0-9]*.bin", ff)))
