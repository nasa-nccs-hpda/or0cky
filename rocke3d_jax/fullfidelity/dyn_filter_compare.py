"""Validate dyn_filter_ff.py (FILTER/SLP/SHAP1D/isotropslp/MAtoPMB/energy functions, D101/D102) against
real-Fortran dumps.  Dumps (instrumentation/ATMDYN_filter.f.patch + ATM_UTILS_filter.f.patch +
ATM_DRV_dynD.f.patch, units 1120-1125), per date in ff_data/<date>/, all big-endian float64 streams,
1 FILTER call per step, 6 steps:
  ffd_filt_consts.bin          once: scalars[32], JM-arrays, LM-arrays, IM*JM arrays (dyn_filter_ff.load_consts)
  ffd_filt_<itime>_in.bin      itime, PEDN(1,:,:)(IM,JM), TSAVG(IM,JM), MA(LM,IM,JM), PK(LM,IM,JM),
                               T,Q,QCL,QCI (IM,JM,LM), QMOM(NMOM=9,IM,JM,LM)         [FILTER entry]
  ffd_slp_<itime>.bin          itime, X,Y (after the SLP loop), X (after SHAP1D), X (after isotropslp),
                               PEDN(1,:,:) (after the row loop: clip + column-mass conservation); (IM,JM) each;
                               rows 1 and JM of X,Y are undefined memory in the Fortran (never used)
  ffd_filt_<itime>_te1.bin     getTotalEnergy call 1 (initial) and _te2 (final): itime, n, MASUM(IM,JM), MA, PK,
                               T, U, V, KEB (B-grid KE before regrid; row 1 undefined), KEIJ, PEIJ, TEIJ, TOTAL
  ffd_filt_<itime>_eadd.bin    addEnergyAsDiffuseHeat: itime, DELTAENERGY, EDIFF, T_in, PK, T_out
  ffd_filt_<itime>_out.bin     FILTER exit: itime, PEDN(LM+1,IM,JM), PMID, PK, MA, MASUM, T,Q,QCL,QCI, QMOM
Usage: python3 dyn_filter_compare.py [--imf-pow]
"""
import sys
import numpy as np
import dyn_filter_ff as ff
from dyn_filter_ff import IM, JM, LM

FF_DEFAULT = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data"
DATES = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
NSTEP = 6
NMOM = 9


def _ij(a):             # Fortran (IM,JM) -> numpy (IM,JM)
    return a.reshape(JM, IM).T.copy()


def _ijl(a):            # Fortran (IM,JM,LM)
    return a.reshape(LM, JM, IM).transpose(2, 1, 0).copy()


def _lij(a, n=LM):      # Fortran (n,IM,JM)
    return a.reshape(JM, IM, n).transpose(2, 1, 0).copy()


class _Rd:
    def __init__(self, path):
        self.raw = np.fromfile(path, dtype='>f8'); self.o = 0

    def take(self, n):
        a = self.raw[self.o:self.o + n]; self.o += n; return a

    def done(self):
        assert self.o == self.raw.size, (self.o, self.raw.size)


def load_in(path):
    r = _Rd(path); d = dict(itime=r.take(1)[0])
    d['pedn1'] = _ij(r.take(IM * JM)); d['tsavg'] = _ij(r.take(IM * JM))
    d['ma'] = _lij(r.take(LM * IM * JM)); d['pk'] = _lij(r.take(LM * IM * JM))
    for n in ('t', 'q', 'qcl', 'qci'):
        d[n] = _ijl(r.take(IM * JM * LM))
    d['qmom'] = r.take(NMOM * IM * JM * LM).reshape(LM, JM, IM, NMOM).transpose(3, 2, 1, 0).copy()
    r.done(); return d


def load_slp(path):
    r = _Rd(path); d = dict(itime=r.take(1)[0])
    for n in ('x1', 'y1', 'x2', 'x3', 'pedn4'):
        d[n] = _ij(r.take(IM * JM))
    r.done(); return d


def load_te(path):
    r = _Rd(path); d = dict(itime=r.take(1)[0], n=int(r.take(1)[0]))
    d['masum'] = _ij(r.take(IM * JM)); d['ma'] = _lij(r.take(LM * IM * JM)); d['pk'] = _lij(r.take(LM * IM * JM))
    for n in ('t', 'u', 'v'):
        d[n] = _ijl(r.take(IM * JM * LM))
    for n in ('keb', 'kea', 'pe', 'te'):
        d[n] = _ij(r.take(IM * JM))
    d['tot'] = r.take(1)[0]; r.done(); return d


def load_eadd(path):
    r = _Rd(path); d = dict(itime=r.take(1)[0], de=r.take(1)[0], ediff=r.take(1)[0])
    d['t_in'] = _ijl(r.take(IM * JM * LM)); d['pk'] = _lij(r.take(LM * IM * JM))
    d['t_out'] = _ijl(r.take(IM * JM * LM)); r.done(); return d


def load_out(path):
    r = _Rd(path); d = dict(itime=r.take(1)[0])
    d['pedn'] = _lij(r.take((LM + 1) * IM * JM), LM + 1)
    for n in ('pmid', 'pk', 'ma'):
        d[n] = _lij(r.take(LM * IM * JM))
    d['masum'] = _ij(r.take(IM * JM))
    for n in ('t', 'q', 'qcl', 'qci'):
        d[n] = _ijl(r.take(IM * JM * LM))
    d['qmom'] = r.take(NMOM * IM * JM * LM).reshape(LM, JM, IM, NMOM).transpose(3, 2, 1, 0).copy()
    r.done(); return d


def paths(date, itime, ff=FF_DEFAULT):
    b = f"{ff}/{date}/ffd_filt_{itime}"
    return dict(inp=b + "_in.bin", out=b + "_out.bin", te1=b + "_te1.bin", te2=b + "_te2.bin",
                eadd=b + "_eadd.bin", slp=f"{ff}/{date}/ffd_slp_{itime}.bin")


def available(date, ff=FF_DEFAULT):
    import os
    return os.path.exists(f"{ff}/{date}/ffd_filt_consts.bin")


def load_call(date, itime, ff=FF_DEFAULT):
    p = paths(date, itime, ff)
    return dict(inp=load_in(p['inp']), slp=load_slp(p['slp']), te1=load_te(p['te1']), te2=load_te(p['te2']),
                eadd=load_eadd(p['eadd']), out=load_out(p['out']))


def load_g(date, ff=FF_DEFAULT):
    return ff_load(f"{ff}/{date}/ffd_filt_consts.bin")


def ff_load(path):
    return ff.load_consts(path)


def _mx(a, b):
    return float(np.max(np.abs(a - b)))


def run_call(date, itime, imf_pow=False, ffd=FF_DEFAULT, stats=None):
    g = load_g(date, ffd); c = load_call(date, itime, ffd)
    rin, rs, t1, t2, ea, ro = c['inp'], c['slp'], c['te1'], c['te2'], c['eadd'], c['out']
    res = {}
    r = slice(1, JM - 1)
    # --- SLP stage (cells J=2..JM-1)
    zs = g['zatmo'] * g['bygrav']
    x1 = ff.slp(rin['pedn1'], rin['tsavg'], zs, g, imf_pow=imf_pow, stats=stats)
    y1 = x1 / rin['pedn1']
    res['slp_x'] = _mx(x1[:, r], rs['x1'][:, r]); res['slp_y'] = _mx(y1[:, r], rs['y1'][:, r])
    res['slp_nbad'] = int(np.sum(x1[:, r] != rs['x1'][:, r]))
    # --- SHAP1D from the recorded post-SLP X (isolates the stage)
    x2 = ff.shap1d(rs['x1'])
    res['shap1d'] = _mx(x2[:, r], rs['x2'][:, r])
    x3 = ff.isotropslp(rs['x2'], g, stats=stats)
    res['isotropslp'] = _mx(x3[:, r], rs['x3'][:, r])
    # --- row loop (clip + conservation) from recorded stage 3
    pn_rec = ff.row_loop(rs['x3'], rs['y1'], rin['pedn1'], g, stats)
    res['rowloop'] = _mx(pn_rec[:, r], rs['pedn4'][:, r])
    pn = ff.slp_filter_pedn(rin['pedn1'], rin['tsavg'], g, imf_pow=imf_pow)
    res['pedn1_chain'] = _mx(pn[:, r], rs['pedn4'][:, r])
    # --- energy functions, initial and final call (self-contained inputs)
    for k, t in (('te1', t1), ('te2', t2)):
        tot, p = ff.total_energy(t['masum'], t['ma'], t['pk'], t['t'], t['u'], t['v'], g, return_parts=True)
        res[k + '_keb'] = _mx(p['keb'][:, 1:], t['keb'][:, 1:])
        res[k + '_kea'] = _mx(p['kea'], t['kea']); res[k + '_pe'] = _mx(p['pe'], t['pe'])
        res[k + '_te'] = _mx(p['te'], t['te']); res[k + '_tot'] = abs(tot - t['tot'])
    # --- addEnergyAsDiffuseHeat
    tn, ediff = ff.add_energy_as_diffuse_heat(ea['de'], ea['t_in'], ea['pk'], g)
    res['eadd_ediff'] = abs(ediff - ea['ediff']); res['eadd_t'] = _mx(tn, ea['t_out'])
    res['eadd_de_matches'] = abs((t2['tot'] - t1['tot']) - ea['de'])
    # --- MAtoPMB from the recorded post-filter MA
    mp = ff.matopmb(ro['ma'], g, imf_pow=imf_pow)
    for n in ('pedn', 'pmid', 'pk', 'masum'):
        res['matop_' + n] = _mx(mp[n], ro[n])
    # --- whole FILTER chain
    o = ff.filter_slp(rin['pedn1'], rin['tsavg'], rin['ma'], rin['pk'], rin['t'], rin['q'], rin['qcl'], rin['qci'],
                      rin['qmom'], t1['u'], t1['v'], g, masum0=t1['masum'], imf_pow=imf_pow)
    for n in ('pedn', 'pmid', 'pk', 'ma', 'masum', 't', 'q', 'qcl', 'qci', 'qmom'):
        res['out_' + n] = _mx(o[n], ro[n])
    res['out_e0'] = abs(o['e0'] - t1['tot']); res['out_e1'] = abs(o['e1'] - t2['tot'])
    res['n_t_exact'] = int(np.sum(o['t'] == ro['t'])); res['n_total'] = o['t'].size
    # non-vacuity scales
    res['chg_pedn1'] = _mx(rs['pedn4'][:, r], rin['pedn1'][:, r]); res['scale_pedn1'] = float(np.max(rin['pedn1']))
    res['chg_t'] = _mx(ro['t'], rin['t']); res['chg_q'] = _mx(ro['q'], rin['q'])
    res['ediff'] = ea['ediff']; res['de'] = ea['de']
    return res


if __name__ == "__main__":
    imf = "--imf-pow" in sys.argv
    worst = {}; stats = {}
    for date, it0 in DATES:
        for k in range(NSTEP):
            r = run_call(date, it0 + k, imf_pow=imf, stats=stats)
            keys = [n for n in r if n.startswith(('slp_x', 'slp_y', 'shap', 'iso', 'rowl', 'pedn1', 'te1_', 'te2_',
                                                   'eadd_', 'matop_', 'out_'))]
            bad = {n: r[n] for n in keys if r[n] != 0.0}
            print(f"{date} itime={it0 + k}: nonzero diffs {len(bad)} of {len(keys)}  "
                  f"slp_nbad={r['slp_nbad']} T_exact={r['n_t_exact']}/{r['n_total']} "
                  f"chg_pedn1={r['chg_pedn1']:.3e} ediff={r['ediff']:.3e}"
                  + ("  " + " ".join(f"{n}={v:.2e}" for n, v in bad.items()) if bad else ""))
            for n in keys:
                worst[n] = max(worst.get(n, 0.), r[n])
    print("imf_pow =", imf)
    print("WORST:", {k: f"{v:.2e}" for k, v in worst.items() if v != 0.0} or "all exactly 0")
    print("branch stats:", stats)
