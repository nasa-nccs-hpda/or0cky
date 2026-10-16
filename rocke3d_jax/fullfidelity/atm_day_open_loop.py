"""D150: one-model-day open-loop run of the chained atmosphere step (atm_step_fast.py) against the real day dumps (D149).

Open loop means: step k+1 starts from OUR end state of step k (atmosphere, ATURB/PBL hidden state, CONDSE cloud/precip carry, LSCOND arrays), and
everything that is not the atmosphere is taken from the real run (recorded inputs, scoping/ATM_STEP_PLAN.md section 3.1):
  * radiation (SOCRATES output): SRHR/TRHR are those of the most recent real radiation step (MOD(Itime-ITIMEI,NRAD=5)==0), held frozen over the
    next four steps exactly as RADIA does; COSZ1 of every step; the RADIA cloud masking of CLDSS/CLDMC is applied to OUR clouds on radiation
    steps (RAD_DRV.f:2611-2615).  The radiation never sees our T/Q/clouds: no radiative feedback.
  * surface: the SURFACE tile/PBL/land-ice/GHY(land patch)/aggregate records of the real step (land_mode 'recorded'), Ent exports, sea-ice, lake,
    ocean surface state, PRECIP_* effects, the non-atmosphere CONDSE entry fields.
  * the day boundary (dailyUpdates after the step that ends at 00:00): DAILY_ATMDYN (D117, dry-mass fixer, mdrya recorded) is applied to OUR
    state; daily_RAD's SNOAGE aging (RAD_DRV.f:1414-1425, uses TDIURN) is taken from the next step's recorded CONDSE entry; daily_orbit/_EARTH/_LAKE/
    _LI/_CAL/_OCEAN/_ch4ox act through recorded inputs only.
Each step is compared with the real end-of-step state (ffa_step_<it>_e.bin) and the real CONDSE exit (ffc_cse_out) of the same step.

Usage (fullfidelity/, conda python):
  python atm_day_open_loop.py --nsteps 48 --tag imf_np [--dyn numpy|jax] [--no-imf] [--land recorded] [--it0 33312] [--daydir nov26_day]
Outputs: ff_data/<daydir>/ours_<tag>/step_<it>.npz (T Q U V QCL QCI P of every step end), ff_data/<daydir>/ours_<tag>/run.json (per-step stats, timings).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _early_env(argv):
    """the JAX dynamics needs the XLA flag of dyn_jax_env before the first `import jax` (also done lazily by atm_step's surface chain)."""
    if '--dyn' in argv and argv[argv.index('--dyn') + 1] == 'jax':
        import dyn_jax_env  # noqa: F401


if __name__ == '__main__':
    _early_env(sys.argv)

import atm_step as A
import atm_step_fast as F
import clouds_condse_io as cio
import dyn_glue_ff as gf
import dyn_glue_io as gio
import dyn_filter_ff as ffl
import atm_day_report as RP

DATE = 'nov26'            # constants of the date (geometry, tables) come from the original dump directory
DAYDIR = 'nov26_day'
IT0 = 33312
CLOUD_KEYS = ('CLDSS', 'CLDMC', 'TAUSS', 'TAUMC', 'CSIZMC', 'CSIZSS', 'QLSS', 'QISS', 'QLMC', 'QIMC', 'W_CLOUD', 'PREC', 'EPREC', 'PRECSS',
              'DDM1', 'DDMS', 'TDN1', 'QDN1')
NDAY = 48


class RealRad(A.Real):
    """A.Real whose radiation record `r` carries the SRHR/TRHR of the last radiation step (frozen heating rates), COSZ1 of this step."""

    def __init__(self, date, itime, ff, rad):
        super().__init__(date, itime, ff)
        self.rad = rad

    @property
    def r(self):
        base = self.site('r')
        d = dict(base)
        d['SRHR'], d['TRHR'] = self.rad['SRHR'], self.rad['TRHR']
        return d

    def r_own(self):
        return self.site('r')


def daily_mdrya(ff, date):
    """recorded constant: the dry-air mass reference MDRYA of the real DAILY_ATMDYN dumps (ffd_glue_daily_<it>.bin of the original dumps)."""
    import glob
    for p in sorted(glob.glob(f"{ff}/{date}/ffd_glue_daily_[0-9]*.bin")):
        d = gio.load_daily(p)
        if 1 in d:
            return float(d[1]['mdrya']), float(d[1]['deltam']), int(d[1]['itime'])
    return None, None, None


def apply_daily(S, ctx, mdrya):
    """DAILY_ATMDYN (D117) on the chained state: MA += DELTAM*MFRAC, then MAtoPMB (PEDN, PMID, PK, PDSIG, MASUM, P) and PEK."""
    o = gf.daily_atmdyn(S['MA'], S['MASUM'], ctx.gg, mdrya, False, True, ctx.imf, mp_g=ctx.fg)
    mp = o['matopmb']
    S['MA'] = np.array(o['ma'], copy=True)
    for k in ('masum', 'pedn', 'pmid', 'pk', 'pdsig', 'p'):
        S[k.upper()] = np.array(mp[k], copy=True)
    S['PEK'] = A._pow(ctx, S['PEDN'], ctx.kapa)
    return dict(deltam=o['deltam'], smass=o['smass'], mdryanow=o['mdryanow'])


def ch4ox_increment(ff, daydir, it_next):
    """DAILY_ch4ox (RAD_DRV.f:1600-1698): Q(i,j,l) += xCH4*dH2O(j,l,month)*byMA(l,i,j), then the polar rows are copied from i=1.  The added water
    MASS per unit area DM(j,l) = xCH4*dH2O(j,l,month) does not depend on the model state; it is recovered from the real step-start state
    of the first step of the new day (ffa_step_<it>_a) and the real end state of the previous step (ffa_step_<it-1>_e): DM = (Qa - Qe) * MAa.
    This is a RECORDED input (the dH2O file and the GHG table are not read here).  Returns (DM (JM,LM), max i-spread of DM/max DM)."""
    e = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next - 1}_e.bin")
    a = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next}_a.bin")
    dm_all = (a['Q'] - e['Q']) * np.transpose(a['MA'], (1, 2, 0))          # (IM,JM,LM)
    dm = dm_all[0].copy()
    spread = 0.0
    for j in range(1, A.JM - 1):
        spread = max(spread, float(np.abs(dm_all[:, j] - dm[j]).max()))
    return dm, spread / max(float(np.abs(dm).max()), 1e-300)


def apply_ch4ox(S, ctx, dm):
    byma = 1.0 / np.transpose(S['MA'], (1, 2, 0))
    q = np.array(S['Q'], copy=True)
    m = A.imaxj_mask(ctx)
    inc = dm[None, :, :] * byma
    q[m] = S['Q'][m] + inc[m]
    q[1:, 0, :] = q[0, 0, :]
    q[1:, A.JM - 1, :] = q[0, A.JM - 1, :]
    S['Q'] = q


def daily_replay_check(ff, daydir, ctx, mdrya, it_next):
    """DAILY_ATMDYN applied to the REAL end state of step it_next-1 versus the real step-start state of it_next (ffa_step_<it_next>_a)."""
    e = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next - 1}_e.bin")
    a = cio.read_cse(f"{ff}/{daydir}/ffa_step_{it_next}_a.bin")
    S = {k: np.array(e[k], copy=True) for k in ('MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P', 'PEK')}
    ms_ = np.zeros(e['MA'].shape[1:])
    for l in range(A.LM - 1, -1, -1):                      # MAtoPMB summation order (top down)
        ms_ = e['MA'][l] + ms_
    S['MASUM'] = ms_
    info = apply_daily(S, ctx, mdrya)
    dm, spread = ch4ox_increment(ff, daydir, it_next)
    S['Q'] = np.array(e['Q'], copy=True)
    apply_ch4ox(S, ctx, dm)
    info['ch4ox_dm_i_spread_rel'] = spread
    out = {}
    for k in ('MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'P', 'PEK', 'Q'):
        d = np.abs(S[k] - a[k])
        out[k] = dict(max_abs=float(d.max()), n_diff=int((S[k] != a[k]).sum()), scale=float(np.abs(a[k]).max()))
        d0 = np.abs(e[k] - a[k])
        out[k]['change_without_daily'] = float(d0.max())
    return dict(info=info, fields=out)


def cloud_stats(X, co, imaxj):
    out = {}
    for k in CLOUD_KEYS:
        if k not in X or k not in co:
            continue
        a, b = np.asarray(X[k], float), np.asarray(co[k], float)
        if a.shape != b.shape:
            continue
        if a.shape == (A.IM, A.JM, A.LM):
            pass
        elif a.shape == (A.LM, A.IM, A.JM):
            a, b = np.transpose(a, (1, 2, 0)), np.transpose(b, (1, 2, 0))
        elif a.shape != (A.IM, A.JM):
            continue
        m = RP.valid_mask('T', imaxj)
        if a.ndim == 3:
            m = m[:, :, None] & np.ones((1, 1, a.shape[2]), bool)
        d = (a - b)[m]
        out[k] = dict(rms=float(np.sqrt(np.mean(d ** 2))), maxabs=float(np.abs(d).max()), scale=float(np.abs(b[m]).max()),
                      n_diff=int((d != 0).sum()), n=int(d.size))
    return out


def run_day(date=DATE, it0=IT0, nsteps=NDAY, imf=True, dyn='numpy', land='recorded', daydir=DAYDIR, ff=cio.FF_DEFAULT, tag='run',
            save=True, log=print, mdrya_dir=None, daily=True):
    ctx = A.make_ctx(date, imf=imf, ff=ff)
    F.ensure_backend(ctx)
    if dyn == 'jax':
        import dyn_step_jax2 as dj2
        kit = dj2.Kit(ctx.dyn)

        def stage_dyn_jax(S, R, ctx_, tm=None):
            w = dj2.dyn_step_jax({k: S[k.upper()] for k in A.ds.STATE_KEYS}, ctx_.dyn, kit, itime=R.itime, timing=tm)
            for k in A.DYN_OUT:
                if k in w:
                    S[k] = np.array(w[k], copy=True)
            S['PEK'] = A._pow(ctx_, S['PEDN'], ctx_.kapa)
            return S
        A.stage_dyn = stage_dyn_jax
    mdrya, deltam_real, it_daily = daily_mdrya(ff, date)
    out_dir = f"{ff}/{daydir}/ours_{tag}"
    if save:
        os.makedirs(out_dir, exist_ok=True)
    axyp = ctx.gg['axyp']
    imaxj = ctx.imaxj
    S, ms, rad, rows = None, {}, None, []
    t_all = time.perf_counter()
    for k in range(nsteps):
        it = it0 + k
        rec_daily = None
        if rad is None or A.is_radiation_step(it):
            rr = A.Real(daydir, it, ff).site('r')
            rad = dict(SRHR=np.array(rr['SRHR']), TRHR=np.array(rr['TRHR']))
        R = RealRad(daydir, it, ff, rad)
        if daily and S is not None and (it % NDAY) == 0 and mdrya is not None:
            rec_daily = apply_daily(S, ctx, mdrya)
            dm, _ = ch4ox_increment(ff, daydir, it)
            apply_ch4ox(S, ctx, dm)
            carry = S.get('_carry', {})
            if 'SNOAGE' in carry:
                sn_new = np.array(R.cse_in['SNOAGE'], copy=True)
                rec_daily['snoage_change_recorded'] = float(np.abs(sn_new - carry['SNOAGE']).max())
                carry['SNOAGE'] = sn_new
        tm = {}
        t0 = time.perf_counter()
        with F.fast_condse():
            S, sn = A.run_step(date, it, ctx, R=R, S=S, ms=ms, land_mode=land, timing=tm)
        wall = time.perf_counter() - t0
        real_e = R.e
        w = RP.weights(real_e['MA'], axyp)
        end = {f: S[f] for f in RP.FIELDS + ('P',)}
        st = RP.state_metrics(end, real_e, w)
        X = S.get('_condse_X')
        cl = cloud_stats(X, R.cse_out, imaxj) if X is not None else {}
        own = R.r_own()
        row = dict(step=k, itime=it, wall=wall, stages={a: b for a, b in tm.items() if a.startswith('stage_')}, stats=st, clouds=cl,
                   radiation_step=bool(A.is_radiation_step(it)),
                   rad_feed_vs_own_r=float(max(np.abs(rad['SRHR'] - own['SRHR']).max(), np.abs(rad['TRHR'] - own['TRHR']).max())),
                   daily=rec_daily)
        rows.append(row)
        if save:
            np.savez(f"{out_dir}/step_{it}.npz", **{f: np.asarray(end[f], float) for f in end})
        carry = {key: np.array(X[key], copy=True) for key in A.CARRY_KEYS if X is not None and key in X}
        if '_cloud_rad' in S:
            carry['CLDSS'], carry['CLDMC'] = (np.array(a, copy=True) for a in S['_cloud_rad'])
        S = {key: v for key, v in S.items() if not key.startswith('_')}
        if carry:
            S['_carry'] = carry
        log(f"step {k:2d} it {it} {wall:5.1f}s  rms T {st['T']['rms']:.2e} Q {st['Q']['rms']:.2e} U {st['U']['rms']:.2e} V {st['V']['rms']:.2e} "
            f"P {st['P']['rms']:.2e} | gmeanT {st['T']['gmean']:+.2e} gmeanQ {st['Q']['gmean']:+.2e}"
            + (f" | daily deltam {rec_daily['deltam']:.3e}" if rec_daily else ''))
        sys.stdout.flush()
    total = time.perf_counter() - t_all
    res = dict(date=date, it0=it0, nsteps=nsteps, imf=imf, dyn=dyn, land=land, tag=tag, total_wall=total, rows=rows, mdrya=mdrya,
               daily_real_itime=it_daily)
    if save:
        json.dump(RP.to_jsonable(res), open(f"{out_dir}/run.json", 'w'))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=DATE)
    ap.add_argument('--daydir', default=DAYDIR)
    ap.add_argument('--it0', type=int, default=IT0)
    ap.add_argument('--nsteps', type=int, default=NDAY)
    ap.add_argument('--no-imf', action='store_true')
    ap.add_argument('--dyn', default='numpy', choices=('numpy', 'jax'))
    ap.add_argument('--land', default='recorded')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--no-daily', action='store_true')
    a = ap.parse_args(argv)
    run_day(a.date, a.it0, a.nsteps, imf=not a.no_imf, dyn=a.dyn, land=a.land, daydir=a.daydir, tag=a.tag, daily=not a.no_daily)
    return 0


if __name__ == '__main__':
    sys.exit(main())
