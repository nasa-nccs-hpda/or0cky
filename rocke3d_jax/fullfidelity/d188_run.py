"""D188: run jax_surface on the REAL entry state of the SURFACE stage of one step and compare with the NumPy stage (C1, reference saved by
d188_ref_numpy.py) and with the real SURFACE records (C2).  Pin the cores the same way as the reference:
    taskset -c 3-5 env OMP_NUM_THREADS=1 python d188_run.py DATE REF.npz OUT.json
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clouds_jax_env  # noqa: F401
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import atm_step as A
import jax_surface as JS
import jax_harness as H

DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}
IM, JM = 72, 46


def sync(x):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, x)
    return x


def cat(a, b):
    r = H.field_category(a, b)
    return r['cat'], r['max_abs'], r['rel'], r['n_diff'], r['n']


def run(date, refpath, outjson, repeats=3):
    it = DATE_IT0[date]
    res = {'date': date, 'itime': it}
    reg = H.RecordedInputRegistry()
    H.declare_d174_inputs(reg, date, it)
    R = A.Real(date, it)
    t0 = time.perf_counter()
    rec = reg.read('surface_records', lambda: A.surface_records(R), stage='d188_template')
    res['t_read_records_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    S = A.real_state_at(R, 'surface', None)
    A._native(S)
    res['t_entry_state_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tpl, host = JS.build_template(rec)
    sync(tpl)
    res['t_build_template_s'] = time.perf_counter() - t0
    res['slots'] = dict(Nw=host['Nw'], Nl=host['Nl'], Ne=host['Ne'], ncell_block=len(host['bcells']), max_substeps=host['max_substeps'],
                        valid_ocean=int(host['wvalid'][0][0].sum()), valid_ice=int(host['wvalid'][0][1].sum()))
    t0 = time.perf_counter()
    Sd = JS.state_from_numpy(S)
    sync(Sd)
    res['t_upload_state_s'] = time.perf_counter() - t0
    stage = JS.make_stage(host)
    times = []
    first_unit = {}
    for rep in range(repeats):
        t0 = time.perf_counter()
        if rep == 0:
            out, aux = stage(Sd, tpl, timers=first_unit)
            sync(out)
        elif rep == repeats - 1:
            with jax.transfer_guard('disallow'):
                out, aux = stage(Sd, tpl)
                sync(out)
        else:
            out, aux = stage(Sd, tpl)
            sync(out)
        times.append(time.perf_counter() - t0)
    res['stage_times_s'] = times
    res['first_call_unit_times_s'] = first_unit
    # per unit steady times
    un = stage.units
    ut = {}
    atm1, cell1, rows1 = un['prep1'](Sd, tpl)
    sync(atm1)
    def tm(name, f):
        t0 = time.perf_counter(); o = f(); sync(o); ut[name] = time.perf_counter() - t0; return o
    atm1, cell1, rows1 = tm('prep1', lambda: un['prep1'](Sd, tpl))
    r1, ex1, tmom, qmom = tm('core1', lambda: un['run_core'](atm1, rows1, tpl, 0, tpl['ghy'][0]['dyn0'], Sd['TMOM'], Sd['QMOM'], Sd['PK']))
    atm2, rows2 = tm('between', lambda: un['between'](Sd, atm1, cell1, rows1, r1, ex1, tpl))
    r2, ex2, tmom2, qmom2 = tm('core2', lambda: un['run_core'](atm2, rows2, tpl, 1, r1['land']['dyn_next'], tmom, qmom, Sd['PK']))
    tm('final', lambda: un['final'](Sd, atm2, r2, ex2, rows2, tpl))
    res['unit_times_s'] = ut
    # --- C1 vs NumPy reference
    ref = dict(np.load(refpath))
    res['numpy_stage_times_s'] = ref['times'].tolist()
    c1 = {}
    for k in ('T', 'Q', 'U', 'V', 'UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA', 'V1AA', 'TSAVG', 'QSAVG',
              'USTARPBL', 'LMONINPBL', 'TMOM', 'QMOM'):
        c1['S.' + k] = cat(np.asarray(out[k]), ref['S.' + k])
    # internals per tile type: map record order -> slots
    Nw = host['Nw']
    wcells = host['wcells']
    lut = np.full(IM * JM, -1)
    lut[wcells] = np.arange(Nw)
    ta = rec['ta']
    cell = (ta[:, 1].astype(int) - 1) * IM + (ta[:, 0].astype(int) - 1)
    slot = (ta[:, 2].astype(int) - 1) * Nw + lut[cell]
    isice = ta[:, 2] == 2
    for pre, r in (('r1', r1), ('r2', r2)):
        for key in ('tg1', 'tg2', 'tr4', 'dth1', 'dq1', 'dmua', 'dmva', 'f0dt', 'f1dt', 'evap', 'shdt', 'evhdt', 'trhdt', 'qg_sat', 'tg'):
            rf = ref.get(f'{pre}.tile.{key}')
            if rf is None:
                continue
            mine = np.asarray(r['tile'][key])[slot]
            for nm, sel in (('ocean', ~isice), ('ice', isice)):
                c1[f'{pre}.tile.{nm}.{key}'] = cat(mine[sel], rf[sel])
        for key in ('us', 'vs', 'ws', 'tsv', 'qsrf', 'cm', 'ch', 'cq', 'dskin', 'ustar', 'lmonin', 'khs', 'z0m', 'psi'):
            rf = ref.get(f'{pre}.pbl.{key}')
            if rf is not None:
                mine = np.asarray(r['pbl'][key])[slot]
                for nm, sel in (('ocean', ~isice), ('ice', isice)):
                    c1[f'{pre}.pbl.{nm}.{key}'] = cat(mine[sel], rf[sel])
        for key in ('uflux1', 'vflux1', 'dth1', 'dq1', 'tg1', 'tg2', 'tr4', 'evap', 'f0dt'):
            rf = ref.get(f'{pre}.li.{key}')
            if rf is not None and key in r['li']:
                c1[f'{pre}.landice.{key}'] = cat(np.asarray(r['li'][key]), rf)
        for key in ('ustar', 'lmonin', 'tsv', 'qsrf', 'cm', 'ch', 'cq'):
            rf = ref.get(f'{pre}.pbl_li.{key}')
            if rf is not None:
                c1[f'{pre}.landice_pbl.{key}'] = cat(np.asarray(r['pbl_li'][key]), rf)
        for key in ('uflux1', 'vflux1', 'dth1', 'dq1', 'tsavg', 'qsavg'):
            rf = ref.get(f'{pre}.land.patch.{key}')
            if rf is not None:
                c1[f'{pre}.land.patch.{key}'] = cat(np.asarray(r['land']['patch'][key]), rf)
        for key in ('tbcs', 'tsns', 'ashg', 'alhg', 'aevap', 'aruns', 'arunu', 'aeruns', 'aerunu', 'ae0', 'abetad', 'w', 'ht', 'nsn', 'dzsn', 'wsn', 'hsn',
                    'fr_snow', 'evap_max_ij', 'fr_sat_ij'):
            rf = ref.get(f'{pre}.land.ghy.{key}')
            if rf is not None and key in r['land']['ghy']:
                c1[f'{pre}.land.ghy.{key}'] = cat(np.asarray(r['land']['ghy'][key]), rf)
        for key in ('ustar', 'lmonin'):
            rf = ref.get(f'{pre}.land.pbl.{key}')
            if rf is not None:
                c1[f'{pre}.land.pbl.{key}'] = cat(np.asarray(r['pbl_land'][key]), rf)
        for key in ('uflux1', 'vflux1', 'dth1', 'dq1', 'tsavg', 'qsavg'):
            rf = ref.get(f'{pre}.comp.{key}')
            if rf is not None:
                c1[f'{pre}.comp.{key}'] = cat(np.asarray(r['comp'][key]), rf)
    for pre, ex in (('ex1', ex1), ('ex2', ex2)):
        for key, rk in (('T', 'T'), ('Q', 'Q'), ('E', 'E'), ('U', 'U'), ('V', 'V'), ('UA', 'UA'), ('VA', 'VA'), ('pblht', 'pblht'), ('dclev', 'dclev')):
            rf = ref.get(f'{pre}.{rk}')
            if rf is not None:
                c1[f'{pre}.{key}'] = cat(np.asarray(ex[key]), rf)
    res['c1'] = c1
    # --- C2 vs real records
    c2 = {}
    ps = R.post_surface()
    for k in ('U', 'V', 'T', 'Q'):
        c2['post_surface.' + k] = cat(np.asarray(out[k]), np.asarray(ps[k]))
    from ffdump_reader import read_dump
    import pbl_compare as PC
    import surface_tile_ff as ST
    import landice_tile_ff as LI
    d_ = f"{R.ff}/{date}"
    dout = read_dump(f"{d_}/ffa_{it}_c2_out.bin", 1)
    tj = lambda x: np.transpose(np.asarray(x), (1, 0, 2))
    mm = np.asarray(JS._VALID)
    for nm, mine, refv in (('T', ex2['T'], tj(dout['T'])), ('Q', ex2['Q'], tj(dout['Q'])), ('U', ex2['U'], tj(dout['U'])), ('V', ex2['V'], tj(dout['V'])),
                           ('E', ex2['E'], np.transpose(dout['EGCM'], (2, 1, 0))), ('UA', ex2['UA'], np.transpose(dout['UALIJ'], (2, 1, 0))),
                           ('VA', ex2['VA'], np.transpose(dout['VALIJ'], (2, 1, 0))), ('pblht', ex2['pblht'], dout['PBLHT'].T)):
        m_ = np.asarray(mine)
        if nm in ('U', 'V'):
            m_, r_ = m_[1:], refv[1:]
        else:
            r_, m_ = refv, m_
            if m_.ndim == 3:
                m_, r_ = m_[mm], r_[mm]
            else:
                m_, r_ = m_[mm], r_[mm]
        c2['atm_exit_substep2.' + nm] = cat(m_, r_)
    # records: PBL outputs, tile outputs, land-ice outputs, GHY outputs, composites
    pa, pb = rec['pa'], rec['pb']
    for ns, (pr, tr_, r) in enumerate(((rec['pa'], rec['ta'], r1), (rec['pb'], rec['tb'], r2)), start=1):
        p12 = pr[pr[:, 2] <= 2]
        for key, col in PC.OUT.items():
            if key in r['pbl']:
                mine = np.asarray(r['pbl'][key])[slot]
                for nm_, sel in (('ocean', ~isice), ('ice', isice)):
                    c2[f'ns{ns}.pbl.{nm_}.{key}'] = cat(mine[sel], p12[sel, col])
        for key, col in ST.OUT.items():
            if key in r['tile']:
                mine = np.asarray(r['tile'][key])[slot]
                for nm_, sel in (('ocean', ~isice), ('ice', isice)):
                    if key == 'tg2' and nm_ == 'ocean':
                        continue      # the open-water tile has no tg2 output (the real record holds 0)
                    c2[f'ns{ns}.tile.{nm_}.{key}'] = cat(mine[sel], tr_[sel, col])
        l_ = rec['la'] if ns == 1 else rec['lb']
        for key, col in LI.OUT.items():
            if key in r['li']:
                c2[f'ns{ns}.landice.{key}'] = cat(np.asarray(r['li'][key]), l_[:, col])
        g_ = rec['g1'] if ns == 1 else rec['g2']
        for key, col in (('tbcs', 245), ('tsns', 246), ('ashg', 247), ('alhg', 248), ('aevap', 249)):
            c2[f'ns{ns}.land.ghy.{key}'] = cat(np.asarray(r['land']['ghy'][key]), g_[:, col])
        blk = rec['blk1'] if ns == 1 else rec['blk2']
        for q, key in enumerate(('uflux1', 'vflux1', 'dth1', 'dq1', 'tsavg', 'qsavg')):
            c2[f'ns{ns}.comp.{key}'] = cat(np.asarray(r['comp'][key]), blk[:, 30 + q])
    res['c2'] = c2
    json.dump(res, open(outjson, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return res, out, aux


def summarize(res):
    print('slots', res['slots'])
    for k in ('t_read_records_s', 't_entry_state_s', 't_build_template_s', 't_upload_state_s'):
        print(k, round(res[k], 3))
    print('stage times jax', [round(t, 3) for t in res['stage_times_s']], 'unit steady', {k: round(v, 3) for k, v in res['unit_times_s'].items()})
    print('numpy stage', res['numpy_stage_times_s'])
    if 'c2' in res:
        cc = {}
        for k, v in res['c2'].items():
            cc.setdefault(v[0], []).append(k)
        print('C2 categories', {c: len(v) for c, v in cc.items()})
        for k, v in res['c2'].items():
            print(f'  C2 {k:42s} {v[0]} max_abs {v[1]:.3e} rel {v[2]:.3e} ndiff {v[3]}/{v[4]}')
    cats = {}
    for k, v in res['c1'].items():
        cats.setdefault(v[0], []).append(k)
    print({c: len(v) for c, v in cats.items()})
    for k, v in res['c1'].items():
        if v[0] != 'A':
            print(f'{k:42s} {v[0]} max_abs {v[1]:.3e} rel {v[2]:.3e} ndiff {v[3]}/{v[4]}')


if __name__ == '__main__':
    r, out, aux = run(sys.argv[1], sys.argv[2], sys.argv[3])
    summarize(r)
