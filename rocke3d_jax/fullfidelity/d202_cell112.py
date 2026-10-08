"""D202: isolate land cell 112 (grid 41,20) at steps >= 28 of the ON (D199 elhx) nov26 run using the d202_day.py captures.
For each step/substep: rebuild the GHY column of the cell from the REAL ffg row (static, Ent exports, constant forcing) with the dynamic state and the PBL-derived
forcing (ts, qs, rho, ch, vs, tprime, qprime, qm1) that the PORT fed to GHY, then run the NumPy Fortran-semantics GHY (ghy_ref / ghy_ref_nit.advnc_full)
(A) with the RECORDED dts/nit (what the port does), (B) with the Fortran time loop (gdtm, GHY.f:2389-2416).  Compare with the captured port result.
    python d202_cell112.py CAPDIR OUT.json [steps...]"""
import sys, json
import numpy as np
import ghy_compare as GC
import ghy_ref as G
import ghy_ref_nit as N

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
CELL = 112
RGAS = 287.048730386149032


def column(rec, dyn, over):
    static, dynamic, forcing, ent_iters, refs, snowm = GC.unpack(rec)
    d = dict(dynamic)
    for k in ('w', 'ht', 'wsn', 'hsn', 'fr_snow', 'nsn'):
        d[k] = np.array(dyn[k])
    d['dzsn'] = np.array(dyn['dzsn'])[:3]
    f = dict(forcing); f.update(over)
    col = G.GhyColumn(static, d, f)
    col.fb, col.fv = f['fb'], f['fv']
    return col, ent_iters, refs, snowm


def run(rec, dyn, over, mode):
    col, ent, refs, snowm = column(rec, dyn, over)
    info = N.advnc_full(col, ent, 900.0, snowm, ffnit=refs['ffnit'], use_recorded_dts=(mode == 'A'))
    out = dict(tbcs=col.tbcs, tsns=col.tsns, ashg=col.ashg, alhg=col.alhg, aevap=col.aevap, nit=info['nit'], dts=[float(x) for x in info['dts']],
               dtm=[float(x) for x in info['dtm']], dts_rec=[None if x is None else float(x) for x in info['dts_rec']])
    return out, col


def main(cap, outp, steps):
    res = {}
    for k in steps:
        z = np.load(f'{cap}/in_{k}.npz')
        it = 33312 + k
        g = GC.load(f'{FF}/ffg_{it}.bin'); ng = len(g) // 2
        for ns in (0, 1):
            tag = 'r1' if ns == 0 else 'r2'
            rows = z['rows/rows1' if ns == 0 else 'rows/rows2'][CELL]
            if ns == 0:
                dyn = {kk: z[f'ghy/g0/dyn0/{kk}'][CELL] for kk in ('w', 'ht', 'nsn', 'dzsn', 'wsn', 'hsn', 'fr_snow')}
            else:
                dyn = {kk: z[f'res/r1/dyn_next/{kk}'][CELL] for kk in ('w', 'ht', 'nsn', 'dzsn', 'wsn', 'hsn', 'fr_snow')}
            pb = {kk: float(z[f'res/{tag}/pbl/{kk}'][CELL]) for kk in ('tsv', 'qsrf', 'ch', 'ws', 'cm', 'us', 'vs')}
            ps = rows[16]
            rho = 100.0 * ps / (RGAS * pb['tsv'])
            ddml = rows[23] > 0.5
            if f'misc/q1_ns{ns+1}' in z.files:
                qm1 = float(z[f'misc/q1_ns{ns+1}'][CELL] * z['misc/ma1'][CELL]); qm1_src = 'port q1*ma1'
            else:     # run1 capture: q1 was captured with the wrong index; the recorded q1*ma1 is used (the evap_min limiter is not binding in these cases, checked at 31/2)
                qm1 = float(z['ghy/g%d/forcing/qm1' % ns][CELL]); qm1_src = 'recorded q1*ma1 (approx)'
            fx = {kk: float(z['ghy/g%d/forcing/%s' % (ns, kk)][CELL]) for kk in ('pr', 'htpr', 'prs', 'htprs', 'srht', 'trht', 'geothermal_heat', 'pres', 'irrig', 'htirrig', 'gusti')}
            over = dict(fx, ts=pb['tsv'], qs=pb['qsrf'], rho=rho, ch=pb['ch'], vs=pb['ws'],
                        tprime=(rows[25] - rows[7]) if ddml else 0.0, qprime=(rows[26] - rows[39]) if ddml else 0.0, qm1=qm1)
            rec = g[ns * ng + CELL]
            over_rec = dict(over)                       # GHY gusti = recorded ffg gusti_in (what the port does): GC.unpack default
            over_pbl = dict(over, gusti=float(rows[24]))  # GHY gusti = the gusti the PBL used (GHY_DRV.f:1267)
            A, colA = run(rec, dyn, over_rec, 'A')
            B, colB = run(rec, dyn, over_rec, 'B')
            Ap, _ = run(rec, dyn, over_pbl, 'A')
            Bp, _ = run(rec, dyn, over_pbl, 'B')
            port = {kk: float(z[f'res/{tag}/ghy/{kk}'][CELL]) for kk in ('tbcs', 'tsns', 'ashg', 'alhg', 'aevap')}
            real = dict(tbcs=rec[245], tsns=rec[246], ashg=rec[247], alhg=rec[248], aevap=rec[249], ffnit=int(round(rec[289])))
            res[f'{k}/{ns+1}'] = dict(pbl=pb, ddml=float(rows[23]), gusti=float(rows[24]), edts_port=[float(x) for x in z['ghy/g%d/edts' % ns][CELL]],
                                     nsub_port=int(z['ghy/g%d/nsub' % ns][CELL]), port=port, A_recorded_schedule_gusti_recorded=A, B_gdtm_gusti_recorded=B, A_recorded_schedule_gusti_pbl=Ap, B_gdtm_gusti_pbl=Bp, qm1=qm1, qm1_src=qm1_src, real_record=real)
            print(k, ns + 1, 'ch %.4f ws %.3f ddml %g gusti_pbl %.2f | port ae %.3e ashg %.0f | A(rec nit,gusti rec) ae %.3e ashg %.0f | A(gusti pbl) ae %.3e ashg %.0f | B(gdtm nit %d, gusti rec) ae %.3e | B(gdtm nit %d, gusti pbl) ae %.3e ashg %.0f | real ae %.3e ashg %.0f nit %d'
                  % (pb['ch'], pb['ws'], rows[23], rows[24], port['aevap'], port['ashg'], A['aevap'], A['ashg'], Ap['aevap'], Ap['ashg'], B['nit'], B['aevap'], Bp['nit'], Bp['aevap'], Bp['ashg'],
                     real['aevap'], real['ashg'], real['ffnit']), flush=True)
    json.dump(res, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
