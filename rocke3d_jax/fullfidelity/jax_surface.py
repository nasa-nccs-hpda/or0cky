"""D188 (build stage S4, step one): the SURFACE stage (2 substeps of 900 s over tiles) as device-resident JAX on FIXED-SHAPE tile slots.

NEW module; nothing existing is edited.  SOCRATES/RADIA untouched.  It composes the already validated JAX pieces
(pbl_ff.advanc_batch, surface_tile_ff.tile_fluxes + ice_props_ff, landice_tile_ff.tile_fluxes, ghy_jax.advnc with max_substeps,
tile_aggregate_ff.aggregate, aturb_ff/aturb_uv_ff through chain_two_substeps._aturb_uv, substep_chain) and ports to jax.numpy the NumPy glue of
atm_step.stage_surface (cell_inputs, override_pbl, override_tiles, predict_ns2, next_land_pbl_columns, first_layer_update, composites).

FIXED-SHAPE LAYOUT (replaces the per-step record row sets by slots with validity masks; D181 section 1 defines the tile existence rule):
  water slots   : every cell that can hold an open-water or sea-ice tile (fwater > 0); TWO slots per water cell (type 1 = ocean/lake water,
                  type 2 = sea/lake ice); a slot is VALID where the tile exists.  Rows [0:Nw] = type 1, [Nw:2Nw] = type 2.
  land-ice slots: static (flice > 0);  land slots: static (fearth > 0, the GHY/Ent cells, in the ffg order).
  All row arrays have the record column layout (ffp 154, ffs 90, ffl 60) so the validated column-indexed code is reused unchanged.  Invalid
  slots hold a copy of a valid row of the same type (finite dummy), never enter a composite (ftype = 0, `where` masks) and are masked in every
  comparison.  The set of slots never changes shape.

WHAT IS RECORDED (read only through jax_harness.RecordedInputRegistry, name 'surface_records'; built ONCE per step on the host by `build_template`):
  template columns of the ffp/ffs/ffl rows that are not recomputed (constants, ddml flags, PBL profiles at step start, ground state, radiation inputs),
  the ffg land rows (GHY start state, precipitation/radiation forcing, Ent exports per sub-iteration, irrigation, MA1), TRUP_in_rad of land
  (infer_trup from the recorded land patch), the recorded fft tile fractions (ftype; equals ptype from RSI), CORIOL.  Ent is RECORDED (D158/D169).
  The substep-2 rows are the substep-1 prediction (predict_ns2 port) on the recorded substep-2 template rows, as in the NumPy stage.

JIT UNITS (device-resident between them; stage_surface_jax):  prep1 (cell inputs + overrides, ns 1) -> core (PBL, tile fluxes, land ice, GHY,
aggregation, first-layer update, ATURB+UV) -> between (predict_ns2 + overrides for ns 2) -> core -> final (composites, merge).
`core` compiles once and is executed twice.
"""
import os
import sys
import time
import functools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import pbl_ff as P  # noqa: E402
import surface_tile_ff as ST  # noqa: E402
import landice_tile_ff as LI  # noqa: E402
import ice_props_ff as IP  # noqa: E402
import tile_aggregate_ff as TA  # noqa: E402
import substep_chain as SC  # noqa: E402
import chain_two_substeps as C2  # noqa: E402
import aturb_compare as AC  # noqa: E402
import ghy_jax as J  # noqa: E402

IM, JM, LM = 72, 46, 40
DT = 900.0
TF = P.TF
RGAS = 287.048730386149032
SHA = 1002.88097573814161
STBO = 5.67037320999999984e-8
C001 = float(np.float32(0.001))           # REAL*4 literal in GHY_DRV.f
FLOAT4_4375 = float(np.float32(4375.0))
PBL_COLS = dict(zs1=5, tkv=7, ps=16, ddml=23, gusti=24, tdns=25, qdns=26, dbl=29, ug=31, vg=32, utop=37, vtop=38, qtop=39, ztop=40,
                mdf=41, dpdxr=42, dpdyr=43, dpdxr0=44, dpdyr0=45, dtdt=46, u1aa=47, v1aa=48)
PBL_GUSTI_OUT = 113
PRED_FILL_COLS = dict(zs1=5, tkv=7, dbl=29, ug=31, vg=32, utop=37, vtop=38, qtop=39, ztop=40)
SKIP2 = ('zs1', 'tkv', 'dbl', 'ug', 'vg', 'utop', 'vtop', 'qtop', 'ztop')
ATM_KEYS = ('T', 'Q', 'U', 'V', 'UALIJ', 'VALIJ', 'EGCM', 'W2GCM', 'PMID', 'PEDN', 'PK', 'PDSIG', 'PEK', 'MA', 'TMOM', 'QMOM', 'DDMS', 'TDN1',
            'QDN1', 'DDML', 'DDM1', 'DPDX', 'DPDY', 'DPDX0', 'DPDY0', 'USTARPBL', 'LMONINPBL', 'PBLHT', 'DCLEV', 'PBLPTOP', 'T1AA', 'U1AA',
            'V1AA', 'TSAVG', 'QSAVG')


# ====================================================================================================== host: the template (once per step)
def _slot_lut(cells_ji, ncell):
    lut = np.full(ncell, -1, np.int64)
    lut[cells_ji] = np.arange(len(cells_ji))
    return lut


def _scatter_rows(rec, slot_of_row, nslot, valid_out):
    """rows -> (nslot, ncol) array in slot order; invalid slots get a copy of the first valid row (finite dummy)."""
    out = np.empty((nslot, rec.shape[1]))
    have = np.zeros(nslot, bool)
    out[slot_of_row] = rec
    have[slot_of_row] = True
    if not have.all():
        fill = rec[0] if len(rec) else np.zeros(rec.shape[1])
        out[~have] = fill
    valid_out[:] = have
    return out


def build_template(rec, reg=None, ghy_builder=None):
    """rec: dict from atm_step.surface_records(R) (pa pb ta tb la lb blk1 blk2 g1 g2).  Returns (tpl, host) where tpl is a pytree of jnp arrays
    (device) and host is a dict of static numpy index arrays / python ints (compile-time constants)."""
    import ghy_advnc_test as AT
    import land_chain as LC
    ncell = IM * JM
    cell_of = lambda r: (r[:, 1].astype(int) - 1) * IM + (r[:, 0].astype(int) - 1)          # flat cell index j*IM+i
    pa, pb, ta, tb, la, lb, g1, g2 = (rec[k] for k in ('pa', 'pb', 'ta', 'tb', 'la', 'lb', 'g1', 'g2'))
    blk1, blk2 = rec['blk1'], rec['blk2']
    # water cells: any open-water/ice PBL row
    wmask = np.zeros(ncell, bool)
    for t in (1, 2):
        wmask[cell_of(pa[pa[:, 2] == t])] = True
    wcells = np.nonzero(wmask)[0]
    Nw = len(wcells)
    wlut = _slot_lut(wcells, ncell)
    lcells = cell_of(la)
    ecells = cell_of(g1)
    assert len(set(lcells)) == len(lcells) and len(set(ecells)) == len(ecells)
    llut = _slot_lut(lcells, ncell)
    elut = _slot_lut(ecells, ncell)
    bcells = cell_of(blk1)
    assert np.array_equal(bcells, cell_of(blk2))
    host = dict(Nw=Nw, Nl=len(lcells), Ne=len(ecells), wj=wcells // IM, wi=wcells % IM, lj=lcells // IM, li=lcells % IM,
                ej=ecells // IM, ei=ecells % IM, bj=bcells // IM, bi=bcells % IM, wcells=wcells, lcells=lcells, ecells=ecells,
                bcells=bcells, blut=_slot_lut(bcells, ncell))
    tpl = {}
    masks = {}
    for ns, (p, t, l, g, blk) in enumerate(((pa, ta, la, g1, blk1), (pb, tb, lb, g2, blk2))):
        d = {}
        # PBL rows: water slots (type 1 then type 2), land ice, land
        vm = np.zeros((2, Nw), bool)
        rows_w = np.zeros((2 * Nw, p.shape[1]))
        trows_w = np.zeros((2 * Nw, t.shape[1]))
        for ty in (1, 2):
            sel = p[:, 2] == ty
            slot = wlut[cell_of(p[sel])]
            v = np.zeros(Nw, bool)
            rows_w[(ty - 1) * Nw:ty * Nw] = _scatter_rows(p[sel], slot, Nw, v)
            vt = np.zeros(Nw, bool)
            selt = t[:, 2] == ty
            trows_w[(ty - 1) * Nw:ty * Nw] = _scatter_rows(t[selt], wlut[cell_of(t[selt])], Nw, vt)
            assert np.array_equal(v, vt), 'PBL and tile row sets differ'
            vm[ty - 1] = v
        # dummy rows of a type with no valid row at all (e.g. no ice): copy the other type's rows
        for ty in (1, 2):
            if not vm[ty - 1].any():
                o = 2 - ty
                rows_w[(ty - 1) * Nw:ty * Nw] = rows_w[o * Nw:(o + 1) * Nw]
                trows_w[(ty - 1) * Nw:ty * Nw] = trows_w[o * Nw:(o + 1) * Nw]
        p3 = p[p[:, 2] == 3]
        p4 = p[p[:, 2] == 4]
        assert np.array_equal(p3[:, :2], l[:, :2]) and np.array_equal(p4[:, :2], g[:, :2])
        d.update(pw=rows_w, tw=trows_w, pl=p3, tl=l, pe=p4)
        masks[ns] = vm
        # fractions of the four types per cell (recorded fft)
        ft = np.zeros((ncell, 4))
        ft[bcells] = blk[:, 2:30:7]
        d['ftype'] = ft
        d['wvalid'] = vm
        tpl[ns] = d
    # slot validity consistent with ftype>0 (water types)
    for ns in (0, 1):
        ft = tpl[ns]['ftype']
        for ty in (0, 1):
            fv = ft[wcells, ty] > 0
            assert np.array_equal(fv, masks[ns][ty]), f'ftype>0 != row existence (ns {ns}, type {ty + 1})'
    assert np.array_equal(masks[0], masks[1]), 'water row sets differ between the substeps'
    host['wvalid'] = [masks[0], masks[1]]
    # CORIOL per cell from the substep-2 PBL rows (as stage_surface)
    cor = np.zeros(ncell)
    cor[cell_of(pb)] = pb[:, 36]
    # TRUP_in_rad of land from the recorded substep-1 land patch (infer_trup), ghy batches
    _, patch1, _ = TA.unpack(blk1)
    idx1 = host['blut'][ecells]
    trup = LC.infer_trup(g1, patch1['dth1'][idx1, 3], DT)
    gb = []
    for g in (g1, g2):
        gb.append(_ghy_batch(g, AT))
    width = max(b['edts'].shape[1] for b in gb)
    for b in gb:
        pad = width - b['edts'].shape[1]
        if pad:
            b['edts'] = np.pad(b['edts'], ((0, 0), (0, pad)))
            b['ecnc'] = np.pad(b['ecnc'], ((0, 0), (0, pad)))
            b['elai'] = np.pad(b['elai'], ((0, 0), (0, pad)))
            b['ebet'] = np.pad(b['ebet'], ((0, 0), (0, pad), (0, 0)))
    host['max_substeps'] = width
    # D195: the recorded GHY rows and the cells with ffnit > 11 (their dts are regenerated from the row, see nit_rebuild)
    host['nit_rows'] = (np.array(g1, copy=True), np.array(g2, copy=True))
    host['nit_hit'] = [np.nonzero(np.round(np.asarray(g)[:, 289]).astype(int) > 11)[0] for g in (g1, g2)]
    out = dict(ns=[jax.tree_util.tree_map(jnp.asarray, tpl[0]), jax.tree_util.tree_map(jnp.asarray, tpl[1])],
               coriol=jnp.asarray(cor.reshape(JM, IM)), trup=jnp.asarray(trup),
               ma1_land=jnp.asarray(g1[:, 165]), ghy=[jax.tree_util.tree_map(jnp.asarray, b) for b in gb])
    return out, host


NIT_KEYS = ('edts', 'ecnc', 'elai', 'ebet', 'nsub', 'dt')
NIT_LOG = []


NIT_MISMATCH = []


def _build_batch_nit_report(rec):
    """ghy_advnc_test_nit.build_batch_nit with the single difference that `info['nit'] == ffnit` is REPORTED (NIT_MISMATCH) instead of asserted: in a free-running
    day the iteration count of a cell may legitimately differ from the recorded one (D194 section 3).  Everything else is a copy, except that the dts row of such a cell is zeroed before it is filled (no stale recorded dts beyond a smaller recomputed count; identical when the count matches)."""
    import ghy_advnc_test as AT
    import ghy_compare as GC
    import ghy_ref_nit as N
    out = list(AT.build_batch_recorded(rec))
    (static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm, ws_can, shc_can, refs) = out
    ffnit = np.round(np.asarray(rec)[:, 289]).astype(int)
    res = [N.run_cell_full(rec[i], dt=900.0, use_recorded_dts=False) for i in np.where(ffnit > 11)[0]]
    width = max(ent_dts.shape[1], int(ffnit.max()), max([r[2]['nit'] for r in res], default=0))
    if width > ent_dts.shape[1]:
        pad = width - ent_dts.shape[1]
        ent_dts = np.pad(ent_dts, ((0, 0), (0, pad)))
        ent_cnc = np.pad(ent_cnc, ((0, 0), (0, pad)))
        ent_lai = np.pad(ent_lai, ((0, 0), (0, pad)))
        ent_betadl = np.pad(ent_betadl, ((0, 0), (0, pad), (0, 0)))
    dt_total = np.array(dt_total, dtype=float)
    n_substeps = np.array(n_substeps)
    for i, (col, r, info) in zip(np.where(ffnit > 11)[0], res):
        if info['nit'] != ffnit[i]:
            NIT_MISMATCH.append(dict(row=int(i), cell=(int(rec[i][0]), int(rec[i][1])), recomputed=int(info['nit']), recorded=int(ffnit[i])))
        u = GC.unpack(rec[i])[3]
        ent_dts[i, :] = 0.0
        for j in range(info['nit']):
            src = u[min(j, len(u) - 1)]
            ent_dts[i, j] = info['dts'][j]; ent_cnc[i, j] = src['cnc']; ent_lai[i, j] = src['lai']; ent_betadl[i, j] = src['betadl']
        n_substeps[i] = info['nit']
        dt_total[i] = 900.0
    return (static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm, ws_can, shc_can, refs)


def nit_rebuild(tpl, host, prec, eprec, precss, strict=True):
    """D195 (merge of D194): the GHY substep schedule of the cells with recorded ffnit > 11 follows OUR precipitation, as in the NumPy chain
    (surface_loop.Loop.stage_surface overwrites ffg columns 143-145 with PREC/EPREC/PRECSS before ghy_advnc_test.build_batch).  build_template builds the
    batch from the RECORDED row; for those cells only (none: no-op and no transfer) ONE declared device->host read of PREC/EPREC/PRECSS rebuilds the batch
    from the rows with our precipitation and replaces edts/ecnc/elai/ebet/nsub/dt of exactly those cells on the device (same shapes).  The assertion
    nit == ffnit of build_batch_nit is NOT caught.  Returns (tpl, n_cells, n_device_to_host_arrays)."""
    import ghy_advnc_test as AT
    hits = host.get('nit_hit')
    if J.schedule() == 'computed':
        return tpl, 0, 0            # D204: the GHY schedule is computed from the state (ghy_jax.advnc_gdtm); the recorded/rebuilt edts/nsub are not used
    if hits is None or not any(len(h) for h in hits):
        return tpl, 0, 0
    P, E, Ps = (np.asarray(x) for x in jax.device_get((prec, eprec, precss)))      # the declared device->host read (1 jax.device_get call, 3 arrays)
    dtsrc, rhow = 1800.0, 1000.0
    w = host['max_substeps']
    ghy = list(tpl['ghy'])
    ncell = 0
    for k, g0 in enumerate(host['nit_rows']):
        idx = hits[k]
        if not len(idx):
            continue
        g = np.array(g0, dtype=np.float64, copy=True)
        i, j = g[:, 0].astype(int) - 1, g[:, 1].astype(int) - 1
        g[:, 143] = P[i, j] / (dtsrc * rhow)
        g[:, 144] = E[i, j] / dtsrc
        g[:, 145] = Ps[i, j] / (dtsrc * rhow)
        gb = _ghy_batch(g, AT, batch=None if strict else _build_batch_nit_report)
        assert gb['edts'].shape[1] <= w, ('batch wider than the compiled max_substeps', gb['edts'].shape, w)
        new = dict(ghy[k])
        rep = {}
        for key in NIT_KEYS:
            v = np.asarray(gb[key])[idx]
            if v.ndim >= 2 and v.shape[1] < w:
                v = np.pad(v, [(0, 0), (0, w - v.shape[1])] + [(0, 0)] * (v.ndim - 2))
            rep[key] = float(np.abs(v - np.asarray(new[key])[idx]).max())
            new[key] = new[key].at[jnp.asarray(idx)].set(jnp.asarray(v).astype(new[key].dtype))
        NIT_LOG.append(dict(substep=k + 1, cells=[(int(i[c]) + 1, int(j[c]) + 1) for c in idx], nsub=[int(x) for x in np.asarray(gb['nsub'])[idx]], max_abs_change=rep))
        ghy[k] = new
        ncell += len(idx)
    return dict(tpl, ghy=ghy), ncell, 3


def _ghy_batch(g, AT, batch=None):
    """Host part of land_chain.run_ghy: the recorded batch (Ent exports, forcing, start state), conditioned as D135/D136 (all recorded data)."""
    (s0, d0, f, edts, ecnc, ebet, elai, ns, dt, snowm, wsc, shc, refs) = (batch or (AT.build_batch_recorded if J.schedule() == 'computed' else AT.build_batch))(g)
    f = dict(f)
    pr = np.maximum(f["pr"], 0.0)
    prs = np.minimum(np.maximum(f["prs"], 0.0), pr)
    f["pr"], f["prs"] = pr, prs
    f["htprs"] = np.where(pr <= 0.0, 0.0, f["htpr"] / np.where(pr <= 0.0, 1.0, pr) * prs)
    fv = np.asarray(f["fv"])
    pos = fv > 0.0
    f["irrig"] = np.where(pos, g[:, 147] / np.where(pos, fv, 1.0), 0.0)
    f["htirrig"] = np.where(pos, g[:, 148] / np.where(pos, fv, 1.0), 0.0)
    st = dict(J.init_static(jnp.asarray(s0["dz"]), jnp.asarray(s0["q"]), jnp.asarray(s0["qk"]), jnp.asarray(f["fb"]), jnp.asarray(f["fv"])))
    st["ws"] = st["ws"].at[:, 0, 1].set(jnp.asarray(wsc))
    st["shc"] = st["shc"].at[:, 0, 1].set(jnp.asarray(shc))
    st = J.init_xklh_static(st)
    st["sl"] = jnp.asarray(s0["sl"])
    return dict(static={k: np.asarray(v) for k, v in st.items()}, dyn0={k: np.asarray(v) for k, v in d0.items()},
                forcing={k: np.asarray(v) for k, v in f.items()}, edts=np.asarray(edts), ecnc=np.asarray(ecnc), ebet=np.asarray(ebet),
                elai=np.asarray(elai), nsub=np.asarray(ns), dt=np.asarray(dt), snowm=np.asarray(snowm))


# ====================================================================================================== device: glue ported from NumPy
def atm_layout(S):
    """atm_step.atm_layout on a dict of jnp arrays in the native chained-state axes."""
    t3 = lambda x: jnp.transpose(x, (1, 0, 2))
    l3 = lambda x: jnp.transpose(x, (2, 1, 0))
    return dict(T=t3(S['T']), Q=t3(S['Q']), U=t3(S['U']), V=t3(S['V']), UA=l3(S['UALIJ']), VA=l3(S['VALIJ']), E=l3(S['EGCM']),
                PMID=l3(S['PMID']), PEDN=l3(S['PEDN']), PK=l3(S['PK']), PEK1=S['PEK'][0].T, PDSIG=l3(S['PDSIG']), MA1=S['MA'][0].T)


def cell_inputs(S, atm, coriol_ji):
    """atm_step.cell_inputs on jnp (the per-cell PBL inputs that depend on the atmosphere), arrays (JM,IM)."""
    e = SC.layer1_exports(atm['T'], atm['Q'], atm['UA'], atm['VA'], atm['MA1'], atm['PEK1'], atm['PMID'][..., 0])
    ug, vg, dbl = SC.get_dbl(S['USTARPBL'].T, S['LMONINPBL'].T, coriol_ji, S['PBLHT'].T, S['DCLEV'].T, atm['T'], atm['Q'], atm['UA'],
                             atm['VA'], atm['PMID'], atm['PK'], e['ztop'])
    ddms, tdn1, qdn1, ddml = S['DDMS'].T, S['TDN1'].T, S['QDN1'].T, S['DDML'].T
    flag = (ddml == 1).astype(jnp.float64)
    mdn = jnp.maximum(ddms, -0.07)
    gusti = jnp.where(flag > 0, jnp.log(1.0 - 600.4 * mdn - FLOAT4_4375 * mdn * mdn), 0.0)
    pek1 = atm['PEK1']
    pk1 = atm['PK'][..., 0]
    c = dict(zs1=e['zs1'], tkv=e['tkv'], ps=atm['PEDN'][..., 0], ddml=flag, gusti=gusti,
             tdns=jnp.where(flag > 0, tdn1 * pek1 / pk1, 0.0), qdns=jnp.where(flag > 0, qdn1, 0.0),
             dbl=dbl, ug=ug, vg=vg, utop=e['utop'], vtop=e['vtop'], qtop=e['qtop'], ztop=e['ztop'],
             mdf=S['DDM1'].T, dpdxr=S['DPDX'].T, dpdyr=S['DPDY'].T, dpdxr0=S['DPDX0'].T, dpdyr0=S['DPDY0'].T,
             u1aa=S['U1AA'].T, v1aa=S['V1AA'].T)
    c['dtdt'] = (c['tkv'] - S['T1AA'].T * pek1) / 900.0
    return c


def override_pbl(rec, cell, cj, ci, skip=()):
    """Overwrite the atmosphere-dependent columns of PBL rows `rec` (N,154) by the per-cell values at the slots' cells (cj, ci)."""
    for nm, c in PBL_COLS.items():
        if nm in skip:
            continue
        rec = rec.at[:, c].set(cell[nm][cj, ci])
    return rec.at[:, PBL_GUSTI_OUT].set(cell['gusti'][cj, ci])


def override_tiles(tile, li, atm, tj, ti, lj, lii):
    """atm_step.override_tiles: ps, ma1, byma1, q1, thv1 of the water-tile rows and ps, ma1, q1 of the land-ice rows."""
    q1 = atm['Q'][..., 0]
    t = tile
    t = t.at[:, ST.IN['ps']].set(atm['PEDN'][..., 0][tj, ti])
    t = t.at[:, ST.IN['ma1']].set(atm['MA1'][tj, ti])
    t = t.at[:, ST.IN['byma1']].set(1.0 / atm['MA1'][tj, ti])
    t = t.at[:, ST.IN['q1']].set(q1[tj, ti])
    t = t.at[:, ST.IN['thv1']].set((atm['T'][..., 0] * (1.0 + q1 * SC.XDELT))[tj, ti])
    l = li
    l = l.at[:, LI.IN['ps']].set(atm['PEDN'][..., 0][lj, lii])
    l = l.at[:, LI.IN['ma1']].set(atm['MA1'][lj, lii])
    l = l.at[:, LI.IN['q1']].set(q1[lj, lii])
    return t, l


def first_layer_update(tmom, qmom, dth1, dq1, t1, q1, pk1):
    """atm_step.first_layer_update on jnp; arrays (IM,JM); tmom/qmom (9,IM,JM,LM)."""
    ft = jnp.where(dth1 * t1 < 0, -dth1 / (t1 * pk1), 0.0)
    fq = jnp.where((dq1 < 0) & (q1 > 0), -dq1 / q1, 0.0)
    tm = tmom.at[:, :, :, 0].set(tmom[:, :, :, 0] * (1.0 - ft)[None])
    qm = qmom.at[:, :, :, 0].set(qmom[:, :, :, 0] * (1.0 - fq)[None])
    small = (q1 + dq1) < 1e-12
    qm = qm.at[:, :, :, 0].set(jnp.where(small[None], 0.0, qm[:, :, :, 0]))
    return tm, qm


def _pbl_batch(rows):
    return P.advanc_batch(P.unpack_records(rows))


def _tiles_water(pbl_rows, tile_rows, out):
    """surface_chain_ff.run_chain body without PC.run / numpy: tile fluxes of the water slots from OUR PBL outputs."""
    d = ST.to_inputs(tile_rows)
    ice = tile_rows[:, 2] == 2
    df, h1, h2, f1, f2 = IP.ice_tile_props(d["tg1"], d["tg2"], d["snow"], d["ssi1"], d["ssi2"], d["srheat"], d["flag_dsws"] > 0.5)
    d.update(dF1dTG=jnp.where(ice, df, 0.0), hcg1=jnp.where(ice, h1, 1.0), hcg2=jnp.where(ice, h2, 1.0),
             fsri1=jnp.where(ice, f1, 0.0), fsri2=jnp.where(ice, f2, 0.0))
    d.update(us=out["us"], vs=out["vs"], ws=out["ws"], gusti=pbl_rows[:, 113], qsrf=out["qsrf"], cm=out["cm"], ch=out["ch"], cq=out["cq"],
             ts=out["tsv"], dskin=out["dskin"], tsv=out["tsv"])
    ddml = pbl_rows[:, 23] > 0.5
    d["tprime"] = jnp.where(ddml, pbl_rows[:, 25] - pbl_rows[:, 7], 0.0)
    d["qprime"] = jnp.where(ddml, pbl_rows[:, 26] - pbl_rows[:, 39], 0.0)
    return ST.tile_fluxes(d)


def _tiles_landice(p3, rec, out):
    d = LI.to_inputs(rec)
    ddml = p3[:, 23] > 0.5
    d.update(us=out["us"], vs=out["vs"], ws=out["ws"], gusti=p3[:, 113], qsrf=out["qsrf"], cm=out["cm"], ch=out["ch"], cq=out["cq"],
             ts=out["tsv"], dskin=out["dskin"], tsv=out["tsv"],
             tprime=jnp.where(ddml, p3[:, 25] - p3[:, 7], 0.0), qprime=jnp.where(ddml, p3[:, 26] - p3[:, 39], 0.0))
    return LI.tile_fluxes(d)


def _land(p4, out, gb, dyn, q1_land, ma1_land, trup, max_substeps):
    """land_chain.land_substep with the PBL outputs `out` already computed: GHY (recorded Ent exports) -> land patch."""
    ps = p4[:, 16]
    tsv, qsrf = out["tsv"], out["qsrf"]
    rho = 100.0 * ps / (RGAS * tsv)
    ddml = p4[:, 23] > 0.5
    forcing = dict(gb['forcing'])
    forcing.update(ts=tsv / (1.0 + qsrf * 0.0), qs=qsrf, rho=rho, ch=out["ch"], vs=out["ws"],
                   gusti=p4[:, PBL_GUSTI_OUT],        # GHY_DRV.f:1267 passes pbl_args%gusti (what the PBL used), not the recorded ffg gusti_in (D202)
                   tprime=jnp.where(ddml, p4[:, 25] - p4[:, 7], 0.0), qprime=jnp.where(ddml, p4[:, 26] - p4[:, 39], 0.0),
                   qm1=q1_land * ma1_land)
    d0 = dict(gb['dyn0'])
    if dyn is not None:
        d0.update(dyn)
    ghy = J.advnc_sched(gb['static'], d0, forcing, gb['edts'], gb['ecnc'], gb['ebet'], gb['elai'], gb['nsub'], gb['dt'], gb['snowm'],
                  max_substeps=max_substeps)
    rcdmws = out["cm"] * out["ws"] * rho
    dlw = DT * (trup - STBO * _pow4(ghy["tbcs"] + TF))
    patch = dict(uflux1=rcdmws * out["us"], vflux1=rcdmws * out["vs"],
                 dth1=-(-ghy["ashg"] + dlw) / (SHA * ma1_land), dq1=ghy["aevap"] / ma1_land, tsavg=tsv, qsavg=qsrf)
    return dict(patch=patch, pbl=out, ghy=ghy, rho=rho, dyn_next={k: ghy[k] for k in J_DYN_KEYS},
                evap_max_ij=ghy["evap_max_ij"], fr_sat_ij=ghy["fr_sat_ij"], elhx=p4[:, 19])


def _pow4(x):
    """NumPy's `x ** 4` on float64 arrays calls libm pow(x, 4.0) (no special case); jnp's `x ** 4` is lax.integer_pow = repeated multiplication, which
    rounds differently in 1 ulp of some elements.  lax.pow with a float exponent is the closest equivalent."""
    return jnp.power(x, 4.0)


J_DYN_KEYS = ("w", "ht", "nsn", "dzsn", "wsn", "hsn", "fr_snow")


def next_land_pbl_columns(rec, land):
    """land_chain.next_land_pbl_columns on jnp."""
    gh = land["ghy"]
    ps = rec[:, 16]
    tg = gh["tsns"] + TF
    qg_sat = P.qsat(tg, rec[:, 19], ps)
    # elhx of the substep that just ran (GHY_DRV.f:1304-1312), see land_chain.next_land_pbl_columns (D199)
    qg_sat_cur = qg_sat if land.get("elhx") is None else P.qsat(tg, land["elhx"], ps)
    qs = land["pbl"]["qsrf"]
    rcdhws = land["pbl"]["ch"] * land["pbl"]["ws"] * land["rho"]
    em, fr = land["evap_max_ij"], land["fr_sat_ij"]
    m = rcdhws > 1e-30
    qn = jnp.where(m, qs + em / (C001 * jnp.where(m, rcdhws, 1.0)), qs)
    qn = jnp.minimum(qn, qg_sat_cur)
    qg = fr * qg_sat_cur + (1.0 - fr) * qn
    rhosrf0 = 100.0 * ps / (RGAS * tg * (1.0 + qg * 0.0))
    rec = rec.at[:, 18].set(tg).at[:, 6].set(tg * (1.0 + qg * 0.0)).at[:, 8].set(qg_sat).at[:, 9].set(qg)
    rec = rec.at[:, 11].set(_pow4(gh["tbcs"] + TF)).at[:, 12].set(em * 1000.0 / rhosrf0).at[:, 13].set(fr)
    return rec


# ====================================================================================================== device: the substep core
def core_substep(atm, rows, ftype, wvalid, hostc, ghy_tpl, dyn, q1_ma1_trup, max_substeps):
    """One substep of the tile loop on the fixed slots.  atm: layout dict (JM,IM,L); rows: dict(pw, tw, pl, tl, pe) already overridden;
    ftype (ncell, 4); hostc: dict of static index arrays; ghy_tpl: GHY batch of this substep; dyn: carried GHY state or None.
    Returns dict(tile, pbl, li, pbl_li, land, comp, ex)."""
    Nw, Nl, Ne = hostc['Nw'], hostc['Nl'], hostc['Ne']
    pw, tw, pl, tl, pe = rows['pw'], rows['tw'], rows['pl'], rows['tl'], rows['pe']
    allp = jnp.concatenate([pw, pl, pe], axis=0)
    out = _pbl_batch(allp)
    o_w = {k: v[:2 * Nw] for k, v in out.items()}
    o_l = {k: v[2 * Nw:2 * Nw + Nl] for k, v in out.items()}
    o_e = {k: v[2 * Nw + Nl:] for k, v in out.items()}
    got = _tiles_water(pw, tw, o_w)
    gli = _tiles_landice(pl, tl, o_l)
    q1_land, ma1_land, trup = q1_ma1_trup
    lr = _land(pe, o_e, ghy_tpl, dyn, q1_land, ma1_land, trup, max_substeps)
    # patches (ncell, 4): zeros where a type is absent (ftype = 0 there)
    ncell = IM * JM
    wc = hostc['wcells']
    patch = {}
    z = jnp.zeros((ncell, 4))
    wv = wvalid                                 # (2, Nw) bool
    vals = {
        'uflux1': (got['dmua'], gli['uflux1'], lr['patch']['uflux1']),
        'vflux1': (got['dmva'], gli['vflux1'], lr['patch']['vflux1']),
        'dth1': (got['dth1'], gli['dth1'], lr['patch']['dth1']),
        'dq1': (got['dq1'], gli['dq1'], lr['patch']['dq1']),
        'tsavg': (o_w['tsv'], o_l['tsv'], lr['patch']['tsavg']),
        'qsavg': (o_w['qsrf'], o_l['qsrf'], lr['patch']['qsavg']),
    }
    for k, (vw, vl, ve) in vals.items():
        a = z
        a = a.at[wc, 0].set(jnp.where(wv[0], vw[:Nw], 0.0))
        a = a.at[wc, 1].set(jnp.where(wv[1], vw[Nw:], 0.0))
        a = a.at[hostc['lcells'], 2].set(vl)
        a = a.at[hostc['ecells'], 3].set(ve)
        patch[k] = a
    comp_all = TA.aggregate(ftype, patch)                      # (ncell,) per field; cells outside the domain have ftype 0
    comp = {k: v[hostc['bcells']] for k, v in comp_all.items()}  # block-cell vectors, as the NumPy stage
    return dict(tile=got, pbl=o_w, li=gli, pbl_li=o_l, land=lr, comp=comp, patch=patch, pbl_land=o_e)


def aturb_step(atm, comp, hostc):
    """first-layer exports -> ATURB flux arrays -> ATURB+UV on the full grid (fill outside the block cells as chain_two_substeps.run_aturb)."""
    bj, bi = hostc['bj'], hostc['bi']
    ma1 = atm['MA1'][bj, bi]
    fl = dict(UFLUX1=comp['uflux1'], VFLUX1=comp['vflux1'], TFLUX1=-comp['dth1'] * ma1 / DT, QFLUX1=-comp['dq1'] * ma1 / DT,
              TSAVG=comp['tsavg'], QSAVG=comp['qsavg'])
    shape = atm['T'].shape[:2]
    full = {}
    for k in ('UFLUX1', 'VFLUX1', 'TFLUX1', 'QFLUX1', 'TSAVG', 'QSAVG'):
        a = jnp.full(shape, {'TSAVG': 280.0, 'QSAVG': 0.0}.get(k, 1e-3))
        full[k] = a.at[bj, bi].set(fl[k])
    args = {k: atm[k] for k in ('T', 'Q', 'UA', 'VA', 'E', 'PMID', 'PEDN', 'PK', 'PEK1', 'PDSIG')}
    args.update(full)
    res, Un, Vn, ua, va = C2._aturb_uv(args, atm['U'], atm['V'], DT)
    return dict(T=res['t'], Q=res['q'], E=res['e'], W2=res['w2'], pblht=res['pblht'], dclev=res['dclev'], pblptop=res['pblptop'],
                U=Un, V=Vn, UA=ua, VA=va)


_VALID = jnp.asarray(AC.valid_mask((JM, IM)))


def merge_valid(new, old, name):
    if name in ('U', 'V'):
        mm = jnp.ones(new.shape, bool).at[0].set(False)
    else:
        mm = _VALID[..., None] if new.ndim == 3 else _VALID
    return jnp.where(mm, new, old)


def grid_from_block(vals, hostc, fill=0.0):
    """block-cell vector -> (IM,JM) array (atm_step._grid)."""
    a = jnp.full((JM, IM), fill)
    return a.at[hostc['bj'], hostc['bi']].set(vals).T


def _substep_with_flu(atm, rows, ftype, wvalid, hostc, ghy_tpl, dyn, qmt, tmom, qmom, pk1, max_substeps):
    r = core_substep(atm, rows, ftype, wvalid, hostc, ghy_tpl, dyn, qmt, max_substeps)
    dth1, dq1 = grid_from_block(r['comp']['dth1'], hostc), grid_from_block(r['comp']['dq1'], hostc)
    tmom, qmom = first_layer_update(tmom, qmom, dth1, dq1, jnp.transpose(atm['T'], (1, 0, 2))[:, :, 0],
                                    jnp.transpose(atm['Q'], (1, 0, 2))[:, :, 0], pk1)
    ex = aturb_step(atm, r['comp'], hostc)
    return r, ex, tmom, qmom


# ====================================================================================================== substep 2 inputs (predict_ns2)
def _accumulate(ft, wvalid, hostc, ustar, lmonin):
    """composite ustar/lmonin of the four types in the order 1,2,3,4 (np.add.at order of the NumPy stage); returns (JM,IM) arrays."""
    ust = jnp.zeros((JM, IM))
    lmo = jnp.zeros((JM, IM))
    Nw = hostc['Nw']
    parts = [(hostc['wj'], hostc['wi'], ft[hostc['wcells'], 0], wvalid[0], ustar[0], lmonin[0]),
             (hostc['wj'], hostc['wi'], ft[hostc['wcells'], 1], wvalid[1], ustar[1], lmonin[1]),
             (hostc['lj'], hostc['li'], ft[hostc['lcells'], 2], jnp.ones(hostc['Nl'], bool), ustar[2], lmonin[2]),
             (hostc['ej'], hostc['ei'], ft[hostc['ecells'], 3], jnp.ones(hostc['Ne'], bool), ustar[3], lmonin[3])]
    for j, i, f, v, u, l in parts:
        ust = ust.at[j, i].add(jnp.where(v, f * u, 0.0))
        lmo = lmo.at[j, i].add(jnp.where(v, f * l, 0.0))
    return ust, lmo


def _split_pbl(r, hostc):
    Nw = hostc['Nw']
    u = (r['pbl']['ustar'][:Nw], r['pbl']['ustar'][Nw:], r['pbl_li']['ustar'], r['pbl_land']['ustar'])
    l = (r['pbl']['lmonin'][:Nw], r['pbl']['lmonin'][Nw:], r['pbl_li']['lmonin'], r['pbl_land']['lmonin'])
    return u, l


def predict_ns2(r1, rows1, tpl2, ftype1, wvalid1, atm2, coriol, hostc, ta_overridden, la_overridden):
    """Port of chain_two_substeps.predict_ns2 to the fixed slots: substep-2 PBL rows (water, land ice, land), tile and land-ice rows from the
    substep-1 results r1 and the substep-2 atmosphere atm2 (our ATURB exit)."""
    Nw = hostc['Nw']
    e = SC.layer1_exports(atm2['T'], atm2['Q'], atm2['UA'], atm2['VA'], atm2['MA1'], atm2['PEK1'], atm2['PMID'][..., 0])
    u, l = _split_pbl(r1, hostc)
    ust, lmo = _accumulate(ftype1, wvalid1, hostc, u, l)
    ug, vg, dbl = SC.get_dbl(ust, lmo, coriol, atm2['pblht'], atm2['dclev'], atm2['T'], atm2['Q'], atm2['UA'], atm2['VA'], atm2['PMID'],
                             atm2['PK'], e['ztop'])
    cellv = dict(utop=e['utop'], vtop=e['vtop'], qtop=e['qtop'], tkv=e['tkv'], zs1=e['zs1'], ztop=e['ztop'], dbl=dbl, ug=ug, vg=vg)

    def fill(rec, out_pbl, cj, ci):
        for nm, c in PRED_FILL_COLS.items():
            rec = rec.at[:, c].set(cellv[nm][cj, ci])
        rec = rec.at[:, 50:58].set(out_pbl['u']).at[:, 58:66].set(out_pbl['v']).at[:, 66:74].set(out_pbl['t'])
        rec = rec.at[:, 74:82].set(out_pbl['q']).at[:, 82:89].set(out_pbl['e'])
        return rec.at[:, 33].set(out_pbl['cm']).at[:, 34].set(out_pbl['ch']).at[:, 35].set(out_pbl['cq'])

    wj = jnp.concatenate([hostc['wj'], hostc['wj']])
    wi = jnp.concatenate([hostc['wi'], hostc['wi']])
    p12 = fill(tpl2['pw'], r1['pbl'], wj, wi)
    p3 = fill(tpl2['pl'], r1['pbl_li'], hostc['lj'], hostc['li'])
    p4 = next_land_pbl_columns(fill(tpl2['pe'], r1['pbl_land'], hostc['ej'], hostc['ei']), r1['land'])
    # ice / land-ice ground state from the tile outputs (ocean tile unchanged in the loop)
    tg_ice = r1['tile']['tg1'][Nw:] + TF
    ice = p12[Nw:]
    qs = P.qsat(tg_ice, ice[:, 19], ice[:, 16])
    ice = ice.at[:, 18].set(tg_ice).at[:, 6].set(tg_ice).at[:, 8].set(qs).at[:, 9].set(qs)
    p12 = p12.at[Nw:].set(ice)
    tg_li = r1['li']['tg1'] + TF
    qs = P.qsat(tg_li, p3[:, 19], p3[:, 16])
    p3 = p3.at[:, 18].set(tg_li).at[:, 6].set(tg_li).at[:, 8].set(qs).at[:, 9].set(qs)
    # tile rows
    tnew = tpl2['tw']
    tj = jnp.concatenate([hostc['wj'], hostc['wj']])
    ti = wi
    tnew = tnew.at[:, ST.IN['q1']].set(e['qtop'][tj, ti])
    tnew = tnew.at[:, ST.IN['thv1']].set(atm2['T'][..., 0][tj, ti] * (1.0 + atm2['Q'][..., 0][tj, ti] * SC.XDELT))
    tnew = tnew.at[:, ST.IN['e0']].set(ta_overridden[:, ST.IN['e0']] + r1['tile']['f0dt'])
    tnew = tnew.at[:, ST.IN['evapor']].set(ta_overridden[:, ST.IN['evapor']] + r1['tile']['evap'])
    isice = jnp.arange(2 * Nw) >= Nw
    for nm, key in (('tg1', 'tg1'), ('tg2', 'tg2'), ('tr4', 'tr4')):
        c = ST.IN[nm]
        tnew = tnew.at[:, c].set(jnp.where(isice, r1['tile'][key], tnew[:, c]))
    lnew = tpl2['tl']
    lnew = lnew.at[:, LI.IN['q1']].set(e['qtop'][hostc['lj'], hostc['li']])
    lnew = lnew.at[:, LI.IN['tg1']].set(r1['li']['tg1'])
    return p12, p3, tnew, lnew, p4


# ====================================================================================================== the stage
def _host_consts(host):
    """static index arrays as device int arrays + python ints (these are closed over by the jitted functions: compile-time constants)."""
    h = dict(host)
    for k in ('wj', 'wi', 'lj', 'li', 'ej', 'ei', 'bj', 'bi', 'wcells', 'lcells', 'ecells', 'bcells'):
        h[k] = jnp.asarray(host[k])
    return h


def make_stage(host):
    """Returns the jitted units and a driver stage_surface_jax(Sd, tpl) -> dict.  `host` from build_template."""
    h = _host_consts(host)
    ms = host['max_substeps']
    Nw = host['Nw']
    wj2 = jnp.concatenate([h['wj'], h['wj']])
    wi2 = jnp.concatenate([h['wi'], h['wi']])

    def rows_override(atm, cell, tpl, skip, tw, tl):
        pw = override_pbl(tpl['pw'], cell, wj2, wi2, skip)
        pl = override_pbl(tpl['pl'], cell, h['lj'], h['li'], skip)
        pe = override_pbl(tpl['pe'], cell, h['ej'], h['ei'], skip)
        return dict(pw=pw, pl=pl, pe=pe)

    @jax.jit
    def prep1(Sd, tpl_all):
        tpl = tpl_all['ns'][0]
        atm1 = atm_layout(Sd)
        cell1 = cell_inputs(Sd, atm1, tpl_all['coriol'])
        rows = rows_override(atm1, cell1, tpl, (), None, None)
        tw, tl = override_tiles(tpl['tw'], tpl['tl'], atm1, wj2, wi2, h['lj'], h['li'])
        rows.update(tw=tw, tl=tl)
        return atm1, cell1, rows

    @jax.jit
    def core(atm, rows, ftype, wvalid, ghy_tpl, ma1_land, trup, dyn, tmom, qmom, pk):
        """One substep; the same compiled executable runs both substeps (all per-substep data are arguments; no eager op outside the jit)."""
        qmt = (atm['Q'][h['ej'], h['ei'], 0], ma1_land, trup)
        return _substep_with_flu(atm, rows, ftype, wvalid, h, ghy_tpl, dyn, qmt, tmom, qmom, pk[0], ms)

    def run_core(atm, rows, tpl_all, k, dyn, tmom, qmom, pk):
        n = tpl_all['ns'][k]
        return core(atm, rows, n['ftype'], n['wvalid'], tpl_all['ghy'][k], tpl_all['ma1_land'], tpl_all['trup'], dyn, tmom, qmom, pk)

    @jax.jit
    def between(Sd, atm1, cell1, rows1, r1, ex1, tpl_all):
        tpl2 = tpl_all['ns'][1]
        atm2 = dict(atm1)
        for k in ('T', 'Q', 'U', 'V', 'UA', 'VA'):
            atm2[k] = merge_valid(ex1[k], atm1[k], k)
        atm2['E'] = merge_valid(ex1['E'], atm1['E'], 'E')
        atm2['pblht'], atm2['dclev'] = ex1['pblht'], ex1['dclev']
        ftype1 = tpl_all['ns'][0]['ftype']
        p12, p3, tnew, lnew, p4 = predict_ns2(r1, rows1, tpl2, ftype1, tpl_all['ns'][0]['wvalid'], atm2, tpl_all['coriol'], h, rows1['tw'], rows1['tl'])
        cell2 = dict(cell1)
        t1aa = atm2['T'][..., 0]
        e2 = SC.layer1_exports(atm2['T'], atm2['Q'], atm2['UA'], atm2['VA'], atm2['MA1'], atm2['PEK1'], atm2['PMID'][..., 0])
        cell2['dtdt'] = (e2['tkv'] - t1aa * atm1['PEK1']) / DT
        cell2['u1aa'], cell2['v1aa'] = atm2['UA'][..., 0], atm2['VA'][..., 0]
        p12 = override_pbl(p12, cell2, wj2, wi2, SKIP2)
        p3 = override_pbl(p3, cell2, h['lj'], h['li'], SKIP2)
        p4 = override_pbl(p4, cell2, h['ej'], h['ei'], SKIP2)
        tnew2, lnew2 = override_tiles(tnew, lnew, atm2, wj2, wi2, h['lj'], h['li'])
        rows2 = dict(pw=p12, pl=p3, pe=p4, tw=tnew2, tl=lnew2)
        return {k: atm2[k] for k in atm1}, rows2        # same pytree structure as atm1: `core` compiles once for both substeps

    @jax.jit
    def final(Sd, atm2, r2, ex2, rows2, tpl_all):
        out = {}
        to_ijl = lambda x: jnp.transpose(x, (1, 0, 2))
        to_lij = lambda x: jnp.transpose(x, (2, 1, 0))
        out['T'] = to_ijl(merge_valid(ex2['T'], atm2['T'], 'T'))
        out['Q'] = to_ijl(merge_valid(ex2['Q'], atm2['Q'], 'Q'))
        out['U'] = to_ijl(merge_valid(ex2['U'], atm2['U'], 'U'))
        out['V'] = to_ijl(merge_valid(ex2['V'], atm2['V'], 'V'))
        out['UALIJ'] = to_lij(merge_valid(ex2['UA'], atm2['UA'], 'UA'))
        out['VALIJ'] = to_lij(merge_valid(ex2['VA'], atm2['VA'], 'VA'))
        out['EGCM'] = to_lij(merge_valid(ex2['E'], atm2['E'], 'E'))
        out['W2GCM'] = to_lij(merge_valid(ex2['W2'], jnp.transpose(Sd['W2GCM'], (2, 1, 0)), 'W2'))
        m = _VALID
        for k, kk in (('PBLHT', 'pblht'), ('DCLEV', 'dclev'), ('PBLPTOP', 'pblptop')):
            out[k] = jnp.where(m.T, ex2[kk].T, Sd[k])
        out['T1AA'] = jnp.where(m.T, out['T'][:, :, 0], Sd['T1AA'])
        out['U1AA'] = jnp.where(m.T, out['UALIJ'][0], Sd['U1AA'])
        out['V1AA'] = jnp.where(m.T, out['VALIJ'][0], Sd['V1AA'])
        out['TSAVG'] = Sd['TSAVG'].at[h['bi'], h['bj']].set(r2['comp']['tsavg'])
        out['QSAVG'] = Sd['QSAVG'].at[h['bi'], h['bj']].set(r2['comp']['qsavg'])
        ust, lmo = _accumulate(tpl_all['ns'][1]['ftype'], tpl_all['ns'][1]['wvalid'], h, *_split_pbl(r2, h))
        out['USTARPBL'], out['LMONINPBL'] = ust.T, lmo.T
        return out

    def stage(Sd, tpl_all, timers=None):
        def T(name, f):
            if timers is None:
                return f()
            t0 = time.perf_counter()
            o = f()
            jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, o)
            timers[name] = timers.get(name, 0.0) + time.perf_counter() - t0
            return o
        atm1, cell1, rows1 = T('prep1', lambda: prep1(Sd, tpl_all))
        r1, ex1, tmom, qmom = T('core_ns1', lambda: run_core(atm1, rows1, tpl_all, 0, tpl_all['ghy'][0]['dyn0'], Sd['TMOM'], Sd['QMOM'], Sd['PK']))
        atm2, rows2 = T('between', lambda: between(Sd, atm1, cell1, rows1, r1, ex1, tpl_all))
        r2, ex2, tmom, qmom = T('core_ns2', lambda: run_core(atm2, rows2, tpl_all, 1, r1['land']['dyn_next'], tmom, qmom, Sd['PK']))
        out = T('final', lambda: final(Sd, atm2, r2, ex2, rows2, tpl_all))
        out['TMOM'], out['QMOM'] = tmom, qmom
        return out, dict(r1=r1, r2=r2, ex1=ex1, ex2=ex2, rows1=rows1, rows2=rows2)

    stage.units = dict(prep1=prep1, core=core, run_core=run_core, between=between, final=final)
    return stage


def state_from_numpy(S):
    """atm_step state dict (native axes, NumPy) -> device dict of the ATM_KEYS used by the stage."""
    return {k: jnp.asarray(np.asarray(S[k], dtype=np.float64)) for k in ATM_KEYS}
