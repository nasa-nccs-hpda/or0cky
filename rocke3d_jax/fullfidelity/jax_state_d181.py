"""D181 (build stage S1 of JAX_COVERAGE_MATRIX section 6), variant "d181": the JAX state pytree with named groups, dict <-> pytree converters
and the static-field builder of the FIXED-SHAPE TILE LAYOUT that replaces the per-step record row sets.  New module; nothing existing is edited;
SOCRATES/RADIA untouched.  (Written under this name because a second agent wrote a different `jax_state.py` into the same directory while this
one was being built; the two files are independent.)

THREE PARTS
 1. STATIC FIELDS + FIXED-SHAPE TILE LAYOUT (`build_static`, `tile_ptype`, `tile_masks`, `rows_in_record_order`, `gather_rows`, `scatter_rows`)
    The real SURFACE records (ffp/ffs/ffl/ffg/fft) list only the tiles that exist, so their row counts change from step to step (D174: 10 distinct
    tile row counts over the 54-step nov26 day).  Here the tiles live in a FIXED layout (type, IM, JM) = (4, 72, 46) with a validity mask:
        type 0 ocean/lake water  exists where (1 - RSI) * FWATER > 0       (FWATER = FOCEAN + FLAKE)
        type 1 sea/lake ice      exists where RSI * FWATER > 0
        type 2 land ice          exists where FLICE > 0                    (static)
        type 3 land (GHY/Ent)    exists where FEARTH > 0                   (static)
    and only cells inside the IMAXJ domain (poles: i = 0 only) can hold a tile.  The rule is the one the ModelE SURFACE loop uses
    (ptype > 0) and the one `surface_loop.apply_state_to_records` already uses to set the PTYPE columns; here it is the DEFINITION of the layout.
    RSI is the ice fraction that SURFACE sees: after MELT_SI of the step (PRECIP_SI does not change RSI; checked in the code of
    `surface_loop.precip_si`).  Everything is written with an array-module argument (numpy or jax.numpy) so the same function runs on the host and
    inside a jit.  The fixed layout never changes shape; rows of a record file are obtained by `rows_in_record_order` (host side) only to
    compare with the real records or to read/write them.
 2. THE PYTREE (`driver_state_to_pytree`, `pytree_to_driver_state`, `build_state`, `describe_tree`)
    A nested dict of jax arrays (all x64) with the groups of JAX_COVERAGE_MATRIX section 4:
      atm        S of atm_step / jax_atm_step (T, U, V, Q, QCL, QCI, P, MA, PK, PMID, PDSIG, PEDN, PEK, GZ, MUS, MVS, MWS, TMOM, QMOM, hidden ATURB/PBL
                 fields, ..., the frozen radiation SRHR/TRHR/COSZ1 when present) in the native axis order of the dict (documented per field in
                 `describe_tree`; the order is NOT unified because the existing stage code relies on it)
      atm_carry  S['_carry']: CONDSE/RADIA carry arrays (CLDSS, CLDMC, SNOAGE, TAUSS, W_CLOUD, FRAC_*, ...)
      atm_ms     the LSCOND module arrays `ms` of the driver (lists of per-level numbers become arrays)
      ocean      surface_loop ocean state (g0m, s0m, mo, uo, vo, gradients, ogeoz, kpl, straits, work arrays), shapes (IM, JM, 13) or (13, NMST)
      ice        sea/lake ice (rsi, snowi, msi, hsi (IM,JM,4), ssi (IM,JM,4), pond_melt, flag_dsws bool)
      ice_dyn    DYNSI/ADVSI carry: usi, vsi, rsix, rsiy, uisurf, visurf
      lake       mwl, gml, tlake, mldlk;  landice: snowli, tlandi (IM,JM,2);  exch: atmosphere-grid exchange fields gtemp, gtemp2, gtempr, sss, mlhc
      land       ghy: GHY prognostic arrays (w, ht, nsn, dzsn, wsn, hsn, fr_snow) ; carry: the land_prev tree of the closed surface loop (when present)
      f3         F3 accumulators (aij/aijl dicts keyed by column, idacc)
      rad_frozen the provider's frozen SRHR/TRHR
      rng        seed0 (uint32 scalar);  clock: itime, k (int64), ss_itime
      tile       DERIVED, fixed layout: ptype (4,IM,JM) float64 and mask (4,IM,JM) bool, refreshed from ice by `refresh_tile`
                 (must be refreshed AFTER MELT_SI of the step; the restart RSI alone gives the wrong tile set)
      tile_pbl   NEW fixed-layout state from the restart (no dict counterpart in the driver): the per-type PBL carry (u,v,t,q,e profiles (4,IM,JM,8),
                 cmgs/chgs/cqgs, ustar_pbl, lmonin_pbl, ipbl (4,IM,JM)); today these live only as columns of the ffp records
      ent_state  the restart's padded Ent state (IM,JM,1023) (no dict counterpart: the driver holds Ent as Python objects)
    plus a separate constant pytree `static` (focean, flake, fland, flice, fearth, fwater, axyp, coriol, hlake, valid, is_ocean, is_lake, tile_static).
    Round trip: `pytree_to_driver_state(tree, meta)` returns a tree equal to the input bitwise (same dtypes, same values, None and empty
    containers and Python scalar types restored).  Host objects that are not numeric (the driver's timing_log, strings) are not converted: they stay
    in the metadata skeleton and are reported by `unconverted(meta)`.
 3. `describe_tree` returns the table (path, shape, dtype, bytes) used by the ledger entry.

No claim is made here about JAX-driven stepping: this module only holds and converts state.
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401  (XLA flags before jax is imported)
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import clouds_condse_io as cio  # noqa: E402

FF = cio.FF_DEFAULT
IM, JM, LM, LMO = 72, 46, 40, 13
NTYPE = 4
TILE_NAMES = ("ocean_lake", "ice", "landice", "land")
RESTART = {'nov26': 'fort1_nov26_itime33312.nc', 'dec01': 'fort1_dec01_itime33552.nc', 'jan01': 'fort1_jan01_itime17520.nc'}
DATE_IT0 = {'nov26': 33312, 'dec01': 33552, 'jan01': 17520}
PBL_SUFFIX = ("ocn01", "ice01", "gla01", "lnd01")          # restart names of the four tile types (type 0..3)


# ====================================================================================================== 1. static fields and tiles
def imaxj_mask():
    """Cells inside the IMAXJ domain: all i for 1 <= j <= JM-2 and only i = 0 at the two poles (0-based)."""
    m = np.zeros((IM, JM), bool)
    m[:, 1:JM - 1] = True
    m[0, 0] = m[0, JM - 1] = True
    return m


def build_static(date, ff=FF, it0=None):
    """Static fields of the exchange grid, as numpy float64 / bool arrays of shape (IM, JM), with the source of every field.
    Sources: FLAKE from the real RESTART (not from a dump); FOCEAN from the ocean geometry (ffo_geom, via surface_loop.load_statics);
    FLICE from the ffc_cse_in dump of step 0 (static topography field, not in the restart: RECORDED); FLAND = 1 - FOCEAN - FLAKE and
    FEARTH = FLAND - FLICE are DERIVED here (verified by `verify_static` to equal the recorded FLAND/FEARTH bitwise); AXYP, CORIOL and HLAKE as in
    surface_loop.load_statics (CORIOL from the ffp record of step 0 and HLAKE from ffl2: RECORDED static columns).
    Returns (static dict, sources dict)."""
    import surface_loop as L
    it0 = it0 or DATE_IT0[date]
    st = L.load_statics(date, ff, it0)
    rs = L.load_restart(date, ff)
    focean = np.asarray(st['ctx']['focean'], dtype=np.float64)
    flake = np.asarray(rs['lake']['flake'], dtype=np.float64)
    flice = np.asarray(st['flice'], dtype=np.float64)
    fland = 1.0 - focean - flake
    fearth = fland - flice
    valid = imaxj_mask()
    static = dict(focean=focean, flake=flake, fland=fland, flice=flice, fearth=fearth, fwater=focean + flake,
                  axyp=np.asarray(st['axyp'], dtype=np.float64), coriol=np.asarray(st['coriol'], dtype=np.float64),
                  hlake=np.asarray(st['hlake'], dtype=np.float64), valid=valid, is_ocean=focean > 0, is_lake=flake > 0)
    static['tile_static'] = np.stack([valid & (flice > 0), valid & (fearth > 0)])      # types 2 and 3
    sources = dict(focean="ocean geometry ffo_geom (ocean ctx), static", flake=f"restart {RESTART[date]}",
                   flice=f"ffc_cse_in_{it0} FLICE (static topography, RECORDED)", fland="derived 1 - FOCEAN - FLAKE", fearth="derived FLAND - FLICE",
                   fwater="derived FOCEAN + FLAKE", axyp="DXYPO(j) (ocean_step)", coriol=f"|ffp_{it0} col 36| (RECORDED static)",
                   hlake=f"ffl2_{it0} col 14 (RECORDED static)", valid="IMAXJ domain", tile_static="valid & (FLICE > 0), valid & (FEARTH > 0)")
    static['_recorded_check'] = dict(fland=np.asarray(st['fland'], dtype=np.float64), fearth=np.asarray(st['fearth'], dtype=np.float64),
                                     flake=np.asarray(st['geo']['flake'], dtype=np.float64))
    return static, sources


def verify_static(static):
    """Maximum |difference| of the derived FLAND/FEARTH and the restart FLAKE against the values recorded in ffc_cse_in (all must be 0.0)."""
    r = static['_recorded_check']
    return dict(fland=float(np.abs(static['fland'] - r['fland']).max()), fearth=float(np.abs(static['fearth'] - r['fearth']).max()),
                flake=float(np.abs(static['flake'] - r['flake']).max()))


def static_pytree(static):
    """The constant pytree (device arrays) of the static fields (drops private entries)."""
    return {k: jnp.asarray(v) for k, v in static.items() if not k.startswith('_')}


def tile_ptype(static, rsi, xp=np):
    """(4, IM, JM) tile fractions: [(1 - RSI) * FWATER, RSI * FWATER, FLICE, FEARTH], zero outside the IMAXJ domain.  xp = numpy or jax.numpy."""
    v = static['valid']
    fw = static['fwater']
    z = xp.zeros_like(rsi)
    return xp.stack([xp.where(v, (1.0 - rsi) * fw, z), xp.where(v, rsi * fw, z), xp.where(v, static['flice'], z), xp.where(v, static['fearth'], z)])


def tile_masks(static, rsi, xp=np):
    """(4, IM, JM) bool validity mask of the fixed tile layout: ptype > 0 inside the IMAXJ domain."""
    return tile_ptype(static, rsi, xp) > 0.0


def refresh_tile(state, static):
    """Pure function: returns the state with tile['ptype'] / tile['mask'] recomputed from ice['rsi'] (jit-able)."""
    p = tile_ptype(static, state['ice']['rsi'], jnp)
    return dict(state, tile=dict(ptype=p, mask=p > 0.0))


# ---- rows of the record files in the order of the real records
def rows_in_record_order(mask4, kind):
    """1-based (i, j, type) integer arrays of the rows a record file would hold for tile mask `mask4` (4, IM, JM), in the order of the real files:
    kind 'ffp' : types 1 and 2 sorted by (j, i, type), then all type 3 by (j, i), then all type 4 by (j, i)  (the PBL record)
         'ffs' : types 1 and 2 sorted by (j, i, type)                                                       (ocean/lake and ice tiles)
         'ffl' : type 3 by (j, i);  'ffg': type 4 by (j, i);  'fft': every valid cell (pass valid in mask4[0])."""
    def cells(t):
        i, j = np.nonzero(np.asarray(mask4[t]))
        o = np.lexsort((i, j))
        return i[o] + 1, j[o] + 1

    if kind in ('ffp', 'ffs'):
        i0, j0 = cells(0)
        i1, j1 = cells(1)
        ii = np.concatenate([i0, i1]); jj = np.concatenate([j0, j1]); tt = np.concatenate([np.full(len(i0), 1), np.full(len(i1), 2)])
        o = np.lexsort((tt, ii, jj))
        ii, jj, tt = ii[o], jj[o], tt[o]
        if kind == 'ffp':
            for t in (2, 3):
                a, b = cells(t)
                ii, jj, tt = np.concatenate([ii, a]), np.concatenate([jj, b]), np.concatenate([tt, np.full(len(a), t + 1)])
        return ii.astype(int), jj.astype(int), tt.astype(int)
    t = {'ffl': 2, 'ffg': 3, 'fft': 0}[kind]
    a, b = cells(t)
    return a.astype(int), b.astype(int), np.full(len(a), t + 1)


def gather_rows(field4, i, j, t):
    """Fixed layout (4, IM, JM[, ...]) -> row vector for 1-based rows (i, j, type)."""
    return np.asarray(field4)[t - 1, i - 1, j - 1]


def scatter_rows(values, i, j, t, shape, fill=0.0, dtype=np.float64):
    """Row vector -> fixed layout of shape (4, IM, JM, ...) = `shape`; cells without a row get `fill`."""
    out = np.full(shape, fill, dtype=dtype)
    out[t - 1, i - 1, j - 1] = values
    return out


# ---- comparison of the mask with the real record row sets
RECORD_TYPES = {'pa': (1, 2, 3, 4), 'pb': (1, 2, 3, 4), 'ta': (1, 2), 'tb': (1, 2), 'la': (3,), 'lb': (3,), 'g1': (4,), 'g2': (4,),
                'blk1': (0,), 'blk2': (0,)}
RECORD_KIND = {'pa': 'ffp', 'pb': 'ffp', 'ta': 'ffs', 'tb': 'ffs', 'la': 'ffl', 'lb': 'ffl', 'g1': 'ffg', 'g2': 'ffg', 'blk1': 'fft', 'blk2': 'fft'}


def record_cellsets(rec):
    """{record name: {type: (IM, JM) bool, 'dup': number of repeated (cell, type) rows}} from a dict of surface records (atm_step.surface_records)."""
    out = {}
    for name, types in RECORD_TYPES.items():
        rows = np.asarray(rec[name])
        d, dup = {}, 0
        for t in types:
            sel = rows[rows[:, 2] == t] if name in ('pa', 'pb', 'ta', 'tb') else rows
            m = np.zeros((IM, JM), bool)
            i, j = sel[:, 0].astype(int) - 1, sel[:, 1].astype(int) - 1
            np.add.at(m, (i, j), True)
            dup += len(i) - int(m.sum())
            d[t] = m
        d['dup'] = dup
        out[name] = d
    return out


def compare_rowsets(mask4, valid, rec, check_order=True):
    """Compare the fixed-layout tile mask with the real record row sets.  mask4 (4, IM, JM) bool, valid (IM, JM) bool, rec = surface_records dict.
    Returns dict: per record name -> dict(n_real, n_mask, only_real [(i, j, type)...], only_mask [...], dup, order_ok).
    'only_real' = rows in the real record without a mask tile; 'only_mask' = mask tiles without a real row (cells are 1-based).
    order_ok: the (i, j, type) columns of the real file equal `rows_in_record_order(mask)` row by row (set AND order identical)."""
    cs = record_cellsets(rec)
    res = {}
    for name, types in RECORD_TYPES.items():
        n_real = n_mask = 0
        only_real, only_mask = [], []
        for t in types:
            exp = valid if t == 0 else np.asarray(mask4[t - 1])
            real = cs[name][t]
            n_real += int(real.sum()); n_mask += int(exp.sum())
            for (i, j) in zip(*np.nonzero(real & ~exp)):
                only_real.append((int(i) + 1, int(j) + 1, t))
            for (i, j) in zip(*np.nonzero(exp & ~real)):
                only_mask.append((int(i) + 1, int(j) + 1, t))
        r = dict(n_real=n_real, n_mask=n_mask, only_real=only_real, only_mask=only_mask, dup=cs[name]['dup'])
        if check_order:
            m4 = np.array(mask4, copy=True)
            if name in ('blk1', 'blk2'):
                m4[0] = valid
            i, j, t = rows_in_record_order(m4, RECORD_KIND[name])
            rows = np.asarray(rec[name])
            if name in ('pa', 'pb', 'ta', 'tb'):
                tt = rows[:, 2].astype(int)
            else:
                tt = np.full(len(rows), {'la': 3, 'lb': 3, 'g1': 4, 'g2': 4, 'blk1': 1, 'blk2': 1}[name])
            r['order_ok'] = bool(len(i) == len(rows) and np.array_equal(i, rows[:, 0].astype(int)) and np.array_equal(j, rows[:, 1].astype(int))
                                 and np.array_equal(t, tt))
        res[name] = r
    return res


def rowsets_match(cmp):
    """True when every record has no only_real / only_mask entry and no duplicate rows."""
    return all(not r['only_real'] and not r['only_mask'] and r['dup'] == 0 for r in cmp.values())


def mismatch_summary(cmp):
    return {name: dict(n_real=r['n_real'], n_mask=r['n_mask'], only_real=len(r['only_real']), only_mask=len(r['only_mask']))
            for name, r in cmp.items() if r['only_real'] or r['only_mask'] or r['n_real'] != r['n_mask'] or r['dup']}


# ---- fixed-layout PBL tile carry from the restart
def tile_pbl_from_restart(date, ff=FF):
    """Per-type PBL carry of the real restart in the fixed layout: u, v, t, q, e (4, IM, JM, 8); cm, ch, cq, ustar, lmonin (4, IM, JM) float64;
    ipbl (4, IM, JM) int64.  Types in the order of TILE_NAMES (restart suffix ocn01, ice01, gla01, lnd01).  Source: the restart, not a dump."""
    import netCDF4 as nc
    R = nc.Dataset(f"{ff}/_pristine_restarts/{RESTART[date]}")
    try:
        def r2(k):
            return np.array(R.variables[k][:], dtype=np.float64).T

        def r3(k):
            return np.transpose(np.array(R.variables[k][:], dtype=np.float64), (1, 0, 2))
        out = {}
        for nm in ('uabl', 'vabl', 'tabl', 'qabl', 'eabl'):
            out[nm[0]] = np.stack([r3(f"{nm}_{s}") for s in PBL_SUFFIX])
        for nm, key in (('cmgs', 'cm'), ('chgs', 'ch'), ('cqgs', 'cq'), ('ustar_pbl', 'ustar'), ('lmonin_pbl', 'lmonin')):
            out[key] = np.stack([r2(f"{nm}_{s}") for s in PBL_SUFFIX])
        out['ipbl'] = np.rint(np.stack([r2(f"ipbl_{s}") for s in PBL_SUFFIX])).astype(np.int64)
        return out
    finally:
        R.close()


def ent_state_from_restart(date, ff=FF):
    """The restart's padded Ent state (IM, JM, 1023) float64."""
    import netCDF4 as nc
    R = nc.Dataset(f"{ff}/_pristine_restarts/{RESTART[date]}")
    try:
        return np.transpose(np.array(R.variables['ent_state'][:], dtype=np.float64), (1, 0, 2)).copy()
    finally:
        R.close()


# ====================================================================================================== 2. dict <-> pytree
# Routing of the paths of ModelDriver.state_dict() into the groups of the pytree.  Longest matching prefix wins; a path matching nothing goes to
# 'extra/<path joined by .>' (never silently dropped).
ROUTES = [
    (('S', '_carry'), ('atm_carry',)),
    (('S',), ('atm',)),
    (('ms',), ('atm_ms',)),
    (('surface', 'SS', 'ocean'), ('ocean',)),
    (('surface', 'SS', 'ice'), ('ice',)),
    (('surface', 'SS', 'lake'), ('lake',)),
    (('surface', 'SS', 'li'), ('landice',)),
    (('surface', 'SS', 'atm'), ('exch',)),
    (('surface', 'SS', 'ghy'), ('land', 'ghy')),
    (('surface', 'SS', 'itime'), ('clock', 'ss_itime')),
    (('surface', 'land_prev'), ('land', 'carry')),
    (('surface', 'usi'), ('ice_dyn', 'usi')),
    (('surface', 'vsi'), ('ice_dyn', 'vsi')),
    (('surface', 'uisurf'), ('ice_dyn', 'uisurf')),
    (('surface', 'visurf'), ('ice_dyn', 'visurf')),
    (('surface', 'adv'), ('ice_dyn',)),
    (('provider', 'rad'), ('rad_frozen',)),
    (('f3', 'aij'), ('f3', 'aij')),
    (('f3', 'aijl'), ('f3', 'aijl')),
    (('f3', 'idacc'), ('f3', 'idacc')),
    (('f3', 's0'), ('f3', 's0')),
    (('seed0',), ('rng', 'seed0')),
    (('itime',), ('clock', 'itime')),
    (('k',), ('clock', 'k')),
]
DTYPE_OVERRIDE = {('rng', 'seed0'): np.uint32}
HOST_PREFIXES = [('timing_log',)]          # bookkeeping of the host driver: kept in the metadata skeleton, never converted


class _Leaf:
    """Placeholder in the metadata skeleton: where a converted leaf belongs and how to restore its Python/numpy type."""
    __slots__ = ('tpath', 'kind', 'dtype', 'shape')

    def __init__(self, tpath, kind, dtype, shape):
        self.tpath, self.kind, self.dtype, self.shape = tpath, kind, dtype, shape

    def __repr__(self):
        return f"<Leaf {'/'.join(self.tpath)} {self.kind} {self.dtype} {self.shape}>"


def _route(path):
    best = None
    for src, dst in ROUTES:
        if tuple(path[:len(src)]) == src and (best is None or len(src) > len(best[0])):
            best = (src, dst)
    if best is None:
        return ('extra', '.'.join(str(p) for p in path))
    src, dst = best
    return tuple(dst) + tuple(str(p) for p in path[len(src):])


def _set(tree, tpath, value):
    d = tree
    for k in tpath[:-1]:
        d = d.setdefault(k, {})
    if tpath[-1] in d:
        raise ValueError(f"two leaves map to the same pytree path {'/'.join(tpath)}")
    d[tpath[-1]] = value


def _numeric_list(x):
    """If x is a non-empty list that is homogeneously (nested) rectangular and made of Python floats, Python ints or numpy float64 scalars only,
    return (array, elem) with elem 'py' or 'np'; else None."""
    if not isinstance(x, list) or not x:
        return None
    kinds = set()

    def scan(v):
        if isinstance(v, list):
            return bool(v) and all(scan(e) for e in v)
        if type(v) is float:
            kinds.add('pyf'); return True
        if type(v) is int:
            kinds.add('pyi'); return True
        if type(v) is np.float64:
            kinds.add('npf'); return True
        return False
    if not scan(x) or len(kinds) != 1:
        return None
    try:
        a = np.array(x, dtype=np.int64 if 'pyi' in kinds else np.float64)
    except (ValueError, TypeError):
        return None
    if a.dtype == object or a.ndim == 0:
        return None
    if a.tolist() != x:                 # not rectangular / lossy (also NaN lists: they stay element-wise)
        return None
    return a, ('np' if 'npf' in kinds else 'py')


def _walk(x, path, tree, host):
    """Returns the skeleton of x; array/scalar leaves go to `tree`, everything non-numeric stays in the skeleton (host)."""
    for hp in HOST_PREFIXES:
        if tuple(path[:len(hp)]) == hp:
            host.append(path)
            return copy.deepcopy(x)
    if isinstance(x, dict):
        return {k: _walk(v, path + (k,), tree, host) for k, v in x.items()}
    nl = _numeric_list(x)
    if nl is not None:
        a, elem = nl
        tpath = _route(path)
        _set(tree, tpath, jnp.asarray(a))
        return _Leaf(tpath, 'list_' + elem, a.dtype.str, a.shape)
    if isinstance(x, (list, tuple)):
        sk = [_walk(v, path + (i,), tree, host) for i, v in enumerate(x)]
        return tuple(sk) if isinstance(x, tuple) else sk
    if x is None:
        return None
    tpath = _route(path)
    if isinstance(x, np.ndarray) and x.dtype.kind in 'fiub':
        dt = DTYPE_OVERRIDE.get(tpath, None)
        a = x if dt is None else x.astype(dt)
        _set(tree, tpath, jnp.asarray(np.ascontiguousarray(a.astype(a.dtype.newbyteorder('=')))))
        return _Leaf(tpath, 'array', x.dtype.str, x.shape)
    if isinstance(x, (bool, np.bool_)):
        _set(tree, tpath, jnp.asarray(bool(x)))
        return _Leaf(tpath, 'bool' if type(x) is bool else 'npscalar', np.asarray(x).dtype.str, ())
    if isinstance(x, (int, np.integer)):
        dt = DTYPE_OVERRIDE.get(tpath, np.int64)
        if type(x) is int and dt == np.uint32:
            assert 0 <= x < 2 ** 32, f"seed {x} does not fit uint32"
        _set(tree, tpath, jnp.asarray(np.asarray(x, dtype=dt)))
        return _Leaf(tpath, 'int' if type(x) is int else 'npscalar', np.asarray(x).dtype.str, ())
    if isinstance(x, (float, np.floating)):
        _set(tree, tpath, jnp.asarray(np.asarray(x, dtype=np.float64)))
        return _Leaf(tpath, 'float' if type(x) is float else 'npscalar', np.asarray(x).dtype.str, ())
    host.append(path)
    return copy.deepcopy(x)


def driver_state_to_pytree(sd):
    """ModelDriver.state_dict()-like tree -> (pytree of jax arrays, meta).  meta = dict(skeleton, host_paths) restores the original tree exactly."""
    tree, host = {}, []
    sk = _walk(sd, (), tree, host)
    return tree, dict(skeleton=sk, host_paths=host)


def _fill(sk, tree):
    if isinstance(sk, _Leaf):
        d = tree
        for k in sk.tpath:
            d = d[k]
        a = np.asarray(d)
        if sk.kind == 'array':
            return a.astype(np.dtype(sk.dtype)).reshape(sk.shape)
        if sk.kind in ('list_py', 'list_np'):
            a = a.astype(np.dtype(sk.dtype)).reshape(sk.shape)
            if sk.kind == 'list_py':
                return a.tolist()

            def mk(b):
                return [b.dtype.type(v) for v in b] if b.ndim == 1 else [mk(r) for r in b]
            return mk(a)
        if sk.kind == 'bool':
            return bool(a)
        if sk.kind == 'int':
            return int(a)
        if sk.kind == 'float':
            return float(a)
        return np.dtype(sk.dtype).type(a)
    if isinstance(sk, dict):
        return {k: _fill(v, tree) for k, v in sk.items()}
    if isinstance(sk, tuple):
        return tuple(_fill(v, tree) for v in sk)
    if isinstance(sk, list):
        return [_fill(v, tree) for v in sk]
    return copy.deepcopy(sk)


def pytree_to_driver_state(tree, meta):
    """Inverse of driver_state_to_pytree (groups tile, tile_pbl, ent_state and static are ignored: they have no dict counterpart)."""
    return _fill(meta['skeleton'], tree)


def unconverted(meta):
    """Paths (original dict paths) of non-numeric host objects kept in the metadata skeleton instead of the pytree."""
    return ['/'.join(str(p) for p in path) for path in meta['host_paths']]


def build_state(sd, date, ff=FF, static=None):
    """Full pytree for one date: converted driver dict + tile layout (derived from ice) + tile_pbl and ent_state from the restart.
    Returns (state, static_pytree, meta)."""
    if static is None:
        static, _ = build_static(date, ff)
    tree, meta = driver_state_to_pytree(sd)
    sp = static_pytree(static)
    tree['tile_pbl'] = {k: jnp.asarray(v) for k, v in tile_pbl_from_restart(date, ff).items()}
    tree['ent_state'] = jnp.asarray(ent_state_from_restart(date, ff))
    tree = refresh_tile(tree, sp)
    return tree, sp, meta


# ---- initial driver dict of a date (what ModelDriver holds before step 0, with the atmosphere state of step 0 filled in)
def initial_driver_state(date, ff=FF, daydir=None, f3=True):
    """The tree ModelDriver(...).state_dict() has at step 0, with S set to the real start state (atm_step.init_state) of the first step.
    Constructs a ModelDriver (a few seconds); surface='closed', ent='record'.  daydir defaults to the date directory (nov26: 'nov26_day')."""
    import model_driver as MD
    import atm_step as A
    daydir = daydir or ('nov26_day' if date == 'nov26' else date)
    it0 = DATE_IT0[date]
    d = MD.ModelDriver(date=date, daydir=daydir, it0=it0, ff=ff, f3=f3, surface='closed', ent='record', rng='chain', daily='none')
    R = A.Real(daydir, it0, ff)
    d.S = A._native(A.init_state(R))
    return MD._np(d.state_dict())


# ====================================================================================================== 3. description / checks
def describe_tree(tree):
    """List of (path, shape, dtype, bytes) of every leaf of a pytree, sorted by path."""
    rows = []

    def rec(x, p):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                rec(x[k], p + [str(k)])
        else:
            a = np.asarray(x) if not hasattr(x, 'shape') else x
            rows.append(('/'.join(p), tuple(a.shape), str(a.dtype), int(np.prod(a.shape, dtype=np.int64)) * np.dtype(a.dtype).itemsize))
    rec(tree, [])
    return rows


def group_bytes(tree):
    out = {}
    for path, shape, dt, nb in describe_tree(tree):
        g = path.split('/')[0]
        out[g] = out.get(g, 0) + nb
    return out


def check_dtypes(tree):
    """Raise if any leaf has a dtype outside the explicit set (float64, int64, uint32, bool, int32, int8, uint8, uint64)."""
    bad = [(p, dt) for p, s, dt, nb in describe_tree(tree)
           if dt not in ('float64', 'int64', 'uint32', 'bool', 'int32', 'int8', 'uint8', 'uint64')]
    if bad:
        raise TypeError(f"unexpected dtypes: {bad}")
    return True


def trees_bitwise_equal(a, b):
    """model_driver.trees_equal (bytes, dtype, shape) for two driver dicts; returns the list of differing paths (empty = identical)."""
    import model_driver as MD
    return MD.trees_equal(a, b)
