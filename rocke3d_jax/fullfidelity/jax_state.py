"""D181 (stage S1 of JAX_COVERAGE_MATRIX section 6): the model STATE PYTREE of the coupled step, the round-trip converters to the dict-of-NumPy
state of model_driver / surface_loop_v2 / atm_step, and the FIXED-SHAPE tile layout.  New module; nothing existing is edited.

Layout conventions (unchanged from the dict world so that every existing kernel can be reused on the pytree leaves):
  atmosphere   exactly the arrays of atm_step.S ((IM,JM,LM) for T,U,V,Q,..; (LM,IM,JM) for MA,PK,PMID,PDSIG,EGCM,W2GCM,UALIJ,..; (LM+1,IM,JM) for
               PEDN,PEK,SRHR,TRHR; (9,IM,JM,LM) for TMOM,QMOM; (IM,JM) for P,PBLHT,...) -- names and shapes unchanged
  surface      SS of surface_loop (ocean (IM,JM,13), ice (IM,JM[,4]), lake, landice, sfc_atm = SS['atm'], ghy (IM,JM,..)) unchanged
  TILES        NEW fixed shape (4, IM, JM, ...): type index 0..3 = itype 1..4 = ocean, sea ice, land ice, land (SURFACE.f itype order)
               plus a boolean validity mask (4, IM, JM).  This replaces the per-step record row sets (ffp 4,536-4,593 rows, ffs 3,437-3,494, ...), whose
               size changes from step to step because ice tiles appear and vanish (D174: 10 distinct row counts over 54 steps).
Everything in the pytree is a jax.Array (float64/int64/bool; x64 enabled).  Python scalars (clock, RNG seed) are 0-d int64/float64 arrays; the Python
type is restored from the `Schema` on the way back.  Nothing in this module runs a model stage: it only holds, moves and checks state.

Two levels:
  generic   to_device(tree) -> (device tree, Schema); to_host(device tree, Schema) -> dict tree.  Bitwise identity for any tree of ndarrays,
            Python scalars, None, dict, list, tuple (dtype, shape and values preserved; non-native byte order is converted to native).
  structured  driver_to_pytree(sd) / pytree_to_driver(js): the state of model_driver.ModelDriver.state_dict() regrouped into the named groups
            of JAX_COVERAGE_MATRIX section 4 (atm, atm_carry, ms, ocean, ice, lake, landice, ghy, sfc_atm, dynsi, advsi, land_carry, tiles, ent,
            rng, clock, provider).  `land_carry` is land_prev (row arrays over the 753 land cells) moved to the fixed (IM,JM) layout with a mask.
"""
import clouds_jax_env  # noqa: F401  (XLA flags --xla_cpu_max_isa=AVX, --xla_disable_hlo_passes=algsimp BEFORE jax, as the rest of the JAX path)
import os
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IM, JM, LM, LMO = 72, 46, 40, 13
TILE_TYPES = ('ocean', 'ice', 'landice', 'land')          # index 0..3 = SURFACE.f itype 1..4
NT = 4

# ---------------------------------------------------------------------------------------------------------------- generic converters
_SCALAR_KINDS = {bool: 'bool', int: 'int', float: 'float'}


class Schema:
    """What the device tree does not carry: the Python kind of every scalar leaf and the paths of None / tuple nodes.
    Hashable-free plain container (it stays on the host and is never traced)."""

    def __init__(self):
        self.scalars = {}        # path -> 'bool'|'int'|'float'|'np.<dtype>'
        self.tuples = set()      # paths that were tuples
        self.nones = set()       # paths that were None (kept as None in the device tree too)
        self.byteswapped = []    # paths that were non-native byte order
        self.rowleaves = {}      # land_carry: path -> original shape (rows)

    def __repr__(self):
        return f"Schema(scalars={len(self.scalars)}, tuples={len(self.tuples)}, nones={len(self.nones)}, byteswapped={len(self.byteswapped)})"


def _list_kind(x):
    """'list:float' / 'list:int' / 'list:bool' for a non-empty list of exact Python scalars of one type, 'list2:float' for a rectangular list of
    such lists (the LSCOND `ms` scratch is 8,400 Python floats); None otherwise."""
    if not x:
        return None
    t = type(x[0])
    if t in _SCALAR_KINDS and all(type(v) is t for v in x):
        return 'list:' + _SCALAR_KINDS[t]
    if t is list and x[0] and all(type(v) is list and len(v) == len(x[0]) for v in x):
        t2 = type(x[0][0])
        if t2 in _SCALAR_KINDS and all(type(w) is t2 for v in x for w in v):
            return 'list2:' + _SCALAR_KINDS[t2]
    return None


def to_device(x, schema=None, path=''):
    """Recursively move a tree of host values to JAX arrays.  Returns (tree, Schema)."""
    top = schema is None
    schema = schema or Schema()
    if isinstance(x, dict):
        out = {k: to_device(v, schema, f"{path}/{k}")[0] for k, v in x.items()}
    elif isinstance(x, tuple):
        schema.tuples.add(path)
        out = tuple(to_device(v, schema, f"{path}[{i}]")[0] for i, v in enumerate(x))
    elif isinstance(x, list):
        k1 = _list_kind(x)
        if k1 is not None:                      # homogeneous list of Python scalars (or rectangular list of lists): ONE array
            schema.scalars[path] = k1
            out = jnp.asarray(np.array(x, dtype={'float': np.float64, 'int': np.int64, 'bool': np.bool_}[k1.split(':')[-1]]))
        else:
            out = [to_device(v, schema, f"{path}[{i}]")[0] for i, v in enumerate(x)]
    elif x is None:
        schema.nones.add(path)
        out = None
    elif isinstance(x, np.ndarray):
        if x.dtype.kind == 'f' and x.dtype.byteorder not in ('=', '|'):
            schema.byteswapped.append(path)
            x = x.astype(x.dtype.newbyteorder('='))
        if x.dtype == object:
            raise TypeError(f"object array at {path}")
        out = jnp.asarray(x)
    elif isinstance(x, (bool, np.bool_)):
        schema.scalars[path] = 'bool' if isinstance(x, bool) else 'np.bool_'
        out = jnp.asarray(x)
    elif isinstance(x, (int, np.integer)):
        schema.scalars[path] = 'int' if isinstance(x, int) else f'np.{np.asarray(x).dtype.name}'
        out = jnp.asarray(np.int64(x)) if not isinstance(x, np.integer) else jnp.asarray(x)
    elif isinstance(x, (float, np.floating)):
        schema.scalars[path] = 'float' if type(x) is float else f'np.{np.asarray(x).dtype.name}'   # np.float64 IS a float subclass: keep its type
        out = jnp.asarray(np.float64(x)) if not isinstance(x, np.floating) else jnp.asarray(x)
    elif hasattr(x, '__jax_array__') or isinstance(x, jax.Array):
        out = jnp.asarray(x)
    else:
        raise TypeError(f"cannot move {type(x).__name__} at {path or '/'} to the device (host-only object)")
    return (out, schema) if top else (out, schema)


def _host_leaf(a, kind):
    h = np.asarray(a)
    if kind.startswith('list'):
        return h.tolist()
    if kind == 'bool':
        return bool(h)
    if kind == 'int':
        return int(h)
    if kind == 'float':
        return float(h)
    if kind.startswith('np.'):
        return np.dtype(kind[3:]).type(h)
    return h


def to_host(x, schema, path=''):
    """Inverse of to_device (dict/list/tuple/None/scalars restored; arrays come back as writable native-endian numpy arrays)."""
    if isinstance(x, dict):
        return {k: to_host(v, schema, f"{path}/{k}") for k, v in x.items()}
    if path in schema.tuples:
        return tuple(to_host(v, schema, f"{path}[{i}]") for i, v in enumerate(x))
    if isinstance(x, (list, tuple)):
        return [to_host(v, schema, f"{path}[{i}]") for i, v in enumerate(x)]
    if x is None:
        return None
    if path in schema.scalars:
        return _host_leaf(x, schema.scalars[path])
    return np.array(x)


def tree_leaves_info(tree):
    """(number of arrays, total bytes) over the jax.Array leaves of a tree."""
    lv = [l for l in jax.tree_util.tree_leaves(tree) if hasattr(l, 'nbytes')]
    return len(lv), int(sum(l.nbytes for l in lv))


# ---------------------------------------------------------------------------------------------------------------- rows <-> fixed grid
def rows_to_grid(rows, ij, shape=(IM, JM)):
    """Scatter row array (N, ...) with 0-based (i,j) indices ij (N,2) to a fixed grid (IM,JM,...) (zeros elsewhere).  Bitwise."""
    rows = np.asarray(rows)
    g = np.zeros(tuple(shape) + rows.shape[1:], rows.dtype)
    g[ij[:, 0], ij[:, 1]] = rows
    return g


def grid_to_rows(grid, ij):
    return np.asarray(grid)[ij[:, 0], ij[:, 1]]


def row_order_ij(mask):
    """(N,2) 0-based (i,j) of the True cells of an (IM,JM) mask in record order: sorted by (j, i) (measured on every record family, see
    tests/test_jax_state.py::test_record_order_is_j_then_i)."""
    ii, jj = np.nonzero(np.asarray(mask))
    o = np.lexsort((ii, jj))
    return np.stack([ii[o], jj[o]], axis=1)


# ---------------------------------------------------------------------------------------------------------------- tile masks
def tile_masks(rsi, focean, flake, flice, fearth, valid):
    """Validity mask (4, IM, JM) of the SURFACE tiles from the state, as a pure function (works on numpy and on jax arrays, jit-able):
      ocean   (1 - RSI) * FWATER > 0       ice  RSI * FWATER > 0       (FWATER = FOCEAN + FLAKE; ocean-domain and lake-domain cells)
      landice FLICE > 0                    land FEARTH > 0
    on the cells where the SURFACE loop runs (valid = i <= IMAXJ(j)).  RSI must be the value AT THE SURFACE CALL, i.e. after MELT_SI and PRECIP_SI
    of the same step (atm_step/model_driver: S1['ice']['rsi']); the RSI of the state between steps (after ADVSI) differs in 74 + 66 cells (D181 test)."""
    xp = jnp if isinstance(rsi, jax.Array) else np
    fw = focean + flake
    return xp.stack([((1.0 - rsi) * fw > 0) & valid, (rsi * fw > 0) & valid, (flice > 0) & valid, (fearth > 0) & valid])


def tile_fractions(rsi, focean, flake, flice, fearth, valid):
    """Tile fractions (4, IM, JM) as the SURFACE records carry them (ptype columns, fft ftype): ocean (1-RSI)*FWATER, ice RSI*FWATER, FLICE, FEARTH,
    zero where the tile does not exist."""
    xp = jnp if isinstance(rsi, jax.Array) else np
    fw = focean + flake
    f = xp.stack([(1.0 - rsi) * fw, rsi * fw, flice, fearth])
    return xp.where(tile_masks(rsi, focean, flake, flice, fearth, valid), f, 0.0)


# ---------------------------------------------------------------------------------------------------------------- record <-> fixed layout
# families (columns): ffp PBL record 154, ffs ocean/ice tile record 90, ffl land-ice record 60, ffg land record 450, fft per-cell composite 40
FAMILIES = dict(pbl=154, tile=90, landice=60, land=450, blk=40)
REC_KEYS = dict(pbl=('pa', 'pb'), tile=('ta', 'tb'), landice=('la', 'lb'), land=('g1', 'g2'), blk=('blk1', 'blk2'))


def _ij0(rows):
    return rows[:, 0].astype(int) - 1, rows[:, 1].astype(int) - 1


def records_to_layout(rec):
    """Dict of the 10 SURFACE record arrays of one step (atm_step.surface_records) -> fixed layout:
         pbl     (2, 4, IM, JM, 154)   substep, type, i, j, column      tile (2, 2, IM, JM, 90)  types ocean, ice
         landice (2, IM, JM, 60)       land (2, IM, JM, 450)            blk  (2, IM, JM, 40)
         mask    (4, IM, JM) bool  tile existence from the pbl records (ffp has a row for every existing tile of every type)
       Entries not present in any row are zero.  Row identity i, j, itype columns are kept in the arrays (they are 'static columns')."""
    out = {}
    pbl = np.zeros((2, NT, IM, JM, 154))
    pmask = np.zeros((2, NT, IM, JM), bool)
    for s, k in enumerate(REC_KEYS['pbl']):
        r = np.asarray(rec[k])
        i, j = _ij0(r)
        t = r[:, 2].astype(int) - 1
        pbl[s, t, i, j] = r
        pmask[s, t, i, j] = True
    tile = np.zeros((2, 2, IM, JM, 90))
    tmask = np.zeros((2, 2, IM, JM), bool)
    for s, k in enumerate(REC_KEYS['tile']):
        r = np.asarray(rec[k])
        i, j = _ij0(r)
        t = r[:, 2].astype(int) - 1
        tile[s, t, i, j] = r
        tmask[s, t, i, j] = True
    li = np.zeros((2, IM, JM, 60))
    lmask = np.zeros((2, IM, JM), bool)
    for s, k in enumerate(REC_KEYS['landice']):
        r = np.asarray(rec[k])
        i, j = _ij0(r)
        li[s, i, j] = r
        lmask[s, i, j] = True
    land = np.zeros((2, IM, JM, 450))
    gmask = np.zeros((2, IM, JM), bool)
    for s, k in enumerate(REC_KEYS['land']):
        r = np.asarray(rec[k])
        i, j = _ij0(r)
        land[s, i, j] = r
        gmask[s, i, j] = True
    blk = np.zeros((2, IM, JM, 40))
    bmask = np.zeros((2, IM, JM), bool)
    for s, k in enumerate(REC_KEYS['blk']):
        r = np.asarray(rec[k])
        i, j = _ij0(r)
        blk[s, i, j] = r
        bmask[s, i, j] = True
    return dict(pbl=pbl, tile=tile, landice=li, land=land, blk=blk,
                masks=dict(pbl=pmask, tile=tmask, landice=lmask, land=gmask, blk=bmask),
                mask=pmask[0])


def layout_to_records(lay):
    """Inverse of records_to_layout: rebuild the 10 record arrays (rows in the record order of the Fortran dump: ocean+ice tiles by (j,i,type),
    then land ice by (j,i), then land by (j,i); fft by (j,i))."""
    rec = {}
    M = lay['masks']
    for s in (0, 1):
        # ffp / ffs: group A = types 1,2 sorted (j, i, type); group B = type 3; group C = type 4
        def rows_types(arr, mask, types):
            ii, jj, tt = [], [], []
            for t in types:
                i, j = np.nonzero(mask[t])
                ii.append(i); jj.append(j); tt.append(np.full(len(i), t))
            ii, jj, tt = np.concatenate(ii), np.concatenate(jj), np.concatenate(tt)
            o = np.lexsort((tt, ii, jj))
            return arr[tt[o], ii[o], jj[o]]
        pm, pb = M['pbl'][s], lay['pbl'][s]
        rec[REC_KEYS['pbl'][s]] = np.concatenate([rows_types(pb, pm, (0, 1)), rows_types(pb, pm, (2,)), rows_types(pb, pm, (3,))])
        rec[REC_KEYS['tile'][s]] = rows_types(lay['tile'][s], M['tile'][s], (0, 1))
        for fam in ('landice', 'land', 'blk'):
            ij = row_order_ij(M[fam][s])
            rec[REC_KEYS[fam][s]] = lay[fam][s][ij[:, 0], ij[:, 1]]
    return rec


# ---------------------------------------------------------------------------------------------------------------- PBL carry <-> fixed layout
_PBL_PROF = {'u': 8, 'v': 8, 't': 8, 'q': 8, 'e': 7}


def pbl_carry_to_grid(st):
    """drv_state_cols.PBLCarry.st ({itype: dict(u,v,t,q,e (JM,IM,n), cm,ch,cq (JM,IM), ipbl (JM,IM))}) -> fixed (type, IM, JM, ...) arrays."""
    out = {}
    for k, n in _PBL_PROF.items():
        out[k] = np.stack([np.transpose(st[t][k], (1, 0, 2)) for t in (1, 2, 3, 4)])
    for k in ('cm', 'ch', 'cq'):
        out[k] = np.stack([st[t][k].T for t in (1, 2, 3, 4)])
    out['ipbl'] = np.stack([st[t]['ipbl'].T for t in (1, 2, 3, 4)])
    return out


def grid_to_pbl_carry(g):
    st = {}
    for n, t in enumerate((1, 2, 3, 4)):
        d = {k: np.ascontiguousarray(np.transpose(g[k][n], (1, 0, 2))) for k in _PBL_PROF}
        for k in ('cm', 'ch', 'cq', 'ipbl'):
            d[k] = np.ascontiguousarray(g[k][n].T)
        st[t] = d
    return st


# ---------------------------------------------------------------------------------------------------------------- land_prev rows <-> fixed layout
def land_prev_to_grid(lp):
    """land_prev (dict of nested dicts of row arrays over the N land cells, plus p4_ij (N,2) 1-based) -> ({'mask': (IM,JM) bool, 'rows': tree of
    (IM,JM,...) arrays, 'other': leaves that are not row arrays}, rowleaves dict path -> original shape).  p4_ij must be in record order."""
    if lp is None:
        return None, {}
    ij = np.asarray(lp['p4_ij']).astype(int) - 1
    n = len(ij)
    mask = np.zeros((IM, JM), bool)
    mask[ij[:, 0], ij[:, 1]] = True
    if not np.array_equal(ij, row_order_ij(mask)):
        raise ValueError('land_prev p4_ij is not in (j,i) record order')
    rowleaves = {}

    def walk(x, path):
        if isinstance(x, dict):
            return {k: walk(v, f"{path}/{k}") for k, v in x.items() if not (path == '' and k == 'p4_ij')}
        if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == n:
            rowleaves[path] = x.shape
            return rows_to_grid(x, ij)
        return x
    tree = walk(lp, '')
    rowleaves['__p4_dtype__'] = np.asarray(lp['p4_ij']).dtype.str
    return dict(mask=mask, tree=tree), rowleaves


def grid_to_land_prev(g, rowleaves):
    if g is None:
        return None
    mask = np.asarray(g['mask'])
    ij = row_order_ij(mask)

    def walk(x, path):
        if isinstance(x, dict):
            return {k: walk(v, f"{path}/{k}") for k, v in x.items()}
        if path in rowleaves:
            return np.ascontiguousarray(np.asarray(x)[ij[:, 0], ij[:, 1]])
        return x
    out = walk(g['tree'], '')
    out['p4_ij'] = (ij + 1).astype(np.dtype(rowleaves['__p4_dtype__']))
    return out


# ---------------------------------------------------------------------------------------------------------------- structured pytree
class JaxState:
    """The device-resident state pytree (`tree`, dict of groups) plus the host-side `Schema`s needed to restore the dict world and the pieces that stay on
    the host by design (`host`: timing log, anything not convertible)."""

    def __init__(self, tree, schema, host):
        self.tree, self.schema, self.host = tree, schema, host

    def __getitem__(self, k):
        return self.tree[k]


def tile_group(rsi, statics):
    """{'mask': (4,IM,JM) bool, 'frac': (4,IM,JM)} of the SURFACE tiles for the ice fraction `rsi` (see tile_masks for which RSI)."""
    a = (rsi, statics['focean'], statics['flake'], statics['flice'], statics['fearth'], statics['valid'])
    return dict(mask=tile_masks(*a), frac=tile_fractions(*a))


def _split_S(S):
    atm = {k: v for k, v in S.items() if not k.startswith('_')}
    extra = [k for k in S if k.startswith('_') and k != '_carry']
    if extra:
        raise ValueError(f"unexpected underscore keys in S: {extra}")
    return atm, S.get('_carry')


def driver_to_pytree(sd, statics=None, pbl_carry=None):
    """ModelDriver.state_dict() (tree of numpy arrays and Python values) -> JaxState.  `statics` (optional, dict with focean, flake, flice, fearth,
    valid (IM,JM)) adds the derived tile mask/fractions of the state RSI to the `tiles` group; `pbl_carry` (drv_state_cols.PBLCarry) adds the
    persistent PBL state of the four tile types in the fixed (type, IM, JM, ...) layout (`tiles.pbl`)."""
    groups = {}
    sc = {}
    host = {'timing_log': sd.get('timing_log')}
    S = sd['S']
    atm, carry = _split_S(S) if S is not None else (None, None)
    groups['atm'] = atm
    groups['atm_carry'] = carry
    groups['ms'] = sd['ms']
    groups['provider'] = sd['provider']
    groups['rng'] = dict(seed0=sd['seed0'])
    clock = dict(itime=sd['itime'], k=sd['k'])
    sf = sd.get('surface')
    rowleaves = {}
    if sf is not None:
        SS = sf['SS']
        for g_, k_ in (('ocean', 'ocean'), ('ice', 'ice'), ('lake', 'lake'), ('landice', 'li'), ('sfc_atm', 'atm'), ('ghy', 'ghy')):
            groups[g_] = SS[k_]
        clock['ss_itime'] = SS['itime']
        groups['dynsi'] = dict(usi=sf['usi'], vsi=sf['vsi'], uisurf=sf['uisurf'], visurf=sf['visurf'])
        groups['advsi'] = sf['adv']
        lg, rowleaves = land_prev_to_grid(sf['land_prev'])
        groups['land_carry'] = lg
    for opt in ('ent', 'f3'):
        if opt in sd:
            groups[opt] = sd[opt]
    groups['clock'] = clock
    tiles = {}
    if statics is not None and sf is not None:
        tiles.update(tile_group(SS['ice']['rsi'], statics))
    if pbl_carry is not None:
        tiles['pbl'] = pbl_carry_to_grid(pbl_carry.st)
    if tiles:
        groups['tiles'] = tiles
    tree, schema = {}, Schema()
    for name, g in groups.items():
        t, _ = to_device(g, schema, f"/{name}")
        tree[name] = t
    schema.rowleaves = rowleaves
    # keys present in sd that were not mapped are an error (nothing may be silently dropped)
    known = {'itime', 'k', 'seed0', 'S', 'ms', 'provider', 'timing_log', 'surface', 'ent', 'f3'}
    missing = set(sd) - known
    if missing:
        raise ValueError(f"state_dict keys not mapped: {sorted(missing)}")
    has = dict(has_S=S is not None, has_surface=sf is not None, surface_keys=sorted(sf) if sf is not None else None, ss_keys=sorted(sf['SS']) if sf else None)
    host['has'] = has
    return JaxState(tree, schema, host)


def pytree_to_driver(js):
    """Inverse of driver_to_pytree: JaxState -> the state_dict tree of numpy arrays / Python values."""
    T, sch, has = js.tree, js.schema, js.host['has']
    H = lambda name: to_host(T[name], sch, f"/{name}")      # noqa: E731
    sd = {}
    clock = H('clock')
    sd['itime'], sd['k'] = clock['itime'], clock['k']
    sd['seed0'] = H('rng')['seed0']
    if has['has_S']:
        S = H('atm')
        c = H('atm_carry')
        if c is not None:
            S['_carry'] = c
    else:
        S = None
    sd['S'] = S
    sd['ms'] = H('ms')
    sd['provider'] = H('provider')
    sd['timing_log'] = js.host['timing_log']
    if has['has_surface']:
        ss_keys = has['ss_keys']
        SS = dict(ocean=H('ocean'), ice=H('ice'), lake=H('lake'), li=H('landice'), atm=H('sfc_atm'), itime=clock['ss_itime'], ghy=H('ghy'))
        SS = {k: SS[k] for k in ss_keys}
        dy = H('dynsi')
        lg = to_host(T['land_carry'], sch, '/land_carry')
        sf = dict(SS=SS, land_prev=grid_to_land_prev(lg, sch.rowleaves), usi=dy['usi'], vsi=dy['vsi'], adv=H('advsi'), uisurf=dy['uisurf'],
                  visurf=dy['visurf'])
        sd['surface'] = {k: sf[k] for k in has['surface_keys']}
    for opt in ('ent', 'f3'):
        if opt in T:
            sd[opt] = H(opt)
    return sd


def pytree_to_pbl_carry(js):
    """tiles.pbl -> drv_state_cols.PBLCarry (None if the state has no PBL carry)."""
    if 'tiles' not in js.tree or 'pbl' not in js.tree['tiles']:
        return None
    import drv_state_cols as D
    return D.PBLCarry(grid_to_pbl_carry(to_host(js.tree['tiles']['pbl'], js.schema, '/tiles/pbl')))


# ---------------------------------------------------------------------------------------------------------------- residency report
def residency_report(js):
    """Which part of the state is device-resident (jax.Array leaves) and what stays on the host, in the vocabulary of ACCEPTANCE_CRITERIA
    section 1: returns dict(device={group: (n_arrays, bytes)}, host=[...])."""
    dev = {}
    for name, g in js.tree.items():
        n, b = tree_leaves_info(g)
        if n:
            dev[name] = (n, b)
    host = []
    if js.host.get('timing_log') is not None:
        host.append('timing_log (Python list of dicts, diagnostics only)')
    host.append(f"Schema (Python kinds of {len(js.schema.scalars)} scalar leaves, {len(js.schema.nones)} None nodes, {len(js.schema.tuples)} tuples; metadata, never traced)")
    return dict(device=dev, host=host, total_arrays=sum(v[0] for v in dev.values()), total_bytes=sum(v[1] for v in dev.values()))


# ---------------------------------------------------------------------------------------------------------------- strict comparison
def strict_diff(a, b, path='', out=None):
    """Bitwise comparison of two host trees INCLUDING Python types (int vs float vs np scalar), dtype, shape, None, list/tuple/dict structure,
    NaN payloads and signed zeros (arrays are compared on their bytes).  Returns the list of differing paths (empty = identical)."""
    import struct
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: key missing in one")
            else:
                strict_diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                strict_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append(f"{path}: shape/dtype {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
        elif np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            d = (a != b) if a.dtype.kind != 'f' else (np.ascontiguousarray(a).view(np.uint64) != np.ascontiguousarray(b).view(np.uint64)) if a.dtype.itemsize == 8 else (a != b)
            out.append(f"{path}: {int(np.count_nonzero(d))} elements differ bitwise")
    elif isinstance(a, (float, np.floating)):
        if struct.pack('>d', float(a)) != struct.pack('>d', float(b)):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a is None:
        pass
    else:
        if not (a == b):
            out.append(f"{path}: {a!r} vs {b!r}")
    return out


# ---------------------------------------------------------------------------------------------------------------- constructors of the dict state
def _np_tree(x):
    """Same normalisation as model_driver._np (jax / big-endian arrays -> native numpy), without importing model_driver."""
    if isinstance(x, dict):
        return {k: _np_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_np_tree(v) for v in x)
    if hasattr(x, "__array__") and not isinstance(x, np.ndarray) and not np.isscalar(x):
        return np.array(x)
    if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.dtype.byteorder not in ("=", "|"):
        return x.astype(np.float64)
    return x


def initial_driver_state(date='nov26', daydir='nov26_day', it0=33312, ff=None):
    """The dict-world state at step `it0` of `date`, built from the restart and the step-start records WITHOUT constructing a ModelDriver:
    atmosphere S = atm_step.init_state (ffd_state s1 + hidden fields of ffa_step_a: recorded, D174 inventory), surface SS = surface_loop.init_surface_state
    (restart), DYNSI USI/VSI and ADVSI RSIX/RSIY from the restart, RNG seed = the first recorded seed.  Equal (strictly) to ModelDriver(...).state_dict()
    at k = 0 with S filled in (tests)."""
    import atm_step as A
    import clouds_condse_io as cio
    import surface_loop as L
    import surface_loop_v2 as V2
    ff = ff or A.FF
    R = A.Real(daydir, it0, ff)
    S = _np_tree(A._native(A.init_state(R)))
    st = L.load_statics(date, ff)
    st['ctx'] = L.make_ocean_ctx(date, ff)
    SS = L.init_surface_state(date, ff, st)
    loop = V2.Loop2(date, daydir, ff, st, SS, flags={}, it0=it0)
    V = loop.V
    seed0 = int(round(float(cio.read_cse(f"{ff}/{daydir}/ffc_cse_in_{it0}.bin")["SEEDS"][0]))) & 0xFFFFFFFF
    return dict(itime=it0, k=0, seed0=seed0, S=S, ms={}, provider=dict(rad=None), timing_log=[],
                surface=dict(SS=_np_tree(loop.SS), land_prev=_np_tree(loop.land_prev), usi=np.array(V.dyn.usi), vsi=np.array(V.dyn.vsi),
                             adv=_np_tree(V.adv), uisurf=_np_tree(V.uisurf), visurf=_np_tree(V.visurf)))
