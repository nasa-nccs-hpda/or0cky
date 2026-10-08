"""D187 shared helpers of the hybrid coupled step (d187_coupled_step.py) and its NumPy reference (d187_ref_numpy.py): flattening of state trees for
bitwise comparison, the surface-state comparison with the real step-boundary records (C2 of the surface half), category counting, I/O audit.

Real records used for the surface state after ONE step (it0), all from ff_data/<date>/ (written by the instrumented real build):
  ocean     ffo_state_<it0>.bin, snapshot tag 14 (ocean exit state of step it0: g0m s0m gxmo ... uo vo uod vod opress ogeoz ...)
  sea ice   advsi_dumps/<date>/ffadv_out_<it0>.bin (ice state after ADVSI = the last ice update of the step; ocean domain)
            and ffadv_in_<it0>.bin (ice state at ADVSI entry = after the ocean step's FORM_SI)
  DYNSI     ffo tag 1 (odmui, odmvi), ffz_undocn_<it0> (ustar)  [inputs of the ocean step in the real model, compared with OUR computed ones]
  RIVERF    ffo tag 1 (oflowo, oeflowo) compared with OUR computed flows
  lake, land ice, sea-ice/ocean tile state: the SURFACE-entry tile records of the NEXT step (ffs_<it0+1>, ffl_<it0+1>) against OUR state after
            surface_loop.surface_pre of step it0+1 (PRECIP_*, MELT_SI of step it0+1 applied with the real PREC/EPREC of step it0+1), i.e. the
            comparison method of D164/D170 (surface_loop.compare_pre_records) at row k=1.
NOT compared with any real record (no end-of-step record exists): the land GHY prognostic state (soil, snow, canopy) and Ent state; the land
influence is visible only through the atmosphere fields."""
import contextlib
import copy
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IM, JM = 72, 46


# ------------------------------------------------------------------------------------------------------------ flatten / categories
def flatten(tree, prefix='', out=None):
    """Nested dict / list / tuple of numpy arrays (and scalars) -> flat dict path -> ndarray (bool, int, float).  Non-numeric leaves are skipped."""
    out = {} if out is None else out
    if isinstance(tree, dict):
        for k in tree:
            flatten(tree[k], f'{prefix}{k}/', out)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            flatten(v, f'{prefix}{i}/', out)
    elif hasattr(tree, 'dtype') or isinstance(tree, (int, float, bool, np.generic)):
        a = np.asarray(tree)
        if a.dtype.kind in 'fiub':
            out[prefix.rstrip('/')] = a if a.dtype.kind != 'f' or a.dtype.byteorder in ('=', '|') else a.astype(np.float64)
    return out


def numeric_state(S):
    return {k: np.asarray(v) for k, v in S.items() if not k.startswith('_') and isinstance(v, (np.ndarray, np.generic)) and np.asarray(v).dtype.kind in 'fiub'}


def count_categories(res):
    c = {}
    for v in res.values():
        c[v['cat']] = c.get(v['cat'], 0) + 1
    return c


def slim(v):
    """Drop the long fields of a harness category record for JSON."""
    keep = ('cat', 'max_abs', 'scale', 'rel', 'n_diff', 'n', 'worst', 'n_cols_over')
    d = {k: v.get(k) for k in keep if k in v}
    if v.get('cols_over'):
        d['cols_over_first10'] = v['cols_over'][:10]
    if v.get('shape_mismatch'):
        d['shape_mismatch'] = v['shape_mismatch']
    return d


def cat_from_abs(maxabs, scale, bound=1e-12):
    """Category from a (max abs diff, scale) pair as returned by surface_loop.compare_pre_records."""
    if maxabs == 0:
        return 'A', 0.0
    rel = maxabs / scale if scale > 0 else float('inf')
    return ('B' if rel <= bound else ('C' if rel <= 1e-6 else 'D')), rel


# ------------------------------------------------------------------------------------------------------------ surface state snapshot
def surface_snapshot(loop):
    """Flat dict of the surface state after the step: ocean, ice, lake, land ice, atm exchange fields, ADVSI/DYNSI persistent state and the carried land state."""
    SS = loop.SS
    d = {}
    for grp in ('ocean', 'ice', 'lake', 'li', 'atm'):
        d.update(flatten(SS[grp], f'surf/{grp}/'))
    V = loop.V
    d.update(flatten(dict(rsix=V.adv['rsix'], rsiy=V.adv['rsiy'], usi=V.dyn.usi, vsi=V.dyn.vsi), 'surf/v2/'))
    lp = getattr(loop, 'land_prev', None)
    if lp:
        d.update(flatten(lp, 'land/'))
    return d


# ------------------------------------------------------------------------------------------------------------ C2 of the surface half
def surface_c2(date, it0, loop, ff=None):
    """Compare the surface state after step it0 (loop.SS, loop.last) with the real records listed in the module docstring.
    Returns a JSON-able dict of category records.  Does not modify the loop state (works on copies)."""
    import jax_harness as H
    import surface_loop as L
    import surface_loop_v2 as V2
    import advsi_ff as ADV
    import ocean_chain_io as C
    import ocean_step as O
    ff = ff or L.FF
    st = loop.st
    SS = loop.SS
    res = {}
    # ---- ocean exit (tag 14)
    ref14 = C.load_step(f'{ff}/{date}', it0)[14]
    fields = [k for k in (O.F3 + ['opress', 'mmi', 'opbot', 'ogeoz', 'kpl', 'vonp', 'must', 'g0mst', 's0mst']) if k in SS['ocean'] and k in ref14]
    res['ocean_exit_vs_ffo_tag14'] = {k: slim(H.field_category(np.asarray(SS['ocean'][k], float), np.asarray(ref14[k], float))) for k in fields}
    # ---- ice after ADVSI and at ADVSI entry
    advdir = f'{ff}/advsi_dumps/{date}'
    din, dout = ADV.read_dump(f'{advdir}/ffadv_in_{it0}.bin', f'{advdir}/ffadv_out_{it0}.bin')
    geo = st['geo']
    oc_ = geo['is_ocean'].copy()
    oc_[1:, 0] = False
    oc_[1:, -1] = False
    post = loop.last['post']

    def masked(cand, ref):
        m = oc_[..., None] if cand.ndim == 3 else oc_
        return H.field_category(np.where(m, cand, 0.0), np.where(m, ref, 0.0))
    res['ice_after_advsi_vs_ffadv_out'] = {k: slim(masked(SS['ice'][k], dout[k])) for k in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')}
    res['ice_at_advsi_entry_vs_ffadv_in'] = {k: slim(masked(post['ice_pre_adv'][k], din[k])) for k in ('rsi', 'msi', 'snowi', 'hsi', 'ssi')}
    # ---- DYNSI / RIVERF computed vs the real values
    rec = L.recorded_dynsi(date, it0, ff)
    d = post['dyn']
    u = rec['undocn']
    ii, jj = u['i'].astype(int) - 1, u['j'].astype(int) - 1
    ocn = geo['is_ocean'] & geo['valid']
    res['dynsi_vs_real'] = dict(
        odmui=slim(H.field_category(np.where(ocn, d['odmui'], 0.0), np.where(ocn, rec['odmui'], 0.0))),
        odmvi=slim(H.field_category(np.where(ocn, d['odmvi'], 0.0), np.where(ocn, rec['odmvi'], 0.0))),
        ustar=slim(H.field_category(d['ustar'][ii, jj], u['ustar'])))
    sn1 = C.load_step(f'{ff}/{date}', it0)[1]
    res['riverf_vs_real'] = dict(oflowo=slim(H.field_category(post['flowo'], sn1['oflowo'])), oeflowo=slim(H.field_category(post['eflowo'], sn1['oeflowo'])))
    # ---- next-step SURFACE-entry records (lake, land ice, ice and ocean tiles)
    inp1 = V2.replay_inputs_v2(date, it0 + 1, ff)
    S1, mid = L.surface_pre(copy.deepcopy(SS), st, dict(prec=inp1['prec'], eprec=inp1['eprec'], irrig_act=inp1['irrig_act']))
    cmp = L.compare_pre_records(S1, mid, st, date, it0 + 1, ff)
    ent = {}
    for k, v in cmp.items():
        if k.startswith('tileset') or k.startswith('tiles.'):
            ent[k] = dict(n=v[0] if k.startswith('tileset') else v[2])
            continue
        cat, rel = cat_from_abs(v[0], v[1])
        ent[k] = dict(cat=cat, max_abs=v[0], scale=v[1], rel=rel, n_tiles=v[2])
    res['next_step_entry_records_vs_our_state'] = ent
    res['note'] = ('records of step it0+1 are the SURFACE-entry state AFTER PRECIP_*/MELT_SI of step it0+1; our state is advanced through the same '
                   'stages of step it0+1 with the real PREC/EPREC of step it0+1 (D164/D170 method, row k=1)')
    return res


def summarize_surface_c2(c2):
    """Counts of categories per group."""
    out = {}
    for g, v in c2.items():
        if not isinstance(v, dict):
            continue
        cats = {}
        for k, r in v.items():
            if isinstance(r, dict) and 'cat' in r:
                cats[r['cat']] = cats.get(r['cat'], 0) + 1
        out[g] = cats
    return out


# ------------------------------------------------------------------------------------------------------------ I/O audit
class IOAudit:
    """Python-level file-open audit (sys.addaudithook 'open'): the distinct files under ff_data and the restart directory that were opened for reading, with
    sizes and the first-open order.  netCDF4 opens are not seen by the Python audit and are added by the caller (restart file)."""

    def __init__(self, roots=('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data',)):
        self.roots = tuple(roots)
        self.files = {}
        self.tag = ''
        self._on = False
        sys.addaudithook(self._hook)

    def _hook(self, event, args):
        if not self._on or event != 'open':
            return
        p = args[0]
        if isinstance(p, str) and p.startswith(self.roots):
            mode = args[1] if len(args) > 1 else 'r'
            if mode in (None, 'r', 'rb', 'rt', 'r+b') or (isinstance(mode, str) and 'w' not in mode and 'a' not in mode):
                self.files.setdefault(p, self.tag)

    @contextlib.contextmanager
    def on(self, tag=''):
        old = self.tag
        self.tag, self._on = tag, True
        try:
            yield self
        finally:
            self._on, self.tag = False, old

    def report(self):
        rows = []
        for p, tag in self.files.items():
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = None
            rows.append(dict(file=p.replace('/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/', ''), bytes=sz, first_read_in=tag))
        return rows
