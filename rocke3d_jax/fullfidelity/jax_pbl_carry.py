"""D222: the persistent PBL state across steps on the device (ROCKE_PBL_CARRY=1, default off).

Replaces the recorded substep-1 PBL input columns (ffp columns 51-89: u,v,t,q,e profiles; 34-36: cm, ch, cq) of the next step's template by the
substep-2 PBL outputs of OUR previous step, with the `loadbl` donor rule of PBL_DRV.f for tiles that had no PBL call in the previous step.
Logic and sources: drv_state_cols.PBLCarry (D177; bitwise from the record carry), see its module docstring.  Same arithmetic, device arrays:
carry = dict(u,v,t,q: (4,JM,IM,8), e: (4,JM,IM,7), cm,ch,cq: (4,JM,IM), ipbl: (4,JM,IM) int32); type index 0..3 = ocean, ice, land ice, land.
"""
import jax
import jax.numpy as jnp

IM, JM = 72, 46
NP = {'u': 8, 'v': 8, 't': 8, 'q': 8, 'e': 7}
SL = {'u': slice(50, 58), 'v': slice(58, 66), 't': slice(66, 74), 'q': slice(74, 82), 'e': slice(82, 89)}
CC = {'cm': 33, 'ch': 34, 'cq': 35}
DONORS = ((1, (2, 4)), (2, (1,)), (3, (4,)), (4, (3, 1)))       # loadbl order of donors (1-based types)
VALS = tuple(NP) + tuple(CC)


def empty():
    c = {k: jnp.zeros((4, JM, IM, n)) for k, n in NP.items()}
    c.update({k: jnp.zeros((4, JM, IM)) for k in CC})
    c['ipbl'] = jnp.zeros((4, JM, IM), jnp.int32)
    return c


def loadbl(c):
    """Tiles whose ipbl is 0 take the state of the first donor type with ipbl == 1 (all from the entry state: the conditions are exclusive)."""
    old = c['ipbl']
    out = dict(c)
    for t, donors in DONORS:
        need = old[t - 1] == 0
        done = jnp.zeros_like(need)
        for dn in donors:
            m = need & ~done & (old[dn - 1] == 1)
            for k in VALS:
                mm = m[..., None] if out[k].ndim == 4 else m
                out[k] = out[k].at[t - 1].set(jnp.where(mm, c[k][dn - 1], out[k][t - 1]))
            done = done | m
    out['ipbl'] = jnp.zeros_like(old)
    return out


def _fill(rows, c, t, j, i, valid):
    for k, sl in SL.items():
        rows = rows.at[:, sl].set(jnp.where(valid[:, None], c[k][t, j, i], rows[:, sl]))
    for k, col in CC.items():
        rows = rows.at[:, col].set(jnp.where(valid, c[k][t, j, i], rows[:, col]))
    return rows


@jax.jit
def fill_template(pw, pl, pe, c, ix):
    """pw (2Nw,154), pl (Nl,154), pe (Ne,154) rows of the substep-1 template; ix: dict of index arrays (tw, wj2, wi2, wvalid2, lj, li, ej, ei)."""
    pw = _fill(pw, c, ix['tw'], ix['wj2'], ix['wi2'], ix['wvalid2'])
    pl = _fill(pl, c, jnp.full(pl.shape[0], 2), ix['lj'], ix['li'], jnp.ones(pl.shape[0], bool))
    pe = _fill(pe, c, jnp.full(pe.shape[0], 3), ix['ej'], ix['ei'], jnp.ones(pe.shape[0], bool))
    return pw, pl, pe


@jax.jit
def update(c, o_w, o_l, o_e, ix):
    """Store the PBL outputs of a substep (o_*: dicts u,v,t,q,e (N,8|7), cm,ch,cq (N,)); c comes from loadbl (ipbl all 0)."""
    out = dict(c)
    for o, t, j, i, valid in ((o_w, ix['tw'], ix['wj2'], ix['wi2'], ix['wvalid2']),
                              (o_l, jnp.full(o_l['cm'].shape[0], 2), ix['lj'], ix['li'], jnp.ones(o_l['cm'].shape[0], bool)),
                              (o_e, jnp.full(o_e['cm'].shape[0], 3), ix['ej'], ix['ei'], jnp.ones(o_e['cm'].shape[0], bool))):
        for k, n in NP.items():
            out[k] = out[k].at[t, j, i].set(jnp.where(valid[:, None], o[k][:, :n], out[k][t, j, i]))
        for k in CC:
            out[k] = out[k].at[t, j, i].set(jnp.where(valid, o[k], out[k][t, j, i]))
        out['ipbl'] = out['ipbl'].at[t, j, i].set(jnp.where(valid, 1, out['ipbl'][t, j, i]))
    return out


def index_arrays(host, wvalid):
    """host: build_template host dict; wvalid: (2, Nw) bool validity of the water slots of the substep."""
    import numpy as np
    Nw = host['Nw']
    wj, wi = np.asarray(host['wj']), np.asarray(host['wi'])
    return dict(tw=jnp.asarray(np.concatenate([np.zeros(Nw, int), np.ones(Nw, int)])),
                wj2=jnp.asarray(np.concatenate([wj, wj])), wi2=jnp.asarray(np.concatenate([wi, wi])),
                wvalid2=jnp.asarray(wvalid).reshape(-1),
                lj=jnp.asarray(host['lj']), li=jnp.asarray(host['li']), ej=jnp.asarray(host['ej']), ei=jnp.asarray(host['ei']))
