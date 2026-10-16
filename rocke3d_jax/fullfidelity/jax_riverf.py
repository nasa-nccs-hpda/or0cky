"""D190 (stage S5): RIVERF (river routing / lake outflow, LAKES.f:1708-2212 original version) as a device program; port of riverf_ff.riverf.

NEW module; riverf_ff.py is used (static tables, `load_statics`) but not edited.

Why this is NOT a loop on the device: in riverf_ff.riverf the sequential cell loop only READS mwl, gml, tlake, mldlk (they are updated afterwards, in a second loop),
and only ACCUMULATES into flow, eflow, flowo, eflowo.  So the per-cell decisions (dmm, dgm, which neighbour) are independent and are computed for all cells at
once.  The accumulation has to keep the order of the Fortran loop (floating-point addition is not associative): the sources of each target cell are STATIC (the river
network is a data file), so the contributions of every target are listed once on the host in loop order ((ju, iu) of the source cell, then the order of the events of
one source cell) and summed on the device in exactly that order, k = 0 .. K-1 (padding entries add +0.0).  `flow - dmm` is written as `flow + (-dmm)` (identical in IEEE).
Result of the two phases is the same as the NumPy loop for every input (checked on random lake states, including emergency, back-water and KDIREC = 9 cases).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import riverf_ff as RF  # noqa: E402

IM, JM = RF.IM, RF.JM
NC = IM * JM
DTSRC, RHOW, SHW, TF, TEENY, BYGRAV = RF.DTSRC, RF.RHOW, RF.SHW, RF.TF, RF.TEENY, RF.BYGRAV
URATE, LAKE_RISE_MAX, RIVER_FAC = RF.URATE, RF.LAKE_RISE_MAX, RF.RIVER_FAC
NSLOT = 7      # own, dest_normal, dest_emergency, kd9(2) a, kd9(2) b, kd9(8) a, kd9(8) b


def _flat(a):
    """(IM, JM) [i, j] -> (NC,) in the loop order (j outer, i inner)."""
    return np.ascontiguousarray(np.asarray(a).T).reshape(-1)


def _unflat(v):
    return v.reshape(JM, IM).T


def make_static(rs, flake, fland, fearth):
    """Host-side static tables from riverf_ff.load_statics (rs) and the static land/lake fractions ((IM, JM) numpy).  Returns a dict of numpy arrays /
    python ints that the jitted function closes over."""
    R = {}
    cell = np.arange(NC)
    ci, cj = cell % IM, cell // IM                                   # 0-based i, j of the flat cell
    domain = (cj >= 1) & (cj <= JM - 2) | ((cj == 0) | (cj == JM - 1)) & (ci == 0)    # IMAXJ domain
    R['domain'] = domain
    for k in ('focean', 'hlake', 'zatmo', 'rate', 'kdirec', 'kd911', 'iflow', 'jflow', 'ifl911', 'jfl911'):
        R[k] = _flat(rs[k])
    R['axyp'] = _flat(np.repeat(rs['axyp_j'][None, :], IM, axis=0))
    R['flake'], R['fland'], R['fearth'] = _flat(flake), _flat(fland), _flat(fearth)
    # destinations (1-based file indices -> flat cell, with validity)
    def dest(idx, jdx):
        ok = (jdx <= JM) & (jdx >= 1) & (idx <= IM) & (idx >= 1)
        d = np.where(ok, (jdx - 1) * IM + (idx - 1), 0)
        return d.astype(np.int64), ok
    R['dn'], R['dn_ok'] = dest(R['iflow'], R['jflow'])
    R['de'], R['de_ok'] = dest(R['ifl911'], R['jfl911'])
    iu, ju = ci + 1, cj + 1
    # KDIREC = 9 partners: kd = 2 -> (iu, ju + 1), kd = 8 -> (iu + 1, ju)
    R['k2'], R['k2_ok'] = dest(iu, ju + 1)
    R['k8'], R['k8_ok'] = dest(iu + 1, ju)
    R['flfac_src'] = np.where((ju == 1) | (ju == JM), float(IM), 1.0)
    # the second override (destination in a polar row) is applied per destination below
    kd = R['kdirec']
    is9 = (kd == 9) & domain
    R['is9'] = is9
    R['k2_act'] = is9 & R['k2_ok'] & (kd[R['k2']] == 9)
    R['k8_act'] = is9 & R['k8_ok'] & (kd[R['k8']] == 9)
    # ---- event tables in loop order
    foc_off_flow = lambda t: R['focean'][t] == 0          # destination receives into flow / eflow (else into flowo / eflowo)
    flow_ev = [[] for _ in range(NC)]
    ocean_ev = [[] for _ in range(NC)]
    for s in range(NC):
        if not domain[s]:
            continue
        if is9[s]:
            for slot_a, slot_b, tgt, act in ((3, 4, R['k2'], R['k2_act']), (5, 6, R['k8'], R['k8_act'])):
                if act[s]:
                    flow_ev[s].append(slot_a * NC + s)
                    flow_ev[int(tgt[s])].append(slot_b * NC + s)
            continue
        flow_ev[s].append(0 * NC + s)
        if R['dn_ok'][s]:
            t = int(R['dn'][s])
            (flow_ev if foc_off_flow(t) else ocean_ev)[t].append(1 * NC + s)
        if R['kd911'][s] > 0 and R['de_ok'][s]:
            t = int(R['de'][s])
            (flow_ev if foc_off_flow(t) else ocean_ev)[t].append(2 * NC + s)
    # events were appended in increasing s (and slot order within s), the order of the Fortran loop; a destination's list may interleave sources: sort by (s, slot)
    def table(ev):
        key = lambda e: (e % NC, e // NC)
        K = max((len(x) for x in ev), default=1)
        tab = np.full((NC, max(K, 1)), NSLOT * NC, np.int64)
        for t, lst in enumerate(ev):
            for k, e in enumerate(sorted(lst, key=key)):
                tab[t, k] = e
        return tab
    R['tab_flow'] = table(flow_ev)
    R['tab_ocean'] = table(ocean_ev)
    R['rate'] = R['rate']
    return R


def riverf(R, lake, flake_g, fland_g, fearth_g):
    """RIVERF on device arrays.  lake: dict mwl, gml, tlake, mldlk (IM, JM).  flake_g / fland_g / fearth_g are the geo fractions the NumPy call passes
    (they must equal the ones given to make_static).  Returns (lake dict (mwl, gml, tlake, mldlk, dlake, glake), flowo, eflowo, gtemp, gtempr, mlhc) (IM, JM)."""
    f = lambda a: jnp.transpose(a).reshape(-1)
    mwl, gml, tl, mld = (f(lake[k]) for k in ('mwl', 'gml', 'tlake', 'mldlk'))
    flake, fland, fearth = (jnp.asarray(R[k]) for k in ('flake', 'fland', 'fearth'))
    focean, hlake, zatmo, axyp, rate = (jnp.asarray(R[k]) for k in ('focean', 'hlake', 'zatmo', 'axyp', 'rate'))
    kd, kd911 = jnp.asarray(R['kdirec']), jnp.asarray(R['kd911'])
    domain = jnp.asarray(R['domain'])
    is9 = jnp.asarray(R['is9'])
    dn, de = jnp.asarray(R['dn']), jnp.asarray(R['de'])
    # ---- per-source decisions (normal branch)
    emerg = (kd == 0) & (flake > .949 * (flake + fearth)) & (mwl > (hlake + LAKE_RISE_MAX) * flake * RHOW * axyp) & (kd911 > 0)
    kdsel = jnp.where(emerg, kd911, kd)
    dsel = jnp.where(emerg, de, dn)
    dok = jnp.where(emerg, jnp.asarray(R['de_ok']), jnp.asarray(R['dn_ok']))
    mwlsill = jnp.where(emerg, RHOW * (hlake + LAKE_RISE_MAX) * flake * axyp, RHOW * hlake * flake * axyp)
    skip = ((kdsel == 0) & (fland * focean == 0)) | ~dok | ~domain | is9
    kd_d, flake_d, fearth_d, mwl_d, tl_d, hlake_d, axyp_d, zat_d = (a[dsel] for a in (kd, flake, fearth, mwl, tl, hlake, axyp, zatmo))
    checkback = ~(((kd_d >= 1) & (kd_d <= 8)) | (flake_d <= .949 * (flake_d + fearth_d)))
    mwlsilld = RHOW * axyp_d * flake_d * (hlake_d + BYGRAV * jnp.maximum(zatmo - zat_d, 0.0))
    c1 = checkback & (mwl_d > mwlsilld)
    lk = flake > 0
    dmm_b1 = URATE * DTSRC * (flake_d * axyp_d * (mwl - mwlsill) - flake * axyp * (mwl_d - mwlsilld)) / (flake * axyp + flake_d * axyp_d)
    dmm_b2 = -(mwl_d - mwlsilld) * URATE * DTSRC
    dmm_b = jnp.where(lk, dmm_b1, dmm_b2)
    do_back = c1 & jnp.where(lk, dmm_b1 < 0, True)
    # normal (non back-water) branch
    cont = ~do_back & (mwl <= mwlsill)
    dmm_n = (mwl - mwlsill) * rate
    dmm_n = jnp.where(mwl - dmm_n < 1e-6, mwl, dmm_n)
    dmm_n = jnp.minimum(dmm_n, .5 * RHOW * axyp)
    mlm = RHOW * mld * flake * axyp
    dmm_n = jnp.where(lk, jnp.minimum(dmm_n, .95 * mlm), dmm_n)
    dmm = jnp.where(do_back, dmm_b, dmm_n)
    dgm = jnp.where(do_back, tl_d * dmm * SHW, tl * dmm * SHW)
    fire = ~skip & ~cont
    dest_pole = (dsel // IM == 0) | (dsel // IM == JM - 1)
    flfac = jnp.where(dest_pole, 1.0 / IM, jnp.asarray(R['flfac_src']))
    foc_d = focean[dsel]
    dmm_s = jnp.where(foc_d == 0, dmm, RIVER_FAC * dmm)
    # ---- KDIREC = 9 internal seas
    def kd9(tgt, act):
        flakeu = jnp.maximum(flake, .01)
        flaked = jnp.maximum(flake[tgt], .01)
        ax_d, mwl_t, fl_t, tl_t, hl_t = axyp[tgt], mwl[tgt], flake[tgt], tl[tgt], hlake[tgt]
        mwlsill9 = RHOW * hlake * flakeu * axyp
        mwlsilld9 = RHOW * hl_t * flaked * ax_d
        d = URATE * DTSRC * (flaked * ax_d * (mwl - mwlsill9) - flakeu * axyp * (mwl_t - mwlsilld9)) / (flakeu * axyp + flaked * ax_d)
        pos = d > 0
        a_lim = 1 * RHOW * flake * axyp
        b_lim = 1 * RHOW * fl_t * ax_d
        cont9 = jnp.where(pos, mwl <= a_lim, mwl_t <= b_lim)
        d_pos = jnp.where(d > mwl - a_lim, mwl - a_lim, d)
        d_neg = jnp.where(d < b_lim - mwl_t, b_lim - mwl_t, d)
        dd = jnp.where(pos, d_pos, d_neg)
        dg = jnp.where(pos, tl * dd * SHW, tl_t * dd * SHW)
        ok = act & (flake + fl_t != 0) & ~cont9
        return dd, dg, ok
    k2t, k8t = jnp.asarray(R['k2']), jnp.asarray(R['k8'])
    d2, g2, ok2 = kd9(k2t, jnp.asarray(R['k2_act']))
    d8, g8, ok8 = kd9(k8t, jnp.asarray(R['k8_act']))
    # ---- event value arrays (7 * NC + 1 zero)
    fire_n = fire
    z = jnp.zeros(NC)
    def ev(kind):
        own = jnp.where(fire_n, -dmm if kind == 'm' else -dgm, z)
        v1 = jnp.where(fire_n & ~emerg, (dmm_s if kind == 'm' else dgm + 0.0) * flfac, z)
        v2 = jnp.where(fire_n & emerg, (dmm_s if kind == 'm' else dgm + 0.0) * flfac, z)
        a2, b2 = (-d2, d2) if kind == 'm' else (-g2, g2)
        a8, b8 = (-d8, d8) if kind == 'm' else (-g8, g8)
        parts = [own, v1, v2, jnp.where(ok2, a2, z), jnp.where(ok2, b2, z), jnp.where(ok8, a8, z), jnp.where(ok8, b8, z), jnp.zeros(1)]
        return jnp.concatenate(parts)
    Vm, Vg = ev('m'), ev('g')
    def accumulate(tab, V):
        acc = jnp.zeros(NC)
        for k in range(tab.shape[1]):
            acc = acc + V[jnp.asarray(tab[:, k])]
        return acc
    flow, eflow = accumulate(R['tab_flow'], Vm), accumulate(R['tab_flow'], Vg)
    flowo, eflowo = accumulate(R['tab_ocean'], Vm), accumulate(R['tab_ocean'], Vg)
    # ---- apply the net flow to the continental reservoirs (domain cells)
    app = domain & (fland + flake > 0.0)
    mwl2 = jnp.where(app, mwl + flow, mwl)
    gml2 = jnp.where(app, gml + eflow, gml)
    clr = app & (mwl2 < 1e-6)
    mwl2 = jnp.where(clr, 0.0, mwl2)
    gml2 = jnp.where(clr, 0.0, gml2)
    hlk1 = (mld * RHOW) * tl * SHW
    mld2 = mld + flow / (RHOW * flake * axyp)
    tl_lake = (hlk1 * flake * axyp + eflow) / (mld2 * RHOW * flake * axyp * SHW)
    tl_land = gml2 / (SHW * mwl2 + TEENY)
    mld_n = jnp.where(app & lk, mld2, mld)
    tl_n = jnp.where(app, jnp.where(lk, tl_lake, tl_land), tl)
    # exports (all cells), dlake / glake (domain), ocean flows
    cond = flake > 0
    gtemp = jnp.where(cond, tl_n, 0.0)
    gtempr = jnp.where(cond, tl_n + TF, 0.0)
    mlhc = jnp.where(cond, SHW * mld_n * RHOW, 0.0)
    dl = domain & cond
    dlake = jnp.where(dl, mwl2 / (RHOW * flake * axyp), 0.0)
    glake = jnp.where(dl, gml2 / (flake * axyp), 0.0)
    oc = domain & (focean > 0.0)
    byo = 1.0 / (axyp * focean)
    flowo = jnp.where(oc, flowo * byo, flowo)
    eflowo = jnp.where(oc, eflowo * byo, eflowo)
    u = lambda v: jnp.transpose(v.reshape(JM, IM))
    new = dict(lake, mwl=u(mwl2), gml=u(gml2), tlake=u(tl_n), mldlk=u(mld_n), dlake=u(dlake), glake=u(glake))
    return new, u(flowo), u(eflowo), u(gtemp), u(gtempr), u(mlhc)
