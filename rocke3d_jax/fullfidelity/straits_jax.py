"""JAX (batched) STBDRA -- straits bottom and side drag, Stage 2, D69 (speed-first; see D62).

STBDRA (OSTRAITS.f:287-328) runs once per model step after STCONV (OCNDYN2.f:488). For each strait
it applies a bottom drag to the lowest layer, then a side drag to every layer, and decays the
cross-strait tracer gradients (20-day restoring to zero). Batched over straits (N) and layers
(1-indexed, padded to LMO+1 as in the other ports).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

LMO = 13
BDRAGX = 1.0
SDRAGX = 1e-1
SECONDS_PER_DAY = 86400.0
REDUCE = 1.0 / (SECONDS_PER_DAY * 20.0)


def stbdra_jax(dts, nmst, lmst, must, mmst, wist, dist, gxmst, sxmst):
    """Inputs: must, mmst, gxmst, sxmst (NMST, LMO+1) 1-indexed; lmst, wist, dist (NMST,).
    Returns (must, gxmst, sxmst) after the step."""
    N = nmst
    L = jnp.arange(LMO + 1)[None, :]
    act = (L >= 1) & (L <= lmst[:, None])
    bottom = L == lmst[:, None]
    w = wist[:, None]
    d = dist[:, None]
    # bottom drag on layer LMST(N)
    mb = mmst
    m_new_b = must * mb ** 2 / (mb ** 2 + dts * BDRAGX * jnp.abs(must) * w * d ** 2)
    must = jnp.where(bottom, m_new_b, must)
    # side drag on every layer 1..LMST(N)
    m_new = must * mmst * w / (mmst * w + dts * SDRAGX * jnp.abs(must) * d)
    must = jnp.where(act, m_new, must)
    gxmst = jnp.where(act, gxmst * (1.0 - REDUCE * dts), gxmst)
    sxmst = jnp.where(act, sxmst * (1.0 - REDUCE * dts), sxmst)
    return must, gxmst, sxmst


def stadvt_jax(am, mm1, mm2, mmst, rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2,
               x1, y1, x2, y2, qlimit):
    """Vectorized STADVT (OSTRAITS.f:171-285) over any batch shape. am, mm1, mm2, mmst, rmst, rxst,
    rzst, and the end-1/end-2 moments (rm1, rx1, ry1, rz1; rm2, rx2, ry2, rz2) broadcast together.
    Returns (rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2, dmass) where dmass is the
    OLN increment (FM1 + FM2). qlimit is a Python bool (STADV's S call uses True, G uses False).

    Branch AM >= 0 (flux from box 1 to box 2) and AM < 0 (flux from box 2 to box 1) are both computed
    and merged with where, as in the Fortran GO TO 200 / 300 structure."""
    tiny = jnp.finfo(jnp.float64).tiny
    # ---- AM >= 0 ----
    a1p = am / mm1
    fm1p = a1p * (rm1 + (1 - a1p) * (x1 * rx1 + y1 * ry1))
    fz1p = a1p * rz1
    a2p = am / mmst
    fm2p = a2p * (rmst + (1 - a2p) * rxst)
    fz2p = a2p * rzst
    rx1p = rx1 * (1 - a1p) * (1 - a1p * x1 * x1)
    ry1p = ry1 * (1 - a1p) * (1 - a1p * y1 * y1)
    rxstp = rxst * (1 - 2 * a2p) - fm1p + fm2p
    rx2p = rx2 + x2 * (fm2p - (rm2 - x2 * rx2) * am / mm2)
    ry2p = ry2 + y2 * (fm2p - (rm2 - y2 * ry2) * am / mm2)
    # ---- AM < 0 ----
    a1n = am / mmst
    fm1n = a1n * (rmst - (1 + a1n) * rxst)
    fz1n = a1n * rzst
    a2n = am / mm2
    fm2n = a2n * (rm2 + (1 + a2n) * (x2 * rx2 + y2 * ry2))
    fz2n = a2n * rz2
    rx1n = rx1 - x1 * (fm1n - (rm1 - x1 * rx1) * am / mm1)
    ry1n = ry1 - y1 * (fm1n - (rm1 - y1 * ry1) * am / mm1)
    rxstn = rxst * (1 + 2 * a1n) + fm1n - fm2n
    rx2n = rx2 * (1 + a2n) * (1 + a2n * x2 * x2)
    ry2n = ry2 * (1 + a2n) * (1 + a2n * y2 * y2)
    pos = am >= 0.0
    fm1 = jnp.where(pos, fm1p, fm1n)
    fz1 = jnp.where(pos, fz1p, fz1n)
    fm2 = jnp.where(pos, fm2p, fm2n)
    fz2 = jnp.where(pos, fz2p, fz2n)
    rx1 = jnp.where(pos, rx1p, rx1n)
    ry1 = jnp.where(pos, ry1p, ry1n)
    rxst = jnp.where(pos, rxstp, rxstn)
    rx2 = jnp.where(pos, rx2p, rx2n)
    ry2 = jnp.where(pos, ry2p, ry2n)
    # ---- common: new tracer masses and moments (OCNKPP.f 300-) ----
    rm1 = rm1 - fm1
    rz1 = rz1 - fz1
    rm2 = rm2 + fm2
    rz2 = rz2 + fz2
    rmst = rmst + (fm1 - fm2)
    rzst = rzst + (fz1 - fz2)
    dmass = fm1 + fm2
    if qlimit:
        rxy = jnp.abs(rx1) + jnp.abs(ry1)
        f1 = jnp.where(rxy > rm1, rm1 / (rxy + tiny), 1.0)
        rx1 = rx1 * f1
        ry1 = ry1 * f1
        rz1 = jnp.where(jnp.abs(rz1) > rm1, jnp.sign(rz1) * rm1, rz1)
        rxy = jnp.abs(rx2) + jnp.abs(ry2)
        f2 = jnp.where(rxy > rm2, rm2 / (rxy + tiny), 1.0)
        rx2 = rx2 * f2
        ry2 = ry2 * f2
        rz2 = jnp.where(jnp.abs(rz2) > rm2, jnp.sign(rz2) * rm2, rz2)
        rxst = jnp.where(jnp.abs(rxst) > rmst, jnp.sign(rxst) * rmst, rxst)
        rzst = jnp.where(jnp.abs(rzst) > rmst, jnp.sign(rzst) * rmst, rzst)
    return rmst, rxst, rzst, rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2, dmass


def parse_straits_nml(path):
    """Straits geometry from the OSTRAITS namelist (run directory copy): ij1, ij2 (1-based I,J)
    and xy1, xy2 (in-cell positions). Returns dict of numpy arrays: ist, jst (N,2), xst, yst (N,2)."""
    import re
    import numpy as np
    txt = open(path).read()
    recs = re.findall(r'&strait(.*?)/', txt, flags=re.S)
    ist, jst, xst, yst = [], [], [], []
    for r in recs:
        def grab(key):
            m = re.search(key + r'\s*=\s*([^\s/]+(?:\s*,\s*[^\s/]+)?)', r)
            return [float(v.strip().replace('d', 'e').replace('D', 'e')) for v in m.group(1).split(',')]
        i1, j1 = grab('ij1')
        i2, j2 = grab('ij2')
        x1, y1 = grab('xy1')
        x2, y2 = grab('xy2')
        ist.append([i1, i2]); jst.append([j1, j2])
        xst.append([x1, x2]); yst.append([y1, y2])
    return dict(ist=np.array(ist, dtype=np.int64), jst=np.array(jst, dtype=np.int64),
                xst=np.array(xst), yst=np.array(yst))


def stadv_jax(dts, lmst, mmst, must, dxyp1, dxyp2, xst, yst, moe, g0me, gxme, gyme, gzme,
              s0me, sxme, syme, szme, g0mst, gxmst, gzmst, s0mst, sxmst, szmst):
    """Batched STADV (OSTRAITS.f:67-169) over straits N and layers (N, LMO+1), 1-indexed.
    moe, g0me, ... are (N, 2, LMO+1): end 1 then end 2. dxyp1/dxyp2 (N,) are DXYPO(J1), DXYPO(J2).
    Returns the updated arrays with the same shapes. Neighbour copy (KN2) is not applied (see
    stadv_compare.py: no shared end points in this build)."""
    L = jnp.arange(LMO + 1)[None, :]
    act = (L >= 1) & (L <= lmst[:, None])
    d1 = dxyp1[:, None]
    d2 = dxyp2[:, None]
    x1 = xst[:, 0:1]; x2 = xst[:, 1:2]
    y1 = yst[:, 0:1]; y2 = yst[:, 1:2]
    am = dts * must
    mm1 = moe[:, 0] * d1
    mm2 = moe[:, 1] * d2
    # G pass (qlimit = False)
    g = stadvt_jax(am, mm1, mm2, mmst, g0mst, gxmst, gzmst,
                   g0me[:, 0], gxme[:, 0], gyme[:, 0], gzme[:, 0],
                   g0me[:, 1], gxme[:, 1], gyme[:, 1], gzme[:, 1],
                   x1, y1, x2, y2, False)
    g0mst_n, gxmst_n, gzmst_n = g[0], g[1], g[2]
    rm1, rx1, ry1, rz1, rm2, rx2, ry2, rz2 = g[3:11]
    # S pass (qlimit = True)
    s = stadvt_jax(am, mm1, mm2, mmst, s0mst, sxmst, szmst,
                   s0me[:, 0], sxme[:, 0], syme[:, 0], szme[:, 0],
                   s0me[:, 1], sxme[:, 1], syme[:, 1], szme[:, 1],
                   x1, y1, x2, y2, True)
    s0mst_n, sxmst_n, szmst_n = s[0], s[1], s[2]
    srm1, srx1, sry1, srz1, srm2, srx2, sry2, srz2 = s[3:11]
    # MOE update (OCNKPP-equivalent: OSTRAITS.f:99-101)
    moe1 = moe[:, 0] - am * (1.0 / d1)
    moe2 = moe[:, 1] + am * (1.0 / d2)
    moe_n = jnp.stack([moe1, moe2], axis=1)
    # gradient limits at both ends (OSTRAITS.f:109-120), 8000 J/kg * mass
    def lim(v, mo, dxy):
        cap = 8000.0 * mo * dxy
        return jnp.where(jnp.abs(v) > cap, jnp.sign(v) * cap, v)
    gx1 = lim(rx1, moe1, d1); gy1 = lim(ry1, moe1, d1); gz1 = lim(rz1, moe1, d1)
    gx2 = lim(rx2, moe2, d2); gy2 = lim(ry2, moe2, d2); gz2 = lim(rz2, moe2, d2)
    sel = lambda new, old: jnp.where(act, new, old)  # noqa: E731
    out = dict(
        moe=jnp.stack([sel(moe1, moe[:, 0]), sel(moe2, moe[:, 1])], axis=1),
        g0me=jnp.stack([sel(rm1, g0me[:, 0]), sel(rm2, g0me[:, 1])], axis=1),
        gxme=jnp.stack([sel(gx1, gxme[:, 0]), sel(gx2, gxme[:, 1])], axis=1),
        gyme=jnp.stack([sel(gy1, gyme[:, 0]), sel(gy2, gyme[:, 1])], axis=1),
        gzme=jnp.stack([sel(gz1, gzme[:, 0]), sel(gz2, gzme[:, 1])], axis=1),
        s0me=jnp.stack([sel(srm1, s0me[:, 0]), sel(srm2, s0me[:, 1])], axis=1),
        sxme=jnp.stack([sel(srx1, sxme[:, 0]), sel(srx2, sxme[:, 1])], axis=1),
        syme=jnp.stack([sel(sry1, syme[:, 0]), sel(sry2, syme[:, 1])], axis=1),
        szme=jnp.stack([sel(srz1, szme[:, 0]), sel(srz2, szme[:, 1])], axis=1),
        g0mst=sel(g0mst_n, g0mst), gxmst=sel(gxmst_n, gxmst), gzmst=sel(gzmst_n, gzmst),
        s0mst=sel(s0mst_n, s0mst), sxmst=sel(sxmst_n, sxmst), szmst=sel(szmst_n, szmst),
    )
    return out


def partners(ist, jst):
    """Shared-cell links (KN2): for each (n, k) the list of (m, j) with the same (I, J) grid cell,
    m != n. Returned as a dict {(n,k): [(m,j), ...]} (0-based)."""
    N = ist.shape[0]
    out = {}
    for n in range(N):
        for k in range(2):
            lst = []
            for m in range(N):
                for j in range(2):
                    if (m, j) != (n, k) and ist[m, j] == ist[n, k] and jst[m, j] == jst[n, k]:
                        lst.append((m, j))
            if lst:
                out[(n, k)] = lst
    return out


def stadv_seq(dts, lmst, mmst, must, dxyp1, dxyp2, xst, yst, ist, jst, me, mst):
    """STADV with the shared-cell copy (OSTRAITS.f:134-150). Straits are processed in order; after
    each strait its end values are copied to every partner end (both directions), then the next strait
    runs. Single-strait batches go through stadv_jax. me and mst are dicts of numpy arrays as in
    stadv_compare.py. Returns (me, mst) dicts."""
    import numpy as np
    me = {k: np.array(v, copy=True) for k, v in me.items()}
    mst = {k: np.array(v, copy=True) for k, v in mst.items()}
    links = partners(ist, jst)
    N = lmst.shape[0]
    for n in range(N):
        sl = slice(n, n + 1)
        out = stadv_jax(dts, lmst[sl], mmst[sl], must[sl], dxyp1[sl], dxyp2[sl], xst[sl], yst[sl],
                        me['moe'][sl], me['g0me'][sl], me['gxme'][sl], me['gyme'][sl],
                        me['gzme'][sl], me['s0me'][sl], me['sxme'][sl], me['syme'][sl],
                        me['szme'][sl], mst['g0'][sl], mst['gx'][sl], mst['gz'][sl],
                        mst['s0'][sl], mst['sx'][sl], mst['sz'][sl])
        for k, v in out.items():
            v = np.asarray(v)
            if k in me:
                me[k][sl] = v
            else:
                mst[{'g0mst': 'g0', 'gxmst': 'gx', 'gzmst': 'gz', 's0mst': 's0',
                     'sxmst': 'sx', 'szmst': 'sz'}[k]][sl] = v
        m_lim = int(lmst[n])
        for k in range(2):
            for (m, j) in links.get((n, k), []):
                for name in ['moe', 'g0me', 'gxme', 'gyme', 'gzme', 's0me', 'sxme', 'syme', 'szme']:
                    me[name][m, j, 1:m_lim + 1] = me[name][n, k, 1:m_lim + 1]
    return me, mst


RRT12 = 1.0 / (12.0 ** 0.5)


def stpgf_jax(vgsp, grav, dts, lmst, lmme, oprese, hoceane, moe, g0me, gzme, s0me, szme,
              dxyp, must, distpg, wist):
    """Batched STPGF (OSTRAITS.f:6-65) with VOLGSP from the OFTAB table (eos_jax). Shapes:
    moe, g0me, gzme, s0me, szme (N, 2, LMO+1) 1-indexed; oprese, hoceane (N, 2); lmme (N, 2) int;
    dxyp (N, 2) = DXYPO(J1), DXYPO(J2); must (N, LMO+1); distpg, wist, lmst (N,).
    Returns the updated must (N, LMO+1)."""
    from eos_jax import volgsp
    N = must.shape[0]
    L = jnp.arange(LMO + 1)[None, None, :]                       # (1,1,L)
    lm = lmme[:, :, None]                                         # (N,2,1)
    act = (L >= 1) & (L <= lm)
    moe_m = jnp.where(act, moe, 0.0)
    dpm = grav * moe_m
    # PEND (top-down cumulative), OCNKPP STPGF:33-36
    cum_above = jnp.cumsum(dpm, axis=2) - dpm                     # sum over l < L
    pend = oprese[:, :, None] + cum_above + 0.5 * dpm
    # specific volumes for each layer (OSTRAITS.f:41-50)
    dxy = dxyp[:, :, None]
    safe = jnp.where(moe_m > 0, moe_m * dxy, 1.0)
    gup = (g0me - 2 * RRT12 * gzme) / safe
    gdn = (g0me + 2 * RRT12 * gzme) / safe
    sup = (s0me - 2 * RRT12 * szme) / safe
    sdn = (s0me + 2 * RRT12 * szme) / safe
    dp = dpm
    pup = pend - RRT12 * dp
    pdn = pend + RRT12 * dp
    vup = volgsp(vgsp, gup, sup, pup)
    vdn = volgsp(vgsp, gdn, sdn, pdn)
    # PHI: bottom-up with PHIE accumulating (OSTRAITS.f:53-58)
    phi_term = (vup * (0.5 - RRT12) + vdn * (0.5 + RRT12)) * 0.5 * dp
    e_add = (vdn + vup) * 0.5 * dp
    e_add = jnp.where(act, e_add, 0.0)
    below = jnp.flip(jnp.cumsum(jnp.flip(e_add, axis=2), axis=2), axis=2) - e_add   # sum over l > L
    phi = -hoceane[:, :, None] * grav + below + phi_term
    dh = moe_m * (vup + vdn) * 0.5
    # MUST update (OSTRAITS.f:61-63)
    Lm = jnp.arange(LMO + 1)[None, :]
    use = (Lm >= 1) & (Lm <= lmst[:, None])
    p1, p2 = pend[:, 0], pend[:, 1]
    h1, h2 = dh[:, 0], dh[:, 1]
    f1, f2 = phi[:, 0], phi[:, 1]
    m1, m2 = moe_m[:, 0], moe_m[:, 1]
    upd = must - 0.5 * dts * wist[:, None] * ((h2 + h1) * (p2 - p1) + (f2 - f1) * (m2 + m1)) / distpg[:, None]
    return jnp.where(use, upd, must)
