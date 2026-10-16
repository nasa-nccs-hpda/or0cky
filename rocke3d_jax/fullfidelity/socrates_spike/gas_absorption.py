"""Gas absorption of radiance_calc for this rundeck (R3): the P/T lookup interpolation (inter_pt_lookup), the k_esft_layer lookup
(with the self-broadening gas-fraction interpolation), the continuum temperature interpolation (inter_t_lookup) and the assembly of
k_abs_layer (scale_absorb branch ip_scale_lookup, the other scaling functions of the band/absorber table, sum_k for the generalised
continua).  Spectral data: spec_data.load(...).   Python loops over bands/absorbers (static), vectorised over (profile, layer).
Fortran: radiance_core/inter_pt_lookup.F90, inter_t_lookup.F90, scale_absorb.F90, sum_k.F90, radiance_calc.F90 (lines ~1070-1520)."""
import numpy as np
import jax.numpy as jnp

EPS = float(np.float32(np.finfo(np.float64).eps))      # REAL, PARAMETER :: eps = EPSILON(1.0_RealK) is a default-REAL parameter
IP_SCALE_LOOKUP = 9        # confirmed by the file header (scale function 9 in the *_ga7_dsa gas blocks, see R2 report)


def _upper_index(table_col, x):
    """MINLOC(x - table, mask x - table >= 0): index (0-based) of the largest table entry <= x; table ascending along axis 0."""
    return jnp.sum(table_col <= x[None], axis=0) - 1


def inter_pt_lookup(p, t, p_lookup, t_lookup):
    """p, t: (n_profile, n_layer); p_lookup: (n_pre,) (log pressure); t_lookup: (n_tmp, n_pre).  Returns 0-based jp, jt, jtt and the four
    interpolation weights."""
    nd_pre = p_lookup.shape[0]
    pmin = p_lookup[0] + max(abs(p_lookup[0]) * EPS, EPS)
    pmax = p_lookup[nd_pre - 1] - max(abs(p_lookup[nd_pre - 1]) * EPS, EPS)
    pl = jnp.minimum(jnp.maximum(jnp.log(p), pmin), pmax)
    jp = _upper_index(jnp.asarray(p_lookup)[:, None, None], pl)
    plj = jnp.asarray(p_lookup)
    fp = (pl - plj[jp]) / (plj[jp + 1] - plj[jp])
    tl = jnp.asarray(t_lookup)
    nd_tmp = tl.shape[0]

    def tidx(jcol):
        col = tl[:, jcol]                                            # (n_tmp, n_profile, n_layer)
        tt = jnp.minimum(jnp.maximum(t, col[0] * (1.0 + EPS)), col[nd_tmp - 1] * (1.0 - EPS))
        j = jnp.sum(col <= tt[None], axis=0) - 1
        lo = jnp.take_along_axis(col, j[None], axis=0)[0]
        hi = jnp.take_along_axis(col, (j + 1)[None], axis=0)[0]
        return j, (tt - lo) / (hi - lo)
    jt, ft = tidx(jp)
    jtt, ftt = tidx(jp + 1)
    compfp = 1.0 - fp
    return dict(jp=jp, jt=jt, jtt=jtt, fac00=compfp * (1.0 - ft), fac10=fp * (1.0 - ftt), fac01=compfp * ft, fac11=fp * ftt)


def k_esft_lookup(ip, k_lookup, i_band, i_gas, n_k):
    """k_lookup: (n_tmp, n_pre, nd_k, nd_species, nd_band) -> (n_profile, n_layer, n_k); i_band, i_gas 1-based."""
    K = jnp.asarray(np.asarray(k_lookup)[:, :, :n_k, i_gas - 1, i_band - 1])
    jp, jt, jtt = ip["jp"], ip["jt"], ip["jtt"]
    g = lambda a, b: K[a, b]                                         # (n_profile, n_layer, n_k)
    v = (ip["fac00"][..., None] * g(jt, jp) + ip["fac10"][..., None] * g(jtt, jp + 1)
         + ip["fac01"][..., None] * g(jt + 1, jp) + ip["fac11"][..., None] * g(jtt + 1, jp + 1))
    return jnp.maximum(0.0, v)


def inter_t_lookup(t, t_lookup_cont):
    """Continuum temperature interpolation: returns 0-based jt_ct and wt_ct."""
    tl = jnp.asarray(t_lookup_cont)
    tt = jnp.minimum(jnp.maximum(t, tl[0] * (1.0 + EPS)), tl[-1] * (1.0 - EPS))
    j = jnp.sum(tl[:, None, None] <= tt[None], axis=0) - 1
    wt = 1.0 - (tt - tl[j]) / (tl[j + 1] - tl[j])
    return j, wt


def scale_absorb_power_quad(p, t, mix, sp, p_ref, t_ref, pressure_offset=0.0):
    """scale_absorb, i_fnc = ip_scale_power_quad: sp = (sp1, sp2, sp3) of the term; returns gas_frac_rescaled (n_profile, n_layer)."""
    p_ref_off_inv = 1.0 / (p_ref + pressure_offset)
    pwk_in = (p + pressure_offset) * p_ref_off_inv
    pwk = jnp.power(pwk_in, sp[0])                      # rtor_v (vector x**y)
    t_inv = 1.0 / t_ref
    tmp = t * t_inv - 1.0
    frac = pwk * (1.0 + tmp * sp[1] + sp[2] * tmp * tmp)
    return jnp.maximum(0.0, frac * mix)


def band_absorption(spec, i_band, p, t, mix, ip=None):
    """k_abs_layer of one band for the k-eqv path: returns (k_abs_layer (n_profile, n_layer, nd_k, n_gas), n_abs_esft (n_gas,),
    w_abs_esft (nd_k, n_gas), index_gas (n_gas,)), plus k_esft_layer of the lookup gases (n_profile, n_layer, nd_k, nd_species).
    spec: dict from spec_data.load; i_band 1-based; mix: (n_profile, n_layer, n_species) gas mixing ratios; no continua / self-broadening
    (asserted: the rundeck's files have none)."""
    assert not np.any(spec["l_self_broadening"] > 0.5) and spec["n_band_cont"][i_band - 1] == 0
    n_gas = int(spec["n_band_absorb"][i_band - 1])
    nd_k = spec["k_lookup"].shape[2]
    n_prof, n_lay = p.shape
    if ip is None:
        ip = inter_pt_lookup(p, t, spec["p_lookup"], spec["t_lookup"])
    kabs = jnp.zeros((n_prof, n_lay, nd_k, n_gas))
    kesft = jnp.zeros((n_prof, n_lay, nd_k, spec["k"].shape[2]))
    nes, wes, idx = [], np.zeros((nd_k, n_gas)), []
    for j in range(1, n_gas + 1):
        g = int(spec["index_absorb"][j - 1, i_band - 1])
        nk = int(spec["i_band_k"][i_band - 1, g - 1])
        assert int(spec["i_scale_k"][i_band - 1, g - 1]) == 2, "only scaling per term is used by the rundeck"
        fn = int(spec["i_scale_fnc"][i_band - 1, g - 1])
        nes.append(nk); idx.append(g)
        wes[:nk, j - 1] = spec["w"][:nk, i_band - 1, g - 1]
        m = mix[:, :, g - 1]
        if fn == IP_SCALE_LOOKUP:
            ke = k_esft_lookup(ip, spec["k_lookup"], i_band, g, nk)
            kesft = kesft.at[:, :, :nk, g - 1].set(ke)
            kabs = kabs.at[:, :, :nk, j - 1].set(jnp.maximum(0.0, ke * m[:, :, None]))
        elif fn == 2:
            for k in range(nk):
                fr = scale_absorb_power_quad(p, t, m, spec["scale"][:3, k, i_band - 1, g - 1], float(spec["p_ref"][g - 1, i_band - 1]),
                                             float(spec["t_ref"][g - 1, i_band - 1]))
                kabs = kabs.at[:, :, k, j - 1].set(spec["k"][k, i_band - 1, g - 1] * fr)
        else:
            raise NotImplementedError("scale function %d (band %d gas %d)" % (fn, i_band, g))
    return kabs, np.array(nes), wes, np.array(idx), kesft


def major_gas_order(spec, i_band, mix_col, mass_col):
    """planet_rad.F90 (lines ~2780-2842): the major absorber of a band is the gas with the smallest column transmission at the layer where
    the product of the gas column transmissions first reaches exp(-1) (or at the surface), found for the LAST profile of the call; it is
    swapped with the first listed gas.  Here one column (mix_col (n_layer, n_species), mass_col (n_layer)); returns the list of gas
    indices (1-based) in band order after the swap.  NOTE: the Fortran does this on the persistent spectrum object, so the list
    carries over from the previous call; starting from the file order each time is exact only for the major gas and the rest of the
    order can differ (affects summation order only).  No continua in the rundeck's files (asserted)."""
    assert spec["n_band_cont"][i_band - 1] == 0
    n = int(spec["n_band_absorb"][i_band - 1])
    gases = [int(spec["index_absorb"][i, i_band - 1]) for i in range(n)]
    colm = np.zeros(n)
    trans = np.ones(n)
    trans_def = np.exp(-1.0)
    for lay in range(mix_col.shape[0]):
        for i, g in enumerate(gases):
            colm[i] = colm[i] + mix_col[lay, g - 1] * mass_col[lay]
            nk = int(spec["i_band_k"][i_band - 1, g - 1])
            w = spec["w"][:nk, i_band - 1, g - 1]
            k = spec["k"][:nk, i_band - 1, g - 1]
            trans[i] = np.sum(w * np.exp(-k * colm[i]))
        if np.prod(trans) <= trans_def:
            break
    nmaj = int(np.argmin(trans))
    gases[0], gases[nmaj] = gases[nmaj], gases[0]
    return gases
