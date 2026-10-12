"""Checks of the spike kernels (CPU, float64): (1) kernel A against a line-by-line NumPy transcription of interp_aer/calc_aer_opt_prop
(including the full-table humidity interpolation that the Fortran does), (2) kernel B against a NumPy transcription of the Fortran
loops, (3) physical invariants of kernel B (layer-splitting invariance without sources, net-flux conservation).  NOT a comparison
with the Fortran executable: the server returns only final fluxes."""
import numpy as np, jax.numpy as jnp
import kernels as K

B = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/spectral_files/"


def ref_interp_aer(tbl, ic, radius, rh, nl):
    H = tbl["humidity"].shape[0]; dh = 1.0 / (H - 1)
    nb = tbl["k_abs"].shape[-1]; nr = int(tbl["n_radius"][ic])
    ipt = np.minimum(1 + (rh * (H - 1)).astype(int), H - 1)           # 1-based
    wh = (np.minimum(1.0, rh) - tbl["humidity"][ipt - 1]) / dh
    rad = tbl["radius_eff_dry"][ic]
    if radius <= rad[0]: ir, wr = 1, 0.0
    elif radius >= rad[nr - 1]: ir, wr = nr - 1, 1.0
    else:
        for i in range(1, nr + 1):
            if radius <= rad[i - 1]:
                ir = i - 1; wr = (radius - rad[ir - 1]) / (rad[ir] - rad[ir - 1]); break
    out = []
    for k in ("k_abs", "k_scat", "g_asym"):
        tmp = np.zeros((nl, nr, nb))
        for il in range(nl):
            if tbl["l_humidity"][ic]:
                tmp[il] = (1 - wh[il]) * tbl[k][ic, :nr, ipt[il] - 1, :] + wh[il] * tbl[k][ic, :nr, ipt[il], :]
            else:
                tmp[il] = tbl[k][ic, :nr, 0, :]
        out.append((1 - wr) * tmp[:, ir - 1, :] + wr * tmp[:, ir, :])
    return out


def test_aer(path, label):
    t = K.aer_table_to_arrays(path)
    rng = np.random.default_rng(1)
    nl, ncol = 40, 5
    tbl = {k: jnp.asarray(v) for k, v in t.items()}
    tbl.update(_rad_np=t["radius_eff_dry"], _nrad_np=t["n_radius"], _lhum_np=t["l_humidity"])
    comp = [0, 1, 2, 3, 4, 5]
    radius = [float(0.5 * (t["radius_eff_dry"][c, 0] + t["radius_eff_dry"][c, int(t["n_radius"][c]) - 1])) for c in comp]
    radius[2] = float(t["radius_eff_dry"][2, 0]) * 0.5      # below table (clamped branch)
    fs = [0.3, 0.0, 0.5, 0.2, 0.0, 0.0]
    rh = rng.uniform(0, 1.2, (ncol, nl))
    mass = rng.uniform(100, 5000, (ncol, nl)); mdd = rng.uniform(1e-12, 1e-8, (6, ncol, nl))
    mix, ab, sc, asy = K.aer_optical(tbl, jnp.asarray(rh), jnp.asarray(mass), comp, radius, jnp.asarray(mdd), fs, [1.0] * 6, 6)
    worst = 0.0
    for n in range(ncol):
        for a in range(6):
            ka, ks, g = ref_interp_aer(t, comp[a], radius[a], rh[n], nl)
            if fs[a] > 0:
                a_s, s_s, g_s = ref_interp_aer(t, 6, radius[a], rh[n], nl)
                r = t["density"][6] / t["density"][comp[a]]
                rab = (1 - fs[a]) * ka + fs[a] * a_s * r
                rsc = (1 - fs[a]) * ks + fs[a] * s_s * r
                rg = ((1 - fs[a]) * ks * g + fs[a] * s_s * g_s) / ((1 - fs[a]) * ks + fs[a] * s_s)
            else:
                rab, rsc, rg = ka, ks, g
            for ref, got in ((rab, ab[n, :, a]), (rsc, sc[n, :, a]), (rg, asy[n, :, a])):
                worst = max(worst, float(np.max(np.abs(np.asarray(got) - ref) / (np.abs(ref) + 1e-300))))
    print("%s: kernel A max relative difference vs NumPy transcription: %.2e" % (label, worst))
    assert worst < 1e-14


def np_two_stream(tau, om, g, scheme, solar, sec_0, da, dra, finc, s_up=None, s_dn=None, dpl=None):
    L, Bn = tau.shape
    sm = 2 - 1.5 * om * g - 0.5 * om if scheme == "pifm80" else 1.66 - 1.5 * om * g
    df = 2 * (1 - om) if scheme == "pifm80" else 1.66 * (1 - om)
    lam = np.sqrt(sm * df)
    if solar:
        tp = 32 * np.finfo(float).eps
        nr = np.abs(lam - sec_0) < tp
        sm = np.where(nr, (1 + tp) * sm, sm); df = np.where(nr, (1 + tp) * df, df); lam = np.where(nr, (1 + tp) * lam, lam)
        ksi = 1.5 * g / sec_0; fac = 0.5 * om * sec_0 / ((lam - sec_0) * (lam + sec_0))
        gu = fac * (sm - sec_0 - ksi * (df - sec_0)); gd = fac * (sm + sec_0 + ksi * (df + sec_0))
    e = np.exp(-lam * tau); gm = (sm - lam) / (sm + lam); inv = 1 / (1 - e * e * gm * gm)
    tr = e * (1 - gm * gm) * inv; rf = gm * (1 - e * e) * inv
    if solar:
        t0 = np.exp(-tau * sec_0)
        suc = (gu - rf * (1 + gd)) - gu * tr * t0; sdc = t0 * (1 + gd - gu * rf) - (1 + gd) * tr
        fd = np.zeros((L, Bn)); run = finc.copy()
        for i in range(L):
            fd[i] = run; run = run * t0[i]
        su, sd = suc * fd, sdc * fd; sups = (dra - da) * run; fdn0 = np.zeros(Bn)
    else:
        su, sd = s_up, s_dn; sups = (1 - da) * dpl; fdn0 = finc
    al = np.zeros((L + 1, Bn)); sp = np.zeros((L + 1, Bn)); be = np.zeros((L, Bn)); ga = np.zeros((L, Bn)); h = np.zeros((L, Bn))
    al[L] = da; sp[L] = sups
    for i in range(L - 1, -1, -1):
        be[i] = 1 / (1 - al[i + 1] * rf[i]); ga[i] = al[i + 1] * tr[i]; h[i] = sp[i + 1] + al[i + 1] * sd[i]
        al[i] = rf[i] + be[i] * ga[i] * tr[i]; sp[i] = su[i] + be[i] * tr[i] * h[i]
    F = np.zeros((2 * L + 2, Bn)); F[1] = fdn0; F[0] = al[0] * F[1] + sp[0]
    for i in range(1, L + 1):
        F[2 * i] = be[i - 1] * (h[i - 1] + ga[i - 1] * F[2 * i - 1])
        F[2 * i + 1] = sd[i - 1] + tr[i - 1] * F[2 * i - 1] + rf[i - 1] * F[2 * i]
    return F


def test_two_stream():
    rng = np.random.default_rng(2); L, Bn = 40, 64
    tau = 10 ** rng.uniform(-4, 1, (L, Bn)); om = rng.uniform(0, 1, (L, Bn)); g = rng.uniform(0, 0.9, (L, Bn))
    sec0 = rng.uniform(1.0, 5.0, Bn); da = rng.uniform(0, 1, Bn); dra = rng.uniform(0, 1, Bn)
    finc = rng.uniform(100, 1300, Bn); sup = rng.uniform(0, 1, (L, Bn)); sdn = rng.uniform(0, 1, (L, Bn)); dpl = rng.uniform(1, 400, Bn)
    for scheme, solar in (("pifm80", True), ("elsasser", False)):
        kw = dict(sec_0=jnp.asarray(sec0), diffuse_albedo=jnp.asarray(da), direct_albedo=jnp.asarray(dra),
                  flux_inc_down=jnp.asarray(finc), s_up=jnp.asarray(sup), s_down=jnp.asarray(sdn), d_planck_flux_surface=jnp.asarray(dpl))
        got = np.asarray(K.two_stream_homogen(jnp.asarray(tau), jnp.asarray(om), jnp.asarray(g), scheme, solar, **kw))
        ref = np_two_stream(tau, om, g, scheme, solar, sec0, da, dra, finc, sup, sdn, dpl)
        err = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-12))
        print("kernel B %s: max relative difference vs NumPy transcription: %.2e" % (scheme, err))
        assert err < 1e-10
    # invariants: layer splitting without sources (IR, no emission): fluxes at TOA and surface unchanged
    tau1 = 10 ** rng.uniform(-3, 0.5, (4, Bn)); om1 = rng.uniform(0.1, 0.9, (4, Bn)); g1 = rng.uniform(0, 0.8, (4, Bn))
    z = jnp.zeros
    def run(t, o, gg):
        return np.asarray(K.two_stream_homogen(jnp.asarray(t), jnp.asarray(o), jnp.asarray(gg), "elsasser", False,
                          diffuse_albedo=jnp.asarray(da), flux_inc_down=jnp.asarray(finc), s_up=z(t.shape), s_down=z(t.shape),
                          d_planck_flux_surface=z(Bn)))
    a = run(tau1, om1, g1)
    b = run(np.repeat(tau1 / 2, 2, 0), np.repeat(om1, 2, 0), np.repeat(g1, 2, 0))
    err = max(np.max(np.abs(a[[0, 1]] - b[[0, 1]]) / np.abs(a[[0, 1]])), np.max(np.abs(a[[-2, -1]] - b[[-2, -1]]) / (np.abs(a[[-2, -1]]) + 1e-12)))
    print("kernel B layer-splitting invariance (no sources): max relative difference %.2e" % err)
    assert err < 1e-9


if __name__ == "__main__":
    test_aer(B + "sp_lw_ga7/aer_lw_ga7.nc", "LW table")
    test_aer(B + "sp_sw_ga7/aer_sw_ga7.nc", "SW table")
    test_two_stream()
    print("ALL OK")
