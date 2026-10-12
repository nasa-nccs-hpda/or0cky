"""JAX/NumPy port of SOCRATES solver_triple_hogan (radiance_core/solver_triple_hogan.F90), the layer solver this rundeck uses
(i_solver = ip_solver_triple_hogan; three regions: clear, stratiform, convective; cloud overlap through the 3x3 matrices v (downward
transfer) and u (upward transfer) per layer boundary).

Layout (all 0-based arrays, layer i of the Fortran is column i-1):
  t, r, s_down, s_up (and *_strat, *_conv): (n_profile, n_layer)
  v, u: (n_profile, n_layer+1, 3, 3); element [:, j, a, b] is the Fortran v{a+1}{b+1}(l, j), j = 0..n_layer (id_ct = 1)
  flux_inc_down, source_ground_{free,strat,conv}, albedo_surface_diff: (n_profile,)
  n_cloud_top: static Python int, 1..n_layer (one value per call in the model)
Returns flux_total (n_profile, 2*n_layer+2): flux_total[:, 2*i] downward flux at the top of layer i+1 ... exactly as the Fortran index
(flux_total(l, k) -> column k-1).

The recurrences run over layers with static Python loops (n_layer = 43), vectorised over profiles; this is a validation port, not the
production scan.  Every arithmetic expression keeps the Fortran operand order so a CPU float64 run can be compared bitwise."""
import jax.numpy as jnp


def solver_triple_hogan(n_cloud_top, t, r, s_down, s_up, t_s, r_s, sd_s, su_s, t_c, r_c, sd_c, su_c, v, u, flux_inc_down,
                        sg_free, sg_strat, sg_conv, albedo_surface_diff):
    n_layer = t.shape[1]
    nct = int(n_cloud_top)
    L = lambda a, i: a[:, i - 1]            # Fortran layer index i (1-based) -> column
    one = 1.0
    V = lambda a, b, j: v[:, j, a, b]       # v{a+1}{b+1}(l, j), j = 0..n_layer
    U = lambda a, b, j: u[:, j, a, b]
    alpha1 = {n_layer + 1: albedo_surface_diff}
    alpha2 = {n_layer + 1: albedo_surface_diff}
    alpha3 = {n_layer + 1: albedo_surface_diff}
    g1 = {n_layer + 1: sg_free}
    g2 = {n_layer + 1: sg_strat}
    g3 = {n_layer + 1: sg_conv}
    gamma1, gamma2, gamma3, binv1, binv2, binv3, h1, h2, h3 = ({} for _ in range(9))
    for i in range(n_layer, nct - 1, -1):
        a1, a2, a3 = alpha1[i + 1], alpha2[i + 1], alpha3[i + 1]
        th1 = a1 * V(0, 0, i) + a2 * V(1, 0, i) + a3 * V(2, 0, i)
        th2 = a1 * V(0, 1, i) + a2 * V(1, 1, i) + a3 * V(2, 1, i)
        th3 = a1 * V(0, 2, i) + a2 * V(1, 2, i) + a3 * V(2, 2, i)
        binv1[i] = one / (one - th1 * L(r, i))
        gamma1[i] = th1 * L(t, i)
        h1[i] = g1[i + 1] + th1 * L(s_down, i)
        binv2[i] = one / (one - th2 * L(r_s, i))
        gamma2[i] = th2 * L(t_s, i)
        h2[i] = g2[i + 1] + th2 * L(sd_s, i)
        binv3[i] = one / (one - th3 * L(r_c, i))
        gamma3[i] = th3 * L(t_c, i)
        h3[i] = g3[i + 1] + th3 * L(sd_c, i)
        lam1 = L(s_up, i) + h1[i] * L(t, i) * binv1[i]
        lam2 = L(su_s, i) + h2[i] * L(t_s, i) * binv2[i]
        lam3 = L(su_c, i) + h3[i] * L(t_c, i) * binv3[i]
        alpha1[i] = L(r, i) + th1 * L(t, i) * L(t, i) * binv1[i]
        g1[i] = U(0, 0, i - 1) * lam1 + U(0, 1, i - 1) * lam2 + U(0, 2, i - 1) * lam3
        alpha2[i] = L(r_s, i) + th2 * L(t_s, i) * L(t_s, i) * binv2[i]
        g2[i] = U(1, 0, i - 1) * lam1 + U(1, 1, i - 1) * lam2 + U(1, 2, i - 1) * lam3
        alpha3[i] = L(r_c, i) + th3 * L(t_c, i) * L(t_c, i) * binv3[i]
        g3[i] = U(2, 0, i - 1) * lam1 + U(2, 1, i - 1) * lam2 + U(2, 2, i - 1) * lam3
    if nct > 1:
        i = nct - 1
        if nct < n_layer:
            th1 = alpha1[i + 1] * V(0, 0, i) + alpha2[i + 1] * V(1, 0, i) + alpha3[i + 1] * V(2, 0, i)
        else:
            th1 = alpha1[i + 1]
        binv1[i] = one / (one - th1 * L(r, i))
        gamma1[i] = th1 * L(t, i)
        h1[i] = g1[i + 1] + th1 * L(s_down, i)
        lam = L(t, i) * binv1[i]
        alpha1[i] = L(r, i) + lam * gamma1[i]
        g1[i] = L(s_up, i) + lam * h1[i]
    for i in range(nct - 2, 0, -1):
        binv1[i] = one / (one - alpha1[i + 1] * L(r, i))
        gamma1[i] = alpha1[i + 1] * L(t, i)
        h1[i] = g1[i + 1] + alpha1[i + 1] * L(s_down, i)
        lam1 = L(t, i) * binv1[i]
        alpha1[i] = L(r, i) + lam1 * gamma1[i]
        g1[i] = L(s_up, i) + lam1 * h1[i]
    ft = {2: flux_inc_down}
    if nct > 1:
        ft[1] = alpha1[1] * ft[2] + g1[1]
    else:
        ft[1] = g1[1] + flux_inc_down * (V(0, 0, 0) * alpha1[1] + V(1, 0, 0) * alpha2[1] + V(2, 0, 0) * alpha3[1])
    for i in range(1, nct):
        ft[2 * i + 1] = (gamma1[i] * ft[2 * i] + h1[i]) * binv1[i]
        ft[2 * i + 2] = L(t, i) * ft[2 * i] + L(r, i) * ft[2 * i + 1] + L(s_down, i)
    fd1, fd2, fd3, fu1, fu2, fu3 = ({} for _ in range(6))
    def finish(i):
        fu1[i] = (gamma1[i] * fd1[i] + h1[i]) * binv1[i]
        fu2[i] = (gamma2[i] * fd2[i] + h2[i]) * binv2[i]
        fu3[i] = (gamma3[i] * fd3[i] + h3[i]) * binv3[i]
        fd1[i] = L(t, i) * fd1[i] + L(r, i) * fu1[i] + L(s_down, i)
        fd2[i] = L(t_s, i) * fd2[i] + L(r_s, i) * fu2[i] + L(sd_s, i)
        fd3[i] = L(t_c, i) * fd3[i] + L(r_c, i) * fu3[i] + L(sd_c, i)
    if nct <= n_layer:
        i = nct
        fd1[i] = V(0, 0, i - 1) * ft[2 * i]
        fd2[i] = V(1, 0, i - 1) * ft[2 * i]
        fd3[i] = V(2, 0, i - 1) * ft[2 * i]
        finish(i)
    for i in range(nct + 1, n_layer + 1):
        p1, p2, p3 = fd1[i - 1], fd2[i - 1], fd3[i - 1]
        fd1[i] = V(0, 0, i - 1) * p1 + V(0, 1, i - 1) * p2 + V(0, 2, i - 1) * p3
        fd2[i] = V(1, 0, i - 1) * p1 + V(1, 1, i - 1) * p2 + V(1, 2, i - 1) * p3
        fd3[i] = V(2, 0, i - 1) * p1 + V(2, 1, i - 1) * p2 + V(2, 2, i - 1) * p3
        finish(i)
    for i in range(nct, n_layer + 1):
        ft[2 * i + 1] = fu1[i] + fu2[i] + fu3[i]
        ft[2 * i + 2] = fd1[i] + fd2[i] + fd3[i]
    return jnp.stack([ft[k] for k in range(1, 2 * n_layer + 3)], axis=1)


def solver_hogan_nct1_scan(t, r, s_down, s_up, t_s, r_s, sd_s, su_s, t_c, r_c, sd_c, su_c, v, u, flux_inc_down, sg_free, sg_strat,
                           sg_conv, albedo_surface_diff):
    """solver_triple_hogan for n_cloud_top = 1 with lax.scan over the layers (same arithmetic as the unrolled version above; compiles
    in seconds instead of minutes).  Shapes as solver_triple_hogan."""
    import jax
    from jax import lax
    n_layer = t.shape[1]
    T = lambda a: jnp.moveaxis(a, 1, 0)                          # (n_layer, n_profile)
    # layer i = 1..n_layer: v(i) is v[:, i], u(i-1) is u[:, i-1]
    vi = jnp.moveaxis(v[:, 1:], 1, 0)                           # (nl, N, 3, 3)
    uim1 = jnp.moveaxis(u[:, :n_layer], 1, 0)
    xs = (vi, uim1, T(t), T(r), T(s_down), T(s_up), T(t_s), T(r_s), T(sd_s), T(su_s), T(t_c), T(r_c), T(sd_c), T(su_c))

    def up_step(carry, x):
        a1, a2, a3, g1, g2, g3 = carry
        vv, uu, t1, r1, sd1, su1, t2, r2, sd2, su2, t3, r3, sd3, su3 = x
        th1 = a1 * vv[:, 0, 0] + a2 * vv[:, 1, 0] + a3 * vv[:, 2, 0]
        th2 = a1 * vv[:, 0, 1] + a2 * vv[:, 1, 1] + a3 * vv[:, 2, 1]
        th3 = a1 * vv[:, 0, 2] + a2 * vv[:, 1, 2] + a3 * vv[:, 2, 2]
        b1 = 1.0 / (1.0 - th1 * r1)
        ga1 = th1 * t1
        h1 = g1 + th1 * sd1
        b2 = 1.0 / (1.0 - th2 * r2)
        ga2 = th2 * t2
        h2 = g2 + th2 * sd2
        b3 = 1.0 / (1.0 - th3 * r3)
        ga3 = th3 * t3
        h3 = g3 + th3 * sd3
        l1 = su1 + h1 * t1 * b1
        l2 = su2 + h2 * t2 * b2
        l3 = su3 + h3 * t3 * b3
        na1 = r1 + th1 * t1 * t1 * b1
        ng1 = uu[:, 0, 0] * l1 + uu[:, 0, 1] * l2 + uu[:, 0, 2] * l3
        na2 = r2 + th2 * t2 * t2 * b2
        ng2 = uu[:, 1, 0] * l1 + uu[:, 1, 1] * l2 + uu[:, 1, 2] * l3
        na3 = r3 + th3 * t3 * t3 * b3
        ng3 = uu[:, 2, 0] * l1 + uu[:, 2, 1] * l2 + uu[:, 2, 2] * l3
        return (na1, na2, na3, ng1, ng2, ng3), (b1, b2, b3, ga1, ga2, ga3, h1, h2, h3)

    a0 = albedo_surface_diff
    init = (a0, a0, a0, sg_free, sg_strat, sg_conv)
    (alpha1, alpha2, alpha3, g1, g2, g3), (B1, B2, B3, GA1, GA2, GA3, H1, H2, H3) = lax.scan(up_step, init, xs, reverse=True)
    ft2 = flux_inc_down
    ft1 = g1 + flux_inc_down * (v[:, 0, 0, 0] * alpha1 + v[:, 0, 1, 0] * alpha2 + v[:, 0, 2, 0] * alpha3)
    vdn = jnp.moveaxis(v[:, :n_layer], 1, 0)                   # v(i-1) for i = 1..nl
    xs2 = (vdn, B1, B2, B3, GA1, GA2, GA3, H1, H2, H3, T(t), T(r), T(s_down), T(t_s), T(r_s), T(sd_s), T(t_c), T(r_c), T(sd_c))

    def down_step(carry, x):
        f1, f2, f3 = carry
        vv, b1, b2, b3, ga1, ga2, ga3, h1, h2, h3, t1, r1, sd1, t2, r2, sd2, t3, r3, sd3 = x
        d1 = vv[:, 0, 0] * f1 + vv[:, 0, 1] * f2 + vv[:, 0, 2] * f3
        d2 = vv[:, 1, 0] * f1 + vv[:, 1, 1] * f2 + vv[:, 1, 2] * f3
        d3 = vv[:, 2, 0] * f1 + vv[:, 2, 1] * f2 + vv[:, 2, 2] * f3
        u1 = (ga1 * d1 + h1) * b1
        u2 = (ga2 * d2 + h2) * b2
        u3 = (ga3 * d3 + h3) * b3
        n1 = t1 * d1 + r1 * u1 + sd1
        n2 = t2 * d2 + r2 * u2 + sd2
        n3 = t3 * d3 + r3 * u3 + sd3
        return (n1, n2, n3), (u1 + u2 + u3, n1 + n2 + n3)

    z = jnp.zeros_like(ft2)
    _, (UP, DN) = lax.scan(down_step, (ft2, z, z), xs2)         # (nl, N)
    cols = [ft1, ft2]
    out = jnp.concatenate([ft1[:, None], ft2[:, None], jnp.stack([UP, DN], axis=2).transpose(1, 0, 2).reshape(ft1.shape[0], 2 * n_layer)],
                          axis=1)
    return out
