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
