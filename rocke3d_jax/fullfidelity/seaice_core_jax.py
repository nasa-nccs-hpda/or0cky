"""JAX-vectorized (batched-array) port of seaice_core_ff.py -- Track B.

Same physics as seaice_core_ff.py (validated bitwise/near-bitwise against real Fortran dumps,
FULL_FIDELITY_DELTAS.md D10/D12); this module operates on whole arrays of cells at once (shape
(N,) for scalars, (N, LMI) for the 4 thermal layers, (N, 2) for the 2 snow/near-surface layers)
instead of Python loops over one cell at a time, using jnp.where in place of Python if/else so
every branch is data-dependent per-cell rather than a Python-level control-flow choice.

Scope: ported here are tfrez/Ti/Ti2b/Ei/dEidTi/Mi/Em/alami, solar_ice_frac_full,
get_snow_ice_layer/set_snow_ice_layer, relayer/relayer_12, tice, sea_ice, ssidec, snowice, and
simelt -- i.e. the whole per-DTsrc-step SEA_ICE/SSIDEC/snowice hot path plus SIMELT.
**ADDICE is intentionally NOT vectorized here** -- it composes 4 sequential decision blocks whose
leaves are themselves nested branches (roughly 15 mutually-exclusive paths overall, several of
which are lead-fraction rebalancing corrections only lightly exercised by the available 3-date
real-Fortran record, see FULL_FIDELITY_DELTAS.md D12). Converting it with the same jnp.where
technique used below is mechanically possible but the bug surface is large relative to how well
validated each rare branch could be made; it stays plain Python (seaice_core_ff.addice) rather
than risk a vectorized version whose rare paths look right but are not actually exercised.

Every branch below was hand-derived from seaice_core_ff.py's Python control flow by writing out
each mutually-exclusive leaf's closed-form result in terms of the ORIGINAL (pre-branch) inputs --
relayer_12 in particular has 9 leaf branches (see its docstring). Divisions that are only valid in
one branch go through `_safe_div`/an inline `jnp.where(x==0,1,x)` guard so the unused branch's
algebra never divides by a literal zero; jnp.where's per-element selection discards the unused
branch's value regardless (no gradient support is needed or provided here -- forward-only).
"""
import jax.numpy as jnp

LHM = 3.34e5
SHW = 4185.0
SHI = 2060.0
BYSHI = 1.0 / SHI
BYLHM = 1.0 / LHM
MU = 0.054
RHOI = 916.6
RHOW = 1000.0
RHOWS = 1030.0
RHOS = 300.0
LMI = 4
XSI = (0.5, 0.5, 0.5, 0.5)
ACE1I = 0.1 * RHOI
AC2OIM = 0.1 * RHOI
ALPHA = 1.0
SSI0 = 0.0032
FSSS = 8.0 / 35.0
SSIMIN = 1e-6
SECONDS_PER_DAY = 86400.0
SILMFAC = 1.0e-7
SILMPOW = 1.36


def _safe_div(a, b):
    return a / jnp.where(b == 0.0, 1.0, b)


def tfrez(sss):
    return (-0.0575 + (-2.154996e-4) * sss) * sss + 1.710523e-3 * sss * jnp.sqrt(jnp.maximum(sss, 0.0))


def Ti(Eit, Si):
    tm = -MU * Si
    lo = jnp.where(jnp.abs(Eit + LHM) < 1e-10, 0.0, (Eit + LHM) * BYSHI)
    b = tm * (SHW - SHI) - (Eit + LHM)
    c = LHM * tm
    det = b * b - 4.0 * SHI * c
    sq = jnp.sqrt(jnp.maximum(det, 0.0))
    hi_quad = -0.5 * (b + sq) * BYSHI
    hi = jnp.where(Eit >= SHW * tm, tm, hi_quad)
    return jnp.where(Si > 1e-10, hi, lo)


def Ti2b(Eit, Si, snowl, mice):
    tm = -MU * Si
    frac = _safe_div(mice, mice + snowl)
    b = frac * tm * (SHW - SHI) - (Eit + LHM)
    c = frac * LHM * tm
    det = b * b - 4.0 * SHI * c
    sq = jnp.sqrt(jnp.maximum(det, 0.0))
    hi = -0.5 * (b + sq) * BYSHI
    lo = jnp.where(jnp.abs(Eit + LHM) < 1e-10, 0.0, (Eit + LHM) * BYSHI)
    return jnp.where(Si > 1e-10, hi, lo)


def Ei(t, si):
    t_safe = jnp.where(t == 0.0, 1.0, t)
    hi = (t + MU * si) * SHI - LHM * (1.0 + MU * si / t_safe) - SHW * MU * si
    lo = t * SHI - LHM
    return jnp.where((si > 0.0) & (t != 0.0), hi, lo)


def dEidTi(t, si):
    t_safe = jnp.where(t == 0.0, 1.0, t)
    hi = SHI + LHM * MU * si / (t_safe * t_safe)
    return jnp.where(si < 1e-10, SHI, jnp.where(t == 0.0, SHI, hi))


def Mi(hsi, ssi, msi):
    msi_safe = jnp.where(msi == 0.0, 1.0, msi)
    frac = 1e3 * ssi / msi_safe
    val = hsi + SHW * MU * 1e3 * ssi
    inner = jnp.where((val > 0.0) | (jnp.abs(val) < 1e-12), msi, 0.0)
    outer_lo = jnp.maximum(0.0, msi + hsi * BYLHM)
    return jnp.where(frac > 1e-10, inner, outer_lo)


def Em(si):
    return -MU * si * SHW


def alami(t, si):
    alami0, alamdt, alamds = 2.11, -0.011, 0.09
    t_safe = jnp.where(t == 0.0, 1.0, t)
    a = alami0 + alamdt * t + alamds * si / t_safe
    hi = jnp.where(a > 0.0, a, alami0)
    mid = jnp.where(t != 0.0, hi, alami0)
    return jnp.where(si < 1e-10, alami0, mid)


def solar_ice_frac_full(snow, msi2, wetsnow):
    kiextvis, kiextnir1 = 1.5, 18.0
    dsnow = snow / RHOS
    hice12 = ACE1I / RHOI
    deep = dsnow > 0.02
    wet_consts = (0.20, 0.33, 10.7, 118.0)
    dry_consts = (0.06, 0.31, 19.6, 196.0)
    shallow_consts = (0.24, 0.43, 10.7, 118.0)
    wetsnow_b = jnp.asarray(wetsnow, dtype=bool)
    deep_consts = [jnp.where(wetsnow_b, w, d) for w, d in zip(wet_consts, dry_consts)]
    fracvis = jnp.where(deep, deep_consts[0], shallow_consts[0])
    fracnir1 = jnp.where(deep, deep_consts[1], shallow_consts[1])
    ksextvis = jnp.where(deep, deep_consts[2], shallow_consts[2])
    ksextnir1 = jnp.where(deep, deep_consts[3], shallow_consts[3])

    cond1 = ACE1I * XSI[0] > snow * XSI[1]
    hice1 = (ACE1I - XSI[1] * (snow + ACE1I)) / RHOI
    fv1_a = jnp.exp(-ksextvis * dsnow - kiextvis * hice1)
    fn1_a = jnp.exp(-ksextnir1 * dsnow - kiextnir1 * hice1)
    dsnow1 = (ACE1I + snow) * XSI[0] / RHOS
    fv1_b = jnp.exp(-ksextvis * dsnow1)
    fn1_b = jnp.exp(-ksextnir1 * dsnow1)
    fv1 = jnp.where(cond1, fv1_a, fv1_b)
    fn1 = jnp.where(cond1, fn1_a, fn1_b)
    fsri0 = fracvis * fv1 + fracnir1 * fn1

    fv2 = jnp.exp(-ksextvis * dsnow - kiextvis * hice12)
    fn2 = jnp.exp(-ksextnir1 * dsnow - kiextnir1 * hice12)
    fsri1 = fracvis * fv2 + fracnir1 * fn2

    fvp, fnp = fv2, fn2
    rest = []
    for l in range(2, LMI):
        hicel = XSI[l] * msi2 / RHOI
        fvp = jnp.exp(-kiextvis * hicel) * fvp
        fnp = jnp.exp(-kiextnir1 * hicel) * fnp
        rest.append(fracvis * fvp + fracnir1 * fnp)
    return jnp.stack([fsri0, fsri1] + rest, axis=-1)


def get_snow_ice_layer(snow, msi2, hsil, ssil, needtemp):
    msi1 = snow + ACE1I
    msi1_safe = jnp.where(msi1 == 0.0, 1.0, msi1)
    condP = ACE1I > XSI[1] * msi1

    mice0_P = ACE1I - XSI[1] * msi1
    mice1_P = XSI[1] * msi1
    snowl0_P = snow
    snowl1_P = jnp.zeros_like(snow)
    si1_P = 1e3 * _safe_div(ssil[..., 0], mice0_P)
    ti1_P_raw = Ti2b(hsil[..., 0] / (XSI[0] * msi1_safe), si1_P, snowl0_P, mice0_P)
    ti1_P = jnp.where((ti1_P_raw > -1e-15) & (ti1_P_raw < 0.0), 0.0, ti1_P_raw)
    hice0_P = jnp.minimum(jnp.maximum(mice0_P * Ei(ti1_P, si1_P), hsil[..., 0]), mice0_P * Em(si1_P))
    hsnow0_P = jnp.where((snowl0_P == 0.0) | (jnp.abs(hsil[..., 0] - hice0_P) < 1e-8),
                          0.0, hsil[..., 0] - hice0_P)
    cond_extra_P = (ti1_P < 0.0) & (snowl0_P > 0.0) & (hice0_P != mice0_P * Em(si1_P))
    hsnow0_P = jnp.where(cond_extra_P, jnp.minimum(hsnow0_P, (ti1_P * SHI - LHM) * snowl0_P), hsnow0_P)
    hsnow1_P = jnp.zeros_like(snow)
    hice1_P = hsil[..., 1]
    sice0_P = ssil[..., 0]
    sice1_P = ssil[..., 1]

    mice0_Q = jnp.zeros_like(snow)
    mice1_Q = ACE1I * jnp.ones_like(snow)
    snowl0_Q = XSI[0] * msi1
    snowl1_Q = XSI[1] * msi1 - ACE1I
    hsnow0_Q = hsil[..., 0]
    si1_Q = 1e3 * _safe_div(ssil[..., 1], mice1_Q)
    ti1_Q_raw = Ti2b(hsil[..., 1] / (XSI[1] * msi1_safe), si1_Q, snowl1_Q, mice1_Q)
    ti1_Q = jnp.where((ti1_Q_raw > -1e-15) & (ti1_Q_raw < 0.0), 0.0, ti1_Q_raw)
    hice0_Q = jnp.zeros_like(snow)
    hice1_Q = jnp.minimum(jnp.maximum(mice1_Q * Ei(ti1_Q, si1_Q), hsil[..., 1]), mice1_Q * Em(si1_Q))
    hsnow1_Q = jnp.where((snowl1_Q == 0.0) | (jnp.abs(hsil[..., 1] - hice1_Q) < 1e-8),
                          0.0, hsil[..., 1] - hice1_Q)
    cond_extra_Q = (ti1_Q < 0.0) & (snowl1_Q > 0.0) & (hice1_Q != mice1_Q * Em(si1_Q))
    hsnow1_Q = jnp.where(cond_extra_Q, jnp.minimum(hsnow1_Q, (ti1_Q * SHI - LHM) * snowl1_Q), hsnow1_Q)
    sice0_Q = jnp.zeros_like(snow)
    sice1_Q = ssil[..., 1]

    mice0 = jnp.where(condP, mice0_P, mice0_Q)
    mice1 = jnp.where(condP, mice1_P, mice1_Q)
    snowl0 = jnp.where(condP, snowl0_P, snowl0_Q)
    snowl1 = jnp.where(condP, snowl1_P, snowl1_Q)
    hsnow0 = jnp.where(condP, hsnow0_P, hsnow0_Q)
    hsnow1 = jnp.where(condP, hsnow1_P, hsnow1_Q)
    hice0 = jnp.where(condP, hice0_P, hice0_Q)
    hice1 = jnp.where(condP, hice1_P, hice1_Q)
    sice0 = jnp.where(condP, sice0_P, sice0_Q)
    sice1 = jnp.where(condP, sice1_P, sice1_Q)

    mice2 = XSI[2] * msi2
    mice3 = XSI[3] * msi2
    hice2 = hsil[..., 2]
    hice3 = hsil[..., 3]
    sice2 = ssil[..., 2]
    sice3 = ssil[..., 3]

    mice = jnp.stack([mice0, mice1, mice2, mice3], axis=-1)
    hice = jnp.stack([hice0, hice1, hice2, hice3], axis=-1)
    sice = jnp.stack([sice0, sice1, sice2, sice3], axis=-1)
    snowl = jnp.stack([snowl0, snowl1], axis=-1)
    hsnow = jnp.stack([hsnow0, hsnow1], axis=-1)

    if not needtemp:
        return snowl, hsnow, hice, sice, None, None, mice

    tsnw0 = jnp.where(snowl0 > 0.0, Ti(_safe_div(hsnow0, snowl0), jnp.zeros_like(snowl0)), 0.0)
    tsnw1 = jnp.where(snowl1 > 0.0, Ti(_safe_div(hsnow1, snowl1), jnp.zeros_like(snowl1)), 0.0)
    tsnw = jnp.stack([tsnw0, tsnw1], axis=-1)

    def _tsil_l(hice_l, sice_l, mice_l):
        val = Ti(_safe_div(hice_l, mice_l), 1e3 * _safe_div(sice_l, mice_l))
        return jnp.where(mice_l > 0.0, val, 0.0)

    tsil = jnp.stack([_tsil_l(hice0, sice0, mice0), _tsil_l(hice1, sice1, mice1),
                      _tsil_l(hice2, sice2, mice2), _tsil_l(hice3, sice3, mice3)], axis=-1)
    return snowl, hsnow, hice, sice, tsnw, tsil, mice


def set_snow_ice_layer(hsnow, hice, sice, mice, snowl):
    snow = snowl[..., 0] + snowl[..., 1]
    msi1 = snow + ACE1I
    msi2 = jnp.sum(mice[..., 2:LMI], axis=-1)
    hsil0 = hsnow[..., 0] + hice[..., 0]
    hsil1 = hsnow[..., 1] + hice[..., 1]
    hsil = jnp.concatenate([hsil0[..., None], hsil1[..., None], hice[..., 2:LMI]], axis=-1)
    ssil = sice
    return snow, msi1, msi2, hsil, ssil


def relayer(fmsi2_in, mice, hice, sice):
    """SEAICE.f relayer. fmsi2_in: (...,) scalar-per-cell; mice/hice/sice: (..., LMI)."""
    m0, m1, m2, m3 = mice[..., 0], mice[..., 1], mice[..., 2], mice[..., 3]
    h0, h1, h2, h3 = hice[..., 0], hice[..., 1], hice[..., 2], hice[..., 3]
    s0, s1, s2, s3 = sice[..., 0], sice[..., 1], sice[..., 2], sice[..., 3]
    msi2 = m2 + m3

    def _layer(f_val, m_l, m_lm1, m_lp1, h_l, h_lm1, h_lp1, s_l, s_lm1, s_lp1):
        cond_pos = f_val > 0.0
        cond_gt = f_val > m_l
        fh_a = h_l + (f_val - m_l) * _safe_div(h_lm1, m_lm1)
        fs_a = s_l + (f_val - m_l) * _safe_div(s_lm1, m_lm1)
        fh_b = h_l * _safe_div(f_val, m_l)
        fs_b = s_l * _safe_div(f_val, m_l)
        fh_pos = jnp.where(cond_gt, fh_a, fh_b)
        fs_pos = jnp.where(cond_gt, fs_a, fs_b)
        fh_neg = h_lp1 * _safe_div(f_val, m_lp1)
        fs_neg = s_lp1 * _safe_div(f_val, m_lp1)
        return jnp.where(cond_pos, fh_pos, fh_neg), jnp.where(cond_pos, fs_pos, fs_neg)

    f1 = (XSI[2] + XSI[3]) * (msi2 + fmsi2_in) - (m2 + m3)
    fh1, fs1 = _layer(f1, m1, m0, m2, h1, h0, h2, s1, s0, s2)
    f2 = XSI[3] * (msi2 + fmsi2_in) - m3
    fh2, fs2 = _layer(f2, m2, m1, m3, h2, h1, h3, s2, s1, s3)

    mice_n = jnp.stack([m0, m1 - f1, m2 + f1 - f2, m3 + f2], axis=-1)
    hice_n = jnp.stack([h0, h1 - fh1, h2 + fh1 - fh2, h3 + fh2], axis=-1)
    sice_n = jnp.stack([s0, s1 - fs1, s2 + fs1 - fs2, s3 + fs2], axis=-1)
    return mice_n, hice_n, sice_n


def relayer_12(hsnow, hice, sice, mice, snowl):
    """SEAICE.f relayer_12 (no tracers). Only layers 0,1 of hice/sice/mice and both snow layers
    change; layers 2,3 pass through. Has 9 mutually-exclusive leaf branches (A1,A2 / B1,B2 /
    C1,C2,C3,C4,C5a,C5b in the naming below) -- each hand-derived from the original sequential
    Python mutations as a closed form in the pre-branch inputs, then selected with nested
    jnp.where matching the original if/elif/else precedence (A, then B, then C)."""
    hs0, hs1 = hsnow[..., 0], hsnow[..., 1]
    hi0, hi1, hi2, hi3 = hice[..., 0], hice[..., 1], hice[..., 2], hice[..., 3]
    si0, si1, si2, si3 = sice[..., 0], sice[..., 1], sice[..., 2], sice[..., 3]
    mi0, mi1, mi2, mi3 = mice[..., 0], mice[..., 1], mice[..., 2], mice[..., 3]
    sl0, sl1 = snowl[..., 0], snowl[..., 1]

    fmsi1 = sl0 + mi0 - XSI[0] * (sl0 + sl1 + ACE1I)
    fmsi1 = jnp.where(jnp.abs(fmsi1) < 1e-14, 0.0, fmsi1)
    fmsi0 = XSI[0] * (sl0 + sl1 + ACE1I)

    cond_A = (sl1 > 0.0) & (mi0 + mi1 > fmsi0)
    cond_B = (mi0 > 0.0) & (mi0 + mi1 < fmsi0)

    # --- A ---
    cond_A1 = fmsi0 > mi1
    fssi1_A1 = (fmsi0 - mi1) * _safe_div(si0, mi0)
    fhsi1_A1 = (fmsi0 - mi1) * _safe_div(hi0, mi0)
    fssi1_A2 = (mi1 - fmsi0) * _safe_div(si1, mi1)
    fhsi1_A2 = (mi1 - fmsi0) * _safe_div(hi1, mi1)
    sice0_A = jnp.where(cond_A1, si0 - fssi1_A1, si0 + fssi1_A2)
    sice1_A = jnp.where(cond_A1, si1 + fssi1_A1, si1 - fssi1_A2)
    hice0_A = jnp.where(cond_A1, hi0 - fhsi1_A1, hi0 + fhsi1_A2)
    hice1_A = jnp.where(cond_A1, hi1 + fhsi1_A1, hi1 - fhsi1_A2)
    mice0_A, mice1_A = mi0 + mi1 - fmsi0, fmsi0
    snowl0_A, snowl1_A = sl0 + sl1, jnp.zeros_like(sl1)
    hsnow0_A, hsnow1_A = hs0 + hs1, jnp.zeros_like(hs1)

    # --- B ---
    cond_B1 = fmsi0 < sl0
    fhsi1_B1 = (sl0 - fmsi0) * _safe_div(hs0, sl0)
    fhsi1_B2 = (fmsi0 - sl0) * _safe_div(hs1, sl1)
    hsnow0_B = jnp.where(cond_B1, hs0 - fhsi1_B1, hs0 + fhsi1_B2)
    hsnow1_B = jnp.where(cond_B1, hs1 + fhsi1_B1, hs1 - fhsi1_B2)
    snowl1_B, snowl0_B = sl0 + sl1 - fmsi0, fmsi0
    hice1_B, hice0_B = hi1 + hi0, jnp.zeros_like(hi0)
    sice1_B, sice0_B = si1 + si0, jnp.zeros_like(si0)
    mice1_B, mice0_B = mi1 + mi0, jnp.zeros_like(mi0)

    # --- C: fmsi1 > 0, mi0 > 0 ---
    cond_Cpos = fmsi1 > 0.0
    cond_mi0pos = mi0 > 0.0
    cond_C1 = fmsi1 > mi0
    d_C1 = fmsi1 - mi0
    hsnow1_C1 = hs1 + d_C1 * _safe_div(hs0, sl0)
    hsnow0_C1 = hs0 - d_C1 * _safe_div(hs0, sl0)
    snowl1_C1, snowl0_C1 = sl1 + d_C1, sl0 - d_C1
    hice1_C1, hice0_C1 = hi1 + hi0, jnp.zeros_like(hi0)
    sice1_C1, sice0_C1 = si1 + si0, jnp.zeros_like(si0)
    mice1_C1, mice0_C1 = mi1 + mi0, jnp.zeros_like(mi0)

    fhsi1_C2 = fmsi1 * _safe_div(hi0, mi0)
    fssi1_C2 = fmsi1 * _safe_div(si0, mi0)
    mice0_C2, mice1_C2 = mi0 - fmsi1, mi1 + fmsi1
    hice0_C2, hice1_C2 = hi0 - fhsi1_C2, hi1 + fhsi1_C2
    sice0_C2, sice1_C2 = si0 - fssi1_C2, si1 + fssi1_C2
    snowl0_C2, snowl1_C2, hsnow0_C2, hsnow1_C2 = sl0, sl1, hs0, hs1

    hsnow0_Cmp = jnp.where(cond_C1, hsnow0_C1, hsnow0_C2)
    hsnow1_Cmp = jnp.where(cond_C1, hsnow1_C1, hsnow1_C2)
    snowl0_Cmp = jnp.where(cond_C1, snowl0_C1, snowl0_C2)
    snowl1_Cmp = jnp.where(cond_C1, snowl1_C1, snowl1_C2)
    hice0_Cmp = jnp.where(cond_C1, hice0_C1, hice0_C2)
    hice1_Cmp = jnp.where(cond_C1, hice1_C1, hice1_C2)
    sice0_Cmp = jnp.where(cond_C1, sice0_C1, sice0_C2)
    sice1_Cmp = jnp.where(cond_C1, sice1_C1, sice1_C2)
    mice0_Cmp = jnp.where(cond_C1, mice0_C1, mice0_C2)
    mice1_Cmp = jnp.where(cond_C1, mice1_C1, mice1_C2)

    # --- C3: fmsi1 > 0, mi0 <= 0 ---
    fhsi1_C3 = fmsi1 * _safe_div(hs0, sl0)
    snowl0_C3, snowl1_C3 = sl0 - fmsi1, sl1 + fmsi1
    hsnow0_C3, hsnow1_C3 = hs0 - fhsi1_C3, hs1 + fhsi1_C3
    mice0_C3, mice1_C3, hice0_C3, hice1_C3, sice0_C3, sice1_C3 = mi0, mi1, hi0, hi1, si0, si1

    hsnow0_Cpos = jnp.where(cond_mi0pos, hsnow0_Cmp, hsnow0_C3)
    hsnow1_Cpos = jnp.where(cond_mi0pos, hsnow1_Cmp, hsnow1_C3)
    snowl0_Cpos = jnp.where(cond_mi0pos, snowl0_Cmp, snowl0_C3)
    snowl1_Cpos = jnp.where(cond_mi0pos, snowl1_Cmp, snowl1_C3)
    hice0_Cpos = jnp.where(cond_mi0pos, hice0_Cmp, hice0_C3)
    hice1_Cpos = jnp.where(cond_mi0pos, hice1_Cmp, hice1_C3)
    sice0_Cpos = jnp.where(cond_mi0pos, sice0_Cmp, sice0_C3)
    sice1_Cpos = jnp.where(cond_mi0pos, sice1_Cmp, sice1_C3)
    mice0_Cpos = jnp.where(cond_mi0pos, mice0_Cmp, mice0_C3)
    mice1_Cpos = jnp.where(cond_mi0pos, mice1_Cmp, mice1_C3)

    # --- C, fmsi1 <= 0 ---
    cond_mi0pos2 = mi0 > 0.0
    fhsi1_C4 = fmsi1 * _safe_div(hi1, mi1)
    fssi1_C4 = fmsi1 * _safe_div(si1, mi1)
    mice0_C4, mice1_C4 = mi0 - fmsi1, mi1 + fmsi1
    hice0_C4, hice1_C4 = hi0 - fhsi1_C4, hi1 + fhsi1_C4
    sice0_C4, sice1_C4 = si0 - fssi1_C4, si1 + fssi1_C4
    snowl0_C4, snowl1_C4, hsnow0_C4, hsnow1_C4 = sl0, sl1, hs0, hs1

    cond_C5a = (sl1 + fmsi1) < 0.0
    hice0_C5a = -(sl1 + fmsi1) * _safe_div(hi1, mi1)
    hice1_C5a = hi1 - hice0_C5a
    sice0_C5a = -(sl1 + fmsi1) * _safe_div(si1, mi1)
    sice1_C5a = si1 - sice0_C5a
    mice0_C5a = -(sl1 + fmsi1)
    mice1_C5a = mi1 - mice0_C5a
    hsnow0_C5a, hsnow1_C5a = hs0 + hs1, jnp.zeros_like(hs1)
    snowl0_C5a, snowl1_C5a = sl0 + sl1, jnp.zeros_like(sl1)

    fhsi1_C5b = fmsi1 * _safe_div(hs1, sl1)
    snowl0_C5b, snowl1_C5b = sl0 - fmsi1, sl1 + fmsi1
    hsnow0_C5b, hsnow1_C5b = hs0 - fhsi1_C5b, hs1 + fhsi1_C5b
    mice0_C5b, mice1_C5b, hice0_C5b, hice1_C5b, sice0_C5b, sice1_C5b = mi0, mi1, hi0, hi1, si0, si1

    hice0_C5 = jnp.where(cond_C5a, hice0_C5a, hice0_C5b)
    hice1_C5 = jnp.where(cond_C5a, hice1_C5a, hice1_C5b)
    sice0_C5 = jnp.where(cond_C5a, sice0_C5a, sice0_C5b)
    sice1_C5 = jnp.where(cond_C5a, sice1_C5a, sice1_C5b)
    mice0_C5 = jnp.where(cond_C5a, mice0_C5a, mice0_C5b)
    mice1_C5 = jnp.where(cond_C5a, mice1_C5a, mice1_C5b)
    hsnow0_C5 = jnp.where(cond_C5a, hsnow0_C5a, hsnow0_C5b)
    hsnow1_C5 = jnp.where(cond_C5a, hsnow1_C5a, hsnow1_C5b)
    snowl0_C5 = jnp.where(cond_C5a, snowl0_C5a, snowl0_C5b)
    snowl1_C5 = jnp.where(cond_C5a, snowl1_C5a, snowl1_C5b)

    hice0_Cneg = jnp.where(cond_mi0pos2, hice0_C4, hice0_C5)
    hice1_Cneg = jnp.where(cond_mi0pos2, hice1_C4, hice1_C5)
    sice0_Cneg = jnp.where(cond_mi0pos2, sice0_C4, sice0_C5)
    sice1_Cneg = jnp.where(cond_mi0pos2, sice1_C4, sice1_C5)
    mice0_Cneg = jnp.where(cond_mi0pos2, mice0_C4, mice0_C5)
    mice1_Cneg = jnp.where(cond_mi0pos2, mice1_C4, mice1_C5)
    hsnow0_Cneg = jnp.where(cond_mi0pos2, hsnow0_C4, hsnow0_C5)
    hsnow1_Cneg = jnp.where(cond_mi0pos2, hsnow1_C4, hsnow1_C5)
    snowl0_Cneg = jnp.where(cond_mi0pos2, snowl0_C4, snowl0_C5)
    snowl1_Cneg = jnp.where(cond_mi0pos2, snowl1_C4, snowl1_C5)

    hice0_C = jnp.where(cond_Cpos, hice0_Cpos, hice0_Cneg)
    hice1_C = jnp.where(cond_Cpos, hice1_Cpos, hice1_Cneg)
    sice0_C = jnp.where(cond_Cpos, sice0_Cpos, sice0_Cneg)
    sice1_C = jnp.where(cond_Cpos, sice1_Cpos, sice1_Cneg)
    mice0_C = jnp.where(cond_Cpos, mice0_Cpos, mice0_Cneg)
    mice1_C = jnp.where(cond_Cpos, mice1_Cpos, mice1_Cneg)
    hsnow0_C = jnp.where(cond_Cpos, hsnow0_Cpos, hsnow0_Cneg)
    hsnow1_C = jnp.where(cond_Cpos, hsnow1_Cpos, hsnow1_Cneg)
    snowl0_C = jnp.where(cond_Cpos, snowl0_Cpos, snowl0_Cneg)
    snowl1_C = jnp.where(cond_Cpos, snowl1_Cpos, snowl1_Cneg)

    # --- combine A / B / C ---
    hice0_BC = jnp.where(cond_B, hice0_B, hice0_C)
    hice1_BC = jnp.where(cond_B, hice1_B, hice1_C)
    sice0_BC = jnp.where(cond_B, sice0_B, sice0_C)
    sice1_BC = jnp.where(cond_B, sice1_B, sice1_C)
    mice0_BC = jnp.where(cond_B, mice0_B, mice0_C)
    mice1_BC = jnp.where(cond_B, mice1_B, mice1_C)
    hsnow0_BC = jnp.where(cond_B, hsnow0_B, hsnow0_C)
    hsnow1_BC = jnp.where(cond_B, hsnow1_B, hsnow1_C)
    snowl0_BC = jnp.where(cond_B, snowl0_B, snowl0_C)
    snowl1_BC = jnp.where(cond_B, snowl1_B, snowl1_C)

    hice0_f = jnp.where(cond_A, hice0_A, hice0_BC)
    hice1_f = jnp.where(cond_A, hice1_A, hice1_BC)
    sice0_f = jnp.where(cond_A, sice0_A, sice0_BC)
    sice1_f = jnp.where(cond_A, sice1_A, sice1_BC)
    mice0_f = jnp.where(cond_A, mice0_A, mice0_BC)
    mice1_f = jnp.where(cond_A, mice1_A, mice1_BC)
    hsnow0_f = jnp.where(cond_A, hsnow0_A, hsnow0_BC)
    hsnow1_f = jnp.where(cond_A, hsnow1_A, hsnow1_BC)
    snowl0_f = jnp.where(cond_A, snowl0_A, snowl0_BC)
    snowl1_f = jnp.where(cond_A, snowl1_A, snowl1_BC)

    hice_n = jnp.stack([hice0_f, hice1_f, hi2, hi3], axis=-1)
    sice_n = jnp.stack([sice0_f, sice1_f, si2, si3], axis=-1)
    mice_n = jnp.stack([mice0_f, mice1_f, mi2, mi3], axis=-1)
    hsnow_n = jnp.stack([hsnow0_f, hsnow1_f], axis=-1)
    snowl_n = jnp.stack([snowl0_f, snowl1_f], axis=-1)
    return hsnow_n, hice_n, sice_n, mice_n, snowl_n


def tice(hsil, ssil, msi1, msi2):
    condA = ACE1I > XSI[1] * msi1
    mice0 = jnp.where(condA, ACE1I - XSI[1] * msi1, jnp.zeros_like(msi1))
    mice1 = jnp.where(condA, XSI[1] * msi1, ACE1I * jnp.ones_like(msi1))
    snowl0 = jnp.where(condA, msi1 - ACE1I, XSI[0] * msi1)
    snowl1 = jnp.where(condA, jnp.zeros_like(msi1), XSI[1] * msi1 - ACE1I)

    tsil0_hi = Ti2b(hsil[..., 0] / (XSI[0] * msi1), 1e3 * _safe_div(ssil[..., 0], mice0), snowl0, mice0)
    tsil0_lo = Ti(hsil[..., 0] / (XSI[0] * msi1), jnp.zeros_like(msi1))
    tsil0 = jnp.where(mice0 != 0.0, tsil0_hi, tsil0_lo)
    tsil1 = Ti2b(hsil[..., 1] / (XSI[1] * msi1), 1e3 * _safe_div(ssil[..., 1], mice1), snowl1, mice1)
    rest = [Ti(hsil[..., l] / (XSI[l] * msi2), 1e3 * ssil[..., l] / (XSI[l] * msi2)) for l in range(2, LMI)]
    return jnp.stack([tsil0, tsil1] + rest, axis=-1)


def sea_ice(dtsrce, snow, hsil, ssil, msi2, f0dt, f1dt, evap, srox0, fmoc, fhoc, fsoc, wetsnow):
    """SEA_ICE. Returns dict: snow, hsil, ssil, msi2, srox2, run, erun, srun, wetsnow, melt12, cmprs.
    Scalar-per-cell args: shape (N,); hsil/ssil: shape (N, LMI)."""
    zeros4 = jnp.zeros_like(hsil)
    fsri_active = solar_ice_frac_full(snow, msi2, wetsnow)
    fsri = jnp.where((srox0 > 0.0)[..., None], fsri_active, zeros4)
    srox2 = srox0 * fsri[..., LMI - 1]

    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)

    f0 = f1dt
    hc1 = dEidTi(tsil[..., 1], 1e3 * _safe_div(sice[..., 1], mice[..., 1])) * mice[..., 1]
    tavg1 = _safe_div(tsil[..., 1] * mice[..., 2] + tsil[..., 2] * mice[..., 1], mice[..., 2] + mice[..., 1])
    savg1 = 1e3 * _safe_div(sice[..., 1] * _safe_div(mice[..., 2], mice[..., 1])
                             + sice[..., 2] * _safe_div(mice[..., 1], mice[..., 2]), mice[..., 2] + mice[..., 1])
    alam1 = alami(tavg1, savg1)
    dfdti1 = 2.0 * alam1 * RHOI * dtsrce / (mice[..., 1] + mice[..., 2])
    f1 = (dfdti1 * (hc1 * (tsil[..., 1] - tsil[..., 2]) + ALPHA * f0) + hc1 * srox0 * fsri[..., 1]) \
        / (hc1 + ALPHA * dfdti1)

    hc2 = dEidTi(tsil[..., 2], 1e3 * _safe_div(sice[..., 2], mice[..., 2])) * mice[..., 2]
    tavg2 = _safe_div(tsil[..., 2] * mice[..., 3] + tsil[..., 3] * mice[..., 2], mice[..., 3] + mice[..., 2])
    savg2 = 1e3 * _safe_div(sice[..., 2] * _safe_div(mice[..., 3], mice[..., 2])
                             + sice[..., 3] * _safe_div(mice[..., 2], mice[..., 3]), mice[..., 3] + mice[..., 2])
    alam2 = alami(tavg2, savg2)
    dfdti2 = 2.0 * alam2 * RHOI * dtsrce / (mice[..., 2] + mice[..., 3])
    f2 = (dfdti2 * (hc2 * (tsil[..., 2] - tsil[..., 3]) + ALPHA * f1) + hc2 * srox0 * fsri[..., 2]) \
        / (hc2 + ALPHA * dfdti2)

    hsil = hsil.at[..., 0].add(f0dt - f1dt)
    hsil = hsil.at[..., 1].add(f1dt)

    snowl, hsnow, hice, sice, _, _, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, False)

    hice = hice.at[..., 1].add(-f1)
    hice = hice.at[..., 2].add(f1)
    hice = hice.at[..., 2].add(-f2)
    hice = hice.at[..., 3].add(f2)
    hice = hice.at[..., LMI - 1].add(-(srox2 + fhoc))
    sice = sice.at[..., LMI - 1].add(-fsoc)
    mice = mice.at[..., LMI - 1].add(-fmoc)

    dew = -evap
    dewi0_init = jnp.maximum(0.0, dew)
    dews_init = dew - dewi0_init

    cond1 = (snowl[..., 0] + dews_init) <= 0.0
    dews_P = -snowl[..., 0]
    dewi0_P = dew - dews_P
    mice0_P = mice[..., 0] + dewi0_P
    mice1_P = mice[..., 1]

    cond2 = mice[..., 0] == 0.0
    dewi1_Q = dewi0_init
    dewi0_Q = jnp.zeros_like(dew)
    mice1_Q = mice[..., 1] + dewi1_Q
    mice0_Q = mice[..., 0]

    dewi0_R = dewi0_init
    dewi1_R = jnp.zeros_like(dew)
    mice0_R = mice[..., 0] + dewi0_init
    mice1_R = mice[..., 1]

    dewi0_notP = jnp.where(cond2, dewi0_Q, dewi0_R)
    dewi1_notP = jnp.where(cond2, dewi1_Q, dewi1_R)
    mice0_notP = jnp.where(cond2, mice0_Q, mice0_R)
    mice1_notP = jnp.where(cond2, mice1_Q, mice1_R)

    dews = jnp.where(cond1, dews_P, dews_init)
    dewi0 = jnp.where(cond1, dewi0_P, dewi0_notP)
    dewi1 = jnp.where(cond1, jnp.zeros_like(dew), dewi1_notP)
    mice = mice.at[..., 0].set(jnp.where(cond1, mice0_P, mice0_notP))
    mice = mice.at[..., 1].set(jnp.where(cond1, mice1_P, mice1_notP))

    melts = jnp.maximum(0.0, hsnow[..., 0] * BYLHM + snowl[..., 0] + dews)
    melts2 = jnp.maximum(0.0, hsnow[..., 1] * BYLHM + snowl[..., 1])

    cond_melt = mice > 0.0
    mi = jnp.where(cond_melt, Mi(hice, sice, mice), 0.0)
    smelti = jnp.where(cond_melt, mi * _safe_div(sice, mice), 0.0)
    hmelti = jnp.where(cond_melt, mi * Em(1e3 * _safe_div(sice, mice)), 0.0)
    melti = jnp.where(cond_melt, mi, 0.0)
    mice_after = mice - melti
    sice_zero_cond = (mice_after == 0.0) & (jnp.abs(sice - smelti) < 1e-12)
    sice_after = jnp.where(cond_melt, jnp.where(sice_zero_cond, 0.0, sice - smelti), sice)
    hice_zero_cond = (mice_after == 0.0) & (jnp.abs(hice - hmelti) < 1e-12)
    hice_after = jnp.where(cond_melt, jnp.where(hice_zero_cond, 0.0, hice - hmelti), hice)
    mice, sice, hice = mice_after, sice_after, hice_after

    snowx = snowl[..., 0] + dews - melts
    cond_snowx = snowx > 0.0
    cmprs_pos = jnp.minimum(0.0, snowx)
    hcmprs_pos = _safe_div(hsnow[..., 0] * cmprs_pos, snowx)
    snowl0_pos = snowx - cmprs_pos
    hsnow0_pos = hsnow[..., 0] - hcmprs_pos
    cond_ice0 = mice[..., 0] > 0.0
    mice0_pos = jnp.where(cond_ice0, mice[..., 0] + cmprs_pos, mice[..., 0])
    mice1_pos = jnp.where(cond_ice0, mice[..., 1], mice[..., 1] + cmprs_pos)
    hice0_pos = jnp.where(cond_ice0, hice[..., 0] + hcmprs_pos, hice[..., 0])
    hice1_pos = jnp.where(cond_ice0, hice[..., 1], hice[..., 1] + hcmprs_pos)

    cmprs_neg = jnp.zeros_like(snowx)
    hice0_neg = hice[..., 0] + hsnow[..., 0]
    snowl0_neg = jnp.zeros_like(snowx)
    hsnow0_neg = jnp.zeros_like(snowx)

    cmprs = jnp.where(cond_snowx, cmprs_pos, cmprs_neg)
    snowl = snowl.at[..., 0].set(jnp.where(cond_snowx, snowl0_pos, snowl0_neg))
    hsnow = hsnow.at[..., 0].set(jnp.where(cond_snowx, hsnow0_pos, hsnow0_neg))
    mice = mice.at[..., 0].set(jnp.where(cond_snowx, mice0_pos, mice[..., 0]))
    mice = mice.at[..., 1].set(jnp.where(cond_snowx, mice1_pos, mice[..., 1]))
    hice = hice.at[..., 0].set(jnp.where(cond_snowx, hice0_pos, hice0_neg))
    hice = hice.at[..., 1].set(jnp.where(cond_snowx, hice1_pos, hice[..., 1]))

    cond_melts2 = melts2 > 0.0
    cond_snowl1_gt = snowl[..., 1] > melts2
    snowl1_new = jnp.where(cond_snowl1_gt, snowl[..., 1] - melts2, jnp.zeros_like(melts2))
    hice1_new = jnp.where(cond_snowl1_gt, hice[..., 1], hice[..., 1] + hsnow[..., 1])
    hsnow1_new = jnp.where(cond_snowl1_gt, hsnow[..., 1], jnp.zeros_like(melts2))
    snowl = snowl.at[..., 1].set(jnp.where(cond_melts2, snowl1_new, snowl[..., 1]))
    hice = hice.at[..., 1].set(jnp.where(cond_melts2, hice1_new, hice[..., 1]))
    hsnow = hsnow.at[..., 1].set(jnp.where(cond_melts2, hsnow1_new, hsnow[..., 1]))

    fmsi2 = -melti[..., 0] - melti[..., 1] + dewi0 + dewi1 + cmprs
    mice, hice, sice = relayer(fmsi2, mice, hice, sice)
    hsnow, hice, sice, mice, snowl = relayer_12(hsnow, hice, sice, mice, snowl)
    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)

    run = melts + melts2 + jnp.sum(melti, axis=-1)
    srun = jnp.sum(smelti, axis=-1)
    erun = srox2 + jnp.sum(hmelti, axis=-1)
    tsil = tice(hsil, ssil, msi1, msi2)
    melt12 = melts + melti[..., 0]
    wetsnow_out = jnp.asarray(wetsnow, dtype=bool) | (melt12 > 0.0)

    return dict(snow=snow, hsil=hsil, ssil=ssil, msi2=msi2, srox2=srox2, run=run, srun=srun,
                erun=erun, wetsnow=wetsnow_out, melt12=melt12, cmprs=cmprs, tsil=tsil)


def ssidec(snow, msi2, hsil, ssil, dt, melt12):
    """SEAICE.f SSIDEC (seaice_thermo='BP'). Returns dict: snow,msi1,msi2,hsil,ssil,melt12,mflux,hflux,sflux."""
    dtssi = 30.0
    bydtssi = 1.0 / (dtssi * SECONDS_PER_DAY)
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)

    tsil_safe = jnp.where(tsil == 0.0, 1.0, tsil)
    mice_safe = jnp.where(mice == 0.0, 1.0, mice)
    cond_ssi_frac = (1e3 * _safe_div(sice, mice)) > 1e-10
    brine_frac = jnp.where(cond_ssi_frac, -MU * 1e3 * sice / (tsil_safe * mice_safe), 0.0)
    brine_frac = jnp.where(sice > 0.0, brine_frac, 0.0)
    bf_safe = jnp.where(brine_frac == 0.0, 1.0, brine_frac)

    msi2_b = msi2[..., None]
    melt12_b = melt12[..., None]
    dt_b = dt[..., None] if hasattr(dt, "ndim") and dt.ndim > 0 else dt

    cond_rate1 = (mice > AC2OIM) & (brine_frac != 0.0)
    r_hi = jnp.minimum(1.0, 0.3 * melt12_b / (mice_safe * bf_safe))
    r_lo = jnp.minimum(0.3, 0.3 * melt12_b / (mice_safe * bf_safe))
    cond_hibrine = (brine_frac > 0.05) & (msi2_b > 2.0 * RHOI)
    rate = jnp.where(cond_rate1, jnp.where(cond_hibrine, r_hi, r_lo), jnp.zeros_like(sice))

    cond_hi2 = brine_frac > 0.05
    rate = jnp.where(cond_hi2, jnp.minimum(rate + dt_b * bydtssi * 5.0 * (brine_frac - 0.05), 1.0), rate)

    cond_low_sice = sice < SSIMIN * mice
    rate = jnp.where(cond_low_sice, 1.0, rate)

    dmsi_lo = rate * brine_frac * mice
    dhsi_lo = -rate * MU * 1e3 * sice * SHW
    dmsi_hi = brine_frac * mice
    dhsi_hi = jnp.maximum(-MU * 1e3 * sice * SHW, hice - (tsil * SHI - LHM) * (mice - dmsi_hi))

    dmsi = jnp.where(cond_low_sice, dmsi_hi, dmsi_lo)
    dhsi = jnp.where(cond_low_sice, dhsi_hi, dhsi_lo)
    dssi = rate * sice   # same formula both branches; rate is already the correct per-lane value

    cond_active = sice > 0.0
    dmsi = jnp.where(cond_active, dmsi, 0.0)
    dhsi = jnp.where(cond_active, dhsi, 0.0)
    dssi = jnp.where(cond_active, dssi, 0.0)
    sice = jnp.where(cond_active, jnp.maximum(0.0, sice - dssi), sice)
    hice = jnp.where(cond_active, hice - dhsi, hice)
    mice = jnp.where(cond_active, mice - dmsi, mice)

    layer01_mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    melt12 = melt12 + jnp.sum(jnp.where(snow[..., None] == 0.0, dmsi * layer01_mask, 0.0), axis=-1)

    fmsi0 = -dmsi[..., 0]
    fhsi0 = fmsi0 * (tsil[..., 1] * SHI - LHM)
    fmsi1 = -dmsi[..., 0] - dmsi[..., 1]
    fhsi1 = fmsi1 * (tsil[..., 2] * SHI - LHM)

    cond_pos01 = (dmsi[..., 0] + dmsi[..., 1]) > 0.0
    hice = hice.at[..., 0].add(jnp.where(cond_pos01, -fhsi0, 0.0))
    hice = hice.at[..., 1].add(jnp.where(cond_pos01, fhsi0 - fhsi1, 0.0))
    mice = mice.at[..., 0].add(jnp.where(cond_pos01, -fmsi0, 0.0))
    mice = mice.at[..., 1].add(jnp.where(cond_pos01, fmsi0 - fmsi1, 0.0))

    snow, msi1, msi2, hsil, ssil = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)

    fmsi2 = XSI[2] * dmsi[..., 3] + XSI[3] * (fmsi1 - dmsi[..., 2])
    cond_fmsi2_pos = fmsi2 > 0.0
    mice2_safe = jnp.where(mice[..., 2] == 0.0, 1.0, mice[..., 2])
    fhsi2 = jnp.where(cond_fmsi2_pos, fmsi2 * hsil[..., 2] / mice2_safe, fmsi2 * (tsil[..., 3] * SHI - LHM))
    fssi2 = jnp.where(cond_fmsi2_pos, fmsi2 * ssil[..., 2] / mice2_safe, jnp.zeros_like(fmsi2))

    hsil = hsil.at[..., 2].add(fhsi1 - fhsi2)
    hsil = hsil.at[..., 3].add(fhsi2)
    ssil = ssil.at[..., 2].add(-fssi2)      # fssi[1] (0-based idx 1) is never assigned in the Fortran, stays 0
    ssil = ssil.at[..., 3].add(fssi2)
    msi2 = msi2 + fmsi1

    mflux = jnp.sum(dmsi, axis=-1)
    hflux = jnp.sum(dhsi, axis=-1)
    sflux = jnp.sum(dssi, axis=-1)
    tsil = tice(hsil, ssil, msi1, msi2)
    return dict(snow=snow, msi1=msi1, msi2=msi2, hsil=hsil, ssil=ssil, melt12=melt12,
                mflux=mflux, hflux=hflux, sflux=sflux)


def snowice(tm, sm, snow, msi2, hsil, ssil, qsfix):
    """SEAICE.f snowice. Returns dict: snow,msi2,hsil,ssil,msnwic,hsnwic,ssnwic,dsnow (no-op if inactive)."""
    active = RHOI * snow > (ACE1I + msi2) * (RHOWS - RHOI)
    msi1 = snow + ACE1I
    z0 = (msi1 + msi2) / RHOWS - (ACE1I + msi2) / RHOI
    maxm0 = z0 * RHOWS * (RHOI - RHOS) / (RHOWS + RHOS - RHOI)
    snowl, hsnow, hice, sice, tsnw, tsil, mice = get_snow_ice_layer(snow, msi2, hsil, ssil, True)
    qsfix_b = jnp.asarray(qsfix, dtype=bool)
    si = jnp.where(qsfix_b, 1e3 * SSI0, FSSS * sm)
    tf = tfrez(sm)
    eic = Ei(tf, si)
    eoc = tm * SHW
    esnow1 = _safe_div(hsnow[..., 0], snowl[..., 0])
    erat1 = (esnow1 - Ei(tf, 0.0)) / (eic - eoc)
    cond_sl1 = snowl[..., 1] > 0.0
    esnow2 = jnp.where(cond_sl1, _safe_div(hsnow[..., 1], snowl[..., 1]), 0.0)
    eratd = (esnow1 - esnow2) / (eic - eoc)
    maxme = (z0 * RHOI * erat1 - snowl[..., 1] * eratd) / (1.0 + erat1 * (RHOWS - RHOI) / RHOWS)
    maxm = jnp.maximum(0.0, jnp.minimum(maxme, maxm0))
    maxm = jnp.maximum(0.0, jnp.minimum(0.9 * RHOWS * (mice[..., 1] / RHOI - z0), maxm))
    dsnow = z0 * RHOI - maxm * (RHOWS - RHOI) / RHOWS
    dsnow = jnp.where(maxm == 0.0, jnp.minimum(0.9 * mice[..., 1], dsnow), dsnow)

    cond_dsnow_gt = dsnow > snowl[..., 1]
    hsnow0_a = hsnow[..., 0] * _safe_div(snowl[..., 0] + snowl[..., 1] - dsnow, snowl[..., 0])
    mice0_a = mice[..., 0] + maxm + dsnow
    sice0_a = sice[..., 0] + maxm * 0.001 * si
    hice0_a = hice[..., 0] + maxm * eoc + snowl[..., 1] * esnow2 + (dsnow - snowl[..., 1]) * esnow1
    snowl0_a = snowl[..., 0] + snowl[..., 1] - dsnow

    hsnow1_b = hsnow[..., 1] * _safe_div(snowl[..., 1] - dsnow, snowl[..., 1])
    mice1_b = mice[..., 1] + dsnow + maxm
    sice1_b = sice[..., 1] + maxm * 0.001 * si
    hice1_b = hice[..., 1] + maxm * eoc + dsnow * esnow2
    snowl1_b = snowl[..., 1] - dsnow

    mice = mice.at[..., 0].set(jnp.where(cond_dsnow_gt, mice0_a, mice[..., 0]))
    mice = mice.at[..., 1].set(jnp.where(cond_dsnow_gt, mice[..., 1], mice1_b))
    hice = hice.at[..., 0].set(jnp.where(cond_dsnow_gt, hice0_a, hice[..., 0]))
    hice = hice.at[..., 1].set(jnp.where(cond_dsnow_gt, hice[..., 1], hice1_b))
    sice = sice.at[..., 0].set(jnp.where(cond_dsnow_gt, sice0_a, sice[..., 0]))
    sice = sice.at[..., 1].set(jnp.where(cond_dsnow_gt, sice[..., 1], sice1_b))
    snowl = snowl.at[..., 0].set(jnp.where(cond_dsnow_gt, snowl0_a, snowl[..., 0]))
    snowl = snowl.at[..., 1].set(jnp.where(cond_dsnow_gt, jnp.zeros_like(dsnow), snowl1_b))
    hsnow = hsnow.at[..., 0].set(jnp.where(cond_dsnow_gt, hsnow0_a, hsnow[..., 0]))
    hsnow = hsnow.at[..., 1].set(jnp.where(cond_dsnow_gt, jnp.zeros_like(dsnow), hsnow1_b))

    fmsi2 = maxm + dsnow
    mice, hice, sice = relayer(fmsi2, mice, hice, sice)
    hsnow, hice, sice, mice, snowl = relayer_12(hsnow, hice, sice, mice, snowl)
    snow_n, msi1_n, msi2_n, hsil_n, ssil_n = set_snow_ice_layer(hsnow, hice, sice, mice, snowl)
    msnwic = -maxm
    hsnwic = msnwic * eoc
    ssnwic = 0.001 * si * msnwic

    zeros = jnp.zeros_like(snow)
    return dict(snow=jnp.where(active, snow_n, snow), msi2=jnp.where(active, msi2_n, msi2),
                hsil=jnp.where(active[..., None], hsil_n, hsil), ssil=jnp.where(active[..., None], ssil_n, ssil),
                msnwic=jnp.where(active, msnwic, zeros), hsnwic=jnp.where(active, hsnwic, zeros),
                ssnwic=jnp.where(active, ssnwic, zeros), dsnow=jnp.where(active, dsnow, zeros))


def simelt(dt, roice, snow, msi2, hsil, ssil, pocean, tm, tfo, enrgmax):
    """SEAICE.f SIMELT (no tracers). Returns dict: roice, snow, msi2, hsil, ssil, tsil, enrgused,
    run0, salt, melted_out. tsil is NaN where roice remains >0 -- the real Fortran leaves TSIL
    (intent(out)) unassigned in that branch, so no reference value is claimed there either (same
    honesty standard as the plain-Python seaice_core_ff.simelt's tsil=None)."""
    hsil_sum = jnp.sum(hsil, axis=-1)
    ssil_sum = jnp.sum(ssil, axis=-1)
    hsil_sum_safe = jnp.where(hsil_sum == 0.0, 1.0, hsil_sum)

    dtemp = jnp.maximum(tm - tfo, 0.0)
    drsi_main = dt * SILMFAC * dtemp ** SILMPOW
    drsi_main = jnp.where((roice - drsi_main) < 1e-3, roice, drsi_main)
    drsi_main = jnp.where((enrgmax + drsi_main * hsil_sum) < 0.0, -enrgmax / hsil_sum_safe, drsi_main)
    drsi_main = jnp.where((roice - drsi_main) > 1.0, 1.0 - roice, drsi_main)
    drsi = jnp.where(roice < 1e-3, roice, drsi_main)

    enrgused = -drsi * hsil_sum
    run0 = drsi * (snow + ACE1I + msi2)
    salt = drsi * ssil_sum
    roice_new = jnp.minimum(1.0, roice - drsi)
    cond_gone = roice_new < 1e-10

    cond_ocean = pocean > 0.0
    ssil_g01 = jnp.stack([SSI0 * XSI[0] * ACE1I * jnp.ones_like(roice),
                           SSI0 * XSI[1] * ACE1I * jnp.ones_like(roice)], axis=-1)
    ssil_g23 = jnp.stack([SSI0 * XSI[2] * AC2OIM * jnp.ones_like(roice),
                           SSI0 * XSI[3] * AC2OIM * jnp.ones_like(roice)], axis=-1)
    ssil_g_ocean = jnp.concatenate([ssil_g01, ssil_g23], axis=-1)
    ssil_g = jnp.where(cond_ocean[..., None], ssil_g_ocean, jnp.zeros_like(hsil))

    denom01 = jnp.stack([XSI[0] * ACE1I * jnp.ones_like(roice), XSI[1] * ACE1I * jnp.ones_like(roice)], axis=-1)
    denom23 = jnp.stack([XSI[2] * AC2OIM * jnp.ones_like(roice), XSI[3] * AC2OIM * jnp.ones_like(roice)], axis=-1)
    denom_g = jnp.concatenate([denom01, denom23], axis=-1)
    hsil_g = denom_g * Ei(tfo[..., None], 1e3 * ssil_g / denom_g)
    tsil_g = jnp.broadcast_to(tfo[..., None], hsil.shape)

    cond_reset = (tfo > Ti(jnp.zeros_like(tfo), 1e3 * SSI0 * jnp.ones_like(tfo))) & (tfo != 0.0)
    hsil_g = jnp.where(cond_reset[..., None], 0.0, hsil_g)

    roice_out = jnp.where(cond_gone, 0.0, roice_new)
    snow_out = jnp.where(cond_gone, 0.0, snow)
    msi2_out = jnp.where(cond_gone, AC2OIM, msi2)
    ssil_out = jnp.where(cond_gone[..., None], ssil_g, ssil)
    hsil_out = jnp.where(cond_gone[..., None], hsil_g, hsil)
    tsil_out = jnp.where(cond_gone[..., None], tsil_g, jnp.nan)

    return dict(roice=roice_out, snow=snow_out, msi2=msi2_out, hsil=hsil_out, ssil=ssil_out,
                tsil=tsil_out, enrgused=enrgused, run0=run0, salt=salt, melted_out=cond_gone)
