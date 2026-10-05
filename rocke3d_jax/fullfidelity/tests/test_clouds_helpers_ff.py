"""Tests for clouds_helpers_ff.py (CLOUDS2.F90 PRECIP_MP, ANVIL_OPTICAL_THICKNESS, MC_CLOUD_FRACTION,
MC_PRECIP_PHASE -- D91; CONVECTIVE_MICROPHYSICS -- D92).

Real-Fortran validation uses ff_data/<date>/ffc_{pmp,anv,mcf,cmp,mpp}_*.bin (skipped if absent).
The scalar-transcription and hand-derived tests do NOT need dumps and are NOT validation against real Fortran:
they only pin the vectorised port to a line-by-line scalar transcription of the Fortran on random inputs that
reach branches the real dumps never exercised.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_helpers_ff as ch  # noqa: E402
import clouds_helpers_compare as cmp  # noqa: E402

DATES = [d for d, _ in cmp.DATES]
HAVE = {d: os.path.exists(f"{cmp.FF_DEFAULT}/{d}/ffc_h_consts.txt") for d in DATES}
needs_dumps = pytest.mark.skipif(not all(HAVE.values()), reason="ff_data ffc_{pmp,anv,mcf,cmp,mpp} dumps not present")
TOL = 1e-12          # relative; observed worst 5.4e-15 (CMP condp), see FULL_FIDELITY_DELTAS.md D91/D92
_cache = {}


def _data(date, rt):
    if (date, rt) not in _cache:
        _cache[(date, rt)] = cmp.load_date(date, rt)
    return _cache[(date, rt)]


def _maxrel(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.max((np.abs(a - b) / np.maximum(np.abs(b), 1e-300))[ok & (b != 0)], initial=0.0))


# ================================================================ real-Fortran validation
@needs_dumps
def test_constants_match_real_model():
    c = cmp.read_consts(f"{cmp.FF_DEFAULT}/nov26/ffc_h_consts.txt")
    for k, v in dict(rgas=ch.RGAS, grav=ch.GRAV, teeny=ch.TEENY, pi=ch.PI, by3=ch.BY3, by6=ch.BY6, twopi=ch.TWOPI,
                     lhe=ch.LHE, lhs=ch.LHS, lhm=ch.LHM, tf=ch.TF, bysha=ch.BYSHA, rhow=ch.RHOW,
                     dtsrc=ch.DTSRC).items():
        assert v == c[k], k


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_precip_mp_real(date):
    p = _data(date, "pmp")
    assert set(np.unique(p["site"]).astype(int)) == set(range(1, 9))        # every call site sampled
    out = cmp.run_pmp(p)
    assert _maxrel(out, p["out"]) < TOL
    assert (out == p["out"]).mean() > 0.99          # last-bit exp() differences only (D54 category)
    for s in range(1, 9):
        m = p["site"] == s
        assert (p["out"][m] > 0).mean() > 0.9, s    # non-vacuous: nonzero precipitating fractions at each site


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_anvil_real(date):
    a = _data(date, "anv")
    rc, tm = cmp.run_anv(a)
    assert _maxrel(rc, a["rcld"]) < TOL and _maxrel(tm, a["taumc"]) < TOL
    assert ((rc == a["rcld"]).mean() > 0.99) and ((tm == a["taumc"]).mean() > 0.99)
    liq = a["svlatl"] == ch.LHE
    assert liq.sum() > 50 and (~liq).sum() > 1000                         # both phases exercised
    assert ((~liq) & (a["rcld"] == a["rimax"])).sum() > 500               # RIMAX cap exercised


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_mc_cloud_fraction_real_bitwise(date):
    m = _data(date, "mcf")
    dts = cmp.read_consts(f"{cmp.FF_DEFAULT}/{date}/ffc_h_consts.txt")["dtsrc"]
    f = cmp.run_mcf(m, dtsrc=dts)
    assert np.array_equal(f, m["fcloud"])                                  # observed 100% bitwise
    deep = (m["plemin"] - m["plel2"]) >= 450
    shallow = (m["plemin"] - m["plemax"]) < 450
    below = m["L"] < m["lmin"]
    assert deep.sum() > 100 and (below & ~shallow).sum() > 50 and (shallow & (m["L"] == m["lmax"] - 1)).sum() > 100
    assert (shallow & below).sum() > 500 and (m["fcloud"] == 1.0).sum() > 5          # virga zeroing and 1.0 cap


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_mc_precip_phase_real_bitwise(date):
    q = _data(date, "mpp")
    lhp, mc, h1 = cmp.run_mpp(q)
    assert np.array_equal(lhp, q["lhp"]) and np.array_equal(mc, q["mcloud"]) and np.array_equal(h1, q["heat1"])
    assert np.array_equal(q["prcp_in"], q["prcp"])                         # PRCP is INOUT but never modified
    melt = (q["lhp1"] == ch.LHS) & (q["told"] > ch.TF) & (q["told1"] <= ch.TF)
    assert melt.sum() > 100                                                # melting branch exercised
    assert ((q["lhp"] != q["vlat"]) & (q["cond"] > 0)).sum() > 30          # phase-conversion heat exercised
    assert (q["lhp1"] == ch.LHS).sum() > 500 and (q["lhp1"] == ch.LHE).sum() > 500


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_convective_microphysics_real(date):
    c = _data(date, "cmp")
    cp, cp1, cip, cgp, dg = cmp.ch.convective_microphysics(
        *[c[k] for k in ("pl wcu dwcu lfrz wcufrz tp ti fitmax pland cn0 cn0i cn0g flamw flamg flami rhoip rhog "
                         "itmax tlmin tlmin1 wmax condip_in condgp_in").split()], return_diag=True)
    for a, b in ((cp, c["condp"]), (cp1, c["condp1"]), (cip, c["condip"]), (cgp, c["condgp"])):
        assert _maxrel(a, b) < TOL
        assert (a == b).mean() > 0.99
    # branch non-vacuity: all three phase branches, FG interior, both LFRZ states, both PLAND sides
    assert dg["water"].sum() > 3000 and dg["ice"].sum() > 2000 and dg["mixed"].sum() > 800
    assert (dg["mixed"] & (dg["fg"] > 0) & (dg["fg"] < 1)).sum() > 800
    assert (c["lfrz"] == 0).sum() > 1000 and (c["lfrz"] > 0).sum() > 1000
    assert (dg["water"] & (c["pland"] < .5)).sum() > 1000 and (dg["water"] & (c["pland"] >= .5)).sum() > 100
    # CONDIP/CONDGP are only assigned in the mixed branch; elsewhere the real routine returns the entry values
    mixed = dg["mixed"]
    assert np.array_equal(c["condip"][~mixed], c["condip_in"][~mixed], equal_nan=True)   # pass-through outside mixed phase


@needs_dumps
def test_single_precision_literal_hazard_is_real():
    """The REAL(4) literals 19.3 / 11.72 / 2.7 / 2.439 / .4 in the DCG/DCI formulas matter: using them as
    double-precision values changes the mixed-phase GRAUPEL result by far more than the validation tolerance."""
    c = _data("nov26", "cmp")
    args = [c[k] for k in ("pl wcu dwcu lfrz wcufrz tp ti fitmax pland cn0 cn0i cn0g flamw flamg flami rhoip rhog "
                           "itmax tlmin tlmin1 wmax condip_in condgp_in").split()]
    good = ch.convective_microphysics(*args)
    saved = (ch._F193, ch._F1172, ch._F27, ch._F2439, ch._F04)
    try:
        ch._F193, ch._F1172, ch._F27, ch._F2439, ch._F04 = 19.3, 11.72, 2.7, 2.439, 0.4     # "naive" double literals
        bad = ch.convective_microphysics(*args)
    finally:
        ch._F193, ch._F1172, ch._F27, ch._F2439, ch._F04 = saved
    assert _maxrel(good[0], c["condp"]) < TOL
    assert _maxrel(bad[0], c["condp"]) > 1e-9          # naive literals are detected as wrong against the real dump


@needs_dumps
@pytest.mark.parametrize("name", ["pi_by6", "flam_pow", "wrong_rhow", "wrong_tf"])
def test_mutations_detected_on_real_data(name, monkeypatch):
    c = _data("nov26", "cmp")
    args = [c[k] for k in ("pl wcu dwcu lfrz wcufrz tp ti fitmax pland cn0 cn0i cn0g flamw flamg flami rhoip rhog "
                           "itmax tlmin tlmin1 wmax condip_in condgp_in").split()]
    if name == "pi_by6":
        monkeypatch.setattr(ch, "BY6", ch.BY6 * 1.001)
    elif name == "wrong_rhow":
        monkeypatch.setattr(ch, "RHOW", 1.001e3)
    elif name == "wrong_tf":
        monkeypatch.setattr(ch, "TF", 273.0)
    out = ch.convective_microphysics(*args, pow4="mul" if name == "flam_pow" else "pow")
    r = max(_maxrel(out[0], c["condp"]), _maxrel(out[1], c["condp1"]))
    if name == "flam_pow":
        # (x*x)*(x*x) instead of pow(x,4.) changes only the last bits: ~10% of records, still < tolerance, and
        # detectable only as a bitwise-fraction drop.  This is the evidence for which form the Fortran uses.
        base = ch.convective_microphysics(*args)
        assert (out[0] == c["condp"]).mean() < (base[0] == c["condp"]).mean() - 0.03
    else:
        assert r > 1e-6


@needs_dumps
def test_other_mutations_detected_on_real_data(monkeypatch):
    a = _data("nov26", "anv")
    rc, tm = cmp.run_anv(a, lhe=ch.LHS)                        # phase test swapped: liquid branch never taken
    assert _maxrel(rc, a["rcld"]) > 1e-3
    a2 = _data("dec01", "anv")                                  # the TAUMC=100 cap is hit by one sampled record (dec01)
    assert (a2["taumc"] == 100.0).sum() >= 1
    rc2, tm2 = cmp.run_anv(a2, cap=99.0)
    assert not np.array_equal(tm2, a2["taumc"])
    m = _data("nov26", "mcf")
    for kw in (dict(anvil5=1.0), dict(virga=False), dict(detr3=False)):
        assert _maxrel(cmp.run_mcf(m, **kw), m["fcloud"]) > 1e-3, kw
    assert _maxrel(cmp.run_mcf(m, dtsrc=3600.0), m["fcloud"]) > 1e-3
    q = _data("nov26", "mpp")
    lhp, mc, h1 = cmp.run_mpp(q, bysha=1.01 * ch.BYSHA)
    assert _maxrel(h1, q["heat1"]) > 1e-3
    lhp, mc, h1 = cmp.run_mpp(q, lhs=ch.LHS + 1.0)               # melt test never fires
    assert not np.array_equal(lhp, q["lhp"]) or not np.array_equal(h1, q["heat1"])
    d = dict(q)
    d["mcrevp"] = np.zeros_like(q["mcrevp"])                    # AR5 (below-cloud-base-only) option not used in the run
    lhp, mc, h1 = cmp.run_mpp(d)
    assert not np.array_equal(mc, q["mcloud"])                  # ...but the port would give different results


# ================================================================ scalar transcriptions (NOT real-Fortran validation)
def _f4(x):
    return float(np.float32(x))


def _s_precip_mp(rho, flam, dc, cn):
    return (rho * (ch.PI * ch.BY6) * cn * math.exp(-flam * dc)
            * (dc * dc * dc / flam + 3. * dc * dc / (flam * flam) + 6. * dc / (flam * flam * flam)
               + 6. / math.pow(flam, 4.0)))


def _s_cmp(pl, wcu, dwcu, lfrz, wcufrz, tp, ti, fitmax, pland, cn0, cn0i, cn0g, flamw, flamg, flami, rhoip, rhog,
           itmax, tlmin, tlmin1, wmax, cip, cgp):
    """Line-by-line transcription of CONVECTIVE_MICROPHYSICS (CLD_AER_CDNC arms excluded)."""
    TF, TEENY = ch.TF, ch.TEENY
    wv = wcu - dwcu
    if wv < 0.0:
        wv = 0.0
    dcg = min(((wv / _f4(19.3)) * (pl / 1000.) ** _f4(.4)) ** _f4(2.7), 1e-2)
    dci = min(((wv / _f4(11.72)) * (pl / 1000.) ** _f4(.4)) ** _f4(2.439), 1e-2)
    tig = TF if lfrz == 0 else TF - 4. * wcufrz
    if tig < ti - 10.0:
        tig = ti - 10.0

    def search(wv):
        dcw = 0.0
        ddcw = 6e-3 * fitmax
        if pland < .5:
            ddcw = 1.5e-3 * fitmax
        for _ in range(1, int(itmax)):
            vt = (-.267 + dcw * (5.15e3 - dcw * (1.0225e6 - 7.55e7 * dcw))) * (1000. / pl) ** .4
            if vt >= 0. and vt >= wv:
                break
            if vt > wmax:
                break
            dcw = dcw + ddcw
        return dcw

    if tp >= TF:
        condp1 = _s_precip_mp(ch.RHOW, flamw, search(wv), cn0)
        condp = _s_precip_mp(ch.RHOW, flamw, search(wcu + dwcu), cn0)
    elif tp <= tig:
        condp1 = _s_precip_mp(rhoip, flami, dci, cn0i)
        wv = wcu + dwcu
        dci = min(((wv / _f4(11.72)) * (pl / 1000.) ** _f4(.4)) ** _f4(2.439), 1e-2)
        condp = _s_precip_mp(rhoip, flami, dci, cn0i)
    else:
        fg = (tp - tig) / ((TF - tig) + TEENY)
        fg = min(fg, 1.0)
        fg = max(fg, 0.0)
        if tlmin <= TF or tlmin1 <= TF:
            fg = 0.0
        fi = 1. - fg
        cip = _s_precip_mp(rhoip, flami, dci, cn0i)
        cgp = _s_precip_mp(rhog, flamg, dcg, cn0g)
        condp1 = fg * cgp + fi * cip
        wv = wcu + dwcu
        dci = min(((wv / _f4(11.72)) * (pl / 1000.) ** _f4(.4)) ** _f4(2.439), 1e-2)
        dcg = min(((wv / _f4(19.3)) * (pl / 1000.) ** _f4(.4)) ** _f4(2.7), 1e-2)
        cip = _s_precip_mp(rhoip, flami, dci, cn0i)
        cgp = _s_precip_mp(rhog, flamg, dcg, cn0g)
        condp = fg * cgp + fi * cip
    return condp, condp1, cip, cgp


def _random_cmp_inputs(rng, n):
    r = rng.uniform
    return dict(pl=r(50, 1000, n), wcu=r(0.0, 12.0, n), dwcu=r(0, 4, n), lfrz=rng.integers(0, 12, n).astype(float),
                wcufrz=r(0, 10, n), tp=r(200, 305, n), ti=np.full(n, 233.16), fitmax=np.full(n, 1 / 3.0),
                pland=r(0, 1, n), cn0=np.full(n, 8e6), cn0i=np.full(n, 4e6), cn0g=np.full(n, 4e6),
                flamw=r(2e3, 6e3, n), flamg=r(800, 3000, n), flami=r(800, 3000, n), rhoip=np.full(n, 100.),
                rhog=np.full(n, 400.), itmax=np.full(n, 3.0), tlmin=r(250, 300, n), tlmin1=r(250, 300, n),
                wmax=np.full(n, 50.), condip_in=r(0, 1e-3, n), condgp_in=r(0, 1e-3, n))


def test_cmp_vectorised_matches_scalar_transcription_all_branches():
    rng = np.random.default_rng(0)
    n = 4000
    a = _random_cmp_inputs(rng, n)
    a["itmax"] = rng.integers(2, 40, n).astype(float)          # exercises the no-exit branch of the DCW search
    a["wmax"] = rng.choice([0.5, 50.0], n)                      # exercises the WMAX exit
    a["fitmax"] = 1.0 / a["itmax"]
    a["ti"] = rng.choice([233.16, 266.0], n)                    # moves TIG so the TIG floor (TIG<TI-10) is hit
    cp, cp1, cip, cgp, dg = ch.convective_microphysics(*a.values(), return_diag=True)
    order = list(a)
    ref = np.array([_s_cmp(*[a[k][i] for k in order]) for i in range(n)])
    for j, v in enumerate((cp, cp1, cip, cgp)):
        assert _maxrel(v, ref[:, j]) < 1e-12, j
    # all branches and loop exits are reached by this random set
    assert dg["water"].sum() > 300 and dg["ice"].sum() > 300 and dg["mixed"].sum() > 300
    assert (dg["exit1"] == 1).sum() > 50 and (dg["exit1"] == 2).sum() > 50 and (dg["exit1"] == 0).sum() > 50
    # FG clip to [0,1] is unreachable in the mixed branch (TIG<TP<TF gives 0<FG<1); only the TLMIN/TLMIN1 zeroing acts
    assert (dg["mixed"] & (dg["fg"] == 1.0)).sum() == 0 and (dg["mixed"] & (dg["fg"] == 0.0)).sum() > 5
    assert (dg["tig"] == a["ti"] - 10).sum() > 50
    # outside the mixed branch CONDIP/CONDGP pass through unchanged
    nm = ~dg["mixed"]
    assert np.array_equal(cip[nm], a["condip_in"][nm]) and np.array_equal(cgp[nm], a["condgp_in"][nm])


def test_cmp_ordering_and_monotonic_sanity():
    # hand-derived: liquid with a very large WCU never exits (VT never reaches WV within ITMAX-1 steps) -> DCW is
    # the full ITMAX-1 increments; with WCU=0, WV=0 and VT(0)=-.267*(1000/pl)^.4<0 so the loop continues until VT>=0
    rng = np.random.default_rng(3)
    a = _random_cmp_inputs(rng, 5)
    a["tp"][:] = 290.0
    a["wcu"][:] = 1e6
    a["dwcu"][:] = 0.0
    a["itmax"][:] = 4.0
    cp, cp1, cip, cgp, dg = ch.convective_microphysics(*a.values(), return_diag=True)
    assert (dg["exit1"] == 2).all() or (dg["exit1"] == 0).all()
    a["wmax"][:] = 1e9
    cp, cp1, cip, cgp, dg = ch.convective_microphysics(*a.values(), return_diag=True)
    assert (dg["exit1"] == 0).all() and (dg["nit1"] == 3).all()
    assert (cp1 > 0).all()


def _s_mcf(L, tl, pl, ccm, wcu, plemin, plel2, plemax, ccmmin, wcumin, lmin, lmax, ccmul, ccmul2, dts):
    rho = pl / (ch.RGAS * tl)
    f = ccmul * ccm / (rho * ch.GRAV * wcu * dts + ch.TEENY)
    if plemin - plel2 >= 450.:
        f = 5. * f
    if L < lmin:
        f = ccmul * ccmmin / (rho * ch.GRAV * wcumin * dts + ch.TEENY)
    if plemin - plemax < 450.:
        if L == lmax - 1:
            f = ccmul2 * ccm / (rho * ch.GRAV * wcu * dts + ch.TEENY)
        if L < lmin:
            f = 0.
    if f > 1.:
        f = 1.
    return f


def test_mcf_matches_scalar_transcription():
    rng = np.random.default_rng(1)
    n = 3000
    r = rng.uniform
    a = [rng.integers(1, 12, n).astype(float), r(200, 300, n), r(50, 1000, n), r(0, 20, n), r(0.01, 3, n),
         r(600, 1000, n), r(100, 1000, n), r(100, 900, n), r(0, 20, n), r(0.01, 3, n),
         rng.integers(1, 8, n).astype(float), rng.integers(1, 12, n).astype(float), np.full(n, 0.06), np.full(n, 0.12)]
    f = ch.mc_cloud_fraction(*a, dtsrc=1800.0)
    ref = np.array([_s_mcf(*[x[i] for x in a], 1800.0) for i in range(n)])
    assert np.array_equal(f, ref)
    assert (f == 1).sum() > 5 and (f == 0).sum() > 100 and ((f > 0) & (f < 1)).sum() > 1000


def _s_mpp(L, airm, fevap, lhp1, prcp, told, told1, vlat, cond, lmin, heat1, revp):
    if revp == 0:
        mc = 2. * fevap * airm if L <= lmin else 0.
    else:
        mc = 2. * fevap * airm
    if mc > airm:
        mc = airm
    lhp = lhp1
    if lhp == ch.LHS and told > ch.TF and told1 <= ch.TF:
        heat1 = heat1 + ch.LHM * prcp * ch.BYSHA
        lhp = ch.LHE
    if lhp == ch.LHE and told <= ch.TF and told1 > ch.TF:
        heat1 = heat1 - ch.LHM * prcp * ch.BYSHA
        lhp = ch.LHS
    if lhp != vlat and cond > 0:
        heat1 = heat1 + (vlat - lhp) * cond * ch.BYSHA
    return lhp, mc, heat1


@pytest.mark.parametrize("revp", [0, 1])
def test_mpp_matches_scalar_transcription(revp):
    rng = np.random.default_rng(2)
    n = 4000
    r = rng.uniform
    L = rng.integers(1, 10, n).astype(float)
    lmin = rng.integers(1, 10, n).astype(float)
    a = [L, r(10, 100, n), r(0, 0.8, n), rng.choice([ch.LHE, ch.LHS], n), r(0, 1e-3, n), r(268, 278, n),
         r(268, 278, n), rng.choice([ch.LHE, ch.LHS], n), r(0, 1e-3, n) * rng.integers(0, 2, n), lmin, r(-5, 5, n) * rng.integers(0, 2, n)]
    lhp, mc, h1 = ch.mc_precip_phase(*a, mc_revp_abv_cldbase=revp)
    ref = np.array([_s_mpp(*[x[i] for x in a], revp) for i in range(n)])
    assert np.array_equal(lhp, ref[:, 0]) and np.array_equal(mc, ref[:, 1]) and np.array_equal(h1, ref[:, 2])
    assert (mc == a[1]).sum() > 100                                   # MCLOUD>AIRM cap exercised here (not in real data)
    assert ((a[10] != 0) & (h1 != a[10])).sum() > 100                 # accumulation onto a nonzero HEAT1 entry value
    if revp == 0:
        assert (mc == 0).sum() > 1000                                 # AR5 below-cloud-base-only option


def test_anvil_hand_cases():
    # liquid branch: rcld = RCLDLX*100*(WTEM/(2*BY3*TWOPI*MCDNCW))**BY3
    rc, tau = ch.anvil_optical_thickness(ch.LHE, 1.0, 1.0, 100.0, 0.06, 1e9, 1.0, 0.5, 1e-3, 1e-3)
    assert abs(rc - 100.0 * (1e-3 / (2 * ch.BY3 * ch.TWOPI * 100.0)) ** ch.BY3) < 1e-12 * rc
    # ice branch capped by RIMAX, optical thickness capped at 100
    rc, tau = ch.anvil_optical_thickness(ch.LHS, 1.0, 1.0, 100.0, 0.06, 1.0, 1.0, 1e-6, 1e3, 1e-3)
    assert rc == 1.0 and tau == 100.0
    # tiny cloud fraction: +1E-20 (REAL(4) literal) keeps the division finite and below the cap
    rc, tau = ch.anvil_optical_thickness(ch.LHE, 1.0, 1.0, 100.0, 0.06, 1.0, 1.0, 0.0, 1e-30, 1e-3)
    assert np.isfinite(tau) and tau < 100.0
    assert ch._F1E20 != 1e-20 and abs(ch._F1E20 - 1e-20) / 1e-20 < 1e-7


def test_precip_mp_hand_values():
    # DC=0: exp(0)=1, bracket = 6/FLAM^4 -> RHO*pi/6*CN*6/FLAM^4 = RHO*pi*CN/FLAM^4
    v = float(ch.precip_mp(1000.0, 2000.0, 0.0, 8e6))
    assert abs(v - 1000.0 * math.pi * 8e6 / 2000.0 ** 4) < 1e-12 * v
    # decreases with the critical size
    assert float(ch.precip_mp(1000.0, 2000.0, 1e-3, 8e6)) < v
