"""Tests for dyn_filter_ff.py (SLP filter FILTER, SLP, SHAP1D, isotropslp, MAtoPMB, getTotalEnergy,
addEnergyAsDiffuseHeat; D101/D102) against real-Fortran dumps (ffd_filt_*, ffd_slp_*), 3 dates x 6 steps.
Dump-based tests skip if the dumps are absent.  Tests labelled HAND / NOT VALIDATED exercise branches that never
occur in the 18 recorded calls (SLP exp branch, FILTER clip limits, shap1 n>1 in isotropslp); they check against
hand-derived/independent formulas, not against the real Fortran."""
import math
import numpy as np
import pytest

import dyn_filter_ff as ff
import dyn_filter_compare as cm
import intel_libm_ff
from dyn_filter_ff import IM, JM, LM
from dyn_filter_compare import DATES, NSTEP

HAVE = all(cm.available(d) for d, _ in DATES)
needs_dump = pytest.mark.skipif(not HAVE, reason="ff_data ffd_filt dumps not present on this host")
needs_imf = pytest.mark.skipif(not intel_libm_ff.available(), reason="Intel libimf not available")
CALLS = [(d, it0 + k) for d, it0 in DATES for k in range(NSTEP)]

KEYS_EXACT = ['shap1d', 'isotropslp', 'rowloop', 'te1_keb', 'te1_kea', 'te1_pe', 'te1_te', 'te1_tot',
              'te2_keb', 'te2_kea', 'te2_pe', 'te2_te', 'te2_tot', 'eadd_ediff', 'eadd_t', 'eadd_de_matches']
KEYS_POW = ['slp_x', 'slp_y', 'pedn1_chain', 'matop_pedn', 'matop_pmid', 'matop_pk', 'matop_masum',
            'out_pedn', 'out_pmid', 'out_pk', 'out_ma', 'out_masum', 'out_t', 'out_q', 'out_qcl', 'out_qci',
            'out_qmom', 'out_e0', 'out_e1']


def _g(date="nov26"):
    return cm.load_g(date)


# ---------------------------------------------------------------- real-dump tests
@needs_dump
@pytest.mark.parametrize("date,itime", CALLS)
def test_stage_and_energy_functions_bitwise(date, itime):
    r = cm.run_call(date, itime, imf_pow=False)
    for k in KEYS_EXACT:
        assert r[k] == 0.0, (k, r[k])
    # eadd_de_matches: recorded DELTAENERGY equals te2.total - te1.total exactly


@needs_dump
@pytest.mark.parametrize("date,itime", CALLS)
def test_numpy_pow_within_1ulp_scale(date, itime):
    """numpy `**` (default): the libimf pow differs by 1 ulp in a few cells: bound the effect."""
    r = cm.run_call(date, itime, imf_pow=False)
    assert r['slp_x'] <= 4e-13 and r['slp_y'] <= 4e-16
    assert r['out_pedn'] <= 1e-12 and r['out_pk'] <= 4e-15 and r['out_masum'] <= 2e-11
    assert r['out_t'] <= 1e-13 and r['out_q'] <= 1e-16 and r['out_qmom'] <= 1e-16
    assert r['out_e1'] <= 1e-9 and r['out_e0'] == 0.0


@needs_dump
@needs_imf
@pytest.mark.parametrize("date,itime", CALLS)
def test_full_chain_bitwise_with_libimf_pow(date, itime):
    r = cm.run_call(date, itime, imf_pow=True)
    for k in KEYS_EXACT + KEYS_POW + ['slp_nbad']:
        assert r[k] == 0.0, (k, r[k])
    assert r['n_t_exact'] == r['n_total']


@needs_dump
@pytest.mark.parametrize("date", [d for d, _ in DATES])
def test_consts_sane(date):
    g = _g(date)
    assert (g['im'], g['jm'], g['lm'], g['nmom']) == (72, 46, 40, 9)
    assert g['mfiltr'] == 1.0 and g['cos_limit'] == 0.15 and g['dt'] == 450.0
    assert (g['lmfrac1'], g['lmfrac2']) == (1, 40) or g['lmfrac1'] < g['lmfrac2']
    assert np.all(g['mfrac'][g['lmfrac1'] - 1:g['lmfrac2']] > 0)
    assert g['byim'] == 1.0 / 72.0
    # imaxj: poles have one cell
    assert g['imaxj'][0] == 1 and g['imaxj'][JM - 1] == 1 and np.all(g['imaxj'][1:JM - 1] == 72)
    assert abs(g['areag'] - 4 * np.pi * g['radius'] ** 2) / g['areag'] < 1e-12


@needs_dump
@pytest.mark.parametrize("date,itime", CALLS[::6])
def test_non_vacuous(date, itime):
    r = cm.run_call(date, itime)
    assert r['chg_pedn1'] > 0.1                    # surface pressure is changed by > 0.1 mb somewhere
    assert r['chg_t'] > 1e-6 and r['chg_q'] > 1e-12
    assert abs(r['ediff']) > 1e-6                  # the energy fix is a real (tiny) nonzero shift
    c = cm.load_call(date, itime)
    assert np.max(np.abs(c['te2']['tot'] - c['te1']['tot'])) > 0
    assert np.max(np.abs(c['te1']['keb'][:, 1:])) > 0 and np.max(np.abs(c['te1']['pe'])) > 0
    assert np.max(np.abs(c['slp']['x2'][:, 1:-1] - c['slp']['x1'][:, 1:-1])) > 1e-3      # SHAP1D acts
    assert np.max(np.abs(c['slp']['x3'][:, 1:3] - c['slp']['x2'][:, 1:3])) > 1e-6        # isotropslp acts


@needs_dump
def test_fast_row_loop_equals_loop():
    g = _g(); c = cm.load_call("dec01", 33552); rin = c['inp']
    a = ff.slp_filter_pedn(rin['pedn1'], rin['tsavg'], g)
    b = ff.slp_filter_pedn_fast(rin['pedn1'], rin['tsavg'], g)
    assert np.array_equal(a, b)


@needs_dump
def test_row_mass_conserved_and_poles_untouched():
    g = _g(); c = cm.load_call("nov26", 33312); rin = c['inp']; rs = c['slp']
    pn = rs['pedn4']; po = rin['pedn1']
    for j in range(1, JM - 1):
        assert abs(np.sum(pn[:, j]) - np.sum(po[:, j])) < 1e-9 * np.sum(po[:, j])
    assert np.array_equal(pn[:, 0], po[:, 0]) and np.array_equal(pn[:, JM - 1], po[:, JM - 1])


@needs_dump
def test_total_energy_invariant_after_fix_is_small():
    """The diffuse-heat fix restores TE to the pre-filter value up to the energy of the heat added in the
    pole-row cells (informational bound: |te2 - te1| << te1, and equals the recorded delta exactly)."""
    c = cm.load_call("nov26", 33313)
    assert abs(c['te2']['tot'] - c['te1']['tot']) < 1e-3 * abs(c['te1']['tot'])
    assert (c['te2']['tot'] - c['te1']['tot']) == c['eadd']['de']


@needs_dump
def test_layers_outside_mfrac_range_unchanged():
    g = _g(); c = cm.load_call("jan01", 17520)
    l1, l2 = g['lmfrac1'] - 1, g['lmfrac2'] - 1
    rin, ro = c['inp'], c['out']
    # MA for layers with MFRAC == 0 is not rewritten
    for l in range(LM):
        if g['mfrac'][l] == 0:
            assert np.array_equal(ro['ma'][l][:, 1:JM - 1], rin['ma'][l][:, 1:JM - 1])
    # q only scaled (not heated) in the filtered layers, rows 1 and JM untouched
    assert np.array_equal(ro['q'][:, 0, :], rin['q'][:, 0, :]) and np.array_equal(ro['q'][:, JM - 1, :], rin['q'][:, JM - 1, :])


# ---------------------------------------------------------------- mutation tests (must be detected)
@needs_dump
def test_mutation_shapiro_order_detected(monkeypatch):
    c = cm.load_call("nov26", 33312); rs = c['slp']
    good = ff.shap1d(rs['x1'])
    assert np.array_equal(good[:, 1:-1], rs['x2'][:, 1:-1])
    assert not np.array_equal(ff.shap1d(rs['x1'], norder=6)[:, 1:-1], rs['x2'][:, 1:-1])


@needs_dump
def test_mutation_clip_bounds_and_pdif_detected():
    g = _g(); c = cm.load_call("nov26", 33312); rs, rin = c['slp'], c['inp']
    ref = rs['pedn4'][:, 1:-1]
    assert np.array_equal(ff.row_loop(rs['x3'], rs['y1'], rin['pedn1'], g)[:, 1:-1], ref)
    g2 = dict(g, byim=g['byim'] * 1.001)                       # wrong BYIM (mass-conservation term)
    assert not np.array_equal(ff.row_loop(rs['x3'], rs['y1'], rin['pedn1'], g2)[:, 1:-1], ref)
    # no conservation at all
    pn = np.where(np.arange(JM)[None, :] >= 0, rs['x3'] / rs['y1'], 0)
    assert not np.array_equal(pn[:, 1:-1], ref)


@needs_dump
def test_mutation_energy_variants_detected():
    g = _g(); t = cm.load_call("dec01", 33552)['te1']
    assert ff.total_energy(t['masum'], t['ma'], t['pk'], t['t'], t['u'], t['v'], g) == t['tot']
    # wrong globalsum order (reversed J): differs
    kea, _ = ff.conserv_ke(t['ma'], t['u'], t['v'], g)
    pe = ff.conserv_pe(t['masum'], t['t'], t['pk'], t['ma'], g)
    te = ((kea + pe) * g['axyp']) / g['areag']
    rev = float(ff.seqsum(ff.seqsum(te, axis=0)[::-1], axis=0))
    assert rev != t['tot']
    # PE sum in reverse layer order differs in at least one cell
    tt = t['t'].transpose(2, 0, 1)
    s_fwd = ff.seqsum((tt * t['pk']) * t['ma'], axis=0)
    s_rev = ff.seqsum(((tt * t['pk']) * t['ma'])[::-1], axis=0)
    assert not np.array_equal(s_fwd, s_rev)
    # PE without the MTOP term, KE without regrid weights
    g2 = dict(g, mtop=0.0)
    assert not np.array_equal(ff.conserv_pe(t['masum'], t['t'], t['pk'], t['ma'], g2), t['pe'])


@needs_dump
def test_mutation_eadd_sign_and_psf_detected():
    g = _g(); ea = cm.load_call("nov26", 33312)['eadd']
    tn, e = ff.add_energy_as_diffuse_heat(ea['de'], ea['t_in'], ea['pk'], g)
    assert np.array_equal(tn, ea['t_out']) and e == ea['ediff']
    tn2, _ = ff.add_energy_as_diffuse_heat(-ea['de'], ea['t_in'], ea['pk'], g)
    assert not np.array_equal(tn2, ea['t_out'])
    g3 = dict(g, pmtop=g['pmtop'] + 1.0)
    assert ff.add_energy_as_diffuse_heat(ea['de'], ea['t_in'], ea['pk'], g3)[1] != ea['ediff']


@needs_dump
def test_mutation_regrid_pole_rows_detected():
    g = _g(); t = cm.load_call("nov26", 33312)['te1']
    keb = t['keb'].copy(); keb[:, 0] = 0.0
    a = ff.regrid_btoa_ext(keb, g) * g['byaxyp']
    assert np.array_equal(a, t['kea'])
    g2 = dict(g, dxyv=g['dxyv'] * 1.0000001)
    assert not np.array_equal(ff.regrid_btoa_ext(keb, g2) * g['byaxyp'], t['kea'])


# ---------------------------------------------------------------- branch coverage inventory
@needs_dump
def test_branch_coverage_inventory():
    """Counts, over the 18 recorded calls, of the branches reached.  Documents what is NOT exercised."""
    st = {}
    for d, it in CALLS:
        cm.run_call(d, it, stats=st)
    for k in ('zs0', 'beta_lapse', 'tasn_warm', 'tasn_cold', 'pow_branch'):
        assert st[k] > 0, k
    assert st['exp_branch'] == 0           # SLP exp() branch (BETA<=1e-6) never reached: HAND test below
    assert st['clip_lo'] == 0 and st['clip_hi'] == 0   # FILTER +-1.18 % clip never reached: HAND test below
    assert all(n == 1 for _, n in st['iso_rows'])      # shap1 n>1 in isotropslp never reached: HAND test below
    assert sorted({j for j, _ in st['iso_rows']}) == [2, 3, 44, 45]


# ---------------------------------------------------------------- HAND-derived / NOT validated against real Fortran
def _g_hand():
    return dict(bmoist=0.0065, grav=9.80665, rgas=287.05, by3=1. / 3., cos_limit=0.15, dt=450.0,
                cosp=np.zeros(JM), dxp=np.ones(JM), byim=1. / 72., mtop=0., mfixs=0.)


def test_hand_slp_zs0_is_identity():
    g = _g_hand()
    ps = np.array([980., 1013.25]); tas = np.array([288., 250.])
    assert np.array_equal(ff.slp(ps, tas, np.zeros(2), g), ps)


def test_hand_slp_standard_lapse_pow_branch():
    """TAS=288, ZS=1000: TSL=294.5>290.5 and TAS<290.5 -> BETA=(290.5-288)/1000; TAS>=255 -> TASn=TAS.
    SLP = PS*(1+BETA*ZS/TAS)**(GRAV/(RGAS*BETA)).  NOT validated against real Fortran (independent math)."""
    g = _g_hand()
    ps, tas, zs = 900., 288., 1000.
    beta = (290.5 - tas) / zs
    ref = ps * (1. + beta * zs / tas) ** (g['grav'] / (g['rgas'] * beta))
    got = float(ff.slp(np.array([ps]), np.array([tas]), np.array([zs]), g)[0])
    assert abs(got - ref) < 1e-12 * ref


def test_hand_slp_cold_and_warm_tasn():
    g = _g_hand()
    # cold: TAS=240 (<255): TASn=.5*(255+240)=247.5, BETA=BMOIST (TSL=246.5+... below 290.5)
    ps, tas, zs = 800., 240., 1500.
    ref = ps * (1. + g['bmoist'] * zs / (0.5 * (255. + tas))) ** (g['grav'] / (g['rgas'] * g['bmoist']))
    assert abs(float(ff.slp(np.array([ps]), np.array([tas]), np.array([zs]), g)[0]) - ref) < 1e-12 * ref
    # warm: TAS=300 (>290.5), TSL>290.5 -> TASn=.5*(290.5+300)
    tas = 300.
    ref = ps * (1. + g['bmoist'] * zs / (0.5 * (290.5 + tas))) ** (g['grav'] / (g['rgas'] * g['bmoist']))
    assert abs(float(ff.slp(np.array([ps]), np.array([tas]), np.array([zs]), g)[0]) - ref) < 1e-12 * ref


def test_hand_slp_exp_branch_never_exercised_in_real_run():
    """BETA<=1e-6: TAS just below 290.5 with TSL>290.5.  SLP = PS*exp((1-.5*BZBYT+BZBYT**(2*by3))*GBYRB*BZBYT)
    (the source's `BZBYT**(2.*by3)` is reproduced as written).  NOT validated against real Fortran."""
    g = _g_hand()
    ps, zs = 950., 800.
    tas = 290.5 - 1e-4                      # BETA = 1e-4/800 = 1.25e-7 < 1e-6
    beta = (290.5 - tas) / zs
    assert beta <= 1e-6
    bz = beta * zs / tas
    gb = g['grav'] / (g['rgas'] * beta)
    ref = ps * math.exp((1. - 0.5 * bz + bz ** (2. * g['by3'])) * gb * bz)
    st = {}
    got = float(ff.slp(np.array([ps]), np.array([tas]), np.array([zs]), g, stats=st)[0])
    assert st['exp_branch'] == 1 and st['pow_branch'] == 0
    assert abs(got - ref) < 1e-12 * ref


def test_hand_shap1d_properties():
    """SHAP1D(8): a constant row is unchanged; the Nyquist mode (-1)^i is annihilated to 0 (the 8th-difference
    operator times 4**-8 on the Nyquist mode is (4/4)**8*... = identity -> x - x = 0); a zonal wave of
    wavenumber 1 is damped by (1 - sin^16(pi k/IM)) (8 passes of -4 sin^2).  NOT validated against real Fortran."""
    x = np.zeros((IM, JM)); x[:, 5] = 3.5
    assert np.array_equal(ff.shap1d(x)[:, 5], x[:, 5])
    nyq = np.zeros((IM, JM)); nyq[:, 5] = (-1.0) ** np.arange(IM)
    assert np.max(np.abs(ff.shap1d(nyq)[:, 5])) < 1e-12
    k = 1
    wave = np.zeros((IM, JM)); wave[:, 5] = np.cos(2 * np.pi * k * np.arange(IM) / IM)
    out = ff.shap1d(wave)[:, 5]
    expect = (1.0 - np.sin(np.pi * k / IM) ** 16) * wave[:, 5]
    assert np.max(np.abs(out - expect)) < 1e-12
    # rows 1 and JM (Fortran J=1, JM) are not filtered
    y = np.random.default_rng(0).standard_normal((IM, JM))
    assert np.array_equal(ff.shap1d(y)[:, 0], y[:, 0]) and np.array_equal(ff.shap1d(y)[:, JM - 1], y[:, JM - 1])


def _shap1_scalar(x, fac):
    """Direct transcription of Fortran shap1 (ATMDYN.f:1934-1955) with scalar loops."""
    x = list(map(float, x)); im = len(x)
    n = int(fac) + 1
    facby4 = fac * .25 / n
    for _ in range(n):
        x1 = x[0]; xim1 = x[im - 1]
        for i in range(im - 1):
            xi = x[i]
            x[i] = x[i] + facby4 * (xim1 - xi - xi + x[i + 1])
            xim1 = xi
        i = im - 1
        x[i] = x[i] + facby4 * (xim1 - x[i] - x[i] + x1)
    return np.array(x)


def test_hand_isotropslp_multi_subiteration():
    """Larger DT forces n=int(fac)+1 > 1 sub-iterations (never reached with DT=450).  NOT validated against
    real Fortran: compared with a scalar transcription of shap1."""
    g = _g_hand()
    g['cosp'] = np.full(JM, 0.05); g['dxp'] = np.full(JM, 1.0e4)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((IM, JM)) * 10 + 1000
    dt = 450.0 * 9
    fac = 1e3 * dt / (g['dxp'][0] ** 2)
    out = ff.isotropslp(x, g, dt=dt)
    assert int(fac) + 1 >= 1
    for j in range(1, JM - 1):
        assert np.array_equal(out[:, j], _shap1_scalar(x[:, j], fac))
    # make n>1: fac >= 1
    dt2 = 450.0 * 1e3
    fac2 = 1e3 * dt2 / (g['dxp'][0] ** 2)
    assert int(fac2) + 1 > 1
    out2 = ff.isotropslp(x, g, dt=dt2)
    assert np.array_equal(out2[:, 3], _shap1_scalar(x[:, 3], fac2))
    # far-from-pole rows are untouched
    g['cosp'] = np.full(JM, 0.5)
    assert np.array_equal(ff.isotropslp(x, g, dt=dt2), x)


def test_hand_row_loop_clip_limits_and_conservation():
    """x/y far from PEDNOLD: clipped to [0.9882,1.0118]*PEDNOLD (never reached in the real windows), then each
    row is shifted by PDIF so its sum equals the old sum (to rounding).  NOT validated against real Fortran."""
    g = _g_hand()
    rng = np.random.default_rng(2)
    po = 1000. + 5 * rng.standard_normal((IM, JM))
    y = np.ones((IM, JM))
    x = po * (1.0 + 0.05 * rng.standard_normal((IM, JM)))    # +-5 % sigma: many hits on both limits
    st = {}
    pn = ff.row_loop(x, y, po, g, stats=st)
    assert st['clip_lo'] > 0 and st['clip_hi'] > 0
    for j in range(1, JM - 1):
        assert abs(np.sum(pn[:, j]) - np.sum(po[:, j])) < 1e-9 * np.sum(po[:, j])
        # before the shift every value was inside the limits; the shift is small relative to 1.18 %
    assert np.array_equal(pn[:, 0], po[:, 0]) and np.array_equal(pn[:, JM - 1], po[:, JM - 1])
    # exact clip arithmetic in one cell: x = 1.3*po -> v = 1.0118*po, then minus PDIF
    po2 = np.full((IM, JM), 1000.); x2 = po2.copy(); x2[0, 5] = 1300.; y2 = np.ones((IM, JM))
    p2 = ff.row_loop(x2, y2, po2, g)
    psumn = 71 * 1000. + 1.0118 * 1000.
    pdif = (psumn - 72 * 1000.) * g['byim']
    assert p2[0, 5] == 1.0118 * 1000. - pdif
    # lower limit
    x2[0, 5] = 500.
    p3 = ff.row_loop(x2, y2, po2, g)
    psumn = 71 * 1000. + 0.9882 * 1000.
    assert p3[0, 5] == 0.9882 * 1000. - (psumn - 72 * 1000.) * g['byim']


def test_hand_add_energy_uniform_shift():
    """T(:,:,L) -= EDIFF/PK(L): with PK==1 every T changes by exactly EDIFF; EDIFF=DELTA/((PSF-PMTOP)*SHA*MB2KG)."""
    g = dict(psf=984., pmtop=2., sha=1004.64, mb2kg=100. / 9.80665)
    t = np.full((IM, JM, LM), 300.); pk = np.ones((LM, IM, JM))
    tn, e = ff.add_energy_as_diffuse_heat(5.0, t, pk, g)
    assert e == 5.0 / ((984. - 2.) * 1004.64 * (100. / 9.80665))
    assert np.array_equal(tn, 300. - e * np.ones_like(t))


def test_hand_regrid_conserves_constant_field():
    """A B-grid field equal to c everywhere: interior A-grid value is (c+c)*RAPVS + (c+c)*RAPVN = 2c*(rapvs+rapvn).
    NOT validated against real Fortran (the real-dump tests do validate the whole function)."""
    g = dict(rapvs=np.full(JM, .3), rapvn=np.full(JM, .2), dxyp=np.ones(JM), dxyv=np.ones(JM), byim=1. / 72.)
    x = np.full((IM, JM), 7.0)
    o = ff.regrid_btoa_ext(x, g)
    assert np.allclose(o[:, 1:JM - 1], 2 * 7.0 * (.3 + .2))
    assert np.allclose(o[:, 0], 7.0) and np.allclose(o[:, JM - 1], 7.0)
