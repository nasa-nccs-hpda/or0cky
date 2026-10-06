"""Tests for atm_step.py (D128/D129): one 30-minute source step chained from the validated ports, against the real-Fortran dumps
(ffa_step_<itime>_{a,r,d,e} written by instrumentation/ATM_DRV_atmstep.f.patch and the older boundary dumps).

All dump-based tests are skipped when ff_data is absent.  Cheap tests (seconds): dump self-consistency, radiation glue, DISSIP+FILTER
replay on all 18 steps, input reconstruction of the SURFACE PBL records, first-layer glue, mutation checks.  Medium (about 1-2 min):
SURFACE replay with the recorded land patch (jit compile).  Slow (about 5 min, only with ATM_STEP_SLOW=1): the whole chain including the
CONDSE port, step 0 of nov26, in the libimf mode.  Bitwise assertions need the Intel libimf (skipped without it).
"""
import os

import numpy as np
import pytest

import atm_step as A
import intel_libm_ff

STEPS = [(d, it0 + k) for d, it0 in A.DATES for k in range(6)]
HAVE = all(A.have_dumps(d, it) for d, it in STEPS)
NEED = pytest.mark.skipif(not HAVE, reason="ff_data atmosphere step dumps (ffa_step_*, ffp/ffs/ffl/ffg/fft, ffc_cse_*) not present")
NO_IMF = not intel_libm_ff.available()
NEEDI = pytest.mark.skipif(not HAVE or NO_IMF, reason="needs the ff_data dumps and the Intel libimf runtime")
SLOW = pytest.mark.skipif(os.environ.get('ATM_STEP_SLOW') != '1' or not HAVE or NO_IMF, reason="set ATM_STEP_SLOW=1 (needs dumps + libimf)")
_CTX, _R = {}, {}


def ctx(date, imf):
    if (date, imf) not in _CTX:
        _CTX[(date, imf)] = A.make_ctx(date, imf=imf)
    return _CTX[(date, imf)]


def real(date, it):
    if (date, it) not in _R:
        _R[(date, it)] = A.Real(date, it)
    return _R[(date, it)]


def is_rad_step(date, it):
    return (it - 16032) % 5 == 0


# ----------------------------------------------------------------------------- the new instrumentation is self-consistent
@NEED
@pytest.mark.parametrize("date,it", STEPS)
def test_step_start_dump_equals_dynamics_start_state(date, it):
    R = real(date, it)
    s1 = R.s1
    for k_a, k_s in (('U', 'u'), ('V', 'v'), ('T', 't'), ('Q', 'q'), ('QCL', 'qcl'), ('QCI', 'qci'), ('MA', 'ma'), ('PEDN', 'pedn'),
                     ('PK', 'pk'), ('P', 'p'), ('TMOM', 'tmom'), ('QMOM', 'qmom')):
        assert np.array_equal(R.a[k_a], s1[k_s]), k_a


@NEED
@pytest.mark.parametrize("date,it", [x for x in STEPS if x[1] != dict(A.DATES)[x[0]] + 5])
def test_step_end_dump_equals_next_step_start(date, it):
    """End of atm_phase2 (ffa_step_e) == start of the next step (ffd_state s1): the end-of-step reference is the real next state."""
    R, R2 = real(date, it), real(date, it + 1)
    for k_e, k_s in (('U', 'u'), ('V', 'v'), ('T', 't'), ('Q', 'q'), ('QCL', 'qcl'), ('QCI', 'qci'), ('MA', 'ma'), ('PEDN', 'pedn'),
                     ('P', 'p'), ('TMOM', 'tmom'), ('QMOM', 'qmom')):
        assert np.array_equal(R.e[k_e], R2.s1[k_s]), k_e


@NEED
@pytest.mark.parametrize("date,it", STEPS)
def test_radiation_library_runs_only_every_fifth_step(date, it):
    """SRHR/TRHR (SOCRATES output stored in RAD_COM) change between step start and the end of phase 1 exactly on the radiation steps
    MOD(Itime-ItimeI,NRAD)==0 (ITIMEI=16032, NRAD=5); COSZ1 is recomputed every step."""
    R = real(date, it)
    changed = bool((R.a['SRHR'] != R.r['SRHR']).any() or (R.a['TRHR'] != R.r['TRHR']).any())
    assert changed == is_rad_step(date, it)
    assert (R.a['COSZ1'] != R.r['COSZ1']).any()


# ----------------------------------------------------------------------------- radiation glue (recorded SOCRATES output)
@NEED
@pytest.mark.parametrize("date,it", STEPS)
def test_radiation_temperature_update_is_bitwise(date, it):
    """T after RADIA = T after CONDSE + (SRHR*COSZ1+TRHR)*DTsrc*bysha*byMA/PK (RAD_DRV.f:5474-5478), from the REAL post-CONDSE T and the
    recorded SRHR/TRHR/COSZ1: bitwise equal to the real post-RADIA T (no libm functions involved)."""
    c = ctx(date, False)
    R = real(date, it)
    S, sn = A.run_step(date, it, c, start='radia', stop='radia', R=R)
    st = A.field_stats(sn['radia']['T'], R.r['T'])
    assert st['cat'] == 'A', st
    assert A.field_stats(R.cse_out['T'], R.r['T'])['max_abs'] > 1e-2       # non-vacuous: radiation changes T by up to ~0.7 K per step


@NEED
@pytest.mark.parametrize("date,it", [x for x in STEPS if x[1] - 16032 >= 0 and x[1] != dict(A.DATES)[x[0]] + 5])
def test_radia_cloud_masking_reproduces_the_next_step_cloud_arrays(date, it):
    """RAD_DRV.f:2611-2615: on radiation steps CLDSS/CLDMC are zeroed where TAUSS/TAUMC <= taulim after CONDSE; on the other steps they
    are unchanged: the real CONDSE exit arrays, so processed, equal the arrays at the next step's CONDSE entry bit for bit."""
    R, Rn = real(date, it), real(date, it + 1)
    o = R.cse_out
    if A.is_radiation_step(it):
        cl, cm = A.radia_cloud_masking(o['CLDSS'], o['CLDMC'], o['TAUSS'], o['TAUMC'])
        assert (cl != o['CLDSS']).any() or (cm != o['CLDMC']).any()          # non-vacuous
    else:
        cl, cm = o['CLDSS'], o['CLDMC']
    assert np.array_equal(cl, Rn.cse_in['CLDSS']) and np.array_equal(cm, Rn.cse_in['CLDMC'])


@NEED
def test_radiation_mutation_is_detected():
    date, it = 'nov26', 33312
    c, R = ctx(date, False), real(date, it)
    S = A.real_state_at(R, 'radia', c)
    S['SRHR'], S['TRHR'], S['COSZ1'] = R.r['SRHR'], R.r['TRHR'], R.r['COSZ1']
    good = A.radia_apply(S['T'], S['SRHR'], S['TRHR'], S['COSZ1'], S['MA'], S['PK'], c)
    assert A.field_stats(good, R.r['T'])['max_abs'] == 0.0
    bad = A.radia_apply(S['T'], S['SRHR'], S['TRHR'] * 1.001, S['COSZ1'], S['MA'], S['PK'], c)          # 0.1 % error in the long-wave term
    assert A.field_stats(bad, R.r['T'])['max_abs'] > 1e-5
    bad = A.radia_apply(S['T'], S['SRHR'], S['TRHR'], S['COSZ1'] * 0.0, S['MA'], S['PK'], c)           # zenith angle dropped
    assert A.field_stats(bad, R.r['T'])['max_abs'] > 1e-3


# ----------------------------------------------------------------------------- dynamics -> CONDSE entry
@NEEDI
@pytest.mark.parametrize("date", [d for d, _ in A.DATES])
def test_chained_dynamics_builds_the_real_condse_entry_state_bitwise(date):
    c = ctx(date, True)
    it = dict(A.DATES)[date]
    R = real(date, it)
    S = A.init_state(R)
    A.stage_dyn(S, R, c)
    inp = A.condse_inputs(S, R)
    for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'TMOM', 'QMOM', 'PK', 'PMID', 'PEDN', 'PDSIG', 'PMIDOLD', 'GZ', 'MWS', 'PEK', 'UKM', 'VKM',
              'UKMSP', 'VKMSP', 'UKMNP', 'VKMNP', 'EGCM', 'W2GCM', 'PBLHT', 'DCLEV', 'PBLPTOP', 'TSAVG', 'QSAVG'):
        assert np.array_equal(inp[k], R.cse_in[k]), k


# ----------------------------------------------------------------------------- DISSIP + FILTER (real post-SURFACE state)
@NEEDI
@pytest.mark.parametrize("date,it", STEPS)
def test_dissip_and_filter_replay_bitwise_with_libimf(date, it):
    c = ctx(date, True)
    R = real(date, it)
    S, sn = A.run_step(date, it, c, start='dissip', stop='filter', R=R)
    assert A.field_stats(sn['dissip']['T'], R.d['T'])['cat'] == 'A'
    st = A.compare_state(sn['filter'], A.end_reference(R), ['T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'PDSIG', 'PEK', 'P', 'QMOM', 'TMOM'])
    bad = {k: v for k, v in st.items() if v['cat'] != 'A'}
    assert not bad, bad


@NEED
@pytest.mark.parametrize("date,it", [x for x in STEPS if x[1] == dict(A.DATES)[x[0]] or x[1] == dict(A.DATES)[x[0]] + 3])
def test_dissip_and_filter_replay_rounding_level_without_libimf(date, it):
    c = ctx(date, False)
    R = real(date, it)
    S, sn = A.run_step(date, it, c, start='dissip', stop='filter', R=R)
    st = A.compare_state(sn['filter'], A.end_reference(R), ['T', 'Q', 'QCL', 'QCI', 'MA', 'PEDN', 'PMID', 'PK', 'P', 'QMOM', 'TMOM'])
    assert all(v['cat'] in 'AB' for v in st.values()), st


@NEED
def test_dissip_filter_mutations_are_detected():
    date, it = 'nov26', 33312
    c, R = ctx(date, False), real(date, it)
    e = A.end_reference(R)
    S, sn = A.run_step(date, it, c, start='dissip', stop='dissip', R=R)
    assert A.field_stats(sn['dissip']['T'], R.d['T'])['cat'] == 'A'
    assert A.field_stats(R.post_surface()['T'], R.d['T'])['max_abs'] > 1e-6          # DISSIP changes T (non-vacuous)
    assert A.field_stats(R.d['T'], e['T'])['max_abs'] > 1e-6                          # FILTER changes T
    assert A.field_stats(R.d['U'], R.post_surface()['U'])['max_abs'] == 0.0           # DISSIP changes only T
    S2, sn2 = A.run_step(date, it, c, start='filter', stop='filter', R=R)
    S2['PEDN'][0, 3, 3] += 1e-3                                                       # perturbed surface pressure
    A.stage_filter(S2, c)
    assert A.field_stats(S2['PEDN'], e['PEDN'])['max_abs'] > 1e-5


# ----------------------------------------------------------------------------- SURFACE glue
def test_first_layer_update_formulas():
    rng = np.random.default_rng(1)
    tmom = rng.normal(size=(9, A.IM, A.JM, A.LM)); qmom = rng.normal(size=(9, A.IM, A.JM, A.LM))
    t1 = 250 + 40 * rng.random((A.IM, A.JM)); q1 = 1e-2 * rng.random((A.IM, A.JM)); pk1 = 0.9 + 0.1 * rng.random((A.IM, A.JM))
    dth1 = rng.normal(size=(A.IM, A.JM)) * 1e-2; dq1 = -1e-3 * rng.random((A.IM, A.JM)); dq1[0, 0] = 1e-4
    tm, qm = A.first_layer_update(tmom, qmom, dth1, dq1, t1, q1, pk1)
    ft = np.where(dth1 * t1 < 0, -dth1 / (t1 * pk1), 0.0)
    assert np.array_equal(tm[:, :, :, 0], tmom[:, :, :, 0] * (1 - ft)[None]) and np.array_equal(tm[:, :, :, 1:], tmom[:, :, :, 1:])
    fq = np.where((dq1 < 0) & (q1 > 0), -dq1 / q1, 0.0)
    exp = qmom[:, :, :, 0] * (1 - fq)[None]
    small = (q1 + dq1) < 1e-12
    exp[:, small] = 0.0
    assert np.array_equal(qm[:, :, :, 0], exp) and np.array_equal(qm[:, :, :, 1:], qmom[:, :, :, 1:])
    assert small.sum() == 0 or (qm[:, small, 0] == 0).all()
    q1b = q1.copy(); q1b[1, 1] = 1e-14; dq1b = dq1.copy(); dq1b[1, 1] = -2e-14                  # Q+DQ1 < qmin: moments zeroed
    qm2 = A.first_layer_update(tmom, qmom, dth1, dq1b, t1, q1b, pk1)[1]
    assert (qm2[:, 1, 1, 0] == 0).all()


@NEEDI
@pytest.mark.parametrize("date", [d for d, _ in A.DATES])
def test_surface_pbl_inputs_rebuilt_from_chained_state_are_bitwise(date):
    """Every atmosphere-dependent column of the substep-1 PBL records (layer-1 scalars, get_dbl, PGRAD_PBL exports, CONDSE downdraft
    exports, surface pressure, dtdt_gcm) rebuilt from the REAL post-RADIA state equals the recorded record column bit for bit."""
    c = ctx(date, True)
    it = dict(A.DATES)[date]
    R = real(date, it)
    S = A.real_state_at(R, 'surface', c)
    rec = A.surface_records(R)
    atm1 = A.atm_layout(S, c)
    pa, pb = rec['pa'], rec['pb']
    coriol = np.zeros(atm1['T'].shape[:2])
    coriol[pb[:, 1].astype(int) - 1, pb[:, 0].astype(int) - 1] = pb[:, 36]
    cell = A.cell_inputs(S, atm1, c, S['USTARPBL'].T, S['LMONINPBL'].T, S['PBLHT'].T, S['DCLEV'].T, coriol, S['T1AA'].T, S['U1AA'].T, S['V1AA'].T)
    for rows in (pa[pa[:, 2] <= 2], pa[pa[:, 2] == 3], pa[pa[:, 2] == 4]):
        o = A.override_pbl(rows, cell)
        for nm, col in list(A.PBL_COLS.items()) + [('gusti_out', A.PBL_GUSTI_OUT)]:
            assert np.array_equal(o[:, col], rows[:, col]), nm
    # mutation: a 1e-6 relative error in the layer-1 temperature changes tkv/zs1/dtdt (detected)
    S2 = dict(S); S2['T'] = S['T'] * (1 + 1e-6)
    cell2 = A.cell_inputs(S2, A.atm_layout(S2, c), c, S['USTARPBL'].T, S['LMONINPBL'].T, S['PBLHT'].T, S['DCLEV'].T, coriol, S['T1AA'].T, S['U1AA'].T, S['V1AA'].T)
    o2 = A.override_pbl(pa[pa[:, 2] <= 2], cell2)
    assert not np.array_equal(o2[:, A.PBL_COLS['tkv']], pa[pa[:, 2] <= 2][:, A.PBL_COLS['tkv']])


@NEEDI
def test_surface_replay_with_recorded_land_is_at_rounding_level_and_mutations_show():
    """Real post-RADIA state -> both SURFACE substeps (PBL, tiles, land-ice, aggregation, first-layer TMOM/QMOM update, ATURB + velocity
    diffusion) with the recorded land patch.  Measured (nov26 step 0): T 8.5e-13 abs on 451 K, U/V 1.1e-12 on 88 m/s, Q 3e-15, EGCM 5.6e-12 on
    8.1, TMOM 1.1e-15, QMOM 1.6e-17.  Bounds below are ~5x those."""
    date, it = 'nov26', 33312
    c, R = ctx(date, True), real(date, it)
    S, sn = A.run_step(date, it, c, start='surface', stop='surface', R=R, land_mode='recorded')
    ps = R.post_surface()
    for k, tol in (('T', 5e-12), ('U', 5e-12), ('V', 5e-12), ('Q', 2e-14)):
        st = A.field_stats(sn['surface'][k], ps[k])
        assert st['max_abs'] < tol, (k, st)
    assert A.field_stats(sn['surface']['TMOM'], R.e['TMOM'])['max_abs'] < 1e-14
    assert A.field_stats(sn['surface']['QMOM'], R.filt_in()['qmom'])['max_abs'] < 1e-15
    assert A.field_stats(sn['surface']['EGCM'], R.e['EGCM'])['max_abs'] < 3e-11
    # non-vacuity / mutations: the stage changes T,U,V,TMOM by far more than the residual
    S0 = A.real_state_at(R, 'surface', c)
    assert A.field_stats(S0['T'], ps['T'])['max_abs'] > 1e-2
    assert A.field_stats(S0['U'], ps['U'])['max_abs'] > 1e-2
    assert A.field_stats(S0['TMOM'], R.e['TMOM'])['max_abs'] > 1e-4          # first-layer TMOM update matters (omitting it fails)


# ----------------------------------------------------------------------------- whole chain (slow)
@SLOW
def test_whole_chain_step0_nov26_with_condse_port_recorded_land():
    """dyn -> CONDSE port -> radiation -> SURFACE (recorded land) -> DISSIP -> FILTER from the real step-start state, libimf mode.
    Measured: 28 of 30 end-state fields A/B; W2GCM 1.7e-12 and LMONINPBL 1.7e-10 scale-relative (C).  Gate bound: every field <= 1e-9."""
    date, it = 'nov26', 33312
    c, R = ctx(date, True), real(date, it)
    S, sn = A.run_step(date, it, c, R=R, land_mode='recorded', ms={})
    st = A.compare_state(sn['filter'], A.end_reference(R), A.END_FIELDS)
    for k, v in st.items():
        assert v['rel'] < 1e-9, (k, v)
    gate = [k for k in ('U', 'V', 'T', 'Q', 'QCL', 'QCI', 'MA', 'TMOM', 'QMOM', 'EGCM', 'PBLHT') if st[k]['cat'] not in 'AB']
    assert not gate, gate
