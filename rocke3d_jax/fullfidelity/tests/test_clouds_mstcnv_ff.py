"""Tests for clouds_mstcnv_ff.py (CLOUDS2.F90 MSTCNV) -- D110-D113.

Real-Fortran validation uses ff_data/<date>/ffc_mc_cols_*.bin and ffc_mc_ck_*.bin (skipped if absent).  Dump tests use an
evenly spaced subset of the boundary records (the full-population results are in the ledger entry; run
`python3 clouds_mstcnv_compare.py [--imf]` for all of them).  libimf tests are skipped without the Intel runtime.
Mutation tests change one piece of the port (or a tunable) and require the real-dump comparison to FAIL, i.e. they
prove the comparison is not vacuous for that piece."""
import os
import sys
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clouds_mstcnv_compare as cmp  # noqa: E402
import clouds_mstcnv_io as io  # noqa: E402
import clouds_mstcnv_ff as m  # noqa: E402

DATES = [d for d, _ in io.DATES]
HAVE = {d: os.path.exists(f"{io.FF_DEFAULT}/{d}/ffc_mc_consts.txt") for d in DATES}
needs_dumps = pytest.mark.skipif(not all(HAVE.values()), reason="ff_data ffc_mc_* dumps not present on this host")
needs_imf = pytest.mark.skipif(not m.imf_available(), reason="Intel libimf not available")
NSUB = 90
_cache = {}


def _data(date):
    if date not in _cache:
        d = io.load_cols(date)
        h = d["hdr"]
        n = h["conv"].size
        conv = np.nonzero(h["conv"] > 0)[0]
        non = np.nonzero(h["conv"] == 0)[0]
        sub = np.concatenate([conv[np.linspace(0, conv.size - 1, NSUB).astype(int)],
                              non[np.linspace(0, non.size - 1, 6).astype(int)] if non.size else non])
        _cache[date] = (d, io.load_consts(date), np.unique(sub), n)
    return _cache[date]


def _run(date, mut=None, tune_over=None, idx=None):
    d, cs, sub, _ = _data(date)
    tune = m.tune_from_consts(cs)
    tune.update(tune_over or {})
    out = []
    for i in (sub if idx is None else idx):
        o = m.mstcnv_column(cmp.record(d, i), tune, mut=mut)
        out.append((i, o))
    return d, out


def _dev(date, **kw):
    """-> (max over fields of max|port - real| / max|real| (field scale over the subset), counted over records WITHOUT
    a discrete threshold flip; number of bitwise-identical records; number of records; number of flip records)."""
    d, res = _run(date, **kw)
    h = d["hdr"]
    mx, nbit, nflip = 0.0, 0, 0
    scale = {k: max(float(np.max(np.abs(d["out"][k][[i for i, _ in res]]))), 1e-300) for k in d["out"]}
    for i, o in res:
        same = True
        fl = bool(cmp.discrete_flips(o, d["out"], i))
        nflip += fl
        for name, _, _ in io.OUT_FIELDS:
            if name in io.STALE_IF_NONCONV and h["conv"][i] == 0:
                continue
            a, b = np.asarray(o[name], float), d["out"][name][i]
            if not np.array_equal(a, b):
                same = False
                if not fl:
                    mx = max(mx, float(np.max(np.abs(a - b))) / scale[name])
        nbit += same
    return mx, nbit, len(res), nflip


# ---------------------------------------------------------------------------------- real-dump validation
@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_boundary_numpy_mode(date):
    """numpy/glibc mode: decision path identical; most records bitwise; threshold flips on cancellation residues and
    ill-conditioned cloud-top (WCU2 ~ 0) quantities are rare and reported, smooth fields within 1e-8."""
    d, res = _run(date)
    n, nbit, nbig = len(res), 0, 0
    smooth = ("sm", "qm", "smom", "qmom", "um", "vm", "tdnl", "qdnl", "prcpmc", "dgdsm", "dgdqm", "mcflx")
    scale = {k: max(float(np.max(np.abs(d["out"][k][[i for i, _ in res]]))), 1e-300) for k in d["out"]}
    for i, o in res:
        same = True
        for name, _, _ in io.OUT_FIELDS:
            if name in io.STALE_IF_NONCONV and d["hdr"]["conv"][i] == 0:
                continue
            a, b = np.asarray(o[name], float), d["out"][name][i]
            if not np.array_equal(a, b):
                same = False
                dev = float(np.max(np.abs(a - b))) / scale[name]
                nbig += dev > 1e-6
                if name in smooth:
                    assert dev < 1e-8, (name, i, dev)
        nbit += same
    assert nbit > 0.7 * n
    assert nbig <= 0.10 * n


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_decision_path_matches(date):
    d, res = _run(date)
    h = d["hdr"]
    for i, o in res:
        for k in ("lmcmin", "lmcmax", "mccont", "ierr", "lerr"):
            assert o[k] == d["out"][k][i], (date, i, k)
        assert (o["lmcmax"] > 0) == (h["conv"][i] > 0)


@needs_dumps
@needs_imf
@pytest.mark.parametrize("date", DATES)
def test_boundary_bitwise_with_libimf(date):
    """With exp/pow routed through the real build's Intel libimf every checked record is bitwise identical."""
    m.set_backend("imf")
    try:
        mx, nbit, n, nflip = _dev(date)
    finally:
        m.set_backend("numpy")
    assert mx < 1e-11 and nflip == 0 and nbit >= 0.97 * n   # observed ~99.8 % bitwise, rest <= 5e-13


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_checkpoints_follow_same_path(date):
    """Stage checkpoints (1-9): same event sequence for the first 12 dumped calls; every field within 1e-9 of the
    field scale (numpy mode), exact for the stages that do not use exp/pow (1,3,6)."""
    r = cmp.compare_ck(date, cmax=12)
    assert r["seq_bad"] == 0 and r["noinp"] == 0 and r["nev"] > 100
    a = r["agg"]
    ill = {"cldmcl", "cldslwij", "clddepij", "taumcl", "csizel", "precnvl"}   # amplified near WCU2 ~ 0 in numpy mode
    for k in a.n:
        if k[1] not in ill:
            assert a.rel(k) < 1e-6, (date, k, a.rel(k))
    assert all(a.bit[k] == a.n[k] for k in a.n if k[0] == 3)


@needs_dumps
def test_nonconvective_records_untouched():
    """Hand-derived property: if no base is unstable, nothing is changed (SM, QM, moments, momentum, TL) and no
    convective output is produced; the real dump agrees."""
    d, res = _run("nov26")
    h = d["hdr"]
    seen = 0
    for i, o in res:
        if h["conv"][i] == 0:
            seen += 1
            assert o["lmcmax"] == 0 and o["lmcmin"] == 0 and o["mccont"] == 0
            r = cmp.record(d, i)
            assert np.array_equal(o["sm"], r["sm"]) and np.array_equal(o["qm"], r["qm"])
            assert np.array_equal(o["smom"], r["smom"]) and np.array_equal(o["um"], r["um"])
            assert np.array_equal(o["tl"], r["tl"])
            assert not o["taumcl"].any() and not o["cldmcl"].any() and o["prcpmc"] == 0 and o["wmsum"] == 0
            assert np.all(o["fssl"] == 1.0)
    assert seen >= 3


@needs_dumps
@pytest.mark.parametrize("date", DATES)
def test_momentum_column_integral_conserved(date):
    """The momentum 'adjustment' step forces the column sum of UM/VM over LDMIN..LMAX to be unchanged by every
    convective event, so the whole-column sum is conserved to round-off (checked on the REAL outputs too)."""
    d, res = _run(date)
    h = d["hdr"]
    for i, o in res:
        if h["conv"][i] == 0:
            continue
        r = cmp.record(d, i)
        for k in ("um", "vm"):
            for src, nm in ((o[k], "port"), (d["out"][k][i], "real")):
                s0, s1 = r[k].sum(axis=1), src.sum(axis=1)
                assert np.all(np.abs(s1 - s0) <= 1e-9 * np.maximum(np.abs(r[k]).sum(axis=1), 1e-30)), (nm, k, i)


@needs_dumps
def test_nonvacuous_coverage():
    """The subset really exercises the code: convective fraction, both plume types, partition 2, downdrafts with
    entrainment and detrainment, deep (anvil/precip partition) and shallow plumes, plume mass cap."""
    br = defaultdict(int)
    for date in DATES:
        d, cs, sub, _ = _data(date)
        tune = m.tune_from_consts(cs)
        for i in sub:
            m.mstcnv_column(cmp.record(d, i), tune, br=br)
    for k in ("base_pass", "area_partition_2", "downdraft_active", "dd_entrain", "dd_detrain_env", "plume_mass_cap",
              "entrain_applied", "detrain_applied", "deep_precip_partition", "optical_anvil", "downdraft_trigger",
              "cond_applied", "dd_detrain_buoyant", "dd_exit_buoyant"):
        assert br[k] > 0, k


# ---------------------------------------------------------------------------------- mutation checks
MUTS = [
    ("pgrad_double", dict(mut=dict(pgrad_double=True))),
    ("f95_double", dict(mut=dict(f95_double=True))),
    ("thetav_old", dict(tune_over=dict(mc_new_ddrft_thetav=0))),
    ("entr_mass_lim_base", dict(tune_over=dict(mc_entr_mass_lim_plume=0))),
    ("fddrt_1", dict(tune_over=dict(mc_fddrt=1.0))),
    ("contce1_changed", dict(tune_over=dict(entrainment_cont1=0.41))),
    ("contce2_changed", dict(tune_over=dict(entrainment_cont2=0.61))),
    ("u00b_changed", dict(tune_over=dict(u00b=0.61))),
    ("radiusl_changed", dict(tune_over=dict(radiusl_multiplier=1.0))),
    ("revp_abv_cldbase_0", dict(tune_over=dict(mc_revp_abv_cldbase=0))),
]


@needs_dumps
@pytest.mark.parametrize("name,kw", MUTS, ids=[x[0] for x in MUTS])
def test_mutation_detected(name, kw):
    """Each mutated port fails the real comparison (deviation >> the 1e-9 noise level) on at least one date."""
    if name == "pgrad_double":
        worst = 0.0
        for d in DATES:
            dd, res = _run(d, **kw)
            sc = {f: max(float(np.max(np.abs(dd["out"][f][[i for i, _ in res]]))), 1e-300) for f in ("um", "vm")}
            for i, o in res:
                for f in ("um", "vm"):
                    worst = max(worst, float(np.max(np.abs(o[f] - dd["out"][f][i]))) / sc[f])
        assert worst > 2e-10, (name, worst)         # numpy-mode noise on um/vm is ~3e-15
        return
    worst = max(_dev(d, **kw)[0] for d in DATES)
    assert worst > 1e-8, (name, worst)


@needs_dumps
def test_mutation_ksub_forced_1():
    """A convective call with a 2-sub-step subsidence (ksub=2; ~0.6 % of calls) found in the checkpoint dumps: forcing
    ksub=1 breaks the real comparison by >1e-5 of the field scale while the unmutated port matches."""
    d = "nov26"
    dd, cs, _, _ = _data(d)
    key = {(int(dd["hdr"]["itime"][i]), int(dd["hdr"]["i"][i]), int(dd["hdr"]["j"][i]), int(dd["hdr"]["ncall"][i])): i
           for i in range(dd["hdr"]["conv"].size)}
    idx = [key[(b["itime"], b["i"], b["j"], b["ncall"])] for b in io.load_ck(d)
           if any(e["stage"] == 6 and e["f"]["ksub"][0] == 2 for e in b["events"])]
    assert idx
    tune = m.tune_from_consts(cs)
    for i in idx:
        r = cmp.record(dd, i)
        base = m.mstcnv_column(r, tune)
        mut = m.mstcnv_column(r, tune, mut=dict(ksub1=True))
        sc = np.max(np.abs(base["sm"]))
        assert np.max(np.abs(base["sm"] - dd["out"]["sm"][i])) < 1e-8 * sc
        assert np.max(np.abs(mut["sm"] - dd["out"]["sm"][i])) > 1e-5 * sc


@needs_dumps
def test_mutation_noqlimit_inactive_in_data():
    """limitq (positivity limiter for Q in the subsidence advection) never changes anything in these windows:
    removing it leaves the real comparison unchanged (so its branches are NOT validated by real data)."""
    for d in DATES:
        assert _dev(d, mut=dict(noqlimit=True))[0] < 1e-6


# ---------------------------------------------------------------------------------- small unit checks
def test_real4_literals_are_not_double():
    for x in (0.001, 0.95, 0.7, 0.33, 0.05, 1e-20):
        assert float(np.float32(x)) != x
    assert m.f4(0.5) == 0.5 and m.f4(0.25) == 0.25 and m.f4(100.0) == 100.0


def test_layout_sizes():
    assert io.total(io.IN_FIELDS) == 1974 and io.total(io.OUT_FIELDS) == 2616
    assert {k: io.total(v) for k, v in io.CK_FIELDS.items()} == {1: 2, 2: 3, 3: 6, 4: 3821, 5: 2249, 6: 1206,
                                                                  7: 884, 8: 1370, 9: 124}
