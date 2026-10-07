"""D163 tests: the F3 AIJ-style diagnostics (f3_diagnostics.py) against the real model's own accumulated diagnostics for the nov26 54-step window.

Reference: ff_data/nov26_day/real_acc54_nov26.npz = (acc of the real model after 54 steps from the nov26 restart) - (acc stored in the restart),
produced by running the real (instrumented-for-radiation-server, otherwise unmodified physics) binary for 54 steps (see the ledger entry D163).
Skipped when that file, the day dumps, the radiation packets or the libimf bridge are absent.  The window run takes about 1-2 min.
Tolerance: relative to the column's own maximum absolute increment, 1e-12 (observed worst 5.3e-14: float64 summation-order roundoff); it is not
loosened by any test.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import f3_diagnostics as fd  # noqa: E402

DAY = f"{fd.FF}/{fd.DAY}"
ACC_NC = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc"
TOL = 1e-12


def _have_window():
    try:
        import intel_libm_ff
        if not intel_libm_ff.available():
            return False
    except Exception:
        return False
    need = [fd.REAL_ACC, f"{DAY}/ffd_aflux_geom.bin", f"{DAY}/ffd_filt_consts.bin"]
    for it in (fd.IT0, fd.IT0 + 53):
        need += [f"{DAY}/ffa_step_{it}_a.bin", f"{DAY}/ffa_step_{it}_r.bin", f"{DAY}/ffa_step_{it}_e.bin", f"{DAY}/ffc_cse_out_{it}.bin",
                 f"{DAY}/ffa_{it}_c1_out.bin", f"{DAY}/ffa_{it}_c2_out.bin", f"{DAY}/ffd_state_{it}_s1.bin"]
    need += [f"{DAY}/rsv_n26_{fd.IT0 + 5 * k}_out.bin" for k in range(11)]
    return all(os.path.exists(p) for p in need)


NEED_WINDOW = pytest.mark.skipif(not _have_window(), reason="real 54-step acc reference / day dumps / radiation packets / libimf not present")
NEED_NC = pytest.mark.skipif(not os.path.exists(ACC_NC), reason="real monthly acc file not present")


# ------------------------------------------------------------------ no-data tests (sampling rules, tables)
def test_surface_sampling_rule_is_one_in_three_substeps():
    """SURFACE.f:386: MODDSF=MOD(NIsurf*ITime+NS-1,3)==0: 36 of the 108 substeps of the window, pattern ns=1,2,none by itime mod 3."""
    n = sum(len(fd.F3Acc.surface_samples(it)) for it in range(fd.IT0, fd.IT0 + 54))
    assert n == 36
    assert fd.F3Acc.surface_samples(33312) == [1] and fd.F3Acc.surface_samples(33313) == [2] and fd.F3Acc.surface_samples(33314) == []


def test_diaga_fires_four_times_in_the_window():
    """MODDA<2 of the even pass (ATMDYN.f:352), NDAA=13, NIdyn=4: real idacc(ia_dga) rose 88 -> 92 over the window."""
    assert fd.diaga_fire_steps() == [33312, 33326, 33339, 33353]


def test_column_table_is_consistent():
    cols = list(fd.AIJ_COLS.values())
    assert len(cols) == len(set(cols))
    for n in fd.PORTED_NAMES:
        assert n in fd.AIJ_COLS
    assert fd.AIJ_COLS["t_500"] == 167 and fd.AIJ_COLS["omega_p5"] == 313 and fd.AIJ_COLS["srnf_toa"] == 377


def test_pole_rows_accumulate_only_i1():
    assert fd.VALID[0, 0] and not fd.VALID[1, 0] and not fd.VALID[1, fd.JM - 1] and fd.VALID[:, 1:fd.JM - 1].all()


def test_field_from_aij_scaling_and_denominator():
    aij = np.zeros((20, 2, 2)); aij[4] = 10.0; aij[9] = 4.0
    meta = dict(ia=4, scale=2.0, denom=10, denom_ia=1)
    an, ad = fd.field_from_aij(aij[4], [10, 0, 0, 5], meta, aij)
    assert np.allclose(an, 10.0 * 2.0 / 5) and np.allclose(ad, 4.0 / 10)
    assert np.allclose(fd.global_mean(an, ad, np.ones((2, 2))), an[0, 0] / ad[0, 0])


@NEED_NC
def test_column_names_match_real_acc_file():
    nm = fd.aij_names_from_nc(ACC_NC)
    for n, c in fd.AIJ_COLS.items():
        assert nm[c]["name"] == n, (n, c, nm[c]["name"])
    for n, c in fd.AIJL_COLS.items():
        import netCDF4 as nc
        d = nc.Dataset(ACC_NC)
        names = [b"".join(x).decode().strip() for x in d["sname_aijl"][:]]
        assert names[c - 1] == n


# ------------------------------------------------------------------ window validation against the real accumulation
@pytest.fixture(scope="module")
def window():
    acc = fd.run_window(log=lambda *a: None)
    return acc, fd.real_window()


@NEED_WINDOW
def test_idacc_counters_match_real(window):
    acc, real = window
    d = real["idacc_delta"]
    assert [acc.idacc[k] for k in (1, 2, 3, 4)] == [int(d[0]), int(d[1]), int(d[2]), int(d[3])] == [54, 11, 36, 4]


@NEED_WINDOW
def test_every_ported_column_matches_real_window(window):
    acc, real = window
    a, _ = acc.to_nc_layout()
    n_checked = 0
    for c in sorted(acc.aij):
        r = real["daij"][c - 1]
        s = np.abs(r).max()
        d = np.abs(a[c - 1] - r).max()
        assert d <= TOL * max(s, 1e-300) if s > 0 else d == 0, (c, d, s)
        n_checked += s > 0
    assert n_checked >= 250


@NEED_WINDOW
@pytest.mark.parametrize("name", ["prsurf", "slp", "qatm", "t_850", "t_500", "t_200", "z_500", "u_200", "v_200", "omega_500", "rh_layer1",
                                  "prec", "evap", "tsurf", "qsurf", "trdn_surf", "tauus", "incsw_toa", "srnf_toa", "trnf_toa", "srnf_grnd"])
def test_f3_key_fields_match_real_window(window, name):
    acc, real = window
    a, _ = acc.to_nc_layout()
    c = fd.AIJ_COLS[name]
    s = np.abs(real["daij"][c - 1]).max()
    assert s > 0
    assert np.abs(a[c - 1] - real["daij"][c - 1]).max() <= TOL * s


@NEED_WINDOW
def test_aijl_columns_match_real_window(window):
    acc, real = window
    _, al = acc.to_nc_layout()
    for n, c in fd.AIJL_COLS.items():
        s = np.abs(real["daijl"][c - 1]).max()
        assert s > 0
        assert np.abs(al[c - 1] - real["daijl"][c - 1]).max() <= TOL * s, n


@NEED_WINDOW
def test_radia_site_covers_the_radiation_columns(window):
    """every column the real RADIA wrote in the window (84 of the 483 that changed) is reproduced from the server output."""
    acc, real = window
    import radiation_server as rs
    tot = sum(rs.read_packet(f"{DAY}/rsv_n26_{fd.IT0 + 5 * k}_out.bin")["AIJD"] for k in range(11))
    cols = [c for c in range(1, 1661) if np.any(tot[:, :, c - 1])]
    assert len(cols) == 84
    a, _ = acc.to_nc_layout()
    for c in cols:
        r = real["daij"][c - 1]
        s = np.abs(r).max()
        assert np.abs(a[c - 1] - r).max() <= TOL * max(s, 1e-300)
