"""D172: scoring of a model-produced month of AIJ columns against the real JAN1950 ensemble noise floor (F3 comparison tool).

Method (RADIATION_AND_F2_PLAN.md section 3.2, stage S2; all thresholds are PROPOSALS of that plan, none is a project decision, they are
parameters here and every statistic is reported so nothing depends on a silent threshold):

  monthly mean of an AIJ column = DIAG_PRT.f ij_mapk rule (f3_diagnostics.field_from_aij); cells where the ratio denominator is zero
  (pressure levels below ground) are undefined and excluded; a cell is used only when it is defined in the model month and in every
  reference member.

  Reference = M real members (D165: ctrl + 7 one-ulp perturbed, `ff_data/ens_jan1950/<m>/JAN1950.accP2SAoM40.nc`).  A new month x is
  one more realisation, so against the mean of M reference members its difference has variance sigma^2 (1 + 1/M); all statistics below
  are normalised by sd*sqrt(1+1/M) with the reference sample sd (ddof=1), which makes the null expectation of
    * the global-mean statistic  t_gm  a Student t with M-1 degrees of freedom,
    * each zonal-bin statistic   t_zon a Student t with M-1 degrees of freedom (neighbouring bins are correlated),
    * the rms ratios R_grid / R_zon  (rms of the difference over cells or bins, divided by sqrt(1+1/M) times the rms of the reference sd
      over the same cells, pooled so that E[R^2] = 1) equal to ~1.
  The plan's 'within 2 sigma' rule uses a sample sd from M=8 members, so even a true new member fails it more than 5 percent of the time
  (P(|t_7|>2) = 0.086; for M=7 reference members P(|t_6|>2) = 0.092): the leave-one-out run (`leave_one_out`) measures that directly.

Public functions: read_acc, monthly_fields, load_reference, score_month, leave_one_out, summarize, ModelMonth.from_f3acc.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f3_diagnostics as f3

ENS_DIR = f"{f3.FF}/ens_jan1950"
STORED = "/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/prod_runs/P2SAoM40/JAN1950.accP2SAoM40.nc"
PROD_DIR = os.path.dirname(STORED)
MEMBERS = ["ctrl", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
JM, IM = 46, 72


# ------------------------------------------------------------------------------------------------------ reading
def read_acc(path):
    """aij (KAIJ,JM,IM) float64, idacc, axyp (JM,IM) of a monthly acc file."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    out = dict(aij=np.asarray(d["aij"][:], dtype=np.float64), idacc=np.asarray(d["idacc"][:], dtype=np.int64),
               axyp=np.asarray(d["axyp"][:], dtype=np.float64) if "axyp" in d.variables else None)
    d.close()
    return out


def column_meta(path=STORED):
    """{col(1-based): dict(name, ia, scale, denom, denom_ia)} from a real acc file."""
    meta = f3.aij_names_from_nc(path)
    for c, m in meta.items():
        m["denom_ia"] = meta[m["denom"]]["ia"] if m["denom"] > 0 else 0
    return meta


def name_to_col(meta):
    """first column carrying each name (the F3 columns have unique names; checked by the tests for the ported ones)."""
    out = {}
    for c in sorted(meta):
        n = meta[c]["name"].strip()
        if n and n not in out:
            out[n] = c
    return out


def monthly_fields(aij, idacc, meta, cols):
    """monthly-mean maps (len(cols),JM,IM), nan where the denominator is zero.  aij: (KAIJ,JM,IM) or dict col->(JM,IM);
    only the columns in `cols` (1-based) are needed, plus the denominator columns they refer to."""
    get = (lambda c: aij[c - 1]) if not isinstance(aij, dict) else (lambda c: aij[c])
    res = np.full((len(cols), JM, IM), np.nan)
    for k, c in enumerate(cols):
        m = meta[c]
        a, b = _ratio(get, c, m, idacc)
        with np.errstate(all="ignore"):
            res[k] = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
    return res


def _ratio(get, c, m, idacc):
    teeny = 1e-30
    anum = get(c) * (m["scale"] / (idacc[m["ia"] - 1] + teeny))
    if m["denom"] > 0:
        adenom = get(m["denom"]) / (idacc[m["denom_ia"] - 1] + teeny)
    else:
        adenom = np.ones_like(anum)
    return anum, adenom


class ModelMonth:
    """A month produced by a model: aij columns as {col: (JM,IM) accumulated sums} (acc-file layout, columns not accumulated are
    absent) and the idacc counters (full array indexed ia-1, or the dict of F3Acc)."""

    def __init__(self, aij, idacc, axyp=None, label="model"):
        self.aij = aij
        if isinstance(idacc, dict):
            arr = np.zeros(max(max(idacc), 12), dtype=np.int64)
            for k, v in idacc.items():
                arr[k - 1] = v
            idacc = arr
        self.idacc = np.asarray(idacc)
        self.axyp = axyp
        self.label = label

    @classmethod
    def from_f3acc(cls, acc, axyp=None, label="model"):
        a, _ = acc.to_nc_layout()
        return cls({c: a[c - 1] for c in acc.aij}, acc.idacc, axyp, label)

    @classmethod
    def from_file(cls, path, label=None):
        r = read_acc(path)
        return cls(r["aij"], r["idacc"], r["axyp"], label or os.path.basename(os.path.dirname(path)))

    def available(self, cols):
        if isinstance(self.aij, dict):
            return [c for c in cols if c in self.aij]
        return list(cols)


def load_reference(cols, ens_dir=ENS_DIR, members=None, meta=None):
    """(X (M,len(cols),JM,IM) monthly means, axyp, member names) from the member acc files."""
    meta = meta or column_meta()
    members = members or [m for m in MEMBERS if os.path.exists(f"{ens_dir}/{m}/JAN1950.accP2SAoM40.nc")]
    X, axyp = [], None
    for m in members:
        r = read_acc(f"{ens_dir}/{m}/JAN1950.accP2SAoM40.nc")
        X.append(monthly_fields(r["aij"], r["idacc"], meta, cols))
        axyp = r["axyp"] if axyp is None else axyp
    return np.stack(X), axyp, members


# ------------------------------------------------------------------------------------------------------ statistics
def _wmean(x, w, ok):
    return (np.where(ok, x, 0.0) * w).sum(axis=(-2, -1)) / (ok * w).sum(axis=(-2, -1))


def t_two_sided_p(t, dof):
    from scipy import stats
    return 2.0 * stats.t.sf(np.abs(t), dof)


def score_field(x, X, w):
    """One field.  x (JM,IM) model monthly mean, X (M,JM,IM) reference members, w (JM,IM) area weights.  Returns a dict of statistics
    (see the module docstring).  Cells undefined (nan) in any member or in x are excluded."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _score_field(x, X, w)


def _score_field(x, X, w):
    M = X.shape[0]
    ok = np.isfinite(x) & np.isfinite(X).all(0)
    out = dict(n_cells=int(ok.sum()), M=M)
    if ok.sum() == 0:
        out["status"] = "undefined"
        return out
    Xo = np.where(ok, X, np.nan)
    mu = np.nanmean(Xo, 0)
    sd = np.nanstd(Xo, 0, ddof=1)
    f = np.sqrt(1.0 + 1.0 / M)
    # --- global mean
    gm_ref = _wmean(X, w, ok)
    gm_x = float(_wmean(x, w, ok))
    gm_mu, gm_sd = float(gm_ref.mean()), float(gm_ref.std(ddof=1))
    d = gm_x - gm_mu
    out.update(gm_model=gm_x, gm_ref_mean=gm_mu, gm_ref_sd=gm_sd, gm_diff=d)
    out["t_gm"] = _safe_div(d, gm_sd * f)
    out["p_gm"] = float(t_two_sided_p(out["t_gm"], M - 1)) if np.isfinite(out["t_gm"]) else (1.0 if d == 0 else 0.0)
    # --- zonal mean (46 bins)
    zx = np.nanmean(np.where(ok, x, np.nan), axis=1)
    zX = np.nanmean(Xo, axis=2)                                   # (M,JM)
    zmu, zsd = zX.mean(0), zX.std(0, ddof=1)
    zd = zx - zmu
    tz = np.array([_safe_div(a, b * f) for a, b in zip(zd, zsd)])
    okz = np.isfinite(zd)
    out["zon_n"] = int(okz.sum())
    out["zon_t_max"] = float(np.nanmax(np.abs(tz[okz]))) if okz.any() else np.nan
    out["zon_frac_within2"] = float(np.mean(np.abs(tz[okz]) <= 2.0)) if okz.any() else np.nan
    out["zon_n_beyond4"] = int(np.sum(np.abs(tz[okz]) > 4.0))
    rs = np.sqrt(np.nanmean(zsd[okz] ** 2)) * f
    out["R_zon"] = _safe_div(float(np.sqrt(np.nanmean(zd[okz] ** 2))), float(rs))
    out["zon_rms_diff"], out["zon_sd_rms"] = float(np.sqrt(np.nanmean(zd[okz] ** 2))), float(np.sqrt(np.nanmean(zsd[okz] ** 2)))
    # --- gridpoint
    dd = np.where(ok, x - mu, 0.0)
    rms_d = float(np.sqrt((dd ** 2 * w).sum() / (ok * w).sum()))
    rms_sd = float(np.sqrt((np.where(ok, sd, 0.0) ** 2 * w).sum() / (ok * w).sum()))
    out["grid_rms_diff"], out["grid_sd_rms"] = rms_d, rms_sd
    out["R_grid"] = _safe_div(rms_d, rms_sd * f)
    # spatial scale for context
    out["spatial_std"] = float(np.sqrt(_wmean((mu - _wmean(mu, w, ok)) ** 2, w, ok)))
    out["status"] = "ok"
    return out


def _safe_div(a, b):
    if b > 0:
        return float(a / b)
    return 0.0 if abs(a) <= 0.0 else float("inf")


def score_month(model, cols, X=None, axyp=None, meta=None, names=None):
    """Score the model month against the reference.  cols: 1-based AIJ columns (only those the model accumulated are scored; the rest
    are listed in 'skipped').  X: reference monthly means (M,len(cols),JM,IM) (default: loaded from the member files)."""
    meta = meta or column_meta()
    have = model.available(cols)
    skipped = [c for c in cols if c not in have]
    if X is None:
        X, axyp0, _ = load_reference(cols, meta=meta)
        axyp = axyp if axyp is not None else axyp0
    axyp = axyp if axyp is not None else (model.axyp if model.axyp is not None else np.ones((JM, IM)))
    xm = monthly_fields(model.aij, model.idacc, meta, have)
    idx = [cols.index(c) for c in have]
    rows = []
    for k, c in enumerate(have):
        r = score_field(xm[k], X[:, idx[k]], axyp)
        r.update(col=c, name=meta[c]["name"].strip())
        rows.append(r)
    return dict(rows=rows, skipped=skipped, M=X.shape[0], label=model.label)


def leave_one_out(cols, members=None, ens_dir=ENS_DIR, meta=None, X=None, axyp=None):
    """Each member scored against the other M-1 (the validation of the scoring method before a model month exists).
    Returns dict(members, per_member{name: rows}, cols, M_ref)."""
    meta = meta or column_meta()
    if X is None:
        X, axyp, members = load_reference(cols, ens_dir, members, meta)
    members = members or MEMBERS[:X.shape[0]]
    per = {}
    for k, m in enumerate(members):
        rest = np.delete(X, k, axis=0)
        rows = []
        for j, c in enumerate(cols):
            r = score_field(X[k, j], rest[:, j], axyp)
            r.update(col=c, name=meta[c]["name"].strip())
            rows.append(r)
        per[m] = rows
    return dict(members=list(members), per_member=per, cols=list(cols), M_ref=X.shape[0] - 1)


def summarize(rows, M, loo_R=None, t_alpha=0.05):
    """Aggregate over fields: counts of |t_gm| beyond the t critical value (expected fraction t_alpha under the null, fields are
    correlated so the count is not binomial), zonal-bin criteria, rms-ratio distribution.  loo_R: dict name -> median LOO R_grid
    for the ratio-to-members criterion."""
    from scipy import stats
    ok = [r for r in rows if r.get("status") == "ok"]
    tcrit = float(stats.t.ppf(1 - t_alpha / 2, M - 1))
    tg = np.array([abs(r["t_gm"]) for r in ok])
    out = dict(n_fields=len(ok), M=M, t_crit=tcrit, n_gm_beyond_tcrit=int((tg > tcrit).sum()), frac_gm_beyond_tcrit=float((tg > tcrit).mean()),
               expected_frac=t_alpha, max_abs_t_gm=float(tg.max()),
               median_R_grid=float(np.median([r["R_grid"] for r in ok])), max_R_grid=float(np.max([r["R_grid"] for r in ok])),
               median_R_zon=float(np.median([r["R_zon"] for r in ok])), max_R_zon=float(np.max([r["R_zon"] for r in ok])),
               zon_frac_within2_mean=float(np.mean([r["zon_frac_within2"] for r in ok])),
               zon_frac_within2_min=float(np.min([r["zon_frac_within2"] for r in ok])),
               n_fields_zon_max_t_beyond4=int(sum(r["zon_n_beyond4"] > 0 for r in ok)),
               p_expected_within2=float(1 - 2 * stats.t.sf(2.0, M - 1)))
    if loo_R:
        rat = [r["R_grid"] / loo_R[r["name"]] for r in ok if r["name"] in loo_R and loo_R[r["name"]] > 0]
        out["max_ratio_to_loo_R_grid"] = float(np.max(rat)) if rat else None
    return out


def verdict(summ, ratio_limit=1.5):
    """The plan's proposed criteria (RADIATION_AND_F2_PLAN.md 3.2) evaluated on a summary, each reported separately and not combined:
    (a) >= 95 percent of zonal bins within 2 sigma and none beyond 4 (mean over fields, count of fields with a bin beyond 4);
    (b) global means within the t critical value for the expected fraction of fields; (c) rms ratio to the members' own <= ratio_limit."""
    return dict(
        a_zonal_mean_frac_within2_ge_0p95=bool(summ["zon_frac_within2_mean"] >= 0.95),
        a_fields_with_bin_beyond4=summ["n_fields_zon_max_t_beyond4"],
        b_gm_frac_beyond_tcrit=summ["frac_gm_beyond_tcrit"], b_expected=summ["expected_frac"],
        c_max_ratio_to_loo=summ.get("max_ratio_to_loo_R_grid"),
        c_within_limit=(None if summ.get("max_ratio_to_loo_R_grid") is None else bool(summ["max_ratio_to_loo_R_grid"] <= ratio_limit)))


# ------------------------------------------------------------------------------------------------------ calibration against leave-one-out
SUMMARY_STATS = ("frac_gm_beyond_tcrit", "max_abs_t_gm", "median_R_grid", "max_R_grid", "median_R_zon", "max_R_zon",
                 "zon_frac_within2_mean", "n_fields_zon_max_t_beyond4")


def loo_summaries(loo):
    """summary (see `summarize`) of every leave-one-out member, M = reference size used for that scoring."""
    return {m: summarize(rows, loo["M_ref"]) for m, rows in loo["per_member"].items()}


def compare_to_loo(summ, loo_summ, stats=SUMMARY_STATS):
    """Where a model-month summary sits relative to the leave-one-out members' summaries (the empirical null of the method with this
    reference size): for each statistic the model value, the LOO min/median/max, and 'inside' = min <= value <= max.  For the statistics
    where larger means worse the flag 'above_max' is the one that matters (zon_frac_within2_mean: 'below_min').  No threshold is
    invented: a model month whose statistics all sit inside the LOO range is indistinguishable from one more real member at this
    sample size; the range itself comes from 8 members only."""
    out = {}
    for s in stats:
        v = np.array([x[s] for x in loo_summ.values()], dtype=float)
        mv = summ[s]
        out[s] = dict(model=mv, loo_min=float(v.min()), loo_median=float(np.median(v)), loo_max=float(v.max()),
                      inside=bool(v.min() <= mv <= v.max()),
                      worse_than_all_loo=bool(mv < v.min()) if s == "zon_frac_within2_mean" else bool(mv > v.max()))
    return out


def per_field_table(rows, keys=("name", "col", "gm_model", "gm_ref_mean", "gm_ref_sd", "t_gm", "R_zon", "R_grid", "zon_frac_within2",
                                "zon_t_max", "grid_rms_diff", "grid_sd_rms", "spatial_std")):
    return [{k: r.get(k) for k in keys} for r in rows if r.get("status") == "ok"]


def flagged_fields(rows, alpha=0.05):
    """Per-field global-mean flags: fields with p_gm < alpha (expected fraction alpha under the null, the fields are correlated) and the
    Bonferroni-level subset p_gm < alpha / n_fields.  The aggregated LOO comparison has little power for one deviating field (a 0.2 K
    t_500 offset, ~4.4 sigma of a new member, leaves every aggregate inside the LOO range), so this per-field list is the primary output."""
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    flagged = sorted([r for r in ok if r["p_gm"] < alpha], key=lambda r: r["p_gm"])
    return dict(n_fields=n, alpha=alpha, n_flagged=len(flagged), expected_flagged=alpha * n,
                bonferroni=[(r["name"], r["t_gm"]) for r in flagged if r["p_gm"] < alpha / max(n, 1)],
                flagged=[(r["name"], r["t_gm"], r["p_gm"]) for r in flagged])
