"""R7 trial integration (nov26, radiation steps 33312-33357): NativeHandoff plugged into the other session's jitted hand-off step
(jax_radiation.make_handoff_step, unchanged) instead of the server callback.  Per radiation step: recorded state of the step (T, Q, clouds, PK..),
live surface fields, carry (SNOAGE, RQT, KLIQ) taken from the HELD outputs of the previous radiation call (as a free run), then the four
non-radiation steps that follow (held SRHR/TRHR with the recorded per-step COSZ1).  Checks: the 21 outputs against the recorded server output,
T_new against the recorded T, the held-branch T against an independent evaluation, the number of compilations, no server process.
  python r6_trial_integration.py products.npz static0.npz specdir out.json"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import clouds_jax_env  # noqa: F401  XLA:CPU flags (no FMA contraction); must precede the jax import
import numpy as np
import jax
import jax.numpy as jnp
import jax_radiation as JR                      # imports clouds_jax_env first (XLA flags) via the project modules
import jax_radiation_check as CK
import radiation_server as rs
import clouds_condse_io as cio
import dyn_step as ds
import atm_step as A
import radia_jax as RJ
import radia_native as RN
import drv_zenith as Z

D = rs.FF + "/nov26_day"
ITIMES = list(range(33312, int(os.environ.get("R6_LAST", 33357)) + 1, 5))      # R6_LAST=33362 crosses midnight (nov27 01:00)
z = np.load(sys.argv[1])
prod = RJ.Products(z, np.load(os.environ["R6_STATIC"]) if "R6_STATIC" in os.environ else None)
day_kw = {}
if os.environ.get("R6_DAYS"):          # run directory of the rundeck: day products from the input files (daily_products.py), selected by itime // 48
    import daily_products as DP
    dp = DP.DailyProducts(os.environ["R6_DAYS"], prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust,
                          solar_constant=prod.solar_constant)
    dlist, day0 = DP.day_products_for_run(dp, prod, ITIMES[0], ITIMES[-1] + 4)
    day_kw = dict(day_products=dlist, day0=day0)
rj = RJ.RadiaJax(sys.argv[3], sys.argv[2], prod, zen=Z.Zenith(), **day_kw)
tab = RN.zenith_table(Z.Zenith(), ds.ITIMEI, (ITIMES[-1] - ds.ITIMEI) // 5 + 1)
nh = RN.NativeHandoff(rj, tab, ds.ITIMEI)
nh.last_out = None
imaxj = np.full(rs.JM, rs.IM); imaxj[0] = imaxj[-1] = 1
mask = np.zeros((rs.IM, rs.JM), bool)
for j in range(rs.JM):
    mask[:imaxj[j], j] = True
ctx = A.make_ctx("nov26", imf=False)
const = JR.RadConst(1800.0, ctx.sha, mask, ds.ITIMEI)
step = JR.make_handoff_step(nh, const)

ncomp = [0]
def _cb(name, *a, **k):
    if "backend_compile" in name:
        ncomp[0] += 1
jax.monitoring.register_event_duration_secs_listener(_cb)

_, (atm0, cloud0, carry0, surf0), _ = CK.device_packet(ITIMES[0])
hold = JR.empty_hold(carry0)
res = dict(device=str(jax.devices()[0]), steps=[])
valid = mask
for it in ITIMES:
    pk, (atm, cloud, carry, surf), _ = CK.device_packet(it)
    state = {**atm, **cloud}
    hold = JR.OrderedDict(hold); hold["SNOAGE"] = carry["SNOAGE"]            # SNOAGE is the model's (daily update), RQT/KLIQ the held outputs
    cosz = jnp.asarray(cio.read_cse(f"{D}/ffa_step_{it}_r.bin")["COSZ1"])
    t0 = time.perf_counter()
    T_new, Q_new, hold, cz = step(state, surf, hold, jnp.int32(it), jnp.int32(0), cosz)
    jax.block_until_ready(T_new)
    wall = time.perf_counter() - t0
    ref = rs.read_packet(f"{D}/rsv_n26_{it}_out.bin")
    row = dict(itime=it, wall_s=wall, compiles_so_far=ncomp[0], fields={})
    for k in JR.OUT_FIELDS:
        m = np.asarray(hold[k]); r = ref[k]
        ax = {"T": (0, 1), "Q": (0, 1), "KLIQ": (2, 3), "ALB": (0, 1)}.get(k, (1, 2) if r.ndim == 3 else (0, 1))
        idx = [slice(None)] * r.ndim
        idx[ax[0]], idx[ax[1]] = np.nonzero(valid)
        d = np.abs(m[tuple(idx)] - r[tuple(idx)])
        row["fields"][k] = float(d.max())
    dT = np.abs(np.asarray(T_new)[valid] - ref["T"][valid]).max()
    row["T_new_vs_recorded_T_maxabs"] = float(dT)
    # ---- the four held steps
    held = []
    for j in range(1, 5):
        it2 = it + j
        try:
            S2, X2 = CK.record_state(it2)
            atm2 = {k: jnp.asarray(S2[k]) for k in ("T", "Q", "PK", "PMID", "PDSIG", "PEDN", "MA")}
            atm2["LTROPO"] = atm["LTROPO"]
            cloud2 = {k: jnp.asarray(X2[k]) for k in JR.CLOUD_IN}
            surf2 = surf
            cz2 = jnp.asarray(cio.read_cse(f"{D}/ffa_step_{it2}_r.bin")["COSZ1"])
        except Exception as e:
            held.append(dict(itime=it2, skipped=repr(e)[:80]))
            continue
        st2 = {**atm2, **cloud2}
        T2, Q2, hold2, cz_used = step(st2, surf2, hold, jnp.int32(it2), jnp.int32(0), cz2)
        # independent evaluation (NumPy, same expression order as RAD_DRV.f label 900)
        srhr, trhr = np.asarray(hold["SRHR"]), np.asarray(hold["TRHR"])
        Tn = np.asarray(atm2["T"]); ma = np.asarray(atm2["MA"]); pkk = np.asarray(atm2["PK"])
        inc = (np.transpose(srhr[1:], (1, 2, 0)) * np.asarray(cz2)[:, :, None] + np.transpose(trhr[1:], (1, 2, 0))) * 1800.0 * (1.0 / ctx.sha) \
              * np.transpose(1.0 / ma, (1, 2, 0)) / np.transpose(pkk, (1, 2, 0))
        Tref = np.where(mask[:, :, None], Tn + inc, Tn)
        held.append(dict(itime=it2, T_maxabs_vs_independent=float(np.abs(np.asarray(T2) - Tref).max()),
                         hold_unchanged=bool(all(np.array_equal(np.asarray(hold2[k]), np.asarray(hold[k])) for k in JR.OUT_FIELDS)),
                         compiles_so_far=ncomp[0]))
    row["held_steps"] = held
    res["steps"].append(row)
    worst = max(row["fields"].items(), key=lambda kv: kv[1])
    print("itime %d: wall %.1f s, compiles %d, worst field %s %.2e, T_new vs recorded %.2e, held steps: %s" % (
        it, wall, ncomp[0], worst[0], worst[1], dT,
        [("%.1e" % h["T_maxabs_vs_independent"]) if "T_maxabs_vs_independent" in h else "skipped" for h in held]), flush=True)
res["total_compilations"] = ncomp[0]
json.dump(res, open(sys.argv[4], "w"), indent=1)
