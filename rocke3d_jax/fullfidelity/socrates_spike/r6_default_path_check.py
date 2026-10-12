"""Default path (no volcanic args) of radia_jax / radia_native / daily_products: day products and one NativeHandoff.call, hashed.
OLD=1: the three modules of 24fc300 are taken first on sys.path."""
import os, sys, json, hashlib
G = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai-socrates-imvi/rocke3d_jax/fullfidelity/socrates_spike"
if os.environ.get("OLD") == "1":
    sys.path.insert(0, "/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/8fd22a28-e2ef-448a-902b-35964613ecf3/scratchpad/old24")
sys.path.insert(0, G); sys.path.insert(0, G + "/..")
if os.environ.get("OLD") == "1":
    sys.path.insert(0, "/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/8fd22a28-e2ef-448a-902b-35964613ecf3/scratchpad/old24")
import clouds_jax_env  # noqa
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import radiation_server as rs, radia_jax as RJ, radia_native as RN, daily_products as DP, drv_zenith as Z
print("modules:", RJ.__file__, RN.__file__, DP.__file__)
R6 = "/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/8fd22a28-e2ef-448a-902b-35964613ecf3/scratchpad/r6"
prod = RJ.Products(np.load(R6 + "/steps_0.npz"), np.load(R6 + "/day0_e.npz"))
dp = DP.DailyProducts("/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/8fd22a28-e2ef-448a-902b-35964613ecf3/scratchpad/socdump/run_c1", prod.c60["plb0"][0], prod.g["drym2g"], prod.g["ppmv80"], prod.taucon_dust,
                      solar_constant=prod.solar_constant)
dlist, day0 = DP.day_products_for_run(dp, prod, 33312, 33312 + 100)
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
res = {"days": [{k: h(v) for k, v in sorted(d.items())} for d in dlist], "solar_irrad": [float(d["solar_irrad"]) for d in dlist], "day0": int(day0)}
rj = RJ.RadiaJax("/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/8fd22a28-e2ef-448a-902b-35964613ecf3/scratchpad/socdump/spec2", R6 + "/static0.npz", prod, zen=Z.Zenith(), day_products=dlist, day0=day0)
tab = RN.zenith_table(Z.Zenith(), 16032, 25)
nh = RN.NativeHandoff(rj, tab, 16032)
pkt = rs.read_packet(rs.FF + "/nov26_day/rsv_n26_33312_in.bin")
pk = {n: jnp.asarray(pkt[n]) for n in pkt}
res["zen_tab"] = h(tab)
res["jaxpr"] = hashlib.sha256(str(jax.make_jaxpr(lambda it: tuple(nh.call(pk, it)[0].values()))(jnp.int32(33362))).encode()).hexdigest()[:16]
for it in (33312, 33362):
    out = jax.jit(lambda i: tuple(nh.call(pk, i)[0].values()))(jnp.int32(it))
    res["out_%d" % it] = [h(o) for o in out]
json.dump(res, open(os.environ["OUT"], "w"), indent=1)
print("done")
