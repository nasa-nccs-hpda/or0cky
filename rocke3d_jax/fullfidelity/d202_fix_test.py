"""D202: validate d202_fix.advnc_gdtm on REAL records (all land cells of one nov26 step/substep): (a) nit and dts vs the recorded ffnit, (b) outputs vs the real record
and vs the shipped ghy_jax.advnc with the recorded schedule.   python d202_fix_test.py STEPK [NS]"""
import sys, json
import numpy as np
import clouds_jax_env  # noqa
import jax, jax.numpy as jnp
import ghy_compare as GC
import ghy_advnc_test as AT
import jax_surface as JS
import ghy_jax as J
import d202_fix as F

FF = '/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26_day'
k = int(sys.argv[1]); ns = int(sys.argv[2]) if len(sys.argv) > 2 else 0
g = GC.load(f'{FF}/ffg_{33312+k}.bin'); n = len(g) // 2
rec = g[ns * n:(ns + 1) * n]
gb = JS._ghy_batch(rec, AT)
ms = gb['edts'].shape[1]
args = lambda: ({k_: jnp.asarray(v) for k_, v in gb['static'].items()}, {k_: jnp.asarray(v) for k_, v in gb['dyn0'].items()}, {k_: jnp.asarray(v) for k_, v in gb['forcing'].items()},
                jnp.asarray(gb['edts']), jnp.asarray(gb['ecnc']), jnp.asarray(gb['ebet']), jnp.asarray(gb['elai']), jnp.asarray(gb['nsub']), jnp.asarray(gb['dt']), jnp.asarray(gb['snowm']))
base = jax.jit(J.advnc, static_argnames=('max_substeps',))(*args(), max_substeps=ms)
new = jax.jit(F.advnc_gdtm, static_argnames=('max_substeps',))(*args(), max_substeps=ms)
ffnit = np.round(rec[:, 289]).astype(int)
nit = np.asarray(new['nit_gdtm'])
out = dict(step=k, ns=ns, ncell=int(n), nit_equal=int((nit == ffnit).sum()), nit_diff_cells=[(int(i), int(ffnit[i]), int(nit[i])) for i in np.where(nit != ffnit)[0]][:30], exhausted=int(np.asarray(new['nit_exhausted']).sum()))
for key, col in (('tbcs', 245), ('tsns', 246), ('ashg', 247), ('alhg', 248), ('aevap', 249)):
    r = rec[:, col]
    d_base = np.asarray(base[key]) - r; d_new = np.asarray(new[key]) - r
    same = nit == ffnit
    out[key] = dict(max_abs_base_vs_real=float(np.nanmax(np.abs(d_base))), max_abs_new_vs_real=float(np.nanmax(np.abs(d_new))),
                    max_abs_new_vs_base=float(np.nanmax(np.abs(np.asarray(new[key]) - np.asarray(base[key])))),
                    max_abs_new_vs_base_nit_equal_cells=float(np.nanmax(np.abs(np.asarray(new[key]) - np.asarray(base[key]))[same])) if same.any() else None,
                    bitwise_equal_to_base=int((np.asarray(new[key]) == np.asarray(base[key])).sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(f'scoping/d202_results/fix_test_step{k}_ns{ns}.json', 'w'), indent=1)
