"""The first OADVT2 X sweep (oadvt_jax._x_sweep, G0M) alone, on this device vs the CPU, from its REAL step-0 inputs. (D214 follow-up; quick: no step compile)

usage: python -u gpu_xsweep_probe.py xsweep_inputs.npz
Inputs are captured on the CPU by  XSWEEP_CAPTURE=<file> python gpu_oadvt_probe.py  (they are the arguments of the first x sweep; finite). Runs the sweep
  A  jit on the default device (the GPU on a GPU node)
  B  jit on the CPU device
  C  with jax.disable_jit() on the default device (op by op: separates a compile/fusion problem from an arithmetic one)
and prints non-finite counts per output, the first non-finite indices (i, j, l), and the largest difference of the finite values against B.
A is the failing configuration (9,679 NaN per tracer array in the full step, job 58802392); B must be all zero.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import oadvt_jax as OJ   # noqa: E402

z = np.load(sys.argv[1])
names = ('rm', 'rx', 'ry', 'rz', 'mm', 'mudt', 'nc', 'lane_pass', 'lmu_l', 'lmm_l', 'rxlimit')
args = [z[n] for n in names]
args[-1] = float(args[-1])
print('backend', jax.default_backend(), 'devices', jax.devices())
print('input non-finite:', {n: int((~np.isfinite(z[n])).sum()) for n in names[:6] if z[n].dtype.kind == 'f'}, ' nc max', int(z['nc'].max()))


def run(device, nojit=False):
    a = [jax.device_put(jnp.asarray(v), device) if not isinstance(v, float) else v for v in args]
    with jax.default_device(device):
        if nojit:
            with jax.disable_jit():
                out = OJ._x_sweep.__wrapped__(*a)
        else:
            out = OJ._x_sweep(*a)
    return [np.asarray(o) for o in out]


def show(tag, out, ref=None):
    nf = [int((~np.isfinite(o)).sum()) for o in out]
    print(f'{tag:34s} non-finite (rm,rx,ry,rz,mm): {nf}')
    if any(nf[:4]):
        idx = np.argwhere(~np.isfinite(out[0]))[:6]
        print('   first non-finite rm indices (i,j,l):', [tuple(int(x) for x in p) for p in idx])
    if ref is not None:
        d = [float(np.max(np.abs(np.where(np.isfinite(o) & np.isfinite(r), o - r, 0.0)))) for o, r in zip(out, ref)]
        print(f'{"":34s} max |diff| vs CPU jit on finite values:', [f'{x:.3g}' for x in d])


cpu = jax.devices('cpu')[0]
dev = jax.devices()[0]
B = run(cpu)
show('B  jit, CPU device', B)
if dev.platform != 'cpu':
    A = run(dev)
    show('A  jit, ' + dev.platform, A, B)
    try:
        Cc = run(dev, nojit=True)
        show('C  op-by-op, ' + dev.platform, Cc, B)
    except Exception as e:     # noqa: BLE001
        print('C  op-by-op failed:', type(e).__name__, str(e)[:200])
else:
    print('default device is the CPU: only B was run (use a GPU node for A and C)')
