"""D187: the D183 libimf host callback (libimf_ops) wired into the D186 atmosphere-phase-1 device modules, WITHOUT editing any existing file.

D183 did this for the D180 modules (jax_atm_step_imf.install).  The D186 modules are different code (jax_p1_glue, jax_p1_dyn, jax_p1_condse), so the
places where the NumPy 'imf' chain (atm_step.make_ctx(imf=True)) routes pow / exp through the Intel libimf are mapped again:

  site (NumPy imf chain)                         D186 device code                              how it is routed here
  ---------------------------------------------  --------------------------------------------  -------------------------------------------------
  ADVECM MAtoP pow(PMID, KAPA)                   dyn_jax_aflux.matop_jax (via DynDevice.kit)   D183 install(): dyn_jax_aflux.jnp = JnpProxy
  PGF pow (PKU/PKD, .01**KAPA)                   dyn_jax_pgf (via DynDevice.kit)               D183 install(): dyn_jax_pgf.jnp = JnpProxy and
                                                                                               DynDevice.kit = jax_atm_step_imf.KitImf (KAPA float
                                                                                               subclass for the host constant .01**KAPA)
  MAtoPMB pow(PMID, KAPA)                        jax_p1_glue.matopmb  (jnp.power)              jax_p1_glue.jnp = JnpProxy (power -> libimf_ops.pow)
  PEK = PEDN**KAPA                               jax_p1_glue.pek_of   (jnp.power)              same
  CALC_TROP pow(zpmk, 1/KAPA), pow(|q|, 1/KAPA)  jax_p1_glue.calc_trop (jnp.power x2)          same (the second one is inside a vmapped while loop:
                                                                                               the callback is evaluated on every lane, results of
                                                                                               inactive lanes are discarded by the loop's selects)
  LSCOND exp / pow / get_dq                      clouds_lscond_jax._core(..., "xla")           jax_p1_condse.lj = shim whose _core uses mode "imf"
                                                                                               (clouds_lscond_jax.OPS['imf'] = OpsImf, D183)
  MSTCNV qsat / size search / microphysics       clouds_mstcnv_jax (jnp, conv_micro_j)         D183 install(): clouds_mstcnv_jax.jnp = JnpProxy and
                                                                                               conv_micro_j = conv_micro_j_imf
  snow-age exp(-PRCP)                            jax_p1_condse._post_fn  jnp.exp(-prcp)        jax_p1_condse.jnp = JnpProxy (the only jnp.exp/power of
                                                                                               that module; evaluated on all columns, used where sa)
  CONDSE pole columns (per-column NumPy port)    jax_p1_condse.PoleHost.run                    PoleHost.run calls cf.set_backend('numpy'): the module's
                                                                                               `cf` is replaced by a shim that maps it to 'imf'
  host NumPy parts (SLP filter pow, dissip,      NumPy in both chains                          unchanged (ctx.imf=True)
  surface half)

Everything here is a LABELLED HOST CALLBACK: libimf_ops evaluates pow / exp on the host with the Intel runtime (jax.pure_callback, scalar loop).
It is not device resident.  Call install() once, in a fresh process, BEFORE any JAX stage is traced, and construct the Phase1 object through
make_phase1() (which builds the context with imf=True and replaces the dynamics kit).
Needs >= 2 cores (host callbacks inside jit deadlock on one core: D185)."""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax; same flags as clouds_jax_env_fast)
import jax_atm_phase1 as P1      # installs the execution counters first (jax_p1_count), then imports the device modules
import jax_atm_step_imf as I
import libimf_ops as L
import jax_p1_glue as G
import jax_p1_condse as CS
import clouds_condse_ff as cf
import clouds_lscond_jax as lj
import atm_step as A

_INSTALLED = {}


class _LjShim:
    """clouds_lscond_jax as seen by jax_p1_condse: identical except that _core always runs with mode "imf" (OpsImf)."""

    def __getattr__(self, name):
        return getattr(lj, name)

    @staticmethod
    def _core(S, P, K, lmcld, mode):
        return lj._core(S, P, K, lmcld, 'imf')


class _CfShim:
    """clouds_condse_ff as seen by jax_p1_condse: set_backend('numpy') (called by PoleHost.run) is mapped to 'imf'."""

    def __getattr__(self, name):
        return getattr(cf, name)

    @staticmethod
    def set_backend(name):
        return cf.set_backend('imf')


def install():
    """Wire libimf into the D186 modules (idempotent within a process).  Returns a dict describing what was rebound (for the header)."""
    if _INSTALLED:
        return dict(_INSTALLED)
    inst = I.install('libimf')               # D183: mode locked, dyn/mstcnv/lscond rebinding (also rebinds three D180 functions; unused here)
    prox = L.JnpProxy()
    G.jnp = prox
    CS.jnp = prox
    CS.lj = _LjShim()
    CS.cf = _CfShim()
    inst = dict(inst)
    inst.update({
        'jax_p1_glue.jnp': 'libimf_ops.JnpProxy (matopmb pk pow, pek_of pow, calc_trop pow x2)',
        'jax_p1_condse.jnp': 'libimf_ops.JnpProxy (snow-age exp)',
        'jax_p1_condse.lj': '_LjShim (LSCOND core mode "imf")',
        'jax_p1_condse.cf': '_CfShim (pole columns keep the imf backend)',
        'DynDevice.kit': 'jax_atm_step_imf.KitImf (set by make_phase1)'})
    _INSTALLED.update(inst)
    return dict(_INSTALLED)


def make_phase1(date, **kw):
    """Phase1 whose context is built with imf=True (NumPy parts and the CONDSE backend use libimf as in the NumPy imf chain) and whose dynamics
    kit is the libimf kit.  install() must have been called."""
    assert _INSTALLED, 'call install() first'
    orig = A.make_ctx

    def make_ctx_imf(date_, imf=False, *a, **k):
        return orig(date_, imf=True, *a, **k)
    A.make_ctx = make_ctx_imf
    try:
        ph = P1.Phase1(date, **kw)
    finally:
        A.make_ctx = orig
    assert ph.ctx.imf, 'context must be an imf context'
    ph.dyn.kit = I.KitImf(ph.ctx.dyn)
    return ph
