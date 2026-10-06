"""D147: CONDSE with the JAX LSCOND (clouds_lscond_jax) and JAX MSTCNV (clouds_mstcnv_jax) wired into the batched chain of clouds_condse_batch.

The chain itself (column set-up, convective post-processing, bookkeeping, hand-off, poles) is clouds_condse_batch.condse_step_batch unchanged (numpy);
only the two heavy kernels are replaced, through a context manager that swaps `lscond_batch` / `mstcnv_batch` for JAX wrappers with the same
call signature and result layout.  Still numpy: the column set-up and post-processing arithmetic of the chain, the two pole columns (per-column
port), the snow-age exp loop, the QUS advection of the MSTCNV subsidence (host callback from the jitted MSTCNV), recalc_agrid_uv.
Usage: X, cnt = condse_step_jax(inp, cfg, ms=None, ls_mode="xla", use_ls=True, use_mc=True)
"""
import clouds_jax_env  # noqa: F401
import contextlib
import os

import clouds_condse_batch as cb
import clouds_lscond_jax as lj
import clouds_mstcnv_jax as mj

if os.environ.get("CLOUDS_JAX_CACHE"):
    import jax
    jax.config.update("jax_compilation_cache_dir", os.environ["CLOUDS_JAX_CACHE"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@contextlib.contextmanager
def jax_kernels(ls_mode="xla", use_ls=True, use_mc=True):
    orig_ls, orig_mc = cb.lb.lscond_batch, cb.mb.mstcnv_batch

    def ls(S, P):
        S2, W = lj.lscond_jax(S, P, mode=ls_mode)
        S.update(S2)
        return S, W

    def mcw(R, tune):
        return mj.mstcnv_jax(R, tune)

    if use_ls:
        cb.lb.lscond_batch = ls
    if use_mc:
        cb.mb.mstcnv_batch = mcw
    try:
        yield
    finally:
        cb.lb.lscond_batch, cb.mb.mstcnv_batch = orig_ls, orig_mc


def condse_step_jax(inp, cfg, ms=None, with_momentum=True, cnt=None, ls_mode="xla", use_ls=True, use_mc=True):
    with jax_kernels(ls_mode, use_ls, use_mc):
        return cb.condse_step_batch(inp, cfg, ms=ms, with_momentum=with_momentum, cnt=cnt)
