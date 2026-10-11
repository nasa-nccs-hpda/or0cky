"""D182 experiment hook for d182_unit.py: replaces the sdrag_jax unit by the dyn_jax_fast variant (same args)."""
import dyn_jax_fast as ff
def apply(dj, kit, PROBES):
    import os
    which = os.environ.get('D182_VARIANTS', 'sdrag').split(',')
    ctx = kit.ctx
    if 'sdrag' in which:
        PROBES['sdrag_jax'].append(ff.make_sdrag2(ctx.sdp, ctx.sdgeo))
    if 'sdrag_b' in which:
        PROBES['sdrag_jax'].append(ff.make_sdrag2b(ctx.sdp, ctx.sdgeo))
