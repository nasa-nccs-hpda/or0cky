"""Per-segment build of the day products and the zenith table for runs longer than a few weeks on a GPU (report 14.4).

The day products are a 16 MB stack per model day and the zenith table 80 kB per radiation step; both are constants of the compiled hand-off
step (the host-callback alternative costs 6-9 s per radiation step on the V100).  A long run is therefore cut into segments of at most
`max_days` model days; each segment gets its own products, table and `make_handoff_step` (one compilation per segment: ~4.5 min on a V100).

    plan = segment_plan(itime_first, itime_last, max_days=30)
    for seg in plan:
        nh, rj = build_segment(seg, dp, prod, specdir, static0, zen, itimei)      # NativeHandoff + RadiaJax of the segment
        step = jax_radiation.make_handoff_step(nh, const)
        for itime in range(seg["itime_first"], seg["itime_last"] + 1): ...

Segments are cut at model-day boundaries, so every radiation step finds its day inside its segment and no step is in two segments.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np

NSPD = 48          # model steps per day (DTsrc = 1800 s)
NRAD = 5


def segment_plan(itime_first, itime_last, max_days=30, nspd=NSPD):
    """Day-aligned segments covering itime_first..itime_last (inclusive): list of dicts(itime_first, itime_last, day0, ndays)."""
    if itime_last < itime_first or max_days < 1:
        raise ValueError("empty range or max_days < 1")
    d_first, d_last = itime_first // nspd, itime_last // nspd
    plan, d = [], d_first
    while d <= d_last:
        d1 = min(d + max_days - 1, d_last)
        plan.append(dict(itime_first=max(itime_first, d * nspd), itime_last=min(itime_last, (d1 + 1) * nspd - 1), day0=d, ndays=d1 - d + 1))
        d = d1 + 1
    return plan


def segment_of(plan, itime):
    """Index of the segment holding `itime`."""
    for i, s in enumerate(plan):
        if s["itime_first"] <= itime <= s["itime_last"]:
            return i
    raise ValueError("itime %d is outside the plan" % itime)


def first_rad_itime(itime, itimei, nrad=NRAD):
    """First radiation step (itime - itimei divisible by nrad) at or after itime."""
    return itime + (-(itime - itimei)) % nrad


def build_segment(seg, dp, prod, specdir, static0, zen, itimei, radia=None):
    """(NativeHandoff, RadiaJax) of one segment.  `radia`: an existing RadiaJax to reuse (its day stack is replaced; no new spectral load)."""
    import daily_products as DP
    import radia_jax as RJ
    import radia_native as RN
    dlist, day0 = DP.day_products_for_run(dp, prod, seg["itime_first"], seg["itime_last"])
    assert day0 == seg["day0"] and len(dlist) == seg["ndays"], (day0, len(dlist), seg)
    rj = radia
    if rj is None:
        rj = RJ.RadiaJax(specdir, static0, prod, zen=zen, day_products=dlist, day0=day0)
    else:
        rj.set_day_products(dlist, day0)
    it0 = first_rad_itime(seg["itime_first"], itimei)
    n = (seg["itime_last"] - it0) // NRAD + 1
    tab = RN.zenith_table(zen, it0, max(n, 1))
    return RN.NativeHandoff(rj, tab, it0), rj
