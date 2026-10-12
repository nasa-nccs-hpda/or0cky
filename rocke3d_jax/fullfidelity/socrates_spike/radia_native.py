"""R6/R7: drop-in replacement of jax_radiation.RadiationHandoff whose `call` evaluates RADIA + SOCRATES natively in the trace (radia_jax.py)
instead of an ordered io_callback to the persistent Fortran server.

    nh = NativeHandoff(radia_jax_instance, zen_table, itimei)
    out, aijc = nh.call(packet, itime, seed)          # same signature and return as RadiationHandoff.call (inside jit)
    step = jax_radiation.make_handoff_step(nh, const) # unchanged

zen_table: the zenith grids of the radiation steps (cosz1, cosz2, cosza), (K, 3, 72, 46), K radiation steps from ITIMEI at NRAD spacing,
precomputed on the host with the exact rational orbit arithmetic (drv_zenith / zenith_coszs); `seed` is accepted and ignored (the seeds feed
only the cloud-overlap diagnostics of the AIJ fields, which are not ported, and the model's random-number chain, which the caller keeps).
AIJ columns: the four accumulators the hand-off asks for (SRNFP0, TRNFP0, SRINCP0, SRNFG) are rebuilt from the outputs; AIJ is otherwise
not produced.  Limits: products of ONE day (see report section 12): the call is only valid for dates the products were made for."""
import os, sys
from collections import OrderedDict
import numpy as np
import jax
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import radia_jax as RJ
import radia_cols as R
import zenith_coszs as ZS

IM, JM, LM = 72, 46, 40
NRAD = 5
OUT_FIELDS = ("T", "Q", "SRHR", "TRHR", "RQT", "KLIQ", "SNOAGE", "CLDSS", "CLDMC", "FSF", "TRSURF", "ALB", "FSRDIR", "SRVISSURF", "FSRDIF",
              "DIRVIS", "DIRNIR", "DIFNIR", "SRDN", "CFRAC", "COSZ1")
AIJ_COLS = OrderedDict([("SRNFP0", 377), ("TRNFP0", 391), ("SRINCP0", 376), ("SRNFG", 385)])


def zenith_table(zen, itimei, n_steps):
    """(K, 3, 72, 46): cosz1, cosz2 (mean over NRAD steps), cosza (sunlit-weighted) for itime = itimei + NRAD*k."""
    tab = np.zeros((n_steps, 3, IM, JM))
    for k in range(n_steps):
        it = itimei + NRAD * k
        cz2, cza = ZS.calc_zenith_angle(zen, it, NRAD)
        tab[k, 0] = zen.cosz1(it)
        tab[k, 1] = cz2
        tab[k, 2] = cza
    return tab


class ZenithStream:
    """zenith grids of the radiation step at `itime` on demand (host, exact rational orbit arithmetic, ~7 ms): the (3, 72, 46) of one row of
    `zenith_table`.  Used by NativeHandoff(zen_source=...) so that no K-row table (80 kB per radiation step) is embedded in the compiled step."""

    def __init__(self, zen):
        self.zen = zen

    def __call__(self, itime):
        it = int(itime)
        cz2, cza = ZS.calc_zenith_angle(self.zen, it, NRAD)
        return np.stack([np.asarray(self.zen.cosz1(it)), np.asarray(cz2), np.asarray(cza)]).astype(np.float64)


class NativeHandoff:
    def __init__(self, radia, zen_table, itimei, log=None, zen_source=None):
        self.radia = radia
        self.zen_source = zen_source                       # callable itime -> (3, 72, 46); zen_table may then be None
        self.table = None if zen_table is None else jnp.asarray(zen_table)
        self.itimei = int(itimei)
        self.log = log
        self.aij_cols = dict(AIJ_COLS)
        self.ii, self.jj = jnp.asarray(radia.ii - 1), jnp.asarray(radia.jj - 1)
        # the day products are selected by itime // 48 and CLIPPED to the window (no exception inside a jit): a zenith table longer than the
        # window would silently reuse the last day's products, so say so now
        if radia.day_arrays is not None and zen_table is not None:
            nd = int(radia.day_arrays["o3col"].shape[0])
            last = self.itimei + NRAD * (int(zen_table.shape[0]) - 1)
            need = last // radia.DAY_LEN - radia.day0 + 1
            self.day_window_ok = need <= nd
            if not self.day_window_ok:
                import warnings
                warnings.warn("NativeHandoff: the zenith table reaches itime %d (day %d of the window) but only %d day products were built; "
                              "steps after day %d will use the last day's products (build day_products for %d days)" % (last, need, nd, nd, need))

    def call(self, packet, itime, seed=None):
        if self.zen_source is not None:
            z = jax.pure_callback(lambda it: np.asarray(self.zen_source(int(it)), dtype=np.float64),
                                  jax.ShapeDtypeStruct((3, IM, JM), jnp.float64), jnp.asarray(itime, jnp.int32))
        else:
            z = self.table[(jnp.asarray(itime, jnp.int32) - self.itimei) // NRAD]
        pa = self.radia.pa_for_itime(itime)
        out = self.radia._core({n: jnp.asarray(packet[n]) for n in packet}, z[0], z[1], z[2], pa)
        res = OrderedDict((n, out[n]) for n in OUT_FIELDS)
        # AIJ increments of the four hand-off columns (RAD_DRV.f 2254-2259, 2453); only boxes I <= IMAXJ are accumulated
        valid = jnp.zeros((IM, JM), bool).at[self.ii, self.jj].set(True)
        csz2 = z[1]
        aij = jnp.stack([out["_SNFS3"] * csz2, -out["_TNFS3"], jnp.where(valid, pa["solar_irrad"] * z[0], 0.0),
                         out["SRHR"][0] * csz2], axis=2)
        return res, aij
