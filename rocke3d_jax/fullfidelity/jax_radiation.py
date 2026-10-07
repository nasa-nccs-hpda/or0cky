"""D185 (stage S3 of the JAX-driven coupled step): the radiation hand-off.

Radiation is NOT computed by JAX.  The real, unmodified RADIA (SOCRATES as a black box) runs in the persistent radiation server
(radiation_server_persist.PersistentServer, D159-D162); this module is the boundary between the JAX step and that server.  Every result
of this module carries the sentence  'radiation computed by the original Fortran (hybrid component)'  and the counters (RadLog.sentence).

Pieces
  1. Packet assembly from ARRAYS (assemble_packet_jax, jitted): the 52-field RADIA input packet (radiation_server.INPUT_FIELDS) from device arrays:
       atmosphere  T Q PK PMID PDSIG PEDN MA LTROPO                  (BYMA = 1/MA is computed in the jit, as the NumPy path does)
       clouds      the 15 CONDSE hand-off arrays + TAUSS TAUMC CLDSS CLDMC
       carry       SNOAGE RQT KLIQ                                    (previous server output, as RADIA carries them)
       surface     the 21 fields of drv_radpacket.FIELDS_SURFACE      (arrays; surface_fields_jax rebuilds the ice/lake/land-ice group
                                                                       from arrays, the land group GTEMPR4/BARESW/SNOWD/FRSNOW stays the host
                                                                       loop of drv_radpacket.land_fields)
     Same field-source split as atm_day_free_rad.assemble_packet (the NumPy path), which is the oracle in the tests.
  2. A host callback usable inside jit: jax.experimental.io_callback(..., ordered=True).  Choice (not pure_callback): the server is STATEFUL
     (its model clock only moves forward, a call is a side effect on the server process, the counters are side effects); pure_callback
     may be deduplicated, reordered, vectorised or dropped when its result is unused; io_callback(ordered=True) is executed exactly once per
     execution of its enclosing computation, in program order, and never dead-code eliminated.  Cost of that choice: ordered effects serialise the
     device stream (one host synchronisation point per call, counted).
  3. The hand-off in the step (handoff_step): lax.cond on 'radiation step' (itime - ITIMEI) % NRAD == 0.  Radiation step: pack, callback, apply
     the outputs with jax_atm_step._radia_T_jax (RAD_DRV.f:5474-5478), Q reset, hold the outputs.  Other four steps of five: NO callback, the held
     SRHR/TRHR are applied with the per-step COSZ1 (an argument: recorded or drv_zenith), exactly as RADIA does.  The hold (all server outputs
     and the carry) is device state between steps.
  4. Seed: RADIA's random-number seed at a step is a trajectory property.  Two sources: seed_from_record(itime) = ffc_cse_out SEEDS[1] (recorded),
     or the D174 chain (drv_rng; implemented on device here as uint32 affine maps: seed_chain_*), starting from the recorded SEEDS[0] of the first
     step.  The seed is a traced int32 scalar passed to the callback (sign convention of the recorded SEEDS[1] values).

Counters (RadLog): calls, packet bytes host<-device and host->device, packet-file bytes, wall seconds (callback total, server request,
server-reported RADIA+IO), host synchronisation points (one per callback: device stream blocked until the host returns), skipped steps.
Limitations: the callback executes on the host thread that runs the XLA CPU client; on GPU the packet is a device-to-host copy of ~33.7 MB
and the result a host-to-device copy of ~21 MB per call (not measured here: no GPU).
"""
import clouds_jax_env  # noqa: F401  (XLA flags BEFORE jax; same as jax_atm_step)
import json
import os
import sys
import threading
import time
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import io_callback

import radiation_server as rs
import drv_rng
import drv_radpacket as DRP
from jax_atm_step import _radia_T_jax

SENTENCE = "radiation computed by the original Fortran (hybrid component)"
IM, JM, LM = rs.IM, rs.JM, rs.LM
NRAD = 5

ATM_IN = ("T", "Q", "PK", "PMID", "PDSIG", "PEDN", "MA", "LTROPO")
CLOUD_IN = ("W_CLOUD", "FRAC_ST_WATER", "FRAC_ST_ICE", "FRAC_CNV_WATER", "FRAC_CNV_ICE", "MIX_ST_WATER", "MIX_ST_ICE", "MIX_CNV_WATER",
            "MIX_CNV_ICE", "DIM_ST_WATER", "DIM_ST_ICE", "DIM_CNV_WATER", "DIM_CNV_ICE", "FRAC_AREA_ST", "FRAC_AREA_CNV",
            "TAUSS", "TAUMC", "CLDSS", "CLDMC")
CARRY_IN = ("SNOAGE", "RQT", "KLIQ")
SURF_IN = DRP.FIELDS_SURFACE
PACKET_FIELDS = tuple(rs.INPUT_FIELDS)
assert set(ATM_IN) | set(CLOUD_IN) | set(CARRY_IN) | set(SURF_IN) | {"BYMA"} == set(PACKET_FIELDS) and len(PACKET_FIELDS) == 52

# server outputs kept on the device (all 21 non-AIJ fields; AIJ reduced to the accumulator columns asked for)
OUT_FIELDS = tuple(k for k in rs.OUTPUT_FIELDS if k != "AIJ")
OUT_SHAPES = OrderedDict([("T", (IM, JM, LM)), ("Q", (IM, JM, LM)), ("SRHR", (LM + 1, IM, JM)), ("TRHR", (LM + 1, IM, JM)),
                          ("RQT", (3, IM, JM)), ("KLIQ", (LM, 4, IM, JM)), ("SNOAGE", (3, IM, JM)), ("CLDSS", (LM, IM, JM)),
                          ("CLDMC", (LM, IM, JM)), ("FSF", (4, IM, JM)), ("TRSURF", (4, IM, JM)), ("ALB", (IM, JM, 9)),
                          ("FSRDIR", (IM, JM)), ("SRVISSURF", (IM, JM)), ("FSRDIF", (IM, JM)), ("DIRVIS", (IM, JM)),
                          ("DIRNIR", (IM, JM)), ("DIFNIR", (IM, JM)), ("SRDN", (IM, JM)), ("CFRAC", (IM, JM)), ("COSZ1", (IM, JM))])
assert tuple(OUT_SHAPES) == OUT_FIELDS
# 1-based AIJ accumulator numbers (atm_day_free_rad.AIJ_COLS); AIJ is a seed-dependent diagnostic (cloud-overlap diagnostics)
AIJ_COLS = {"SRNFP0": 377, "TRNFP0": 391, "SRINCP0": 376, "SRNFG": 385}


# ============================================================================================================ 1. packet from arrays
@jax.jit
def assemble_packet_jax(atm, cloud, carry, surf):
    """atm/cloud/carry/surf: dicts of device arrays (names in ATM_IN, CLOUD_IN, CARRY_IN, SURF_IN).  Returns the 52-field packet as a dict in
    INPUT_FIELDS order (float64, Fortran-packet shapes)."""
    out = OrderedDict()
    for k in PACKET_FIELDS:
        if k == "BYMA":
            out[k] = 1.0 / jnp.asarray(atm["MA"], jnp.float64)
        else:
            src = atm if k in atm else cloud if k in cloud else carry if k in carry else surf
            out[k] = jnp.asarray(src[k], jnp.float64)
    return out


def split_packet(pk):
    """dict of the 52 packet arrays (e.g. a live packet) -> (atm, cloud, carry, surf) dicts for assemble_packet_jax."""
    return ({k: pk[k] for k in ATM_IN}, {k: pk[k] for k in CLOUD_IN}, {k: pk[k] for k in CARRY_IN}, {k: pk[k] for k in SURF_IN})


@jax.jit
def surface_fields_jax(rsi, snowi, pond_melt, flag_dsws, zsi_ag, zsnowi_ag, gtempr_ag, fwater, flake, mwl, axyp, flice, fland, fearth,
                       gtempr_atm, tlandi1, snowli):
    """Array version of drv_radpacket.ice_lake_landice_fields (same where-expressions): RSI ZSI SNOWI POND_MELT FLAG_DSWS FLAKE DLAKE FLICE FLAND FEARTH
    GTEMPR1 GTEMPR2 GTEMPR3 SNOWLI ZSNOWI from the ice state after MELT_SI (rsi snowi pond_melt flag_dsws), seaice_to_atmgrid (zsi_ag zsnowi_ag
    gtempr_ag), geometry (fwater flake axyp flice fland fearth), lake mass mwl, atmocn gtempr, land-ice tlandi1 (layer 1) and snowli."""
    TF, RHOW = DRP.TF, DRP.RHOW
    fw = fwater > 0
    poice = rsi * fwater > 0
    fli = flice > 0
    return dict(
        RSI=rsi, SNOWI=snowi, POND_MELT=pond_melt, FLAG_DSWS=jnp.asarray(flag_dsws, jnp.float64),
        ZSI=jnp.where(poice, zsi_ag, 0.2), ZSNOWI=jnp.where(poice, zsnowi_ag, 0.0), GTEMPR2=gtempr_ag, FLAKE=flake,
        DLAKE=jnp.where(flake > 0, mwl / (RHOW * jnp.where(flake > 0, flake, 1.0) * axyp), 0.0),
        FLICE=flice, FLAND=fland, FEARTH=fearth, GTEMPR1=jnp.where(fw, gtempr_atm, TF),
        GTEMPR3=jnp.where(fli, tlandi1 + TF, TF), SNOWLI=jnp.where(fli, snowli, 0.0))


# ============================================================================================================ 4. seeds
def seed_from_record(itime, ff=None, daydir="nov26_day"):
    """RADIA seed of the real model at this step = SEEDS[1] of ffc_cse_out (signed value as recorded)."""
    import clouds_condse_io as cio
    ff = ff or rs.FF
    return int(round(float(cio.read_cse(f"{ff}/{daydir}/ffc_cse_out_{itime}.bin")["SEEDS"][1])))


def seed0_from_record(itime, ff=None, daydir="nov26_day"):
    import clouds_condse_io as cio
    ff = ff or rs.FF
    return int(round(float(cio.read_cse(f"{ff}/{daydir}/ffc_cse_out_{itime}.bin")["SEEDS"][0]))) % drv_rng.M


def _affine(n):
    c = drv_rng.jump(0, n)
    a = (drv_rng.jump(1, n) - c) % drv_rng.M
    return np.uint32(a), np.uint32(c)


_A_RAD, _C_RAD = _affine(drv_rng.N_CONDSE)               # seed at RADIA entry from SEEDS[0]
_A_STEP_R, _C_STEP_R = _affine(drv_rng.N_STEP_RAD)       # SEEDS[0] of the next step after a radiation step
_A_STEP_N, _C_STEP_N = _affine(drv_rng.N_STEP_NORAD)     # ... after a non-radiation step


def seed_chain_radia(seed0):
    """device: RADIA seed (int32 bit pattern, as the recorded SEEDS[1]) from SEEDS[0] (uint32 scalar), D174 chain."""
    s = jnp.asarray(seed0, jnp.uint32) * _A_RAD + _C_RAD       # uint32 wraps modulo 2^32
    return jax.lax.bitcast_convert_type(s, jnp.int32)


def seed_chain_next(seed0, is_rad):
    s = jnp.asarray(seed0, jnp.uint32)
    return jnp.where(is_rad, s * _A_STEP_R + _C_STEP_R, s * _A_STEP_N + _C_STEP_N)


# ============================================================================================================ logging
class RadLog:
    """Counters of the hand-off.  Thread-safe (callbacks may run on a runtime thread)."""

    def __init__(self, server_binary=None, restart=None):
        self.lock = threading.Lock()
        self.server_binary, self.restart = server_binary, restart
        self.calls = []
        self.skipped_steps = 0
        self.dev_to_host_arrays = self.host_to_dev_arrays = 0
        self.bytes_d2h = self.bytes_h2d = 0
        self.file_bytes_in = self.file_bytes_out = 0
        self.callback_s = self.request_s = self.server_s = 0.0
        self.sync_points = 0

    def add(self, **rec):
        with self.lock:
            self.calls.append(rec)
            self.bytes_d2h += rec["bytes_d2h"]
            self.bytes_h2d += rec["bytes_h2d"]
            self.dev_to_host_arrays += rec["arrays_d2h"]
            self.host_to_dev_arrays += rec["arrays_h2d"]
            self.file_bytes_in += rec["file_bytes_in"]
            self.file_bytes_out += rec["file_bytes_out"]
            self.callback_s += rec["callback_s"]
            self.request_s += rec["request_s"]
            self.server_s += rec["server_s"]
            self.sync_points += 1

    def reset(self):
        self.__init__(self.server_binary, self.restart)

    def totals(self):
        return dict(radiation=SENTENCE, calls=len(self.calls), bytes_device_to_host=self.bytes_d2h, bytes_host_to_device=self.bytes_h2d,
                    arrays_device_to_host=self.dev_to_host_arrays, arrays_host_to_device=self.host_to_dev_arrays,
                    packet_file_bytes_in=self.file_bytes_in, packet_file_bytes_out=self.file_bytes_out,
                    wall_s_callback=self.callback_s, wall_s_server_request=self.request_s, wall_s_server_radia_and_io=self.server_s,
                    host_sync_points=self.sync_points, steps_without_callback=self.skipped_steps,
                    server_binary=self.server_binary, restart=self.restart, SOCRATES_modified=False,
                    seeds=[c["seed"] for c in self.calls], itimes=[c["itime"] for c in self.calls])

    def sentence(self):
        t = self.totals()
        return ("%s: calls=%d, bytes device->host=%d, host->device=%d, wall callback=%.2f s (server request %.2f s, RADIA+IO in server %.2f s), "
                "host sync points=%d, steps without callback=%d, SOCRATES/RADIA modified=no"
                % (SENTENCE, t["calls"], t["bytes_device_to_host"], t["bytes_host_to_device"], t["wall_s_callback"],
                   t["wall_s_server_request"], t["wall_s_server_radia_and_io"], t["host_sync_points"], t["steps_without_callback"]))

    def to_json(self):
        return json.dumps(dict(totals=self.totals(), calls=self.calls), indent=1, default=float)


def say(log, title, printer=print):
    """Print the mandatory sentence line above a result."""
    printer("[%s] %s" % (title, log.sentence() if log is not None else SENTENCE + " (no counters)"))


# ============================================================================================================ 2. the callback
class RadiationHandoff:
    """Owns the server connection and the log; `callback` is the io_callback target; `call_eager` is the same call outside jit (for timing).
    server: radiation_server_persist.PersistentServer (started) or any object with request(state, itime, seed) -> dict (+ _server_s, _wall_s)."""

    def __init__(self, server, log=None, aij_cols=AIJ_COLS, keep_outputs=False):
        self.server = server
        self.keep_outputs = keep_outputs
        self.history = []                                 # (itime, seed, out dict of the server) when keep_outputs
        self.log = log or RadLog(getattr(server, "binary", None), getattr(server, "restart", None))
        self.aij_cols = dict(aij_cols)
        self.last_out = None
        self.out_spec = tuple([jax.ShapeDtypeStruct(OUT_SHAPES[k], jnp.float64) for k in OUT_FIELDS]
                              + [jax.ShapeDtypeStruct((IM, JM, len(self.aij_cols)), jnp.float64)])

    # host side ------------------------------------------------------------------------------------------------------------
    def _serve(self, packet_arrays, itime, seed):
        t0 = time.perf_counter()
        arrs = [np.asarray(a) for a in packet_arrays]                  # the device -> host copies
        state = OrderedDict((k, a) for k, a in zip(PACKET_FIELDS, arrs))
        itime, seed = int(np.asarray(itime)), int(np.asarray(seed))
        out = self.server.request(state, itime, seed)
        cols = [c - 1 for c in self.aij_cols.values()]
        res = [np.ascontiguousarray(out[k], dtype=np.float64) for k in OUT_FIELDS]
        res.append(np.ascontiguousarray(out["AIJ"][:, :, cols], dtype=np.float64))
        self.last_out = out
        if self.keep_outputs:
            self.history.append((itime, seed, out))
        pin = os.path.join(getattr(self.server, "rundir", "."), "rsv_in.bin")
        pout = os.path.join(getattr(self.server, "rundir", "."), "rsv_out.bin")
        self.log.add(itime=itime, seed=seed, bytes_d2h=int(sum(a.nbytes for a in arrs)) + 16, arrays_d2h=len(arrs) + 2,
                     bytes_h2d=int(sum(a.nbytes for a in res)), arrays_h2d=len(res),
                     file_bytes_in=os.path.getsize(pin) if os.path.exists(pin) else 0,
                     file_bytes_out=os.path.getsize(pout) if os.path.exists(pout) else 0,
                     callback_s=time.perf_counter() - t0, request_s=float(out["_wall_s"]), server_s=float(out["_server_s"]),
                     fields_in=len(arrs), fields_out=len(res))
        return tuple(res)

    def callback(self, packet_arrays, itime, seed):
        return self._serve(packet_arrays, itime, seed)

    # traced side ----------------------------------------------------------------------------------------------------------
    def call(self, packet, itime, seed):
        """Inside jit: packet = dict of the 52 arrays; returns (dict of OUT_FIELDS arrays, AIJ-columns array)."""
        tup = tuple(packet[k] for k in PACKET_FIELDS)
        res = io_callback(self.callback, self.out_spec, tup, itime, seed, ordered=True)
        return OrderedDict(zip(OUT_FIELDS, res[:-1])), res[-1]

    def call_eager(self, packet, itime, seed):
        """The same host call outside jit (device arrays or numpy in; returns the same tuple as numpy)."""
        return self._serve(tuple(packet[k] for k in PACKET_FIELDS), itime, seed)


# ============================================================================================================ 3. the hand-off in the step
class RadConst:
    """Constants of the step the hand-off needs: dtsrc, 1/sha, the IMAXJ mask (IM,JM,1), ITIMEI phase.  From an atm_step ctx or explicit."""

    def __init__(self, dtsrc, sha, imaxj_mask, itimei):
        self.dtsrc, self.bysha = float(dtsrc), 1.0 / float(sha)
        self.mask3 = jnp.asarray(np.asarray(imaxj_mask, bool))[:, :, None]
        self.itimei = int(itimei)

    @classmethod
    def from_ctx(cls, ctx):
        import atm_step as A
        import dyn_step as ds
        return cls(ctx.dtsrc, ctx.sha, A.imaxj_mask(ctx), ds.ITIMEI)


def empty_hold(carry=None):
    """Hold = the held server outputs (all OUT_FIELDS + AIJ columns), device state between steps.  Zeros except the carry given
    (SNOAGE RQT KLIQ: the restart/live values at the first call)."""
    h = OrderedDict((k, jnp.zeros(OUT_SHAPES[k], jnp.float64)) for k in OUT_FIELDS)
    h["AIJC"] = jnp.zeros((IM, JM, len(AIJ_COLS)), jnp.float64)
    for k, v in (carry or {}).items():
        h[k] = jnp.asarray(v, jnp.float64)
    return h


def hold_from_outputs(out, aij_cols=None):
    h = OrderedDict((k, jnp.asarray(out[k], jnp.float64)) for k in OUT_FIELDS)
    h["AIJC"] = jnp.zeros((IM, JM, len(AIJ_COLS)), jnp.float64) if aij_cols is None else jnp.asarray(aij_cols, jnp.float64)
    return h


def make_handoff_step(handoff, const):
    """Returns a jitted function
        step(state, surf, hold, itime, seed, cosz1_step) -> (T_new, Q_new, hold_new, cosz1_used)
    state: dict with ATM_IN + CLOUD_IN arrays of THIS step (T, Q after CONDSE; the other atmosphere arrays of the step);
    surf: dict of SURF_IN arrays; hold: see empty_hold; itime int32 scalar; seed int32 scalar (used on radiation steps only);
    cosz1_step (IM,JM): per-step COSZ1 for the four non-radiation steps (recorded or drv_zenith); on a radiation step COSZ1 is the server's.
    The carry (SNOAGE RQT KLIQ) is read from `hold`; on a radiation step the masked CLDSS/CLDMC of the server are held for the next CONDSE entry."""

    def rad_branch(T, Q, packet_state, surf, hold, itime, seed, cosz1_step):
        carry = {k: hold[k] for k in CARRY_IN}
        pk = assemble_packet_jax({k: packet_state[k] for k in ATM_IN}, {k: packet_state[k] for k in CLOUD_IN}, carry, surf)
        out, aijc = handoff.call(pk, itime, seed)
        new = OrderedDict(out)
        new["AIJC"] = aijc
        t_new = _radia_T_jax(T, out["SRHR"], out["TRHR"], out["COSZ1"], packet_state["MA"], packet_state["PK"], const.mask3, const.dtsrc,
                             const.bysha)
        return t_new, out["Q"], new, out["COSZ1"]

    def held_branch(T, Q, packet_state, surf, hold, itime, seed, cosz1_step):
        t_new = _radia_T_jax(T, hold["SRHR"], hold["TRHR"], cosz1_step, packet_state["MA"], packet_state["PK"], const.mask3, const.dtsrc,
                             const.bysha)
        return t_new, Q, hold, cosz1_step

    @jax.jit
    def step(state, surf, hold, itime, seed, cosz1_step):
        is_rad = ((itime - const.itimei) % NRAD) == 0
        T, Q = state["T"], state["Q"]
        return jax.lax.cond(is_rad, rad_branch, held_branch, T, Q, state, surf, hold, itime, seed, cosz1_step)

    return step


def is_radiation_step(itime, itimei):
    return (int(itime) - int(itimei)) % NRAD == 0
