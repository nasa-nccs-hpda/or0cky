"""D176: wiring of drv_radcols into the closed coupled path (surface_loop_v2.Loop2 / surface_loop.run_coupled), without editing any existing file.

The coupled path reads the real tile records of every step (atm_step.surface_records) and overwrites the columns that are functions of OUR state
(surface_loop.apply_state_to_records).  `CoupledRad.install()` additionally

  * overwrites the RADIATION columns (drv_radcols.fill_records) after apply_state_to_records, from the RadSurf state and COSZ1;
  * replaces land_chain.infer_trup (land TRUP inferred from the recorded patch) by TRSURF(4);
  * keeps the RadSurf state current: radiation steps load the server output; between radiation steps the ice-fraction legs of RESET_SURF_FLUXES
    are applied from OUR sea-ice / lake-ice fractions at the three (ocean cells) and two (lake cells) points described in drv_radcols:
        (a) after surface_pre (MELT_SI + seaice_to_atmgrid; v2: S1): RSI at the end of the previous step -> S1 RSI      [ocean and lake cells]
        (b) ocean form_si:  S1 RSI -> post['ice_pre_adv'] RSI                                                          [ocean cells]
        (c) ADVSI:          post['ice_pre_adv'] RSI -> end-of-step RSI                                                  [ocean cells]
        (F) lake form_si:   S1 RSI -> post['ice_after_gsi'] RSI (lake cells; GROUND_SI does not change RSI)            [lake cells]
    through wrappers of surface_loop.surface_pre and surface_loop_v2.surface_post_v2 (module attributes swapped and restored).

Inputs from outside: server_out(it) -> radiation-server output dict on radiation steps (the persistent server in the free-radiation day, or the
recorded rsv_n26_<it>_out.bin in teacher tests); cosz1(it) -> COSZ1 (IM,JM) for the step (server output on radiation steps; on the other steps NOT
computed here: pass the recorded one until D177).  The day boundary (daily_LAKE: FLAKE changes -> RESET_SURF_FLUXES) is NOT computed: daily_LAKE is not
ported; `day_boundary_calls` counts the steps where it should have been applied (itime % 48 == 0) so that a run cannot pass over it silently.
"""
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import drv_radcols as DR  # noqa: E402

NDAY = 48


class CoupledRad:
    def __init__(self, server_out, cosz1, ca=None, is_rad_step=None):
        import atm_step as A
        self.server_out, self.cosz1_fn = server_out, cosz1
        self.ca = DR.ghg_ca() if ca is None else ca
        self.is_rad = is_rad_step or A.is_radiation_step
        self.rad = DR.RadSurf()
        self.loaded_it = None
        self.cur_it = None
        self.S1_rsi = None
        self.prev_end_rsi = None
        self.geo = None
        self.log = []
        self.day_boundary_calls = []
        self.legs_applied = 0

    # -- state for the step about to run
    def state(self, it):
        self.cur_it = it
        if self.is_rad(it) and self.loaded_it != it:
            self.rad.from_server(self.server_out(it))
            self.loaded_it = it
        if not self.rad.ready:
            raise RuntimeError("RadSurf has no radiation-server output yet (first step must be a radiation step)")
        if it % NDAY == 0 and it != self.first_it and it not in self.day_boundary_calls:
            self.day_boundary_calls.append(it)
        cz = self.rad.cosz1_server if self.is_rad(it) else np.asarray(self.cosz1_fn(it), float)
        return self.rad, cz

    first_it = None

    # -- leg hooks
    def _geo(self, st):
        g = st['geo']
        return g['focean'] > 0, g['flake'] > 0

    def after_pre(self, S, S1, st):
        """(a): RSI at the end of the previous step (S) -> after MELT_SI (S1)."""
        ocn, lake = self._geo(st)
        if self.rad.ready and not self.is_rad(self.cur_it):
            self.rad.ice_leg(S['ice']['rsi'], S1['ice']['rsi'], ocn | lake)
            self.legs_applied += 1
        self.S1_rsi = np.array(S1['ice']['rsi'])

    def after_post(self, post, S2, st):
        """(b), (c) ocean cells and (F) lake cells; applied at the end of the step (they act on the NEXT step's records)."""
        ocn, lake = self._geo(st)
        self.rad.ice_leg(self.S1_rsi, post['ice_pre_adv']['rsi'], ocn)
        self.rad.ice_leg(post['ice_pre_adv']['rsi'], S2['ice']['rsi'], ocn)
        self.rad.ice_leg(self.S1_rsi, post['ice_after_gsi']['rsi'], lake)
        self.legs_applied += 3

    # -- installation
    @contextlib.contextmanager
    def install(self, it0):
        import atm_step as A
        import land_chain as LC
        import surface_loop as L
        import surface_loop_v2 as V2
        self.first_it = it0
        orig = dict(rec=A.surface_records, pre=L.surface_pre, app=L.apply_state_to_records, post=V2.surface_post_v2, trup=LC.infer_trup)
        me = self
        stash = {}

        def rec_w(R):
            r = orig['rec'](R)
            stash['R_it'] = R.itime
            me.state(R.itime)
            return r

        def pre_w(S, st, inp, melt_done=None):
            S1, mid = orig['pre'](S, st, inp, melt_done=melt_done)
            me.after_pre(S, S1, st)
            stash['st'] = st
            return S1, mid

        def app_w(rec, S1, mid, st, ps_ij):
            r, rep = orig['app'](rec, S1, mid, st, ps_ij)
            rad, cz = me.state(me.cur_it)
            return DR.fill_records(r, rad, cz, me.ca), rep

        def post_w(S, st, inp, mid, V, ocean_stages=None):
            S2, post = orig['post'](S, st, inp, mid, V, ocean_stages)
            me.after_post(post, S2, st)
            return S2, post

        def trup_w(g, patch_dth1, dtsurf):
            return DR.land_trup(g, me.rad)

        A.surface_records, L.surface_pre, L.apply_state_to_records, V2.surface_post_v2, LC.infer_trup = rec_w, pre_w, app_w, post_w, trup_w
        try:
            yield self
        finally:
            A.surface_records, L.surface_pre, L.apply_state_to_records, V2.surface_post_v2, LC.infer_trup = (
                orig['rec'], orig['pre'], orig['app'], orig['post'], orig['trup'])
