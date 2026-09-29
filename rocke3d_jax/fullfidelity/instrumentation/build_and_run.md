# Building and running the instrumented ModelE (P2SAoM40)

Never edit the original tree; work in a copy (236 MB, everything but ModelE_Support):

    SRC=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0
    COPY=<scratch>/mE2
    rsync -a --exclude=ModelE_Support $SRC/ $COPY/
    cd $COPY/model
    patch ATM_DRV.f < <this dir>/ATM_DRV.f.patch     # adds ffdump + 5 call sites
    patch MODELE.f  < <this dir>/MODELE.f.patch      # pre/post SURFACE call sites
    patch ATURB.f   < <this dir>/ATURB.f.patch       # entry/exit dumps of atm_diffus (ffa_* files)
    patch PBL_DRV.f < <this dir>/PBL_DRV.f.patch     # one record per PBL call (ffp_* files)
    patch SURFACE.f < <this dir>/SURFACE.f.patch     # one record per ocean/ice tile (ffs_* files)
    patch SURFACE_LANDICE.f < <this dir>/SURFACE_LANDICE.f.patch  # land-ice tile (ffl_* files)
    patch SEAICE_DRV.f < <this dir>/SEAICE_DRV_precsi.f.patch     # PRECIP_SI (ffw_* files, D26) -- apply
      AFTER SEAICE_DRV.f.patch (below); adds ffdump_precsi call site
    patch LAKES.f < <this dir>/LAKES_precip_lk.f.patch            # PRECIP_LK (ffv_* files, D27) -- apply
      AFTER LAKES.f.patch (below); adds ffdump_precip_lk call site
    patch ATM_DRV.f < <this dir>/ATM_DRV_precsi.f.patch           # adds ffdump_precsi itself -- apply
      AFTER ATM_DRV.f.patch (above)
    patch ATM_DRV.f < <this dir>/ATM_DRV_precip_lk.f.patch        # adds ffdump_precip_lk itself -- apply
      AFTER ATM_DRV_precsi.f.patch (immediately above)
    patch LANDICE_DRV.f < <this dir>/LANDICE_DRV_precli.f.patch   # PRECIP_LI (ffx_* files, D28)
    patch ATM_DRV.f < <this dir>/ATM_DRV_precli.f.patch           # adds ffdump_precli itself -- apply
      AFTER ATM_DRV_precip_lk.f.patch (immediately above)
    patch ICEDYN_DRV.f < <this dir>/ICEDYN_DRV_dynsi.f.patch      # DYNSI/VPICEDYN (ffy_*/ffz_* files, D29)
    patch ATM_DRV.f < <this dir>/ATM_DRV_dynsi.f.patch            # adds ffdump_geom/ffdump_dynsi_in/out --
      apply AFTER ATM_DRV_precli.f.patch (immediately above); units 970-972 (981-983 conflict with
      real PBL_DRV.f/other usage -- checked, do not reuse without re-grepping the whole tree first)
    patch SEAICE_DRV.f < <this dir>/SEAICE_DRV_apress.f.patch     # CALC_APRESS (ffz_apress_* files, D30)
    patch ATM_DRV.f < <this dir>/ATM_DRV_apress.f.patch           # adds ffdump_apress itself -- apply
      AFTER ATM_DRV_dynsi.f.patch (immediately above); unit 977
    patch SEAICE_DRV.f < <this dir>/SEAICE_DRV_s2ag.f.patch       # seaice_to_atmgrid (ffz_s2ag_* files, D31)
      -- apply AFTER SEAICE_DRV_apress.f.patch (above)
    patch ATM_DRV.f < <this dir>/ATM_DRV_s2ag.f.patch             # adds ffdump_s2ag itself -- apply
      AFTER ATM_DRV_apress.f.patch (immediately above); unit 978
    patch SEAICE_DRV.f < <this dir>/SEAICE_DRV_underice.f.patch   # UNDERICE (ffz_undocn_*/ffz_undlk_*
      files, D32) -- apply AFTER SEAICE_DRV_s2ag.f.patch (above)
    patch ATM_DRV.f < <this dir>/ATM_DRV_underice.f.patch         # adds ffdump_underice_ocn/_lake --
      apply AFTER ATM_DRV_s2ag.f.patch (immediately above); units 979/980
    patch OCNDYN.f < <this dir>/OCNDYN_precip_oc.f.patch          # PRECIP_OC (ffz_precoc_* files,
      D33 -- first Stage 2/ocean-core item)
    patch ATM_DRV.f < <this dir>/ATM_DRV_precip_oc.f.patch        # adds ffdump_precip_oc itself --
      apply AFTER ATM_DRV_underice.f.patch (immediately above); unit 973 (reused from D29's vacated
      debug-only units, re-checked for conflicts first)
    patch OCNDYN.f < <this dir>/OCNDYN_osourc.f.patch             # OSOURC (ffz_osourc_* files, D34)
      -- apply AFTER OCNDYN_precip_oc.f.patch (above)
    patch ATM_DRV.f < <this dir>/ATM_DRV_osourc.f.patch           # adds ffdump_osourc itself --
      apply AFTER ATM_DRV_precip_oc.f.patch (immediately above); unit 974
    patch OCNDYN.f < <this dir>/OCNDYN_ground_oc_sweep.f.patch    # GROUND_OC below-freezing sweep
      (ffz_gocsw_* files, D35) -- apply AFTER OCNDYN_osourc.f.patch (above)
    patch ATM_DRV.f < <this dir>/ATM_DRV_ground_oc_sweep.f.patch  # adds ffdump_ground_oc_sweep --
      apply AFTER ATM_DRV_osourc.f.patch (immediately above); unit 975
    (ATM_DRV.f.patch already includes ffdump_aturb; apply it once)
    source <repo>/rocke3d_jax/fullfidelity/env_modele.sh
    export SOCRATESPATH=$SRC/ModelE_Support/socrates  # required, else socrates depend fails
    cd $COPY/decks && gmake RUN=P2SAoM40 $COPY/model/P2SAoM40.bin    # ~1 min (3 ifort calls)

Run (scratch dir; copy I, P2SAoM40, P2SAoM40ln/uln, runtime_opts, fort.1.nc from
`ModelE_Support/huge_space/P2SAoM40`; use the *instrumented* P2SAoM40.bin):

    # short run: edit I so YEARE/MONTHE/DATEE/HOURE ends after N steps, e.g.
    #   YEARE=1950,MONTHE=11,DATEE=26,HOURE=3   (6 steps from 1950-11-26 00:00)
    export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH
    export OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
    FFD_START=33312 FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1
    # (corrected 2026-09-28, D26: `-l run.PRT` is not a real flag -- see MODELE_DRV.f's arg parser,
    # only -r/-cold-restart/-i/--time exist; must `sh P2SAoM40ln` first to set up input-file symlinks)
    # CRITICAL (found in D29, cost real debugging time): `./P2SAoM40` (the file actually executed) is
    # a SEPARATE FILE from `P2SAoM40.bin` (the gmake build output), NOT a symlink -- copy the rebuilt
    # binary to BOTH names in the run dir (`cp P2SAoM40.bin P2SAoM40.bin` AND `cp P2SAoM40.bin P2SAoM40`)
    # after every rebuild, or the run silently keeps executing a stale binary with none of your new
    # instrumentation. Symptom: your new dump files never appear and even an unconditional `STOP`
    # statement placed as literally the first line of the routine never fires. Don't trust a timer-table
    # trip count as proof this run executed a given routine either -- GISS ModelE checkpoints its
    # cumulative CPU-timer table into the restart file, so a stale count carries forward unchanged
    # across reruns from the same restart even when the routine never actually runs.
    # CRITICAL (found in D27, sharpened in D36): before EVERY rerun in a shared/reused scratch dir,
    # reset BOTH fort.1.nc AND fort.2.nc to the same pristine start itime -- not just fort.1.nc. The
    # restart reader picks whichever of the two has the LATER itime (look for "RESTART DISK READ,
    # UNIT 1" vs "UNIT 2" in run.PRT), and GISS ModelE double-buffers its checkpoint writes across
    # both files, so a prior run in the same dir can leave fort.1.nc pristine but fort.2.nc advanced
    # (or vice versa) -- the next run silently starts from the advanced one and does zero *useful*
    # steps inside your FFD window even though it "terminates normally" and the timer table looks
    # plausible. Symptom: even an unconditional, ungated dump call never fires. Verify with
    # `python3 -c "import netCDF4 as nc; print(nc.Dataset('fort.1.nc').variables['itime'][...])"`
    # on both files before trusting a run. Fix: copy a known-pristine fort.1.nc over BOTH names.
    # PERMANENT FIX (D37): all 3 test dates' pristine restarts are archived at
    #   ff_data/_pristine_restarts/fort1_{nov26,dec01,jan01}_itime{33312,33552,17520}.nc
    # Restore before EVERY rerun with (per run dir, using that date's archived file as SRC):
    #   rm -f fort.1.nc fort.2.nc && cp $SRC fort.1.nc && cp $SRC fort.2.nc
    # nov26's source is also the untouched master at
    # `ModelE_Support/huge_space/P2SAoM40/fort.1.nc` (itime=33312, never run in place); jan01's is
    # `ModelE_Support/huge_space/P2SAoM40/1JAN1950.rsfP2SAoM40.nc` (itime=17520, an exact match,
    # found in D37). dec01 (itime=33552) has NO such external archive -- D36's rerun consumed the
    # one that existed at the time -- so its ff_data/_pristine_restarts copy is the ONLY pristine
    # source; if it's ever lost, regenerate by copying the nov26 master into a scratch run dir,
    # editing `I`'s `YEARE=1950,MONTHE=12,DATEE=1,HOURE=0` (a full 5-day/240-step run instead of
    # the usual 6-step window), running to completion (~13 min), and re-archiving the result.

Dumps: `ffd_<itime>_<tag>.bin`, tags pre_condse, post_condse, post_radia,
pre_surface, post_surface, pre_aturb, post_aturb (last two are the dummy
phase-2 ATM_DIFFUS slot; real ATURB runs inside SURFACE, SURFACE.f:1172).
Big-endian stream; read with `ffdump_reader.py`. Array layouts as allocated:
U,V,T,Q,QCL,QCI = (I,J,L); MA,PK,PMID,PEDN,PDSIG = (L,I,J); P=(I,J).
Validated: instrumented binary vs original after 6 steps — 256/257 restart
variables bitwise identical (only `cputime` differs: wall-clock timers).

ATURB dumps: `ffa_<itime>_c<k>_{in,out}.bin` (k=1,2: NIsurf=2 substeps per step;
dtime=900 s), begin with one big-endian float64 (dtime), read with
`read_dump(path, header_f8=1)`. Fields: U,V,T,Q (I,J,L); UALIJ,VALIJ,EGCM,W2GCM,
MA,PK,PEK,PMID,PEDN,PDSIG (L,I,J); UFLUX1,VFLUX1,TFLUX1,QFLUX1,TSAVG,QSAVG,PBLHT,
DCLEV,PBLPTOP (I,J). NOTE: cells not computed by the model (polar i>1, halo)
contain uninitialised garbage in the flux arrays (e.g. 3e208) - mask to valid
cells (i=1 only at j=1 and j=JM) before comparing.

PBL dumps: `ffp_<itime>.bin`, one 154-double big-endian record per PBL call (all tile types;
~9.2k records/step). Column layout: see rec(1..154) assignments in PBL_DRV.f.patch
(inputs 1-89: i,j,itype,ihc, pbl_args inputs, profiles; outputs 90-154). Read with
`np.fromfile(path,'>f8').reshape(-1,154)`. `ffa_consts.txt` also holds tf, stbo, lhs, mrat, rvap, ...
Note: fixed-form source, keep every added line <= 72 columns.

SURFACE tile dumps: `ffs_<itime>.bin`, 90-double big-endian records, one per ocean (itype 1) or sea-ice
(itype 2) tile per surface substep (~7k/step). Columns: see srec(1..81) in SURFACE.f.patch
(inputs at BL entry, PBL outputs, tile outputs). Entries not set for a tile type hold -1e300 or stale values.

Land-ice tile dumps: `ffl_<itime>.bin`, 60-double records, ~692/step (both substeps); columns lrec(1..46) in SURFACE_LANDICE.f.patch.

Land (GHY) dumps: patch `giss_LSM/GHY.f` and `GHY_DRV.f` with `GHY.f.patch`/`GHY_DRV.f.patch` in
addition to the ATM_DRV.f patch. `ffg_<itime>.bin`: 450-double records, one per land (`fearth>0`) tile
per DTsrc step; `ffg_thm.txt` (soil property table, written once) is needed by the reference port only
for cross-checking, not required at runtime (ghy_ref.py recomputes it). Record layout: see
`fullfidelity/ghy_compare.py`'s `unpack()`.

Sea-ice ground-thermodynamics dumps: patch `SEAICE_DRV.f` with `SEAICE_DRV.f.patch` in addition to the
ATM_DRV.f patch. `ffi_<itime>.bin`: 60-double records, one per sea-ice/lake-ice cell (`POICE>0`) per
GROUND_SI call. Column layout: see `fullfidelity/seaice_compare.py` module docstring.

Tile-aggregation dumps: `fft_<itime>.bin`, 40-double records, one per grid cell (3312/step): 4 patches'
ftype/uflux1/vflux1/dth1/dq1/tsavg/qsavg (cols 3-30) then the composite atmsrf values (cols 31-36).
Same SURFACE.f patch as the ocean/ice tile dump.

ADDICE dumps: `ffn_<itime>.bin`, 40-double records, one per `FORM_SI` call (all water-covered cells).
SIMELT dumps: `ffm_<itime>.bin`, 30-double records, one per `MELT_SI` call. Same `SEAICE_DRV.f.patch`.

Lake mixing dumps: patch `LAKES.f` with `LAKES.f.patch` in addition to the ATM_DRV.f patch (adds
`ffdump_lakes`). `ffl2_<itime>.bin`: 40-double records, one per lake cell (`FLAKE>0`) per `GROUND_LK`
call (~600-1000/step depending on date, since not every cell has a lake). Column layout: see
`fullfidelity/lakes_compare.py` module docstring. Note: `GROUND_LK` always calls `LKMIX` with `TKE=0.`
(the `U2rho` entrainment term is commented out in this rundeck's source), so the TKE-driven entrainment
branch inside `LKMIX` is never exercised by these dumps -- ported faithfully but unvalidated in practice.

PRECIP_SI dumps (Stage 1 of the DYNSI/ocean port, D26): patch `SEAICE_DRV.f` with
`SEAICE_DRV_precsi.f.patch` (after `SEAICE_DRV.f.patch`) and `ATM_DRV.f` with `ATM_DRV_precsi.f.patch`
(after `ATM_DRV.f.patch`; adds `ffdump_precsi`, unit 986). `ffw_<itime>.bin`: 40-double records, one per
sea-ice cell (`POICE>0`) per `PRECIP_SI` call, ~700-800/step (same cell population as `ffi_*`/GROUND_SI,
since both are `si_ocn`-based). Columns: 1:i 2:j 3:dtsrc 4:snow(in) 5:msi2(in) 6:9=hsil(in) 10:13=ssil(in)
14:prcp 15:enrgp -- after `PREC_SI` -- 16:snow 17:msi2 18:21=hsil 22:25=ssil 26:29=tsil 30:run0 31:srun0
32:erun0 33:wetsnow 34:cmprs (0-based columns are all -1, see `fullfidelity/precsi_compare.py`).
PRECIP_LK dumps (D27): patch `LAKES.f` with `LAKES_precip_lk.f.patch` (after `LAKES.f.patch`) and `ATM_DRV.f`
with `ATM_DRV_precip_lk.f.patch` (after `ATM_DRV_precsi.f.patch`; adds `ffdump_precip_lk`, unit 985).
`ffv_<itime>.bin`: 40-double records, one per lake+land-ice cell (`FLAKE+FLICE>0`, both scalars not
per-tile) per `PRECIP_LK` call, ~950-1000/step. Columns: 1:i 2:j 3:flake 4:flice 5:rsi 6:prcp 7:enrgp
8:runpsi 9:runo_li(atmgla%RUNO) 10:melti 11:emelti 12:axyp 13:mwl(in) 14:gml(in) 15:tlake(in) 16:mldlk(in)
-- after `PRECIP_LK` -- 17:mwl 18:gml 19:tlake 20:mldlk 21:dlake 22:glake 23:gtemp 24:gtemp2 25:gtempr
(0-based columns are all -1, see `fullfidelity/precip_lk_compare.py`).
`IRRIG_LK` (irrigation withdrawal, `LAKES.f`, 128 lines, real for this rundeck -- `IRRIGATION_ON` is
defined) is called just before `PRECIP_LK` (`SURFACE.f:~300-322`) and remains not-yet-instrumented
(depends on an external prescribed irrigation-demand dataset, deferred as its own item).

PRECIP_LI dumps (D28): patch `LANDICE_DRV.f` with `LANDICE_DRV_precli.f.patch` and `ATM_DRV.f` with
`ATM_DRV_precli.f.patch` (after `ATM_DRV_precip_lk.f.patch`; adds `ffdump_precli`, unit 984). `ffx_<itime>.bin`:
40-double records, one per land-ice tile (`ftype>0 and prcp>0`) per `PRECIP_LI` call (~200-250/step; only
`ihc=1` present, since `NHC=1` for this rundeck, D22). Columns: 1:i 2:j 3:ihc 4:ftype 5:prcp 6:enrgp 7:snow(in)
8:tg1(in) 9:tg2(in) -- after `PRECIP_LI` -- 10:snow 11:tg1 12:tg2 13:run0(=RUNO) 14:edifs(=E1) 15:difs(=IMPLM)
16:erun2(=IMPLH) (0-based columns are all -1, see `fullfidelity/precli_compare.py`). Note: the real record is
entirely cold precipitation (`ENRGP<0`); the "rain" branch is checked against synthetic inputs instead (see
`tests/test_precli_jax.py`).

DYNSI/VPICEDYN dumps (D29): patch `ICEDYN_DRV.f` with `ICEDYN_DRV_dynsi.f.patch` and `ATM_DRV.f` with
`ATM_DRV_dynsi.f.patch` (after `ATM_DRV_precli.f.patch`; adds `ffdump_geom` unit 970, `ffdump_dynsi_in` unit
971, `ffdump_dynsi_out` unit 972). Whole-grid snapshots, not per-cell records (this is a real 2D iterative
solve on the ice-dyn B-grid, NX1=IMICDYN+2 x NY1=JMICDYN = 74x46 for this rundeck). `ffz_geom.bin`: written
ONCE per run (static: analytic lat-lon geometry + FOCEAN-derived masks, no dependence on any prognostic
field) -- header `dble(nx1,ny1,imicdyn,jmicdyn)` then `DXT/DXU/BYDX2/BYDXR(1:nx1)`, `DYT/DYU/BYDY2/BYDYR/
CST/CSU/TNGT/TNG/BYCSU(1:ny1)`, `SINEN/BYDXDY/HEFFM/UVM(1:nx1,1:ny1)` (Fortran column-major), `FOCEAN
(1:imicdyn,1:ny1)`. `ffy_<itime>_in.bin`/`ffy_<itime>_out.bin`: one pair per `DYNSI` call (once per step, since
this non-cubed-sphere/non-STANDALONE_HYCOM rundeck takes `OCN_DRV.f`'s single `CALL DYNSI(atmice,iceocn,
si_ocn)` branch). `_in`: header `dble(itime)` then `iRSI/iMSI(1:imicdyn,1:ny1)`, `GAIRX/GAIRY/GWATX/GWATY/
PGFUB/PGFVB/HEFF/AREA/AMASS/COR/UICE(:,:,1)/VICE(:,:,1)(1:nx1,1:ny1)` -- the real recorded inputs to
`VPICEDYN` (the atm-stress/ocean-current regrid and `HEFF`/`AREA`/`AMASS`/`COR` derivation happen earlier in
`DYNSI`'s own body and are not yet re-derived, since they depend on not-yet-ported ocean-model fields
`OGEOZA`/`UOSURF`/`VOSURF`). `_out`: header `dble(itime)` then `UICE(:,:,1)/VICE(:,:,1)/DMU/DMV
(1:nx1,1:ny1)`, `USI/VSI/DMUI/DMVI(1:imicdyn,1:ny1)`. See `fullfidelity/icedyn_geom_compare.py` (geometry,
bitwise exact) and `fullfidelity/dynsi_compare.py` (VPICEDYN, bitwise/float64-exact after D29's 3 bug fixes
-- see `FULL_FIDELITY_DELTAS.md`) for the exact unpack layout.

CALC_APRESS dumps (D30): patch `SEAICE_DRV.f` with `SEAICE_DRV_apress.f.patch` and `ATM_DRV.f` with
`ATM_DRV_apress.f.patch` (after `ATM_DRV_dynsi.f.patch`; adds `ffdump_apress`, unit 977). `ffz_apress_
<itime>.bin`: 10-double records, one per atm-grid cell per `CALC_APRESS` call (~3,170/step, once per full
step, same call cadence as `DYNSI`). Columns: 1:i 2:j 3:srfp(hPa) 4:rsi 5:snowi 6:msi 7:apress(Pa, output)
(0-based columns are -1, see `fullfidelity/apress_compare.py`). `GRAV` is a `USE_PLANET_RAD` runtime
parameter like D29's `RADIUS`; the compare script infers it algebraically from a real ice-covered cell.

seaice_to_atmgrid dumps (D31): patch `SEAICE_DRV.f` with `SEAICE_DRV_s2ag.f.patch` (after
`SEAICE_DRV_apress.f.patch`) and `ATM_DRV.f` with `ATM_DRV_s2ag.f.patch` (after `ATM_DRV_apress.f.patch`; adds
`ffdump_s2ag`, unit 978). `ffz_s2ag_<itime>.bin`: 20-double records, one per atm-grid cell per call, from
EVERY call site (`ATM_DRV.f`/`OCN_DRV.f`/`SURFACE.f` all call the same pure function; not distinguished in the
dump, just pooled -- ~12,680-19,020 records/step). Columns: 1:i 2:j 3:ncall(diagnostic only, increments each
time i=j=1 is seen) 4:rsi 5:snowi 6:msi 7:hsi(1) 8:hsi(2) 9:ssi(1) 10:ssi(2) 11:ssi(3) 12:ssi(4) -- after the
2nd loop -- 13:gtemp 14:gtemp2 15:gtempr 16:zsnowi 17:zsi 18:fwsim (0-based columns are -1, see
`fullfidelity/seaice_to_atmgrid_compare.py`). Note: the 3rd loop's `RESET_SURF_FLUXES` call (radiation-
adjacent, touches only `RAD_COM`'s FSF/TRSURF) is deliberately not dumped/ported.

UNDERICE dumps (D32): patch `SEAICE_DRV.f` with `SEAICE_DRV_underice.f.patch` (after `SEAICE_DRV_s2ag.f.patch`)
and `ATM_DRV.f` with `ATM_DRV_underice.f.patch` (after `ATM_DRV_s2ag.f.patch`; adds `ffdump_underice_ocn` unit
979, `ffdump_underice_lake` unit 980). `ffz_undocn_<itime>.bin` (ocean domain, from `OCN_DRV.f`'s call):
20-double records, one per real (`DOPOINT`) sea-ice cell (~480-800/step). Columns: 1:i 2:j 3:tic 4:si 5:tm
6:sm 7:dh 8:ustar 9:coriol 10:mlsh -- after `iceocean_fluxes` -- 11:mflux 12:sflux 13:hflux (all un-scaled by
DTsrc; scale by 1800s to match `FMSI_IO`/`FHSI_IO`/`FSSI_IO`). `ffz_undlk_<itime>.bin` (lakes domain, from
`SURFACE.f`'s call): 20-double records, one per real lake-ice cell (~300-500/step). Columns: 1:i 2:j 3:tic
4:tm 5:dh 6:mlsh 7:dlake 8:glake -- after `icelake_fluxes`+flux-limiting -- 9:mflux 10:hflux (0-based columns
are -1, see `fullfidelity/underice_compare.py`).

PRECIP_OC dumps (D33, first Stage 2/ocean-core item): patch `OCNDYN.f` with `OCNDYN_precip_oc.f.patch` and
`ATM_DRV.f` with `ATM_DRV_precip_oc.f.patch` (after `ATM_DRV_underice.f.patch`; adds `ffdump_precip_oc`, unit
973). `ffz_precoc_<itime>.bin`: 20-double records, one per real (`FOCEAN>0` and `oPREC>0`) ocean-grid cell
per call (~1,810-1,893/step, once per full step). Columns: 1:i 2:j 3:focean 4:oprec 5:orsi 6:orunpsi
7:oeprec 8:oerunpsi 9:osrunpsi 10:dxypo 11:mo(in) 12:g0m(in) 13:s0m(in) -- after `PRECIP_OC` -- 14:mo
15:g0m 16:s0m (0-based columns are -1, see `fullfidelity/precip_oc_compare.py`). Ocean grid is IMO=72,
JMO=46 (`ORES_5x4.F90`) -- same resolution/index space as the atmosphere grid for this rundeck.

OSOURC dumps (D34, called from GROUND_OC): patch `OCNDYN.f` with `OCNDYN_osourc.f.patch` and `ATM_DRV.f`
with `ATM_DRV_osourc.f.patch` (after `ATM_DRV_precip_oc.f.patch`; adds `ffdump_osourc`, unit 974).
`ffz_osourc_<itime>.bin`: 78-double records (20 header + 4x13 layer arrays + 6 output scalars; LMO=13 fixed
for this rundeck's `OCN_LAYERING L13`), one per real (`FOCEAN>0`) ocean cell per call (~2,095/step). Header
columns: 1:i 2:j 3:roice 4:mo(in) 5:s0m(in) 6:dxypj 7:bydxypj 8:lmij 9:runo 10:runi 11:eruno 12:eruni
13:sruno 14:sruni 15:srox(1) 16:srox(2) 17:lmo 18:mo(out) 19:s0m(out); then (all length LMO): g0ml(in),
gzml(in), g0ml(out), gzml(out); then 6 trailing doubles: dmoo,deoo,dmoi,deoi,dsoo,dsoi (see
`fullfidelity/osourc_compare.py` for the exact unpack). `FSR`/`FSRZ`/`LSRPD` (solar-penetration profile) are
NOT dumped -- derived analytically in `osourc_ff.py`/`osourc_jax.py` from `OCEAN_COM.f`'s `init_solar`.

GROUND_OC below-freezing sweep dumps (D35): patch `OCNDYN.f` with `OCNDYN_ground_oc_sweep.f.patch` and
`ATM_DRV.f` with `ATM_DRV_ground_oc_sweep.f.patch` (after `ATM_DRV_osourc.f.patch`; adds
`ffdump_ground_oc_sweep`, unit 975). `ffz_gocsw_<itime>.bin`: 20-double records, one per real `(i,j,l)`
layer triple (`FOCEAN>0`, `L=2..LMM(I,J)`) per call (~22,224/step). Columns: 1:i 2:j 3:l 4:mo(in) 5:g0m(in)
6:s0m(in) 7:dxypj 8:pcorr(=SHCGS(GF00,S0L)*8.19d-8*P0L, recorded since SHCGS needs the OFTAB table) 9:p0l
(recorded, used directly in the TF0 correction too, not just via pcorr) -- after the sweep -- 10:mo 11:g0m
12:s0m (0-based columns are -1, see `fullfidelity/ground_oc_sweep_compare.py`).
