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

get_dq_cond / get_dq_evap dumps (D89, first cloud delta): patch `CLOUDS2.F90` with `CLOUDS2_dq.f90.patch`
(adds `call ffdq_site(k)` before each of the 6 call sites, CLOUDS2.F90:1364/2067/2716/4002/4013/4425, and a
`call ffdump_dq(...)` before the `return` of get_dq_cond/get_dq_evap) and `ATM_DRV.f` with
`ATM_DRV_clouds_dq.f.patch` (adds `ffdq_site` and `ffdump_dq`, unit 1050, inserted before the unchanged
`#ifdef CACHED_SUBDD` / `subroutine accum_subdd_atm` block, so it is independent of the order of the other
ATM_DRV patches; verified to apply both to pristine ATM_DRV.f and after ATM_DRV.f.patch). Both patches were
generated by `diff -u` against pristine files. No other patch is needed: the helper reads FFD_START/FFD_NSTEP
itself. The validated build used ONLY these two patches on a fresh rsync copy (mE_cloud/mE2). Unit 1050 was
grepped across the whole tree (model/*.f, *.F90, *.h) and all existing instrumentation patches: unused
(other sessions use e.g. 1070 for fltruv; units 970-999 and 1000-1040 are taken by earlier patches).

    # build (note: the make target must be the ABSOLUTE path to model/P2SAoM40.bin, '../model/...' fails with
    # "No rule to make target")
    source <repo>/rocke3d_jax/fullfidelity/env_modele.sh
    export SOCRATESPATH=$SRC/ModelE_Support/socrates
    C=<scratch>/mE_cloud/mE2; cd $C/decks && gmake RUN=P2SAoM40 $C/model/P2SAoM40.bin     # ~1 min incremental
    # The rsync copy keeps the source files read-only (mode r--): `chmod u+w` before patching/editing.

    # run dirs (per date; steps as in the earlier deltas, 6 steps): copy I, P2SAoM40ln, P2SAoM40uln, runtime_opts
    # from ModelE_Support/huge_space/P2SAoM40, `sed` the YEARE/MONTHE/DATEE/HOURE line in I
    # (nov26: 1950,11,26,3; dec01: 1950,12,1,3; jan01: 1950,1,1,3), copy P2SAoM40.bin to BOTH P2SAoM40.bin and
    # P2SAoM40, restore BOTH fort.1.nc and fort.2.nc from ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc,
    # `sh P2SAoM40ln`, then:
    export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH
    export OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1      # the dump counters assume a single thread
    FFD_START=<itime> FFD_NSTEP=6 FFC_DQ_STRIDE=10 ./P2SAoM40 -i I > run.PRT 2>&1     # ~30 s per date
    # (the three dates were run concurrently in separate run dirs)
    # Copy only the new ffc_dq_* files into ff_data/<date>/ (use `\cp -n`).

Dumps: `ffc_dq_<itime>.bin`: 13-double big-endian records, one per SAMPLED call: itime, site, kind (1=cond,
2=evap), ncall, sm, qm, plk, mass, lhx, pl, cond (0 for kind 1), dqsum(out), f(out). site: 1 MSTCNV updraft cond
(CLOUDS2.F90:1364), 2 MSTCNV downdraft evap (2067), 3 MSTCNV precip evap (2716), 4 LSCOND evap liquid (4002),
5 LSCOND evap ice (4013), 6 LSCOND cond (4425). Sampling: every FFC_DQ_STRIDE-th call (default 10, used 10) of
each (itime,site) pair, `ncall` is the running count of ALL calls at that site in that step (so ncall is a
multiple of the stride). ~10k records/step (~1 MB), ~56-60k records per date. `ffc_dq_consts.txt` (written once)
holds bysha, mrat, rvap, tf, lhe, lhs and the stride. Read with `fullfidelity/clouds_dq_compare.py`
(`load_date`). SCM.F90's call to get_dq_cond is not live in this rundeck and not instrumented.

Patches (apply to a fresh `rsync` copy, never the original tree; only these two on top of the base set are needed, they touch no unit number used by earlier patches):

    cd $COPY/model
    patch ATM_DRV.f < <this dir>/ATM_DRV.f.patch          # base (ffdump + call sites)
    patch MODELE.f  < <this dir>/MODELE.f.patch           # base
    patch ATMDYN.f  < <this dir>/ATMDYN_fltruv.f.patch    # D90: 4 dump calls in DYNAM (end-of-DYNAM filter chain)
    patch ATM_DRV.f < <this dir>/ATM_DRV_fltruv.f.patch   # D90: adds ffdump_fltruv_on/_geom/_in/_uv/_ny, unit 1070
      -- apply AFTER ATM_DRV.f.patch; in a full stack apply it last (it only appends at end of file; the
      patch was generated against ATM_DRV.f.patch alone, so re-diff if applied after other ATM_DRV_*.patch files:
      the end-of-file context will differ and `patch` will report an offset/fuzz or reject)
    (ATM_DRV.f and ATMDYN.f are read-only in the rsync copy: `chmod u+w` first, as the original tree has them r/o)

Build: `source env_modele.sh` in the SAME shell (not in a pipeline subshell, or ifort is not found), then
`export SOCRATESPATH=$SRC/ModelE_Support/socrates; cd $COPY/decks && gmake RUN=P2SAoM40 $COPY/model/P2SAoM40.bin` (~50 s).

Run (per date, own run dir; used here: `$COPY/run_<date>`): copy `I P2SAoM40ln P2SAoM40uln runtime_opts modules` from
`ModelE_Support/huge_space/P2SAoM40`, then `\cp model/P2SAoM40.bin` to BOTH `P2SAoM40.bin` and `P2SAoM40` (the binary itself, not
the 1.6 kB wrapper script that is in huge_space), `sh P2SAoM40ln`, set in `I`: `YEARE=1950,MONTHE=11,DATEE=26,HOURE=3`
(nov26) / `MONTHE=12,DATEE=1` (dec01) / `MONTHE=1,DATEE=1` (jan01), `rm -f fort.1.nc fort.2.nc` and copy
`ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc` to BOTH names, then
`FFD_START=<N> FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1` with `LD_LIBRARY_PATH` and `OMP_NUM_THREADS=1` as in build_and_run.md
(~1 min each; the three runs were executed concurrently, rc=0). Itimes: nov26 33312, dec01 33552, jan01 17520.
Copy `ffd_fltruv_*.bin` (25 files per date: geom + 6 steps x {in,flt,ny,out}) into `ff_data/<date>/`; do not overwrite existing files.

Dumps (add to the dump list in build_and_run.md): `ffd_fltruv_<itime>_{in,flt,ny,out}.bin`, `ffd_fltruv_geom.bin`; layouts and
loader in `fullfidelity/dyn_fltruv_compare.py` (`load_in/load_uv/load_ny/load_geom`). Validate: `python3 dyn_fltruv_compare.py [--analytic]`,
`pytest fullfidelity/tests/test_dyn_fltruv_ff.py`.

## D96-D98 (AFLUX/ADVECM/MAtoP, PGF, ADVECV) build and run lines

Patches (apply to a fresh `rsync` copy, never the original tree; independent of the other D-patches: the hunks are local and anchored in unchanged code; verified to apply to pristine files and after `ATM_DRV.f.patch`, `ATM_DRV_clouds_dq`, `ATM_DRV_fltruv`, `ATMDYN_fltruv`). Only these three are needed on top of pristine source (the helper reads FFD_START/FFD_NSTEP itself, no base ATM_DRV/MODELE patch is required):

    cd $COPY/model      # all three files are read-only in the rsync copy: chmod u+w first
    patch ATMDYN.f  < <this dir>/ATMDYN_aflux_pgf_advecv.f.patch   # D96/D97: hooks in AFLUX, ADVECM(+MAtoP), live PGF
    patch MOMEN2ND.f < <this dir>/MOMEN2ND_advecv.f.patch          # D98: hooks in ADVECV
    patch ATM_DRV.f < <this dir>/ATM_DRV_dynB.f.patch              # helpers ffdb_on/ffdb_pass_inc/ffdb_open/ffdb_geom/
                                                                    #   ffdb_aflux_in/out, ffdb_advecm_in/out, ffdb_pgf_in/out,
                                                                    #   ffdb_advecv_in/out; units 1081-1085

Units: 1081 AFLUX, 1082 ADVECM, 1083 geometry, 1084 PGF, 1085 ADVECV (1081-1099 grepped over model/*.f, *.F90, *.h and all instrumentation patches: unused;
D89 uses 1050, D90 1070, another agent 1071-1080). The ATM_DRV patch is anchored after `end subroutine finalize_atm`, so it does not clash with D89 (before `#ifdef CACHED_SUBDD` near `read_aic`) or D90 (end of file).

Build (as for D89/D90): `source env_modele.sh` in the SAME shell, `export SOCRATESPATH=$SRC/ModelE_Support/socrates`,
`cd $COPY/decks && gmake RUN=P2SAoM40 $COPY/model/P2SAoM40.bin` (absolute target; ~50 s incremental on the rsync copy, 0 errors). The real flags are
`-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict` (no FMA, no reassociation), which is why strict left-to-right numpy order matches.

Run (per date, own run dir `$COPY/run_<date>`; the three runs concurrently, rc=0, about a minute each): copy `I P2SAoM40ln P2SAoM40uln runtime_opts` from
`ModelE_Support/huge_space/P2SAoM40`, `\cp model/P2SAoM40.bin` to BOTH `P2SAoM40.bin` and `P2SAoM40`, set the `YEARE/MONTHE/DATEE/HOURE` line of `I`
to 1950,11,26,3 / 1950,12,1,3 / 1950,1,1,3, `rm -f fort.1.nc fort.2.nc` and copy `ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc` to BOTH, `sh P2SAoM40ln`, then
`export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1` and
`FFD_START=<N> FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1` (N = 33312 nov26, 33552 dec01, 17520 jan01). Copy `ffd_aflux_*`, `ffd_pgf_*`, `ffd_advecv_*` into
`ff_data/<date>/` with `\cp -n`.

Dumps (add to the dump list in build_and_run.md; per date 241 files, ~1.8 GB: geom + 6 steps x 5 passes x {aflux, aflux_advecm, pgf, advecv} x {in,out}):
`ffd_aflux_geom.bin`; `ffd_aflux_<itime>_p<k>_{in,out}.bin`; `ffd_aflux_advecm_<itime>_p<k>_{in,out}.bin`; `ffd_pgf_<itime>_p<k>_{in,out}.bin`;
`ffd_advecv_<itime>_p<k>_{in,out}.bin`, with k = leapfrog pass within DYNAM (1 fwd MRCH=0 NS=4, 2 bwd MRCH=-1 NS=4, 3 even MRCH=2 NS=4, 4 odd MRCH=-2 NS=3,
5 even MRCH=2 NS=2; counted at AFLUX entry, reset when itime changes). Record layouts are documented in the docstrings of `dyn_aflux_compare.py`,
`dyn_pgf_compare.py`, `dyn_advecv_compare.py` (loaders `load_*`, geometry loader `dyn_aflux_ff.load_geom`).

Validate: from `fullfidelity/` (bare module imports, as for D90): `python3 dyn_aflux_compare.py [--recorded-avrx] [--imf-pow]`, `python3 dyn_pgf_compare.py [--imf-pow]`,
`python3 dyn_advecv_compare.py`; `PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_dyn_aflux_ff.py test_dyn_pgf_ff.py test_dyn_advecv_ff.py` (572 passed, ~55 s).
`--imf-pow` / the libimf tests need the Intel runtime (`libimf.so`, path in `intel_libm_ff.py`, override with `INTEL_LIBIMF_DIR`); they are skipped elsewhere.
New code files: `dyn_fft72_ff.py` (FFT72 radix FFT, batched), `intel_libm_ff.py`, `dyn_aflux_ff.py`, `dyn_pgf_ff.py`, `dyn_advecv_ff.py` and the three `*_compare.py`.

## D94/D95 (FFT72/AVRX, isotropuv/shap1, SDRAG) build and run lines

Patches (generated by `diff -u` against the PRISTINE source; only these two are needed, no base patches; apply to a fresh `rsync` copy;
`ATM_DRV_dynA.f.patch` is anchored after the unchanged `end subroutine read_aic` and also applies after `ATM_DRV.f.patch` (offset 5)):

    cd $COPY/model; chmod u+w ATMDYN.f ATM_DRV.f
    patch ATMDYN.f  < <this dir>/ATMDYN_dynA.f.patch    # hooks: ffd_avrx_site(1|3) before CALL AVRX in AFLUX/PGF (and 2 in the dead V2 PGF),
                                                        # ffd_avrx/ffd_avrx_consts inside AVRX, ffd_iso inside isotropuv, ffd_sdrag around CALL SDRAG
    patch ATM_DRV.f < <this dir>/ATM_DRV_dynA.f.patch   # adds ffd_dynA_on/_envi, ffd_avrx_site/_consts, ffd_avrx, ffd_iso, ffd_sdrag; units 1071-1079
    (units 1071-1080 were grepped over the whole model tree and all existing patches before use: unused; 1080 spare)

Build and run exactly as in the D89/D90 recipes (absolute make target, `source env_modele.sh` in the same shell, `\cp` the binary to BOTH `P2SAoM40.bin` and
`P2SAoM40`, restore BOTH `fort.1.nc` and `fort.2.nc` from `ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc`, `sh P2SAoM40ln`, `OMP_NUM_THREADS=1`,
`FFD_START=<N> FFD_NSTEP=6 ./P2SAoM40 -i I`; ~1 min incremental build ~45 s; the three dates were run concurrently, rc=0). In `I` change ONLY the first
`YEARE=1950,MONTHE=..,DATEE=..,HOURE=0` end-date line (nov26: 11,26,3; dec01: 12,1,3; jan01: 1,1,3) -- do not touch the `ISTART=2 ... YEARE=1949` line.
Optional env: `FFD_AVRX_STRIDE` (default 17), `FFD_ISO_LSTRIDE` (default 4), `FFD_SDRAG_ISTRIDE` (default 24).
Copy ONLY the new files into `ff_data/<date>/` with `\cp -n`: `ffd_avrx.bin ffd_avrx_calls.bin ffd_avrx_consts.bin ffd_isotr.bin ffd_isotr_stat.bin ffd_isotr_geom.bin
ffd_sdrag.bin ffd_sdrag_stat.bin ffd_sdrag_consts.bin` (9 files, ~19-20 MB per date; sampled so that each file is 2-7 MB). Writes are big-endian float64
streams (ifort `-convert big_endian`), files stay open for the run and are flushed after each record.

Dumps (add to the dump list in build_and_run.md; layouts and loaders in the compare scripts):
- `ffd_avrx_consts.bin` (once): `jm, dlon, bydyp(3)`, `DRAT(jm)`, `NMIN(jm)` (as doubles; undefined entries for rows J=1,JM), `DXP`, `DYP`, `BYSN(1:36)`, `C(0:72)`, `S(0:72)`.
  `ffd_avrx_calls.bin`: 4 doubles per AVRX call [itime, site, ncall, sampled] (all calls). `ffd_avrx.bin`: 222 doubles per ACTIVE row of a sampled call:
  itime, site (1 AFLUX, 3 PGF), ncall, j, X_in(72), AN(0:36), BN(0:36) (FFT of X_in, before truncation scaling), X_out(72). Loader: `dyn_avrx_compare.py`.
- `ffd_isotr_geom.bin` (once): `jm, im, dt, fjeq, radius`, `COSV(jm)`, `DXV(jm)`, `COSIV(im)`, `SINIV(im)`. `ffd_isotr_stat.bin`: 8 doubles for EVERY (j,l) row of every call:
  itime, ncall(1..5), j, l, k, fac, n, max|u_in|. `ffd_isotr.bin`: 440 doubles per sampled row (l=1,5,..,37 or n>1): itime, ncall, j, l, k, fac, n, hemi, U_in, V_in,
  UA, VA (after both shap1, before the pole FFT), U_out, V_out (each 72). Loader: `dyn_isotropuv_compare.py`.
- `ffd_sdrag_consts.bin` (once): `lm, ls1, lsdrag, lpsdrag, ang_sdrag, wc_jdrag, wmax, x_sdrag(1:2), linear_sdrag; jm; CSDRAGL(1:lm), VSDRAGL(1:lm)` (entries below LS1
  are memory outside the Fortran arrays: ignore), `COSV,RAPVN,RAPVS,DXYV,DXYN,DXYS (jm each)`, `RGAS`. `ffd_sdrag_stat.bin`: two 10-double records per call (full-grid input
  statistics; max |dU|,|dV|). `ffd_sdrag.bin`: 445 doubles per sampled column (itime, ncall, i, j, dt1; U,V,T,PK,PEDN(L+1),MA(L,I+1,J-1),MA(L,I,J-1),MA(L,I+1,J),MA(L,I,J) inputs;
  U,V outputs; 40 each). Loader: `dyn_sdrag_compare.py`.
Validate: `python3 dyn_avrx_compare.py; python3 dyn_isotropuv_compare.py; python3 dyn_sdrag_compare.py`;
`pytest fullfidelity/tests/test_dyn_avrx_ff.py test_dyn_isotropuv_ff.py test_dyn_sdrag_ff.py` (dump-based tests skip if the dumps are absent).
Scratch tree used here: `<scratchpad>/mE_dynA/mE2` (the original ModelE tree and other sessions' scratch builds were not touched).

## D99/D100 (AADVT) build and run lines

Patches (fresh `rsync` copy, never the original tree; hunks local; each verified to apply to PRISTINE files; units 1100-1102
were grepped over model/*.f, *.F90, *.h and all instrumentation patches: unused):

    cd $COPY/model      # chmod u+w the three files first
    patch ATMDYN.f  < <this dir>/ATMDYN_aadvt.f.patch     # ffdc_aadvt_in/_out around CALL AADVT (ATMDYN.f:337)
    patch QUS_DRV.f < <this dir>/QUS_DRV_aadvt.f.patch    # ffdc_setx/ffdc_ckpt (after X1,Y,Z), ffdc_nsx/ffdc_nsz in aadvtx/aadvtz
    patch ATM_DRV.f < <this dir>/ATM_DRV_dynC.f.patch     # helpers ffdc_*; inserted after 'end subroutine alloc_drv_atm'

Build and run exactly as the D96-D98 section (absolute make target, ~1-2 min). NOTE the run end time: for a restart run the
FIRST `YEARE=...,HOURE=0` line (I:109, the &INPUTZ namelist) is the end time, not the &INPUTZ_cold one; set it to
1950,11,26,3 / 1950,12,1,3 / 1950,1,1,3 (editing the cold line makes nov26/jan01 run days and dec01 stop at once).
`FFD_START=<itime> FFD_NSTEP=6 ./P2SAoM40 -i I` (33312 / 33552 / 17520), ~50 s per date. Stress runs: additionally
`FFD_AADVT_STRESS=4` (files get prefix `ffd_aadvtS4_`; the model stops on its own courmax>1 error after ~3.5 steps).
Copy `ffd_aadvt*` into `ff_data/<date>/` with `\cp -n`.

Dumps (per date 72 files, ~868 MB; big-endian f8): `ffd_aadvt_<itime>_c<k>_{in,out,s1,s2,s3,ns}.bin`, k=1,2 = AADVT call within
DYNAM. Layouts in the docstring of `dyn_aadvt_compare.py` (loaders `load_in/load_out/load_ckpt/load_ns`).
Validate: from `fullfidelity/`: `python3 dyn_aadvt_compare.py [aadvt|aadvtS4]`, `python3 dyn_adv1d_compare.py`,
`PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_dyn_aadvt_ff.py` (82 passed, ~40 s).
Standalone harness: `{ sed -n 1,221p QUSDEF.f; sed -n 636,758p QUSDEF.f; } > qus1d_ext.f; ifort -O2 -ftz -convert big_endian
-assume protect_parens -fp-model strict -c qus1d_ext.f; ifort <same> instrumentation/qus1d_standalone_drv.f90 qus1d_ext.o -o drv`;
inputs by `python3 dyn_adv1d_compare.py --gen <dir>` then `./drv` in that dir (in.bin -> out.bin).

# D91-D93 build/run lines (for build_and_run.md)

CLOUDS2 helper dumps (D91-D93: PRECIP_MP, ANVIL_OPTICAL_THICKNESS, MC_CLOUD_FRACTION, CONVECTIVE_MICROPHYSICS,
MC_PRECIP_PHASE, MASS_FLUX): patch `CLOUDS2.F90` with `CLOUDS2_helpers.f90.patch` (a dump call before the `return` of each
routine; `call ffh_site(k)` before each of the 8 PRECIP_MP statements in CONVECTIVE_MICROPHYSICS; saved entry values of
CONDIP/CONDGP, PRCP/HEAT1) and `ATM_DRV.f` with `ATM_DRV_clouds_helpers.f.patch` (adds `ffh_site` and `ffh_rec`, units
1051-1056 for the six dumps and 1057 for the consts/counts text files, inserted before the unchanged `#ifdef CACHED_SUBDD` /
`subroutine accum_subdd_atm` block, so it is independent of the order of the other ATM_DRV patches; verified to apply to
pristine ATM_DRV.f and after ATM_DRV_clouds_dq.f.patch, offset 72 fuzz 2). Both patches were generated by `diff -u`
against pristine files, and the resulting patched files are byte-identical to the built tree. The CLOUDS2 patch applies
to pristine CLOUDS2.F90 and after CLOUDS2_dq.f90.patch (offset 6). The validated build used ONLY these two patches on a
fresh rsync copy (scratch dir mE_cloud2/mE2). Units 1051-1057 were grepped across the tree (model/*.f, *.F90, *.h) and all
instrumentation patches: unused (D89 uses 1050, the dynamics patches 1070).

    # build: identical to D89 (absolute make target; chmod -R u+w the rsync copy before patching)
    source <repo>/rocke3d_jax/fullfidelity/env_modele.sh
    export SOCRATESPATH=$SRC/ModelE_Support/socrates
    C=<scratch>/mE_cloud2/mE2; cd $C/decks && gmake RUN=P2SAoM40 $C/model/P2SAoM40.bin      # ~1 min

    # run dirs: exactly as D89 (I, P2SAoM40ln, P2SAoM40uln, runtime_opts from ModelE_Support/huge_space/P2SAoM40; sed the
    # YEARE line to nov26 1950,11,26,3 / dec01 1950,12,1,3 / jan01 1950,1,1,3; binary copied to P2SAoM40.bin and P2SAoM40;
    # BOTH fort.1.nc and fort.2.nc from ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc; `sh P2SAoM40ln`)
    export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH
    export OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
    export FFC_H_STRIDE_PM=20 FFC_H_STRIDE_AN=3 FFC_H_STRIDE_MC=40 FFC_H_STRIDE_CM=20 FFC_H_STRIDE_PP=40 FFC_H_STRIDE_MF=3
    FFD_START=<itime> FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1       # about 1 min per date, three dates concurrently
    # Copy only the new ffc_{pmp,anv,mcf,cmp,mpp,mf}_*.bin, ffc_h_consts.txt, ffc_h_counts.txt into ff_data/<date>/ (`\cp -n`).
    # Pitfall found: the record length argument `n` of ffh_rec must equal the array-constructor length (mismatch is
    # silent and truncates the record); the loaders assert the file size is a multiple of the record length.

Dumps (all big-endian f8 stream, header itime, site, ncall; inputs, then outputs):
`ffc_pmp_<itime>.bin` 8 doubles (rho, flam, dc, cn, out; site 1-8 = PRECIP_MP call in CONVECTIVE_MICROPHYSICS: 1 liquid
lower CONDP1 (5504), 2 liquid upper CONDP (5518), 3 ice lower (5541), 4 ice upper (5552), 5/6 mixed lower CONDIP/CONDGP
(5564/5565), 7/8 mixed upper (5578/5579)); `ffc_anv_` 15 (CLOUDS2.F90:3114); `ffc_mcf_` 18 (2675); `ffc_cmp_` 30 = header 3 +
27 (1938); `ffc_mpp_` 19 (2703); `ffc_mf_` 31 (1056). Column lists are in `clouds_helpers_compare.py` and
`clouds_massflux_compare.py`. Sampling: first 2 calls of each (itime, site) plus every Nth (strides above); `ncall` is the
running count of ALL calls at that site in the step. `ffc_h_counts.txt` has one line per routine per step (name, itime,
routine index, call counts per site 1-8) for every step except the last of each window (its totals are not flushed; ncall
of the last record is a lower bound within one stride). `ffc_h_consts.txt` has dtsrc (1800), rgas, grav, teeny, pi, by3,
by6, twopi, lhe, lhs, lhm, tf, bysha, deltx, rhow, mrat, rvap and the strides. About 7.1-7.7 MB per date for all six dumps.
Read with `fullfidelity/clouds_helpers_compare.py` and `fullfidelity/clouds_massflux_compare.py` (`load_date`).

## D101/D102 (FILTER/SLP, energy functions) build and run lines

Patches (generated by `diff -u` against the PRISTINE source; apply to a fresh `rsync` copy; verified to apply to pristine files and after the `ATM_DRV_dynA/B/C` and `ATMDYN_dynA/aflux_pgf_advecv/aadvt` patches (offset only)):

    cd $COPY/model; chmod u+w ATMDYN.f ATM_DRV.f ATM_UTILS.f
    patch ATMDYN.f   < <this dir>/ATMDYN_filter.f.patch     # hooks in FILTER (ffdd_in, ffdd_stage 1-4, ffdd_out) and CONSERV_KE (ffdd_ke)
    patch ATM_UTILS.f < <this dir>/ATM_UTILS_filter.f.patch # ffdd_te in getTotalEnergy, ffdd_eadd in addEnergyAsDiffuseHeat
    patch ATM_DRV.f  < <this dir>/ATM_DRV_dynD.f.patch      # helpers ffdd_on/_consts/_fname/_in/_stage/_out/_ke/_te/_eadd; units 1120-1125
    (units 1120-1139 were grepped over model/*.f, *.F90, *.f90, *.h and all patches before use: unused; 1126-1139 spare)

Build and run exactly as in the D89/D90/D94 recipes: `source env_modele.sh` in the SAME shell, `export SOCRATESPATH=$SRC/ModelE_Support/socrates`,
`cd $COPY/decks && gmake RUN=P2SAoM40 $COPY/model/P2SAoM40.bin` (absolute target; ~55 s on the rsync copy, 0 errors); per date a run dir with `I P2SAoM40ln
P2SAoM40uln runtime_opts`, `\cp` the binary to BOTH `P2SAoM40.bin` and `P2SAoM40`, first `YEARE=1950,MONTHE=..,DATEE=..,HOURE=3` line of `I` (11,26 / 12,1 / 1,1; the
`ISTART=2 ... YEARE=1949` line untouched), `fort.1.nc` and `fort.2.nc` both restored from `ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc`, `sh P2SAoM40ln`,
`LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib`, `OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1`, `FFD_START=<N> FFD_NSTEP=6 ./P2SAoM40 -i I` (N = 33312 nov26,
33552 dec01, 17520 jan01); the three dates ran concurrently, rc=0, about 1 minute in all. Scratch tree used: `<scratchpad>/mE_dynD/mE2` (the original tree and other scratch builds untouched).
Copy only the new files into `ff_data/<date>/` with `\cp -n`: `ffd_filt_consts.bin` and, per step, `ffd_filt_<itime>_{in,out,te1,te2,eadd}.bin` and `ffd_slp_<itime>.bin`
(37 files per date, 342 MB).

Dumps (add to the dump list in build_and_run.md; layouts and loaders in `fullfidelity/dyn_filter_compare.py`):
`ffd_filt_consts.bin`; `ffd_filt_<itime>_in.bin` (FILTER entry: PEDN(1), TSAVG, MA, PK, T, Q, QCL, QCI, QMOM); `ffd_slp_<itime>.bin` (SLP-filter stages); `ffd_filt_<itime>_out.bin`
(FILTER exit); `ffd_filt_<itime>_te{1,2}.bin` (getTotalEnergy initial/final records, self-contained inputs + KEB/KEIJ/PEIJ/TEIJ/TOTAL); `ffd_filt_<itime>_eadd.bin`
(addEnergyAsDiffuseHeat). FILTER is called once per step (atm_phase2, every step because NFILTR=1).

Validate: from `fullfidelity/`: `python3 dyn_filter_compare.py [--imf-pow]`; `PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_dyn_filter_ff.py` (79 tests, ~30 s; dump tests skip
if the dumps are absent, libimf tests skip without the Intel runtime). New code files: `dyn_filter_ff.py`, `dyn_filter_compare.py`.

# D107-D109 build/run lines (for build_and_run.md)

LSCOND dumps (D107-D109): patch `CLOUDS2.F90` with `CLOUDS2_lscond.f90.patch` (hooks inside LSCOND: entry after the parameter
set-up and before "initialise vertical arrays", checkpoint after `PRCPSS=max(...)` (post main loop, before CTEI), entry of the "COMPUTE
CLOUD PARTICLE SIZE AND OPTICAL THICKNESS" block, tail exit and call exit before `return`; plus `ffls_pack_state` inserted after
`end subroutine LSCOND`, and the local declaration `integer :: ffls_b, ffls_t`) and `ATM_DRV.f` with `ATM_DRV_clouds_lscond.f.patch`
(helpers `ffls_want/ffls_put/ffls_put1/ffls_stash/ffls_flush`, units 1170 bnd, 1171 mid, 1172 tail, 1173 consts text, inserted after
`end subroutine atm_phase2`, an anchor used by no other patch). Both generated by `diff -u` against the PRISTINE files and verified to
apply to pristine and after `CLOUDS2_dq`/`CLOUDS2_helpers` and `ATM_DRV_clouds_dq`/`ATM_DRV_clouds_helpers` (offset 6 / offset 72 fuzz 2).
Units 1170-1173 were grepped over the model tree (model/*.f, *.F90, *.h), all instrumentation patches and the other sessions'
scratch builds before use: unused. The validated build used ONLY these two patches on a fresh rsync copy (mE_cloud3/mE2).

    SRC=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0
    rsync -a --exclude=ModelE_Support $SRC/ <scratch>/mE_cloud3/mE2/ && chmod -R u+w <scratch>/mE_cloud3/mE2
    cd <scratch>/mE_cloud3/mE2/model
    patch CLOUDS2.F90 < <this dir>/CLOUDS2_lscond.f90.patch ; patch ATM_DRV.f < <this dir>/ATM_DRV_clouds_lscond.f.patch
    source <repo>/rocke3d_jax/fullfidelity/env_modele.sh
    export SOCRATESPATH=$SRC/ModelE_Support/socrates
    C=<scratch>/mE_cloud3/mE2; cd $C/decks && gmake RUN=P2SAoM40 $C/model/P2SAoM40.bin      # ~1 min, 0 errors
    # run dirs: as D89/D99 (I, P2SAoM40ln, P2SAoM40uln, runtime_opts from huge_space; BOTH fort.1.nc and fort.2.nc from
    # ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc; binary to P2SAoM40.bin and P2SAoM40; sh P2SAoM40ln); in I edit the FIRST
    # `YEARE=1950,MONTHE=12,DATEE=1,HOURE=0,` line (I:109) to 1950,11,26,3 / 1950,12,1,3 / 1950,1,1,3
    export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
    FFD_START=<itime> FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1       # ~1.5 min per date, three dates concurrently, rc=0
    # itimes: nov26 33312, dec01 33552, jan01 17520.  Copy ffc_ls_* into ff_data/<date>/ with `\cp -n`.
    # Optional env: FFC_LS_STRIDE_B (default 10: bnd+mid records), FFC_LS_STRIDE_T (default 4: tail records).

Dumps (big-endian f8 stream; every record: 6-double header itime, kind, ncall, i, j, nbody, then the body; per date 18 files + consts,
~170 MB): `ffc_ls_bnd_<itime>.bin` (5,077-double body, 317 records/step; boundary = 18 parameters + 4 scalars + 18 input LM arrays +
RNDSSL(3,LM) + PRECNVL(LM+1) + RA(4) + state-in + state-out + 6 exit scalars), `ffc_ls_mid_<itime>.bin` (2,526-double body, 317/step;
PRCPSS, HCNDSS, state, 9 local LM arrays, PREBAR, PREICE), `ffc_ls_tail_<itime>.bin` (1,494-double body, 792-793/step; entry 12 scalars +
23 LM arrays + LHP, exit WMSUM + 13 LM arrays). Sampling: call taken when `mod(ncall+(itime-FFD_START),stride)==0` (ncall = running
count of LSCOND calls in the step, 3170 per step: 72x44 columns + 2 pole columns computed for I=1), so every step samples a different
column subset; UM/VM are recorded for K=1..4 only (pole rows J=1,JM have KMAX=72). State arrays (24 LM + LHP + prebar1 + QMOM/SMOM + UM/VM): see
`fullfidelity/clouds_lscond_io.py` (the single source of the layouts, loaders `load_bnd/load_mid/load_tail`, `read_consts`).
`ffc_ls_consts.txt`: dtsrc rgas grav bygrav teeny pi by3 twopi lhe lhs lhm tf sha bysha deltx mrat bymrat rvap and the two strides.

Validate (from `fullfidelity/`, bare imports): `python3 clouds_lscond_size_compare.py [--imf]`; `python3 clouds_lscond_compare.py [--imf] [--quiet] [--max N]`;
`PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_clouds_lscond_size_ff.py fullfidelity/tests/test_clouds_lscond_ff.py` (78 passed, ~90 s;
libimf tests skip when the Intel runtime is absent, dump tests skip when ff_data is absent).
Pitfalls found: (1) `**5` in the CTEI SIGK formula is a libm pow call, not a multiplication chain; (2) `WMUI=WMUIX*.001` carries a REAL(4)
literal (value taken from the dump); (3) the pole columns have KMAX=72 (UM/VM are recorded for K<=4 only); (4) CLEARA, RHF, RH00 are
uninitialised Fortran stack above LMCLD (=29): compare only L<=LMCLD; (5) the LSCOND routine reads and writes module arrays, so the hooks
live inside LSCOND and the packing routine `ffls_pack_state` inside module CLOUDS.
Scratch tree used here: `<scratchpad>/mE_cloud3/mE2` (the original ModelE tree and other sessions' scratch builds were not touched).

## D114-D117 (coupling glue) build and run lines

Patches (diff -u against PRISTINE files, each verified with patch on a fresh copy; ATM_DRV_dynF verified byte-identical to the built file). Units 1240-1251 grepped over model/*.f,*.F90,*.f90,*.h and all instrumentation patches: unused (1252-1269 spare).
    cd $COPY/model; chmod -R u+w .
    patch ATM_DRV.f < ATM_DRV_dynF.f.patch   # helpers ffdg_* after 'end subroutine atm_phase1_exports' + efix/site hooks in atm_phase1
    patch ATMDYN.f < ATMDYN_glue.f.patch; patch ATM_UTILS.f < ATM_UTILS_glue.f.patch; patch ATMDYN_COM.F90 < ATMDYN_COM_glue.f90.patch
    patch SURFACE.f < SURFACE_glue.f.patch; patch ATURB.f < ATURB_glue.f.patch; patch CLOUDS2_DRV.F90 < CLOUDS2_DRV_glue.f90.patch; patch DIAG.f < DIAG_glue.f.patch
Build as D89/D99 (rsync copy, absolute make target, source env_modele.sh in same shell, SOCRATESPATH set): about 2 min, 0 errors. Scratch tree: <scratchpad>/mE_dynF/mE2.
Run: as D99 (I, P2SAoM40ln, P2SAoM40uln, runtime_opts; both restarts from _pristine_restarts; OMP_NUM_THREADS=1; FFD_START/FFD_NSTEP=6) BUT end date = one day + 1 h so DAILY_ATMDYN fires: first YEARE line 1950,11,27,1 / 1950,12,2,1 / 1950,1,2,1 (50 steps, ~6 min, three dates concurrent, rc=0). Run script: <scratchpad>/mE_dynF/work/rundates.sh. Dumps other than DAILY are gated to the 6-step window; DAILY only needs FFD_START.
Copy ffd_glue_* into ff_data/<date>/ with \cp -n (51 files, ~480 MB per date: recalc 7 calls/step x 4.2 MB dominates, then kea/rg3/dissip/efix/trop/wsave; nothing sampled; consts 2.3 MB).
Dump layouts: docstring of fullfidelity/dyn_glue_io.py. Call-site ids for recalc_agrid_uv: 1 SURFACE.f:426, 2 ATURB.f:619, 3 CLOUDS2_DRV.F90:565, 4 :2594, 5 DIAG.f:213 (DIAGA, fires on some steps), 0 unlabelled (none seen). calc_kea_3d sites: 1 ATM_DRV step-7, 2 inside DISSIP.
Pitfalls: `mv`/`cp` are interactive aliases (use \mv, \cp); fixed-form lines must stay <=72 columns; efix needs Q and QCI for CONSERV_SE (added after the first build); stream files are opened with position='append' (works in ifort 19.1).
Validate (fullfidelity/): python3 dyn_glue_compare.py [--imf-pow] [--analytic-geom]; PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_dyn_glue_ff.py (43 passed).

## D103-D106 (QDYNAM: AADVQ0, AADVQ sweeps, glue) build and run lines

Patches (generated by `diff -u` against the PRISTINE source; hunks local; each verified to apply to pristine files; units 1140-1144 and 1149
were grepped over model/*.f, *.F90, *.h and all instrumentation patches: unused):

    cd $COPY/model; chmod u+w ATMDYN.f QUS3D.f ATM_DRV.f
    patch ATMDYN.f < <this dir>/ATMDYN_qdynam.f.patch   # ffde_in/q0/ain/out/fin inside QDYNAM (+ second USE TRACER_ADV for the AADVQ0 outputs)
    patch QUS3D.f  < <this dir>/QUS3D_qdynam.f.patch    # ffde_stg/carry/full inside AADVQ (the live routines are in QUS3D.f)
    patch ATM_DRV.f < <this dir>/ATM_DRV_dynE.f.patch   # helpers ffde_*; inserted after 'end subroutine new_io_atmvars'

Build: `source env_modele.sh` in the same shell, `export SOCRATESPATH=$SRC/ModelE_Support/socrates`, `cd $COPY/decks && gmake RUN=P2SAoM40
$COPY/model/P2SAoM40.bin` with the ABSOLUTE target path built from the copy root (a target containing `..` made gmake say "Nothing to be done").
~1 min. Run exactly as the D99 section (end-date line is I:109; nov26 11,26,3 / dec01 12,1,3 / jan01 1,1,3; restore BOTH fort.1.nc and fort.2.nc from
`ff_data/_pristine_restarts`; `FFD_START=<itime> FFD_NSTEP=6`; ~1 min, three dates concurrently, rc=0). Stress: `FFD_QDYN_STRESS=<f>` (optional
`FFD_QDYN_COMP=<subset of uvw>` selects which of MUs/MVs/MWs are scaled; files `ffd_qdynS<f>[<comp>]_*`). f=8 completes 6 steps; f=16 reaches ncyc 7-8 and then
the model's own ncyc>ncmax stop; u-only or v-only scaling stops at ncyc>ncmax already at 8. Copy `ffd_qdyn*` to `ff_data/<date>/` with `\cp -n`.
(`cp` is aliased to `cp -i`: use `\cp`.)

Dumps per date: `ffd_qdyn_geom.bin` + 6 x `ffd_qdyn_<itime>_{in,q0,ain,ck,out,fin}.bin` (37 files, ~750 MB). Layouts: docstring of `dyn_qdynam_io.py`.
Validate (from the repo root, the conda python): `PYTHONPATH=fullfidelity python fullfidelity/dyn_qdynam_compare.py [qdyn|qdynS8]` (run from `fullfidelity/` for bare imports),
`PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_dyn_aadvq_ff.py`.
Standalone harness of the real QUS3D.f: `sed -n 1,41p QUSDEF.f > qusdef_mod.f; F="-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict";
ifort $F -c instrumentation/qus3d_standalone_stubs.f90; ifort $F -c qusdef_mod.f; ifort $F -fpp -I$MODEL/include -c $MODEL/QUS3D.f -o qus3d.o;
ifort $F -c instrumentation/qus3d_standalone_drv.f90; ifort $F *.o -o drv`; inputs `python3 dyn_qus3d_harness.py --gen DIR`, `cd DIR && ../drv`, then `--compare DIR`.
New files: `dyn_aadvq_ff.py`, `dyn_qdynam_io.py`, `dyn_qdynam_compare.py`, `dyn_qus3d_harness.py`, `tests/test_dyn_aadvq_ff.py`,
`instrumentation/{ATMDYN_qdynam,QUS3D_qdynam,ATM_DRV_dynE}.f.patch`, `qus3d_standalone_{stubs,drv}.f90`. Scratch tree: `<scratchpad>/mE_dynE/mE2`.

# D110-D113 (MSTCNV) build and run lines (for build_and_run.md)

Patches (generated by `diff -u` against the PRISTINE source; apply to a fresh `rsync` copy; verified to apply to pristine files
and, with offsets only, on top of `CLOUDS2_dq/_helpers/_lscond` and `ATM_DRV_clouds_dq/_helpers/_lscond/_dynA-E`):

    cd $COPY/model; chmod -R u+w .            # the rsync copy is read-only
    patch CLOUDS2.F90 < <this dir>/CLOUDS2_mstcnv.f90.patch    # decl block after 'integer :: lborrow1', entry hook (ffm_const/
                                                               #   ffm_in), 9 checkpoint hooks (ffm_ck), exit hook (ffm_out)
    patch ATM_DRV.f   < <this dir>/ATM_DRV_clouds_mstcnv.f.patch   # helpers ffm_init/ffm_const/ffm_in/ffm_ck/ffm_out; units 1200-1202
    (units 1200-1239 were grepped over model/*.f, *.F90, *.h and all instrumentation patches before use: unused; 1203-1239 spare.
     The ATM_DRV hunk is anchored after the unchanged 'end subroutine atm_exports_phasesrf', used by no other patch.)

Build exactly as the D89/D96 recipes (absolute make target; the target path must not contain `..`):

    source <repo>/rocke3d_jax/fullfidelity/env_modele.sh      # same shell
    export SOCRATESPATH=$SRC/ModelE_Support/socrates
    C=<scratch>/mE_cloud4/mE2; cd $C/decks && gmake RUN=P2SAoM40 $C/model/P2SAoM40.bin     # about 70 s, 0 errors

Run (per date, own run dir; the three dates concurrently, rc=0, ~1 min wall): copy `I P2SAoM40ln P2SAoM40uln runtime_opts` from
`ModelE_Support/huge_space/P2SAoM40`; `\cp` the binary to BOTH `P2SAoM40.bin` and `P2SAoM40`; set ONLY the first
`YEARE=1950,MONTHE=..,DATEE=..,HOURE=0` line of `I` to 1950,11,26,3 / 1950,12,1,3 / 1950,1,1,3; `rm -f fort.1.nc fort.2.nc` and copy
`ff_data/_pristine_restarts/fort1_<date>_itime<N>.nc` to BOTH; `sh P2SAoM40ln`; then

    export LD_LIBRARY_PATH=/app/netcdf4/platform/x86_64/rocky/8.10/4.9.3s/lib:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MP_SET_NUMTHREADS=1
    export FFM_CSTRIDE=2 FFM_CKSTRIDE=24 FFM_NSTRIDE=25
    FFD_START=<N> FFD_NSTEP=6 ./P2SAoM40 -i I > run.PRT 2>&1        # N = 33312 nov26, 33552 dec01, 17520 jan01

Copy only the new files into `ff_data/<date>/` with `\cp -n`: `ffc_mc_cols_<itime>.bin` (6), `ffc_mc_ck_<itime>.bin` (6),
`ffc_mc_consts.txt` (19 files, about 390 MB per date; 1.17 GB for the three dates).
Sampling: MSTCNV is called for J=2..45 only (the poles are not called): about 3165-3169 calls per step. With the default strides
(FFM_CSTRIDE=1, FFM_CKSTRIDE=3) the dumps were 520 MB per step (3.1 GB per date) and were reduced: every 2nd convective call
(FFM_CSTRIDE=2), every 25th non-convective call (FFM_NSTRIDE=25), plus every call whose checkpoints are dumped (every 24th call
of the step, kept only if convective). About 66 % of the columns convect, so ~1040 convective + ~45 non-convective boundary
records per step (6399-6540 records per date).

Dumps (big-endian f8 streams; layouts, field tables and loaders in `fullfidelity/clouds_mstcnv_io.py`, which is also the
generator of the Fortran array constructors of the patch):
- `ffc_mc_cols_<itime>.bin`: per sampled MSTCNV call 8 header doubles (itime, i, j, ncall, conv, nin=1974, nout=2616, ckflag), the
  IN fields (PEARTH, PLAND, DCL, LMCM, XMASS, BYDTsrc, DTsrc, BYBR, KMAX; PL, PLE(41), PLK, AIRM, BYAM, ETAL, TL, TVL, SM, QM, QCLL,
  QCIL, SDL, WTURB, GZL; SMOM, QMOM (9,40); RA, UM, VM, U_0, V_0 for the K-sample) and the OUT fields (scalars; 37 layer arrays;
  LHP, PRECNVL (41); SMOM, QMOM, UM, VM). KMAX is 4 on this grid (non-polar columns), so all four K are recorded.
- `ffc_mc_ck_<itime>.bin`: per convective call with mod(ncall,24)==0 a block of events (stage, LMIN, IC, NPPL, n, payload):
  1 DMSE test, 2 after MASS_FLUX, 3 plume set-up, 4 end of ascent, 5 end of downdraft, 6 end of subsidence, 7 before the
  precip/evap loop, 8 end of one cloud type, 9 after the cloud-base-loop post-processing. 479-526 blocks / 18 800-20 700 events per date.
- `ffc_mc_consts.txt`: tunables of this run (entrainment_cont1/2 .4/.6, radiusl_multiplier 1.01, U00a .695, U00b .6, MC_FDDRT .5,
  LMCM 23, MC_* switches 1) and the constants.

Validate: from `fullfidelity/`: `python3 clouds_mstcnv_compare.py [--imf] [--dates ..]` (all records, ~2 min numpy, ~7 min --imf per
date), `PYTHONPATH=fullfidelity pytest fullfidelity/tests/test_clouds_mstcnv_ff.py` (dump tests skip if the dumps are absent;
libimf tests skip without the Intel runtime). New code files: `clouds_mstcnv_ff.py`, `clouds_mstcnv_io.py`,
`clouds_mstcnv_compare.py`. Scratch tree used here: `<scratchpad>/mE_cloud4/mE2` (the original ModelE tree and the other
sessions' scratch builds were not touched).
