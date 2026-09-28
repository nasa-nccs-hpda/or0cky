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
