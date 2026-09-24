# Building and running the instrumented ModelE (P2SAoM40)

Never edit the original tree; work in a copy (236 MB, everything but ModelE_Support):

    SRC=/panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0
    COPY=<scratch>/mE2
    rsync -a --exclude=ModelE_Support $SRC/ $COPY/
    cd $COPY/model
    patch ATM_DRV.f < <this dir>/ATM_DRV.f.patch     # adds ffdump + 5 call sites
    patch MODELE.f  < <this dir>/MODELE.f.patch      # pre/post SURFACE call sites
    patch ATURB.f   < <this dir>/ATURB.f.patch       # entry/exit dumps of atm_diffus (ffa_* files)
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
    FFD_START=33312 FFD_NSTEP=6 ./P2SAoM40 -l run.PRT

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
