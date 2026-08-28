#include "rundeck_opts.h"
      SUBROUTINE initTracerGriddedData(is_coldstart)
!@sum init_tracer initializes trace gas attributes
!@calls sync_param, SET_TCON
      USE DOMAIN_DECOMP_ATM, only:GRID,getDomainBounds,AM_I_ROOT
      USE RESOLUTION, only : jm,lm
      USE ATM_COM, only: pmidl00
      USE ATM_COM, only: MA  ! Air mass of each box (kg/m^2)
      use OldTracer_mod, only: trname
      USE TRACER_COM, only: ntm, tracers
#ifdef TRACERS_ON
      USE TRDIAG_COM
#endif
      USE Dictionary_mod
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP)  || (defined TRACERS_TOMAS)
      use trdust_drv, only : init_soildust
#endif
#ifdef TRACERS_PHOTOLYSIS
      use photolysis, only: fastj2_init
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_SPECIAL_Shindell
      use model_com, only: master_yr, modelEclock
      use dist_grid_mod, only : dist_grid
      use timestream_mod, only: init_stream, read_stream_ijless,
     &     timestream
      USE TRCHEM_Shindell_COM,only:LCOalt,PCOalt,
     &     CH4altINT,CH4altINX,LCH4alt,PCH4alt,
     &     CH4altX,CH4altT,ch4_init_shnh,scale_ch4_IC_file,
     &     OxIC,fix_CH4_chemistry,allowSomeChemReinit,
     &     CH4ICX,use_rad_ch4,
     &     COIC,Lmax_rad_O3,Lmax_rad_CH4,N2OICX,CFCIC,
     &     use_rad_n2o,use_rad_cfc,cfc_rad95,
     &     topLevelOfChemistry,rjphoto,ijlprn,prnrts,
     &   ICfact_N,ICfact_COt,ICfact_COs,ICfact_Oth,ICfact_N2O,ICfact_CFC
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_AEROSOLS_SOA
      USE TRACERS_SOA, only: soa_init
#endif  /* TRACERS_AEROSOLS_SOA */
#ifdef TRACERS_AEROSOLS_VBS
      USE TRACERS_VBS, only: vbs_init
#ifndef TRACERS_AMP
      use AEROSOL_SOURCES, only: vbs_conc
#else
      USE TRACERS_VBS, only: vbs_bins
      use AMP_AEROSOL, only: vbs_conc
      use AERO_CONFIG, only: mname
#endif  /* not TRACERS_AMP */
#endif  /* TRACERS_AEROSOLS_VBS */
#if (defined TRACERS_AMP)
      use AERO_CONFIG, only: nmodes
      USE AERO_COAG, only : SETUP_KIJ
      USE AERO_SETUP, only: SETUP_CONFIG,SETUP_SPECIES_MAPS,SETUP_DP0,
     &                      SETUP_AERO_MASS_MAP,SETUP_COAG_TENSORS,
     &                      SETUP_EMIS,SETUP_KCI
      USE AERO_NPF, only: SETUP_NPFMASS
      USE AERO_DIAM, only: SETUP_DIAM,DP
      USE AMP_AEROSOL, only: DIAM
#endif
      USE FILEMANAGER, only: openunit,closeunit,nameunit

      implicit none
      logical, intent(in) :: is_coldstart
c
      integer :: l,k,n,kr,m,ns
#ifdef TRACERS_SPECIAL_O18
      real*8 fracls
#endif
#ifdef TRACERS_TOMAS
      integer :: bin
      real*8 :: TOMAS_dens,TOMAS_radius
#endif
#ifdef TRACERS_SPECIAL_Shindell
!@var iu_data unit number
!@var title header read in from file
      integer iu_data,i,j,nq
      character*80 title
      character(len=300) :: out_line
      real*8, dimension(6) :: temp_ghg
      integer :: temp_year, xyear, year, day
      type(timestream) :: trICratN, trICratCOt, trICratCOs,
     & trICratOth, trICratN2O, trICratCFC, trICch4
#endif /* TRACERS_SPECIAL_Shindell */
#if defined(TRACERS_AEROSOLS_VBS) && defined(TRACERS_AMP)
      integer :: ivbs,igasm2,iaerm2
#endif  /* TRACERS_AEROSOLS_VBS and TRACERS_AMP */

! temp storage for new tracer interfaces
      integer :: values(ntm)
      integer :: val

      INTEGER J_0, J_1, I_0, I_1

C****
C**** Extract useful local domain parameters from "grid"
C****
      call getDomainBounds(grid, J_STRT=J_0,       J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

      do n=1,ntm
#ifdef TRACERS_ON
        select case (trname(n))

        case ('N2O')
#ifdef TRACERS_SPECIAL_Shindell
          call getIC('N2O_IC',N2OICX)
#endif

      case ('CH4')
#ifdef TRACERS_SPECIAL_Shindell
C**** determine initial CH4 distribution if set from rundeck
C**** This is only effective with a complete restart.
          call sync_param("fix_CH4_chemistry",fix_CH4_chemistry)
          call sync_param("scale_ch4_IC_file",scale_ch4_IC_file)
C         Interpolate CH4 altitude-dependence to model resolution:
          CALL LOGPINT(LCH4alt,PCH4alt,CH4altINT,LM,PMIDL00,CH4altT,
     &         .true.)
          CALL LOGPINT(LCH4alt,PCH4alt,CH4altINX,LM,PMIDL00,CH4altX,
     &         .true.)
          if(fix_CH4_chemistry.eq.-1)then
            call getIC('CH4_IC',CH4ICX)
            do j=J_0,J_1  ; do i=I_0,I_1
              CH4ICX(i,j,:) = CH4ICX(i,j,:) * scale_ch4_IC_file
            end do ; end do
          end if
#endif /* TRACERS_SPECIAL_Shindell */

      case ('Ox')
#ifdef TRACERS_SPECIAL_Shindell
          call getIC('Ox_IC',OxIC)
#endif /* TRACERS_SPECIAL_Shindell */

      case ('CFC')
#ifdef TRACERS_SPECIAL_Shindell
          if(AM_I_ROOT( ))then
C          check on GHG files 1995 value for CFCs:
           call openunit('GHG',iu_data,.false.,.true.)
           do i=1,5; read(iu_data,'(a80)') title; enddo
           temp_year=0
           do while(temp_year <= 1995)
             read(iu_data,*,end=101) temp_year,(temp_ghg(j),j=1,6)
             if(temp_year==1995)then
               temp_ghg(1)=cfc_rad95*0.95d0
               temp_ghg(2)=cfc_rad95*1.05d0
               temp_ghg(3)=(temp_ghg(4)+temp_ghg(5))*1.d-9
               if(temp_ghg(3) < temp_ghg(1) .or.
     &         temp_ghg(3) > temp_ghg(2))then
                 call stop_model('please check on cfc_rad95 2',255)
               endif
             endif
           enddo
 101       continue
           if(temp_year<1995)
     &     call stop_model('please check on cfc_rad95 1',255)
           call closeunit(iu_data)
          endif
          ! read the CFC initial conditions:
          call getIC('CFC_IC',CFCIC)
#endif /* TRACERS_SPECIAL_Shindell */

      case ('CO'
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
     *     ,'dC17O','dC18O','d13CO'
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
     *     )
#ifdef TRACERS_SPECIAL_Shindell
          call getIC('CO_IC',COIC)
#endif /* TRACERS_SPECIAL_Shindell */

        end select
#endif
      end do

#ifdef TRACERS_ON
#ifdef TRACERS_SPECIAL_Shindell
      ! Read time-dependant factors to scale chemical tracer initial conditions.
      ! Some of these are not single-tracer specific, so not done under trname
      ! select case above. For now, no horizontal variation, but since using
      ! timestream, it should be easy to read these on the model grid instead:
      call modelEclock%get(year=year, dayOfYear=day)
      call get_param( "O3_yr", xyear, default=master_yr )
      if(xyear==0) xyear=year
      call getIC2(trICratN,'trICratN',ICfact_N,grid)
      call getIC2(trICratCOt,'trICratCOt',ICfact_COt,grid)
      call getIC2(trICratCOs,'trICratCOs',ICfact_COs,grid)
      call getIC2(trICratOth,'trICratOth',ICfact_Oth,grid)
      call getIC2(trICratN2O,'trICratN2O',ICfact_N2O,grid)
      call getIC2(trICratCFC,'trICratCFC',ICfact_CFC,grid)
      call getIC2hems(trICch4,'trICch4',ch4_init_shnh,grid,'ch4')
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_AEROSOLS_SOA
      call soa_init
#endif  /* TRACERS_AEROSOLS_SOA */

#ifdef TRACERS_AEROSOLS_VBS
! initialize vbs
#ifndef TRACERS_AMP
      call vbs_init(vbs_conc(1), ntm)
#else
! find index of first VBS gas tracer. Brute force, but only happens once.
      igasm2=0
      do n=1,ntm
        if (trim(trname(n)).eq.'vbsGm2') then
          igasm2=n
          exit
        endif
      enddo
      if (igasm2==0) call stop_model('Could not find vbsGm2 tracer',255)

      do i=1,nmodes
! find index of first VBS aerosol tracer. Brute force, but only happens once.
        iaerm2=0
        do n=1,ntm
          if (trim(trname(n)).eq.'M_'//mname(i)//'_OCM2') then
            iaerm2=n
            exit
          endif
        enddo
        if (iaerm2==0) cycle ! No VBS species in this population

        do ivbs=1,vbs_bins
          vbs_conc(i)%igas(ivbs)=igasm2-1+ivbs
          vbs_conc(i)%iaer(ivbs)=iaerm2-1+ivbs
        enddo
        call vbs_init(vbs_conc(i), ntm)
      enddo
#endif  /* not TRACERS_AMP */
#endif  /* TRACERS_AEROSOLS_VBS */

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS)
c**** soil dust aerosol initializations
      call init_soildust
#endif

C**** Miscellaneous initialisations

#ifdef TRACERS_SPECIAL_Shindell
      call cheminit ! **** Initialize the chemistry ****
#ifdef TRACERS_ACETONE
      ! right now, this resides in TRACERS_SPECIAL_Shindell.f
      ! and only is in use for acetone; thus inside ifdefs.
      call init_oceanEmissions
#endif /* TRACERS_ACETONE */
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef TRACERS_PHOTOLYSIS
! initialize fastj, always AFTER chemistry
#ifdef TRACERS_SPECIAL_Shindell
      call fastj2_init(topLevelOfChemistry,rjphoto,ijlprn(1:2),prnrts,
     &                 allowSomeChemReinit)
#else
      call fastj2_init(LM, ! just ask for one reaction below, as an example
     &                 reshape((/'O3      ','O(1D)   ','O2     '/), ! 8 chars
     &                         (/3,1/)), ! reshape needed for 1 reaction only
     &                 (/0,0/), ! i,j indices (not used if next line is .false.)
     &                 .false., ! print or not column debug output
     &                 0) ! allowSomeChemReinit
#endif  /* TRACERS_SPECIAL_Shindell */
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_COSMO
      call init_cosmo
#endif
#endif /* TRACERS_ON */

#ifdef TRACERS_AMP
      CALL SETUP_CONFIG
      CALL SETUP_SPECIES_MAPS
      CALL SETUP_DP0
      CALL SETUP_AERO_MASS_MAP
      CALL SETUP_COAG_TENSORS
      CALL SETUP_DP0
      CALL SETUP_KIJ
      CALL SETUP_EMIS
      CALL SETUP_KCI
      CALL SETUP_NPFMASS
      if (is_coldstart) then ! do not overwrite diam during warm starts
        CALL SETUP_DIAM
        do n=1,nmodes
          DIAM(:,:,:,n)=DP(n) ! all gridboxes get the default value at init
        enddo
      endif
      CALL SETUP_RAD
#endif

      call init_src_dist

      return

#ifdef TRACERS_SPECIAL_Shindell
      CONTAINS

        subroutine getIC2(Dstream,Dfile,Dvar,mainGrid,vname)
        implicit none
        type(dist_grid) :: mainGrid
        type(timestream) :: Dstream
        character(len=*) :: Dfile
        character(len=*), optional :: vname
        character*80 :: rvname
        real*8 :: Dvar
        rvname='ICscale'
        if (present(vname) ) rvname=vname

        call init_stream
     &  (maingrid,Dstream,Dfile,trim(rvname),0.d0,1.d30,'none',
     &       xyear,day,ijless=.true.)
        call read_stream_ijless(maingrid,Dstream,xyear,day,Dvar)


        if(am_i_root())
     &    write(6,*)trim(rvname),' from ',trim(Dfile),' = ',Dvar
        end subroutine getIC2

        subroutine getIC2hems(Dstream,Dfile,Dvar,mainGrid,vname)
        implicit none
        type(dist_grid) :: mainGrid
        type(timestream) :: Dstream
        character(len=*) :: Dfile
        character(len=*), optional :: vname
        character*80 :: rvname
        real*8 :: Dvar(2)
        rvname='ICscale'
        if (present(vname) ) rvname=vname

        call init_stream
     &  (maingrid,Dstream,Dfile,trim(rvname),0.d0,1.d30,'none',
     &       xyear,day,ijless=.true.)
        call read_stream_ijless(maingrid,Dstream,xyear,day,Dvar)


        if(am_i_root())
     &    write(6,*)trim(rvname),' from ',trim(Dfile),' = ',Dvar
        end subroutine getIC2hems

        subroutine getIC(fn,ICs)
        use resolution, only: im
        use pario, only : par_open, par_close, read_dist_data, 
     &   get_dimlen, read_data
        implicit none
        integer :: fid, nlev
        character(len=*), intent(in) :: fn
        real*8, dimension(:,:,:), allocatable :: loc3D ! mass mixing ratio
        real*8, dimension(:), allocatable :: IClevs
        real*8, dimension(:,:,:), allocatable :: ICs

        ! read the file's data:
        fid = par_open(grid,trim(fn),'read')
        nlev = get_dimlen(grid,fid,'pressures')
        allocate(IClevs(nlev))
        allocate(loc3D(grid%i_strt:grid%i_stop,
     &                 grid%j_strt:grid%j_stop,nlev))
        call read_data(grid,fid,'pressures',IClevs,bcast_all=.true.)
        call read_dist_data(grid,fid,trim(fn),loc3D) ! var name same as file short name
        call par_close(grid,fid)

        ! interpolate in pressure and convert units:
        if(size(ICs,3)/=LM) call stop_model(
     &   'Expected LM 3rd dim size in getIC for file '//trim(fn),255)
        do j=J_0,J_1
          do i=I_0,I_1
            call logpint(nlev,IClevs,loc3D(i,j,:),LM,PMIDL00,ICs(i,j,:),
     &                   .true.)
            ICs(i,j,:) = ICs(i,j,:) * MA(:,i,j) ! kg/m2 mass
          end do
        end do

        deallocate(loc3D,IClevs)
        end subroutine getIC
#endif /* TRACERS_SPECIAL_Shindell */

      end subroutine initTracerGriddedData
