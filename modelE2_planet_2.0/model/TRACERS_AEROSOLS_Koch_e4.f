#include "rundeck_opts.h" 
      MODULE AEROSOL_SOURCES
!@sum repository for Koch aerosol sources, features, etc.
!@auth Dorothy Koch
!@ subroutines in this file include:
!@ alloc_aerosol_sources
!@ read_DMS_sources
!@ aerosol_gas_chem_prep
!@ aerosol_gas_chem
!@ get_oxidants
!@ GET_SULFATE
!@ GET_BC_DALBEDO
!@ GRAINS
!@ read_seawifs_chla
      use resolution, only : LM
      use timestream_mod, only : timestream
#ifdef TRACERS_AEROSOLS_VBS
      use TRACERS_VBS, only: vbs_tracers
#endif /* TRACERS_AEROSOLS_VBS */
      IMPLICIT NONE
      SAVE
!@var DMSinput           DMS ocean source (kg/s/m2)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: DMSinput ! DMSinput(im,jm,12)
#ifndef TRACERS_AEROSOLS_SOA
!@var OCT_src    OC Terpene source (kg/m2/s)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: OCT_src !(im,jm,12)
#endif  /* TRACERS_AEROSOLS_SOA */
!@var SO2_src_3D SO2 volcanic sources (and biomass) (kg m-2 s-1)
!@var H2O_src_3D H2O volcanic sources (kg kg-1 s-1)
!@var SU_src_3D SO4 direct injection source for simplified volc experiment (kg kg-1 s-1)
!@var DD1_src_3D dust as proxy volcanic ash sources (fine, kg kg-1 s-1) TRACERS_AMP only
!@var DD2_src_3D dust as proxy volcanic ash sources (coarse, kg kg-1 s-1) TRACERS_AMP only
!@var BC_src_3D BC wildfire sources (kg kg-1 s-1)
!@var OC_src_3D OC wildfire sources (kg kg-1 s-1)
      INTEGER :: nso2src_3d=0,iso2volcano=0,iso2volcanoexpl=0
      INTEGER :: iso2directinj=0
      real*8, ALLOCATABLE, DIMENSION(:,:,:,:) :: SO2_src_3D !(im,jm,lm,nso2src_3d)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: H2O_src_3D !(im,jm,lm)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: SU_src_3D !(im,jm,lm)
#ifdef TRACERS_AMP
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: DD1_src_3D !(im,jm,lm)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: DD2_src_3D !(im,jm,lm)
#endif
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: BC_src_3D !(im,jm,lm)
      real*8, ALLOCATABLE, DIMENSION(:,:,:) :: OC_src_3D !(im,jm,lm)

      type oxidants
        real*8 :: OH,NO3,O3 ! both for online and offline
        real*8 :: HO2,H2O2  ! COUPLED_CHEM.ne.1 only
      end type oxidants
      type(oxidants) :: oxid
      real*8, ALLOCATABLE, DIMENSION(:,:,:) ::
     &  ohr,dho2r,perjr,tno3r,o3_offline, ! COUPLED_CHEM.ne.1 only
     &  off_HNO3 !@var off_HNO3 offline HNO3 for nitrate + AMP when gas phase chem off
      real*8, allocatable, dimension(:,:,:) :: readCache

#ifdef BC_ALB
      real*8, ALLOCATABLE, DIMENSION(:,:) :: snosiz
#endif  /* BC_ALB */
#ifdef TRACERS_AEROSOLS_VBS
!@var VBSemifactFF factor that distributes organic aerosols in volatility bins
!@+                from fossil fuel sources
!@var VBSemifactBB factor that distributes organic aerosols in volatility bins
!@+                from biomass burning sources
      integer, parameter :: vbs_sets=1
      type(vbs_tracers), dimension(vbs_sets) :: vbs_conc
      real*8, allocatable, dimension(:) :: VBSemifactFF,VBSemifactBB
#endif /* TRACERS_AEROSOLS_VBS */
      integer, parameter :: nAeroStream=6
      type(timestream), dimension(nAeroStream) :: AeroStream
      logical :: AeroFirst=.true.

!@var oh_live for on-line radical exporting from chemistry
!@var no3_live for on-line radical exporting from chemistry
!@var o3_live for on-line radical exporting from chemistry
      real*8, dimension(LM) :: oh_live, no3_live, o3_live

!@dbparam tune_DMS Multiplication factor for DMS emissions
      real*8 :: tune_DMS=1.

      END MODULE AEROSOL_SOURCES

      SUBROUTINE alloc_aerosol_sources(grid)
!@auth D. Koch
      use domain_decomp_atm, only: dist_grid, getDomainBounds
      use TRACER_COM, only: NTM
      use TRACER_COM, only: coupled_chem
      use TRACER_COM, only: direct_inject_num
      use AEROSOL_SOURCES, only: DMSinput,
#ifndef TRACERS_AEROSOLS_SOA
     * OCT_src,
#endif  /* TRACERS_AEROSOL   S_SOA */
     * nso2src_3d,SO2_src_3D,iso2volcano,iso2volcanoexpl,H2O_src_3d,
     * SU_src_3D,BC_src_3D,OC_src_3D,
#ifdef TRACERS_AMP
     * DD1_src_3D,DD2_src_3D,
#endif
     * iso2directinj,ohr,dho2r,perjr, tno3r, readCache,
     * o3_offline,off_HNO3
#ifdef TRACERS_AEROSOLS_VBS
      use AEROSOL_SOURCES, only: vbs_sets,vbs_conc
      use AEROSOL_SOURCES, only: VBSemifactFF,VBSemifactBB
#endif
#ifdef BC_ALB
      use AEROSOL_SOURCES, only: snosiz
#endif  /* BC_ALB */
      use filemanager, only: file_exists

      use RESOLUTION, only: lm
      use TimeConstants_mod, only: INT_MONTHS_PER_YEAR
      
      IMPLICIT NONE
      type (dist_grid), intent(in) :: grid
      integer :: I_0, I_1, J_0, J_1
      integer :: v
      logical :: init = .false.

      if(init)return
      init=.true.

      call getDomainBounds(grid)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      J_0 = grid%J_STRT
      J_1 = grid%J_STOP

      allocate( DMSinput(I_0:I_1,J_0:J_1,INT_MONTHS_PER_YEAR) )
#ifndef TRACERS_AEROSOLS_SOA
      allocate( OCT_src(I_0:I_1,J_0:J_1,INT_MONTHS_PER_YEAR) )
#endif  /* TRACERS_AEROSOLS_SOA */
      if (file_exists('SO2_VOLCANO')) then
        nso2src_3d=nso2src_3d+1
        iso2volcano=nso2src_3d
      endif
      if (file_exists('SO2_VOLCANO_EXPL')) then
        nso2src_3d=nso2src_3d+1
        iso2volcanoexpl=nso2src_3d
      endif
      if (direct_inject_num>0) then
        nso2src_3d=nso2src_3d+1
        iso2directinj=nso2src_3d
      endif
      allocate( SO2_src_3D(I_0:I_1,J_0:J_1,lm,nso2src_3d) )
      allocate( H2O_src_3D(I_0:I_1,J_0:J_1,lm) )
      allocate( SU_src_3D(I_0:I_1,J_0:J_1,lm) )
#ifdef TRACERS_AMP
      allocate( DD1_src_3D(I_0:I_1,J_0:J_1,lm) )
      allocate( DD2_src_3D(I_0:I_1,J_0:J_1,lm) )
#endif
      allocate( BC_src_3D(I_0:I_1,J_0:J_1,lm) )
      allocate( OC_src_3D(I_0:I_1,J_0:J_1,lm) )
      if (coupled_chem.le.0) then
        allocate(        ohr(I_0:I_1,J_0:J_1,lm),
     *                 dho2r(I_0:I_1,J_0:J_1,lm),
     *                 perjr(I_0:I_1,J_0:J_1,lm),
     *                 tno3r(I_0:I_1,J_0:J_1,lm),
     *                 o3_offline(I_0:I_1,J_0:J_1,lm),
     *                 off_HNO3(I_0:I_1,J_0:J_1,lm))
        allocate(  readCache(I_0:I_1,J_0:J_1,lm) )
      endif
#ifdef BC_ALB
      allocate( snosiz(I_0:I_1,J_0:J_1) )
#endif  /* BC_ALB */
#ifdef TRACERS_AEROSOLS_VBS
      do v=1,vbs_sets
        allocate(VBSemifactFF(vbs_conc(v)%nbins))
        allocate(VBSemifactBB(vbs_conc(v)%nbins))
      enddo
#endif

      return
      end SUBROUTINE alloc_aerosol_sources


      SUBROUTINE read_DMS_sources(swind,itype,i,j,DMS_flux) !!! T
!@sum generates DMS ocean source
!@auth Koch
c Monthly DMS ocean concentration sources are read in and combined
c  with wind and ocean temperature functions to get DMS air surface
c  concentrations
c want kg DMS/m2/s
      use TimeConstants_mod, only: SECONDS_PER_DAY
      use OldTracer_mod, only: tr_mm
      USE TRACER_COM, only: n_DMS
      use model_com, only: modelEclock
      USE AEROSOL_SOURCES, only: tune_DMS
!@var DMSinput DMS concentration in sea water [nM, or moles lt-1, or moles dm-3]
      USE AEROSOL_SOURCES, only: DMSinput
#ifdef old_DMS_emis
      USE FLUXES, only: GTEMP
#endif
      implicit none
      integer jread
!@var akw Sea-air transfer velocity [cm hr-1, but also m s-1]
      REAL*8 akw,SCH,SCHR
#ifdef old_DMS_emis
      real*8 Tc ! YHL - FOR another DMS source
#endif
      real*8, PARAMETER :: SCHT=600.d0
      real*8, INTENT(OUT) :: DMS_flux
!@var swind Surface wind speed [m s-1]
      real*8, INTENT(IN) :: swind
      integer, INTENT(IN) :: itype,i,j

      DMS_flux=0.d0
        if (itype.eq.1) then
#ifndef old_DMS_emis
c Nightingale et al
        akw = 0.23d0*swind*swind + 0.1d0 * swind ! cm hr-1
        akw = akw * 0.24d0 ! m day-1 ; 0.24d0 is 24 [hr day-1] * 0.01 [m cm-1]
! In the line below, 1.d-9 converts nM to M; tr_mm converts M to g dm-3, which
! is the same with kg m-3; 1/SECONDS_PER_DAY converts akw from m day-1 to m s-1.
! tune_DMS is just a dimensionless tuning factor. DMS_flux in kg m-2 s-1.
        DMS_flux=akw*DMSinput(i,j,modelEclock%getMonth())*1.d-9*
     &           tr_mm(n_DMS)/SECONDS_PER_DAY*tune_DMS
#endif

#ifdef old_DMS_emis
c YUNHA - Liss and Merlivat (1986) code is from GISS GCM II-prime. 
c Liss and Merlivat (1986), use for > lm=40 to moderate DMS flux

       Tc=GTEMP(1,1,I,J) ! YUNHA GTEMP is already Celcius. 

       SCH=2674.d0-147.12d0*Tc+3.726d0*Tc*Tc-0.038d0*Tc*Tc*Tc
       IF(Tc.gt.47.) print*,'BAD_TEMPERATURE_DMS',i,j,Tc,SCH
       SCHR=SCHT/SCH
       if (swind.lt.3.6) then
        akw=0.041*(SCHR)**(2.d0/3.d0)*swind
       else if (swind.lt.13.) then
         akw=(0.68*SWIND - 2.31)*DSQRT(SCHR)
       else
       akw=(1.42*SWIND - 11.8)*DSQRT(SCHR)
       endif  !swind
       DMS_flux=akw*DMSinput(i,j,modelEclock%month())*1.d-9*62.d0/
     *          SECONDS_PER_DAY*tune_DMS     !not sure of units

#endif
        endif !itype
c
      return
      end SUBROUTINE read_DMS_sources

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS)

      subroutine aerosol_gas_chem_prep
!@sum prepare info for aerosol gas phase chemistry
!@auth Dorothy Koch
      use timestream_mod, only : init_stream,read_stream
      use Dictionary_mod, only : get_param
      use domain_decomp_atm, only: getDomainBounds, grid
      use model_com, only: modelEclock, master_yr
      use aerosol_sources, only: ohr,dho2r,perjr,tno3r,o3_offline,
     & off_HNO3, AeroStream, AeroFirst, nAeroStream, readCache
      use atm_com, only: ma
      use constant, only: mair
      use OldTracer_mod, only: vol2mass
      use tracer_com, only: n_HNO3
      use resolution, only: LM

      implicit none

      integer :: xday,xyear,clockYear,k,L
      logical :: cyclic
      character*10, dimension(nAeroStream) :: AeroVars = (/
     &'ohr       ','dho2r     ','perjr     ','tno3r     ','o3_offline',
     &'off_HNO3  '/)

      cyclic=.true.
      call modelEclock%get(year=clockYear, dayOfYear=xday)
      call get_param( "aer_int_yr", xyear, default=master_yr )
      if(xyear==0)then
        xyear=clockYear ; cyclic=.false.
      end if

      ! Read the monthly data and interpolate to current day:
      ! Except for HNO3, the old code did not do interpolation -
      ! just monthly step-function. I think you could reproduce this by
      ! changing 'linm2m' to 'none; in init_stream call.
      if(AeroFirst) then
        AeroFirst=.false.
        do k = 1,nAeroStream
          call init_stream(grid,AeroStream(k),'OFFLINE_CHEM',
     &    trim(AeroVars(k)),0d0, 1d30,'linm2m',xyear,xday,
     &    cyclic=cyclic )
        end do
      end if

      do k = 1,nAeroStream
        call read_stream(grid,AeroStream(k),xyear,xday,readCache)
        select case(k)
        case (1) ; ohr = readCache    ! OH [molecules cm-3]
        case (2) ; dho2r = readCache ! HO2 [molecules cm-3]
        case (3) ; perjr = readCache ! H2O photolysis rate [s-1]
        case (4) ; tno3r = readCache ! NO3 [molecules cm-3]
        case (5) ; o3_offline = readCache  ! mole O3 / mole air (converted later)
        case (6) ! mole HNO3 / mole air.
          do L=1,LM ! Convert here to kg HNO3 m-2 layer-1, like trm_col
            off_HNO3(:,:,L)=readCache(:,:,L)*ma(L,:,:)*vol2mass(n_HNO3)
          end do
        end select
      end do

      end subroutine aerosol_gas_chem_prep


      SUBROUTINE aerosol_gas_chem(i,j)
!@sum aerosol gas phase chemistry
!@vers 2013/03/27
!@auth Dorothy Koch
      use OldTracer_mod, only: trname, tr_mm
      use TRACER_COM, only: ntm, trm_col
      use TRACER_COM, only: coupled_chem, n_BCIA, n_BCII, n_DMS,n_H2O2_s
      use TRACER_COM, only: rsulf1, rsulf2, rsulf3, rsulf4
      use TRACER_COM, only: n_MSA, N_OCII, n_SO2, n_OCIA
      use TRACER_COM, only: n_SO4, n_SO4_d1, n_SO4_d2, n_SO4_d3, n_H2SO4
      use TRACER_COM, only: nChemistry, nChemprod, nChemLoss, nOther
#if (defined TRACERS_HETCHEM) || (defined TRACERS_NITRATE)
      use TRACER_COM, only: rxts1, rxts2, rxts3
#endif
      USE TRDIAG_COM, only : 
     *     jls_OHconk,jls_HO2con,jls_NO3,jls_phot
      use resolution, only: lm
      USE MODEL_COM, only: dtsrc
      use atmcol_com, only: tl   ! layer temperature (K)
      use atmcol_com, only: qv   ! layer humidity (kg/kg)
      use atmcol_com, only: pl   ! layer pressure (mb)
      use atmcol_com, only: ma   ! layer mass (kg/m2)
      use atmcol_com, only: byma ! 1/ma
      USE PBLCOM, only : dclev
      USE FLUXES, only: tr3Dsource
      USE AEROSOL_SOURCES, only: oxid
      USE CONSTANT, only : mair
#ifdef TRACERS_TOMAS
      USE TOMAS_AEROSOL, only : h2so4_chem
#endif
      use trdiag_com, only : taijls=>taijls_loc,ijlt_prodSO4gs

c Aerosol chemistry
      implicit none
      integer, intent(in) :: i,j
!
      real*8 ppres,tt,dmm,r1,d1,r2,d2,ttno3,r3,d3,
     * ddno3,dddms,ddno3a,fmom
      real*8 rk4,ek4,r4,d4
      real*8 r6,d6,ek9,ek9t,ch2o,eh2o,dho2mc,dho2kg,eeee,xk9,
     * r5,d5,dmssink
#ifdef TRACERS_HETCHEM
     *       ,d41,d42,d43
#endif
      integer l,n,iuc,iun,itau,itt,
     * ittime,isp,iix,jjx,llx,ii,jj,ll,iuc2,it,mmm

      do l=1,lm
        call get_oxidants(i,j,l) ! get oxidant concentrations

        ppres=pl(l)*9.869d-4 ! [atm]
        dmm=ppres/(.082d0*tl(l))*6.02d20! number density of air [molecules/cm3]

! ===== THIS IS CHEMISTRY OF Koch AEROSOLS =====
        do n=1,NTM

        select case (trname(n))
        case ('DMS')
C***1.DMS + OH -> 0.75SO2 + 0.25MSA
C***2.DMS + OH -> SO2
C***3.DMS + NO3 -> HNO3 + SO2

          r1=rsulf1(l)*oxid%OH
          d1 = exp(-r1*dtsrc)
          r2=rsulf2(l)*oxid%OH
          d2 = exp(-r2*dtsrc)

          if (l.gt.dclev(i,j)) then
            ttno3=0.d0
          else
            ttno3=oxid%NO3
          endif
          if (jls_NO3>0) call inc_tajls2(i,j,l,jls_NO3,ma(l)*ttno3)

          r3=rsulf3(l)*ttno3
          d3= exp(-r3*dtsrc)
          ddno3=r3*trm_col(l,n)/tr_mm(n)*1000.d0*dtsrc
          dddms=trm_col(l,n)/tr_mm(n)*1000.d0
          if (ddno3.gt.dddms) ddno3=dddms

          ddno3=ddno3*0.9
C DMS losses: eqns 1, 2 ,3

          tr3Dsource(l,nChemistry,n) = trm_col(l,n)*(d1*d2-1.)/dtsrc

          dmssink=ddno3*tr_mm(n)/1000.d0

          if (dmssink.gt.trm_col(l,n)+tr3Dsource(l,nChemistry,n)*dtsrc)
     *      dmssink=trm_col(l,n)+tr3Dsource(l,nChemistry,n)*dtsrc
          tr3Dsource(l,nChemistry,n)=
     *      tr3Dsource(l,nChemistry,n)-dmssink/dtsrc

        case ('MSA')
C MSA gain: eqn 1

          tr3Dsource(l,nChemistry,n) = 0.25d0*Tr_mm(n)/Tr_mm(n_dms)*
     *         trm_col(l,n_dms)*(1.d0 -D1)*SQRT(D2)/dtsrc
          
        case ('SO2')
c SO2 production from DMS
          tr3Dsource(l,nChemprod,n) = (
     * 0.75*tr_mm(n)/tr_mm(n_dms)*trm_col(l,n_dms)*(1.d0 - d1)*sqrt(d2)+
     *      tr_mm(n)/tr_mm(n_dms)*trm_col(l,n_dms)*(1.d0 - d2)*sqrt(d1)+
     *      dmssink*tr_mm(n)/tr_mm(n_dms)
     *                                 )/dtsrc

c oxidation of SO2 to make SO4: SO2 + OH -> H2SO4
          r4=rsulf4(l)*oxid%OH
          d4 = exp(-r4*dtsrc)

          IF (d4.GE.1.) d4=0.99999d0
#ifdef TRACERS_HETCHEM
          d41 = exp(-rxts1(l)*dtsrc)     
          d42 = exp(-rxts2(l)*dtsrc)     
          d43 = exp(-rxts3(l)*dtsrc)     
          tr3Dsource(l,nChemloss,n) = (-trm_col(l,n)*(1.d0-d41)/dtsrc)
     .                              + (-trm_col(l,n)*(1.d0-d4)/dtsrc)
     .                              + (-trm_col(l,n)*(1.d0-d42)/dtsrc)
     .                              + (-trm_col(l,n)*(1.d0-d43)/dtsrc)
#else
          tr3Dsource(l,nChemloss,n) = -trm_col(l,n)*(1.d0-d4)/dtsrc 
#endif  /* TRACERS_HETCHEM */

        end select

        enddo                     ! tracer loop

! ===== END OF CHEMISTRY OF Koch AEROSOLS ====

c diagnostics to save oxidant fields
c No need to accumulate Shindell version here because it
c   is done elsewhere
        if (jls_OHconk>0) call inc_tajls2(i,j,l,jls_OHconk,
     &       ma(l)*oxid%OH)
        if (jls_HO2con>0) call inc_tajls2(i,j,l,jls_HO2con,
     &       ma(l)*oxid%HO2)

! SO4 and H2O2_s formation MUST be in a separate loop, since SO2 does not have
! to be before SO4 or H2SO4 or H2O2_s in the tracer list
        do n=1,NTM
        select case (trname(n))
#ifdef TRACERS_HETCHEM
       case ('SO4_d1')
c sulfate production from SO2 on mineral dust aerosol due to O3 oxidation

       tr3Dsource(l,nChemistry,n)=tr3Dsource(l,nChemistry,n)
     *         +tr_mm(n)/tr_mm(n_so2)
     *         *(1.d0-d41)*trm_col(l,n_so2)            !  SO2
     *           /dtsrc

       case ('SO4_d2')
c sulfate production from SO2 on mineral dust aerosol

       tr3Dsource(l,nChemistry,n) = tr3Dsource(l,nChemistry,n)+
     *   tr_mm(n)/tr_mm(n_so2)*(1.d0-d42)*trm_col(l,n_so2)
     *           /dtsrc

       case ('SO4_d3')
c sulfate production from SO2 on mineral dust aerosol

       tr3Dsource(l,nChemistry,n) = tr3Dsource(l,nChemistry,n)+
     *   tr_mm(n)/tr_mm(n_so2)*(1.d0-d43)*trm_col(l,n_so2)
     *           /dtsrc

#endif
        case('SO4','H2SO4')
C SO4 production
#ifndef TRACERS_TOMAS
          tr3Dsource(l,nChemistry,n)=trm_col(l,n_so2)*(1.d0 -d4)/dtsrc
     *         *tr_mm(n)/tr_mm(n_so2)
#else
          H2SO4_chem(l)=trm_col(l,n_so2)*(1.d0 -d4)/dtsrc
     *         *tr_mm(n)/tr_mm(n_so2)
          tr3Dsource(l,nChemistry,n_H2SO4)=H2SO4_chem(l)
#endif
          taijls(i,j,l,ijlt_prodSO4gs)=taijls(i,j,l,ijlt_prodSO4gs)+
     &      tr3Dsource(l,nChemistry,n)

        case('H2O2_s')

          if (coupled_chem.ne.1) then

c hydrogen peroxide formation and destruction:
C***5.H2O2 +hv -> 2OH
C***6.H2O2 + OH -> H2O + HO2
C***7.H2O2 + SO2 -> H2O + SO3 (in-cloud, in CB)
C***9.HO2 + HO2 ->H2O2 + O2
C     HO2 + HO2 + M ->
C     HO2 + HO2 + H2O ->
C     HO2 + HO2 + H2O + M ->

          tt = 1.d0/tl(l)
          r6 = 2.9d-12 * exp(-160.d0*tt)*oxid%OH
          d6 = exp(-r6*dtsrc)
          ek9 = 2.2d-13*exp(600.d0*tt)
          ek9t = 1.9d-20*dmm*0.78d0*exp(980.d0*tt)*1.d-13
          ch2o = qv(l)*6.02d20*28.97d0/18.d0*ppres/(.082d0*tl(l))
          eh2o = 1.+1.4d-21*exp(2200.d0*tt)*ch2o
          dho2mc=oxid%HO2
          dho2kg=oxid%HO2*ma(l)*tl(l)*.082056d0/(ppres*28.97d0*6.02d20)
          eeee = eh2o*(ek9+ek9t)*dtsrc*dho2mc
          xk9 = dho2kg*eeee
c         if (i.eq.2.and.l.eq.1.and.j.eq.46) write(6,*) 
c    *    'RRR CHEM DEBUG ',i,j,xk9,dho2kg,eeee,dho2mc
c    *    ,eh2o,ek9,ek9t,dtsrc
c         if (i.eq.72.and.l.eq.1.and.j.le.46) write(6,*) 
c    *    'RRR CHEM DEBUG ',i,j,xk9,dho2kg,eeee,dho2mc
c H2O2 production: eqn 9
         
          tr3Dsource(l,nChemprod,n) = tr_mm(n)*xk9/dtsrc
c        if (i.eq.10.and.j.eq.45.and.l.eq.1) then
c        write(6,*) 'RRR OXID H2O2',xk9,dho2kg,eeee
c         endif
c H2O2 losses:5 and 6
          r5 = oxid%H2O2
          d5 = exp(-r5*dtsrc)

          tr3Dsource(l,nChemLoss,n)=(trm_col(l,n))*(d5*d6-1.d0)
     *         /dtsrc

          if (jls_phot>0) call inc_tajls2(i,j,l,jls_phot,oxid%H2O2)
          endif ! coupled_chem.ne.1
        end select
        enddo ! tracer loop

      enddo ! level loop

      RETURN
      END SUBROUTINE aerosol_gas_chem

#endif /* oma or matrix or tomas */

      SUBROUTINE get_oxidants(i,j,l)

      use constant, only : pi
      use ATMCOL_COM, only: pl,tl,byma
      use TRACER_COM, only: coupled_chem
      use AEROSOL_SOURCES, only: oxid,ohr,dho2r,perjr,tno3r,o3_offline,
     &    oh_live, no3_live, o3_live
      USE DOMAIN_DECOMP_ATM, only:GRID, getDomainBounds
      use RAD_COM, only: cosz1,cosz_day,sunset
      implicit none
      integer, intent(in) :: i,j,l
      real*8, parameter :: night_frac_min=0.01d0 !min night_frac for NO3 scaling
      real*8 stfac,night_frac
      real*8 :: ppres,dmm

      if (coupled_chem.eq.1) then
! coupled mode: use online radical concentrations
        oxid%OH=oh_live(l)
        oxid%NO3=no3_live(l)
        oxid%O3=o3_live(l)
        oxid%HO2=0.d0
        oxid%H2O2=0.d0
      else
! offline: read-in radical concentrations and impose diurnal variability
        if (cosz1(i,j).gt.0.) then
          stfac=cosz1(i,j)/cosz_day(i,j)
          oxid%OH=ohr(i,j,l)*stfac
          oxid%NO3=0.d0
          oxid%HO2=dho2r(i,j,l)*stfac
          oxid%H2O2=perjr(i,j,l)*stfac
        else
c Get NO3 only if dark, weighted by number of dark hours
          oxid%OH=0.d0
          night_frac = 1.-sunset(i,j)/pi
          if (night_frac.gt.night_frac_min) then
            oxid%NO3=tno3r(i,j,l)/night_frac !DMK jmon
          else
            oxid%NO3=0.d0
          endif
          oxid%HO2=0.d0
          oxid%H2O2=0.d0
        endif
        ppres=pl(l)*9.869d-4 ! [atm]
        dmm=ppres/(.082d0*tl(l))*6.02d20! number density of air [molecules/cm3]
        oxid%O3=o3_offline(i,j,l)*dmm ! mole O3 / mole air --> molecules O3 cm-3
      endif

      END SUBROUTINE get_oxidants


      SUBROUTINE GET_SULFATE(pl,temp_in,fcloud,
     *  wa_vol,wmxtr,sulfin,sulfinom,sulfinc,sulfout,tr_left,
     *  tmg,tmd,airm,lhx,dt_sulf,fcld0,dtime)

!@sum  GET_SULFATE calculates formation of sulfate from SO2 and H2O2
!@+    within or below convective or large-scale clouds. Gas
!@+    condensation uses Henry's Law if not freezing.
!@auth Dorothy Koch

!**** GLOBAL parameters and variables:
      USE CONSTANT, only: bygasc, MAIR,teeny,mb2kg,gasc,lhe
      use OldTracer_mod, only: trname, mass2vol, tr_mm
      use OldTracer_mod, only: tr_RKD, tr_DHD
      USE TRACER_COM, only: n_H2O2_s,n_SO2
     *     ,NTM
     *     ,n_SO4,n_H2O2,coupled_chem
      use tracer_com, only: aqchem_count,aqchem_list
      USE CLOUDS, only: NTX

      IMPLICIT NONE

!**** Local parameters and variables and arguments:

!@var sulfin amount of precursor used to make product from the gas phase (kg)
!@var sulfinom sulfin=sulfinom*tmg, for avoiding divisions sulfinom=sulfin/tmg
!@+   (dimensionless)
!@var sulfinc amount of precursor used to make product from the condensate (kg)
!@var sulfout total amount of product generated (kg)
!@var tr_left is the amount of precursor left after product is made
!@+   and is now available to condense
!@+   This is a very strange variable, probably wrong!!!
      real*8, dimension(aqchem_count), intent(out) :: sulfin,sulfinom,
     &                                        sulfinc,sulfout,tr_left
!@var fcloud cloud fraction available for tracer condensation. fcloud=fplume
!@+   for convective clouds, and fcloud=fcld for large-scale clouds
!@var fcld0 updated cloud fraction, given the current state of large-scale
!@+   clouds. fcld0=0.d0 for convective clouds.
!@var lhx latent heat of evaporation or sublimation (J/Kg). When equal to lhe
!@+   the cloud is in the ice phase.
!@var finc XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
      real*8, intent(in) :: fcloud,fcld0,lhx
      real*8 :: finc
!@var airm layer pressure depth (mb). Multiply by mb2kg to convert to air mass
!@+   per m2, based on the hydrostatic pressure equation:
!@+   pressure (Pa=kg/m/s2) = height (m) * density (kg/m3) * g (m/s2)
!@var amass airmass in kg/m2, calculated by airm*mb2kg
      real*8, intent(in) :: airm
      real*8 :: amass

!@var dtime time step over which these chemical reactions take place.
!@+   Units are in seconds (s).
      real*8, intent(in) :: dtime

!@var pl pressure at current altitude (mbar)
!@var press pressure at current altitude (Pa)
!@var temp temperature to be used (K), always greater than 230K
!@var tfac exponent factor for temperature dependence calculations (mol/J).
!@+   tfac = (1/temp - 1/298)/R; R=8.31451 J/mol/K
!@var clwc cloud liquid water content (volume water/volume air)
!@var temp_in temperature at current altitude (K)
!@var wmxtr cloud water mixing ratio (kg water/kg air)
      real*8 :: press, temp, tfac, clwc
      real*8, intent(in) :: pl,temp_in,wmxtr

!@param k1so2dissoc0 first dissociation rate of dissolved SO2 at 298K.
!@+     SO2.H2O <--> H+ + HSO3-
!@param dh1so2dissoc enthalpy of dissociation for the first dissociation rate
!@+     of dissolved SO2
!@var k1so2dissoc first dissociation rate of dissolved SO2 at current temp.
      real*8, parameter :: k1so2dissoc0=1.3d-2 ! M (molar)
      real*8, parameter :: dh1so2dissoc=-1.6736d4 !J/mol
      real*8 :: k1so2dissoc ! M (molar)
!@param Hplus concentration of H+ (molar) for pH=4.5
!@var henry modified henry constant of the current tracer at current
!@+   conditions, taking into account the current pH, if needed (moles/J).
!@+   multiply with convert_HSTAR to convert to moles/liter/atm
      real*8, parameter :: Hplus=10.d0**(-4.5d0)
      real*8 :: henry
!@param kso2h2o20 reaction rate of SO2 + H2O2 at 298K. SO2 + H2O2 --> SO3 + H2O
!@param dhso2h2o2 enthalpy of reaction of SO2 + H2O2
!@var kso2h2o2 reaction rate of SO2 + H2O2 at current temp.
      real*8, parameter :: kso2h2o20=6.357d14    !1/(M*M*s)
      real*8, parameter :: dhso2h2o2=3.95d4 !J/mol
      real*8 :: kso2h2o2

!@var ix index of current species
!@var is index of current species in ntx array
!@var ih index of current species in ntx array
!@var isx index of SO2 species in aqchem_list array
!@var ihx index of H2O2 species in aqchem_list array
      integer :: ix,is,ih,isx,ihx

!@var tmg amount of tracer in the gas phase in the cloudy area (kg).
!@var tmd amount of tracer in the aqueous phase (kg).
!@var tmgmol amount of gas phase tracer in cloudy area (moles)
!@var tmdmol amount of gas phase tracer in cloudy area (moles)
!@var tmgrate is the new concentration of species that resulted from the
!@+   dissolution of its gas phase precursor species, on top of what was there
!@+   from the previous timestep. The units are M/kg, meaning molarity
!@+   produced per kilogram reacted.
!@var tmdrate is the new concentration of species that resulted from the
!@+   already dissolved precursor species, on top of what was there
!@+   from the previous timestep. The units are M/kg, meaning molarity
!@+   produced per kilogram reacted.
      real*8, dimension(ntx), intent(in) :: tmg,tmd
      real*8, dimension(aqchem_count) :: tmgmol,tmdmol,tmgrate,tmdrate

!@var wa_vol cloud water volume (liters)
!@var dso4g amount of sulfate produced from the gas phase (moles/kg/kg)
!@var dso4d amount of sulfate produced from the condensate phase (moles/kg/kg)
!@var dso4gt amount of sulfate produced from the gas phase (moles)
!@var dso4dt amount of sulfate produced from the condensate phase (moles)
      real*8,  intent(in) :: wa_vol
      real*8 :: dso4g,dso4d,dso4gt,dso4dt

!@var n index for tracer number loop
      integer :: n

!@var dt_sulf accumulated diagnostic of sulfate chemistry changes
      real*8, dimension(ntx), intent(inout) :: dt_sulf

      sulfin(:)=0.d0
      sulfinom(:)=0.d0
      sulfinc(:)=0.d0
      sulfout(:)=0.d0
      tr_left(:)=1.d0

! if no water clouds or no clouds at all, do nothing
      if (lhx.ne.lhe.or.fcloud.lt.teeny.or.wmxtr.le.teeny) return

! calculate the fraction of tracer mass that becomes condensate
      finc=(fcloud-fcld0)/fcloud
      if (finc.lt.0.d0) finc=0.d0

! calculate some variables for later
      amass=airm*mb2kg ! kg/m2
      press = pl*1.d2 ! Pa
! comment from Dorothy Koch:
! calls to this subroutine are sometimes made at stages of the cloud scheme
! at which some but not all tendencies have been applied to temp_in, so we
! impose a lower limit (liquid water is very unlikely to exist below 230 K)
      temp = max(temp_in, 230.d0) ! K
      tfac = (1.d0/temp - 1.d0/298.d0)*bygasc  ! mol/J
! cloud liquid water content
      clwc=wmxtr*mair*press/temp*bygasc/1.d6/fcloud ! volume water/volume air

      k1so2dissoc=k1so2dissoc0*exp(-dh1so2dissoc*tfac) ! SO2.H2O <--> H+ + HSO3-
      kso2h2o2=kso2h2o20*exp(-dhso2h2o2/(gasc*temp)) ! SO2 + H2O2 --> SO3 + H2O

! First allow for formation of sulfate from SO2 and H2O2. Then remaining
! gases may be allowed to dissolve (amount given by tr_left)
! H2O2 + SO2 -> H2O + SO3 -> H2SO4
      do n=1,aqchem_count
        ix=aqchem_list(n)

        select case (trname(ix))
        case('SO2', 'H2O2', 'H2O2_s')

! save some per-tracer values needed later
          select case (trname(ix))
          case('SO2')
            is=ix
            isx=n
          case('H2O2','H2O2_s')
            ih=ix
            ihx=n
            select case (trname(ix))
            case('H2O2')
              if (coupled_chem.le.0) goto 400
            case('H2O2_s')
              if (coupled_chem.eq.1) goto 400
            end select
          end select

! initial amount of species in the gas and aqueous phases
          tmgmol(n)=1.d3*tmg(ix)/tr_mm(ix) ! gas-phase, in moles
          tmdmol(n)=tmd(ix)*1.d3/tr_mm(ix) ! aqueous phase, in moles

! henry coefficient
          henry=tr_rkd(ix)*exp(-tr_dhd(ix)*tfac) ! moles/J

! partial pressure of gas x henry's law coefficient
          tmgrate(n)=mass2vol(ix)*1.d-3*press/amass*henry

! modified Henry's Law coefficient assuming pH of 4.5
          select case (trname(ix))
          case('SO2')
            henry=henry*(1.d0+ k1so2dissoc/Hplus)
          end select

! rate of production from gaseous and dissolved precursors.
          tmgrate(n)=tmgrate(n)/(1.d0+henry*clwc*gasc*temp) ! dimless henry
          tmdrate(n)=mass2vol(ix)*press/amass*bygasc/temp*1.d-3/clwc

 400      continue
        end select
      enddo

! do not calculate dso4g if there is not enough gas phase to react
      if (tmg(ih).lt.teeny.or.tmg(is).lt.teeny) then
        dso4g=0.d0
        dso4gt=0.d0
        go to 21
      endif

! production from the gas phase, moles/kg/kg
      dso4g=kso2h2o2*k1so2dissoc*tmgrate(ihx)*tmgrate(isx)*dtime*wa_vol
      dso4g=dso4g*finc ! increase production based on current cloud water volume
      dso4gt=dso4g*tmg(ih)*tmg(is) ! moles

! can't be more than the moles we started with
      if (dso4gt.gt.tmgmol(isx)) then ! so2
        dso4g=tmgmol(isx)/(tmg(ih)*tmg(is))
        dso4gt=tmgmol(isx)
      endif
      if (dso4gt.gt.tmgmol(ihx)) then ! h2o2
        dso4g=tmgmol(ihx)/(tmg(ih)*tmg(is))
        dso4gt=tmgmol(ihx)
      endif
 21   continue

! do not calculate dso4d if there is not enough dissolved phase to react
      if (tmd(ih).lt.teeny.or.tmd(is).lt.teeny) then
        dso4d=0.d0
        dso4dt=0.d0
        go to 22
      endif

! production from the already-dissolved aqueous phase, moles/kg/kg
      dso4d=kso2h2o2*k1so2dissoc*tmdrate(ihx)*tmdrate(isx)*dtime*wa_vol
      dso4dt=dso4d*tmd(ih)*tmd(is) ! moles

! can't be more than the moles we started with
      if (dso4dt.gt.tmdmol(isx)) then
        dso4d=tmdmol(isx)/(tmd(ih)*tmd(is))
        dso4dt=dso4d*tmd(ih)*tmd(is)
      endif
      if (dso4dt.gt.tmdmol(ihx)) then
        dso4d=tmdmol(ihx)/(tmd(ih)*tmd(is))
        dso4dt=dso4d*tmd(ih)*tmd(is)
      endif
 22   continue

! save final concentrations and diagnostics
      do n=1,aqchem_count
        ix=aqchem_list(n)
        select case (trname(ix))
        case('SO4','M_ACC_SU','ASO4__01')
          sulfout(n)=tr_mm(ix)/1.d3*(dso4gt+dso4dt) ! kg
          dt_sulf(ix) = dt_sulf(ix) + sulfout(n)

        case('SO2','H2O2','H2O2_s')
          select case (trname(ix))
          case('SO2')
            sulfinom(n)=-dso4g*tmg(ih)*tr_mm(ix)/1.d3 ! dimensionless
            sulfinc(n)=-dso4d*tmd(ih)*tr_mm(ix)/1.d3*tmd(ix) ! kg
          case('H2O2','H2O2_s')
            select case (trname(ix))
            case('H2O2')
              if (coupled_chem.le.0) goto 401
            case('H2O2_s')
              if (coupled_chem.eq.1) goto 401
            end select
            sulfinom(n)=-dso4g*tmg(is)*tr_mm(ix)/1.d3 ! dimensionless
            sulfinc(n)=-dso4d*tmd(is)*tr_mm(ix)/1.d3*tmd(ix) ! kg
          end select
          sulfin(n)=sulfinom(n)*tmg(ix) ! kg

          sulfinom(n)=max(-1.d0,sulfinom(n))
          sulfin(n)=max(-tmg(ix),sulfin(n))
          sulfinc(n)=max(-tmd(ix),sulfinc(n))
          tr_left(n)=0.d0
          if (fcloud.gt.abs(sulfinom(n))) then
            tr_left(n)=fcloud+sulfinom(n)
          endif
 401      continue
          dt_sulf(ix)=dt_sulf(ix)+sulfin(n)+sulfinc(n)

        end select
      enddo

      END SUBROUTINE GET_SULFATE

#ifdef BC_ALB
      SUBROUTINE GET_BC_DALBEDO(i,j,bc_dalb,snow_present)
!@sum Calculates change to albedo of snow on ice and snow on land due
!@+     to BC within the snow.
!@+     Parameterization based on Warren and Wiscombe (1980) (21 inputs)
!@+     or actually on Flanner et al. fig 2 r_e=500 (14 inputs)
!@+     or Warren and Wiscombe (1985) (18 input, old vs new, then I
!@+     continue linearly from 19-29 off the plot)
!@auth Dorothy Koch, modified by Kostas Tsigaridis

c gtracer(n,i,j) is tracer concentration in snow on sea ice?

!@param rhow density of pure water [kg m-3]
      USE CONSTANT, only: rhow
!@var tr_wsn_ij tracer amount in snow over land (multiplied by fr_snow) [kg m-2]
c wsn_ij(nsl,2,i,j)
      USE GHY_COM, only: tr_wsn_ij, wsn_ij
!@var si_atm%snowi snow amount on sea ice [kg m-2]
      USE SEAICE_COM, only : si_atm
      USE FLUXES, only: atmice
      USE TRACER_COM, only: itrBC
      IMPLICIT NONE
c Warren and Wiscombe 1985 includes age dependence
      real*8, parameter :: bc(29)=(/1.d0,2.d0,3.d0,4.d0,5.d0,
     * 6.d0,7.d0,8.d0,9.d0,10.d0,20.d0,30.d0,40.d0,50.d0,60.d0,
     * 70.d0,80.d0,90.d0,100.d0,110.d0,120.d0,130.d0,140.d0,
     * 150.d0,160.d0,170.d0,180.d0,190.d0,200.d0/)
      real*8, parameter :: daln(29)=(/0.d0,0.1d0,0.1d0,0.2d0,
     * 0.2d0,0.2d0,0.2d0,0.3d0,0.3d0,0.4d0,0.7d0,0.9d0,1.1d0,
     * 1.3d0,1.5d0,1.6d0,1.8d0,2.d0,2.2d0,2.4d0,2.6d0,2.8d0,
     * 3.d0,3.2d0,3.4d0,3.6d0,3.8d0,4.d0,4.2d0/)
      real*8, parameter :: dalo(29)=(/0.1d0,0.2d0,0.4d0,0.5d0,
     * 0.6d0,0.7d0,0.8d0,0.9d0,1.d0,1.d0,2.d0,2.6d0,3.2d0,
     * 3.8d0,4.3d0,4.8d0,5.2d0,5.5d0,5.9d0,6.3d0,6.7d0,7.1d0,
     * 7.5d0,7.9d0,8.3d0,8.7d0,9.1d0,9.5d0,9.9d0/)
c Flanner et al
c     real*8, parameter :: bc(14)=(/25.d0,50.d0,100.d0,150.d0,
c    * 200.d0,
c    *250.d0,300.d0,400.d0,500.d0,600.d0,700.d0,800.d0,900.d0,
c    * 1000.d0/)
c     real*8, parameter :: dal(14)=(/1.d0,2.d0,3.d0,4.d0,5.d0,
c    * 6.d0,7.d0,8.d0,9.d0,10.d0,11.d0,11.5d0,12.d0,12.5d0/)
c Warren and Wiscomb 1980
c     real*8, parameter :: bc(21)=(/0.05d0,0.075d0,0.1d0,0.2d0,
c    * 0.3d0,0.4d0,0.5d0,0.6d0,0.7d0,0.8d0,0.9d0,1.d0,2.d0,
c    * 3.d0,4.d0,5.d0,6.d0,7.d0,8.d0,9.d0,10.d0/)
c     real*8, parameter :: dal(21)=(/2.d0,3.d0,4.d0,6.d0,8.d0,
c    * 9.d0,10.d0,11.d0,12.d0,13.d0,14.d0,16.d0,18.d0,20.d0,
c    * 22.d0,24.d0,26.d0,28.d0,30.d0,32.d0,34.d0/)

!@var bcsnowb BC amount in snow over bare soil [kg m-2]
!@var bcsnowv BC amount in snow over vegetation [kg m-2]
!@var sconb BC concentration in snow over bare soil [kg kg-1]
!@var sconv BC concentration in snow over bare soil [kg kg-1]
!@var scon BC concentration in snow over land [ppm by mass]
!@var icon BC concentration in snow over sea ice [ppm by mass]
      real*8 ::  bcsnowb,bcsnowv,sconb,sconv,scon,icon
!@var fb fraction of land with bare soil (1.-fv)
!@var fv fraction of land with vegetation (1.-fb)
!@var bcc BC concentration (=max(scon,icon)) to be used for albedo calculations
!@var rads snow grain size determined in GRAINS
      real*8 :: fv,fb,bcc,rads
      INTEGER n,ib
      INTEGER, INTENT(IN) :: i,j
      REAL*8, INTENT(OUT) :: bc_dalb
      logical, intent(out) :: snow_present
      integer :: ntrBC

      ntrBC = size(itrBC)

! initialize
      bcsnowb=0.d0
      bcsnowv=0.d0
      sconb=0.d0
      sconv=0.d0
      scon=0.d0
      icon=0.d0
      bc_dalb=0.d0
      snow_present = .false.

! get bare soil and vegetation fractions (fb+fv=1.)
      call get_fb_fv( fb, fv, i, j )

! calculate BC concentration in snow layer 1 over bare soil
      if (wsn_ij(1,1,i,j).gt.0.d0) then
        snow_present = .true. ! should this ignore trace amounts of snow?
        do n=1,ntrBC
          bcsnowb=bcsnowb+tr_wsn_ij(itrBC(n),1,1,i,j)
        enddo
        sconb=bcsnowb/wsn_ij(1,1,i,j)/rhow
      endif

! calculate BC concentration in snow layer 1 over vegetation
      if (wsn_ij(1,2,i,j).gt.0.d0) then
        snow_present = .true. ! should this ignore trace amounts of snow?
        do n=1,ntrBC
          bcsnowv=bcsnowv+tr_wsn_ij(itrBC(n),1,2,i,j)
        enddo
        sconv=bcsnowv/wsn_ij(1,2,i,j)/rhow
      endif

! calculate mean BC concentration in total snow in layer 1
      scon=(fb*sconb+fv*sconv)*1.d9

! calculate BC concentration in snow over sea ice
      if (si_atm%snowi(i,j).gt.0.d0) then
        snow_present = .true. ! should this ignore trace amounts of snow?
        do n=1,ntrBC
          icon=icon+atmice%gtracer(itrBC(n),i,j)*1.d9
        enddo
      endif

! use the maximum BC concentration between snow over land and over sea ice
      bcc=max(icon,scon)

! calculate snow grain size
      call GRAINS(i,j,rads)

! calculate BC albedo effect
      do ib=1,28
        if (bcc.gt.bc(ib).and.bcc.lt.bc(ib+1)) then
          bc_dalb=-(daln(ib)
     *       +(rads-100.d0)/900.d0*(dalo(ib)-daln(ib)))/100.d0
          exit
        endif
      enddo
      if (bcc.ge.bc(29)) bc_dalb=-(daln(29)
     *    + (rads-100.d0)/900.d0*(dalo(29)-daln(29)))/100.d0
c     if (bc_dalb.ne.0.) write(6,*) 'alb_write',i,j,bc_dalb,bcc,rads

      END SUBROUTINE GET_BC_DALBEDO

      SUBROUTINE GRAINS(i,j,rads)
!@sum Estimates snow grain size (microns) based on air temperature
!@+     and snow age. From Susan Marshall's PhD thesis
!@auth Dorothy Koch

      USE CONSTANT, only: pi,gasc,tf
      USE FLUXES, only: atmsrf
      use TimeConstants_mod, only: DAYS_PER_YEAR
      USE RAD_COM, only: snoage
      USE AEROSOL_SOURCES, only: snosiz
      IMPLICIT none
      REAL*8 E,A,age,r0,radmm,ert,
     * tfac,area,delrad
      INTEGER, INTENT(IN) :: i,j
      REAL*8, INTENT(OUT) :: rads
      DATA E,A /26020.d0, 29100.d0/
c tsavg(i,j) surface air temperature
c snoage(k,i,j) k=1 ocean ice, k=2 land ice, k=3 land
c   (do these differ within a gridbox?) in days
c RN1 radius from previous timestep
c RADS snow grain radius
c
c Find the age of snow, I assume the age does not
c  vary within the gridbox so just take the max?
       age=DMAX1(snoage(1,i,j),snoage(2,i,j),snoage(3,i,j))
c Use Temperature to check if melting or non-melting snow
       IF (atmsrf%tsavg(i,j).le.tf) then
c Non-melting snow; distinguish between initial or
c  secondary growth rate
        IF (age.lt.13.5) then
c initial growth
         r0=50.
         radmm=r0+ (0.008d0+age)
         rads=radmm*1000.d0
         snosiz(i,j)=rads+r0
        ELSE
c secondary growth
         r0=150.d0
         ert = dexp(-E/(GASC*atmsrf%TSAVG(I,J)))
         tfac = a*ert
         area = (TFAC/DAYS_PER_YEAR) * (AGE-12.5d0)
         radmm=dsqrt(area/pi)
         delrad = radmm*1000.d0
         rads=delrad + r0
         snosiz(i,j)=rads
        ENDIF
       ELSE
c melting snow
        radmm=dsqrt(0.05d0)
        rads=snosiz(i,j)+(radmm*100.d0)
        snosiz(i,j)=rads
       ENDIF
       rads=DMIN1(rads,1000.d0)
       rads=DMAX1(rads,100.d0)
      RETURN
      END SUBROUTINE GRAINS
#endif  /* BC_ALB */
