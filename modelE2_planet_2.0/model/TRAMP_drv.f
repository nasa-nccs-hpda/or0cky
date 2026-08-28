#include "rundeck_opts.h"
      SUBROUTINE MATRIX_DRV(i,j)
!@vers 2013/03/27
      USE AmpTracersMetadata_mod, only: AMP_MODES_MAP, AMP_NUMB_MAP,
     *  AMP_AERO_MAP
      USE TRACER_COM, only: n_H2SO4, n_M_ACC_SU, n_M_AKK_SU, n_M_BC1_BC,
     *  n_M_DD1_DU, n_M_DD2_DU, n_M_OCC_OC, n_M_SSA_SS, n_M_SSC_SS, n_M_SSS_SS,
     *  n_NH3, nBiomass, nAircraft, ntmAMPe, nVolcanic, trm_col, ntmAMPi,
     *  nMicrophys, nThermo, n_M_OCC_OCM2, n_M_OCC_OCM1, n_M_OCC_OCM0,
     *  n_M_OCC_OCP1, n_M_OCC_OCP2, n_M_OCC_OCP3, n_M_OCC_OCP4,
     *  n_M_OCC_OCP5, n_M_OCC_OCP6
      use TRACER_COM, only: nRocket
#ifdef  TRACERS_SPECIAL_Shindell
      USE TRACER_COM, only: n_HNO3
#endif
#ifdef TRACERS_AMP_M9
      use TRACER_COM, only: n_vbsGm2,n_vbsGm1,n_vbsGz,n_vbsGp1,n_vbsGp2,
     &                      n_vbsGp3,n_vbsGp4,n_vbsGp5,n_vbsGp6,nOther
#endif  /* TRACERS_AMP_M9 */
      use OldTracer_mod, only: trname
      USE TRDIAG_COM, only : taijs=>taijs_loc,taijls=>taijls_loc
     *     ,ijts_AMPp,ijlt_AMPm,ijts_AMPpdf, ijts_AMPe
     *     ,itcon_AMP,itcon_AMPm
      USE AMP_AEROSOL
      use RunTimeControls_mod, only: tracers_special_shindell
      use TRACER_COM, only: coupled_chem
      USE AEROSOL_SOURCES, only: off_HNO3

      USE RESOLUTION, only : lm     ! dimensions
      USE MODEL_COM, only : dtsrc
      USE CONSTANT,   only:  lhe,mair,gasc,rgas  
      USE FLUXES, only: tr3Dsource,trsource,Fland
      use ATMCOL_COM, only: tl   ! layer temperature (K)
      use ATMCOL_COM, only: rhl  ! layer relative humidity (0-1)
      use ATMCOL_COM, only: pl   ! layer pressure (mb)
      use ATMCOL_COM, only: ma   ! layer mass (kg/m2)
      use ATMCOL_COM, only: byma ! 1/ma
      USE AERO_CONFIG
      USE AERO_INIT
      USE AERO_PARAM, only: ILAY, NEMIS_SPCS
      USE AERO_DIAM, only: DP, DP_DRY
      USE AERO_ACTV, only: NACTIV
      USE AERO_SETUP 
      USE PBLCOM,     only: EGCM !(LM,IM,JM) 3-D turbulent kinetic energy [m^2/s^2]
      USE AERO_SUBS, only: SPCMASSES
!      USE AERO_SUBS, only:  SIZE_PDFS
      USE AEROSOL_SOURCES, only: oxid

      IMPLICIT NONE
      integer, intent(in) :: i,j

      REAL(8):: TK,RH,PRES,TSTEP,AQSO4RATE,PM(3)
      REAL(8):: AERO(NAEROBOX)     ! aerosol conc. [ug/m^3] or [#/m^3]
      REAL(8):: GAS(NGASES)        ! gas-phase conc. [ug/m^3]
      REAL(8):: EMIS_MASS(NEMIS_SPCS) ! mass emission rates [ug/m^3]
      REAL(8):: SPCMASS(NMASS_SPCS+2)
      REAL(8):: DT_AERO(NDIAG_AERO,NAEROBOX) !NDIAG_AERO=15
      REAL(8):: yS, yM, WUP,AVOL
      REAL(8) :: PDF1(NBINS)               ! number or mass conc. at each grid point [#/m^3] or [ug/m^3]       
      REAL(8) :: PDF2(NBINS)               ! number or mass conc. at each grid point [#/m^3] or [ug/m^3]       
      INTEGER:: l,n,J_0, J_1, I_0, I_1, m,nAMP
C**** functions
      REAL(8):: QSAT

      NACTV(I,J,:,:)      = 0.d0 
      DIAM(I,J,:,:)       = 0.d0
      DIAM_dry(I,J,:,:)   = 0.d0

      DO L=1,LM                            

      ILAY = L
      DT_AERO(:,:) = 0.d0
      AERO(:)      = 0.d0
! meteo
      RH = MIN(1.d0,rhl(l)) ! rH [0-1]
      PRES= pl(l)*100. ! PRES in [Pa]
      TSTEP=dtsrc
      WUP = SQRT(.6666667*EGCM(l,i,j))  ! updraft velocity

!@var avol volume of air per m2 of surface area per layer [m3/m2/layer]
      AVOL = MA(l)/mair*1000.d0*gasc*tl(l)/pres 
! in-cloud SO4 production rate [ug/m^3/s] ::: AQsulfRATE [kg] 
      AQSO4RATE = AQsulfRATE (l,i,j)* 1.d9  / AVOL /dtsrc

c conversion trm_col [kg/m2/layer] -> GAS [ug/m^3]
      GAS(GAS_H2SO4) = trm_col(l,n_H2SO4)* 1.d9 / AVOL ! [ug H2SO4/m^3]
#ifndef TRACERS_AMP_M11 /* not */
      if (tracers_special_shindell.and.coupled_chem==1) then
        GAS(GAS_HNO3) = trm_col(l,n_HNO3)*1.d9 / AVOL  ! [ug HNO3/m^3]
      else
        GAS(GAS_HNO3) = off_HNO3(i,j,l)*1.d9 /AVOL     ! [ug HNO3/m^3]
      endif
      GAS(GAS_NH3) = trm_col(l,n_NH3)* 1.d9 / AVOL     ! [ug NH3/m^3]
#ifdef TRACERS_AMP_M9
      GAS(GAS_OCM2) = trm_col(l,n_vbsGm2)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCM1) = trm_col(l,n_vbsGm1)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCM0) = trm_col(l,n_vbsGz)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP1) = trm_col(l,n_vbsGp1)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP2) = trm_col(l,n_vbsGp2)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP3) = trm_col(l,n_vbsGp3)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP4) = trm_col(l,n_vbsGp4)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP5) = trm_col(l,n_vbsGp5)* 1.d9 / AVOL ! [ug OM/m^3]
      GAS(GAS_OCP6) = trm_col(l,n_vbsGp6)* 1.d9 / AVOL ! [ug OM/m^3]
      call get_oxidants(i,j,l)
      GAS(GAS_OH)   = oxid%OH
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */
!  [kg/s] -> [ug/m3/s]

       DO n=ntmAMPi,ntmAMPe
         nAMP=n-ntmAMPi+1
c conversion trm_col [kg/m2/layer] -> AERO [ug/m3]
         if(AMP_NUMB_MAP(nAMP).eq. 0) then
       AERO(AMP_AERO_MAP(nAMP)) =trm_col(l,n)*1.d9 / AVOL ! ug/m3
          else

       AERO(AMP_AERO_MAP(nAMP)) =trm_col(l,n)/ AVOL       !  #/m3
          endif
       ENDDO

! save emissions to EMIS_MASS
      EMIS_MASS(:) = 0.d0
      if (L.eq.1) then
!      Emis Mass [ug/m3/s] <-- trsource[kg/m2/s]
        if (n_M_AKK_SU>0) EMIS_MASS(1) =sum(trsource(i,j,:,n_M_AKK_SU))*1.d9/ AVOL
        if (n_M_ACC_SU>0) EMIS_MASS(2) =sum(trsource(i,j,:,n_M_ACC_SU))*1.d9/ AVOL
#ifndef TRACERS_AMP_M11 /* not */
        if (n_M_BC1_BC>0) EMIS_MASS(3) =sum(trsource(i,j,:,n_M_BC1_BC))*1.d9/ AVOL
        if (n_M_OCC_OC>0) EMIS_MASS(4) =sum(trsource(i,j,:,n_M_OCC_OC))*1.d9/ AVOL
        if (n_M_DD1_DU>0) EMIS_MASS(5) =sum(trsource(i,j,:,n_M_DD1_DU))*1.d9/ AVOL
        if (n_M_SSS_SS>0) EMIS_MASS(6) =sum(trsource(i,j,:,n_M_SSS_SS))*1.d9/ AVOL ! only for M4/M8
        if (n_M_SSA_SS>0) EMIS_MASS(6) =sum(trsource(i,j,:,n_M_SSA_SS))*1.d9/ AVOL ! all but M4/M8
        if (n_M_SSC_SS>0) EMIS_MASS(7) =sum(trsource(i,j,:,n_M_SSC_SS))*1.d9/ AVOL ! all but M4/M8
        if (n_M_DD2_DU>0) EMIS_MASS(10)=sum(trsource(i,j,:,n_M_DD2_DU))*1.d9/ AVOL
#ifdef TRACERS_AMP_M9
        if (n_M_OCC_OCM2>0) EMIS_MASS(11) =sum(trsource(i,j,:,n_M_OCC_OCM2))*1.d9/ AVOL
        if (n_M_OCC_OCM1>0) EMIS_MASS(12) =sum(trsource(i,j,:,n_M_OCC_OCM1))*1.d9/ AVOL
        if (n_M_OCC_OCM0>0) EMIS_MASS(13) =sum(trsource(i,j,:,n_M_OCC_OCM0))*1.d9/ AVOL
        if (n_M_OCC_OCP1>0) EMIS_MASS(14) =sum(trsource(i,j,:,n_M_OCC_OCP1))*1.d9/ AVOL
        if (n_M_OCC_OCP2>0) EMIS_MASS(15) =sum(trsource(i,j,:,n_M_OCC_OCP2))*1.d9/ AVOL
        if (n_M_OCC_OCP3>0) EMIS_MASS(16) =sum(trsource(i,j,:,n_M_OCC_OCP3))*1.d9/ AVOL
        if (n_M_OCC_OCP4>0) EMIS_MASS(17) =sum(trsource(i,j,:,n_M_OCC_OCP4))*1.d9/ AVOL
        if (n_M_OCC_OCP5>0) EMIS_MASS(18) =sum(trsource(i,j,:,n_M_OCC_OCP5))*1.d9/ AVOL
        if (n_M_OCC_OCP6>0) EMIS_MASS(19) =sum(trsource(i,j,:,n_M_OCC_OCP6))*1.d9/ AVOL
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */
      endif
!      Emis Mass [ug/m3/s] <-- tr3Dsource[kg/s]
      if (n_M_AKK_SU>0)
     * EMIS_MASS(1) = EMIS_MASS(1) + ((tr3Dsource(l,nVolcanic,n_M_AKK_SU)+
     *                                 tr3Dsource(l,nBiomass,n_M_AKK_SU)+
     *                                 tr3Dsource(l,nAircraft,n_M_AKK_SU)+
     *                                 tr3Dsource(l,nRocket,n_M_AKK_SU)
     *                                )*1.d9 / AVOL)
      if (n_M_ACC_SU>0)
     * EMIS_MASS(2) = EMIS_MASS(2) + ((tr3Dsource(l,nVolcanic,n_M_ACC_SU)+
     *                                 tr3Dsource(l,nBiomass,n_M_ACC_SU)+
     *                                 tr3Dsource(l,nAircraft,n_M_ACC_SU)+
     *                                 tr3Dsource(l,nRocket,n_M_ACC_SU)
     *                                )*1.d9 / AVOL)
#ifndef TRACERS_AMP_M11 /* not */
      if (n_M_BC1_BC>0)
     * EMIS_MASS(3) = EMIS_MASS(3) + ((tr3Dsource(l,nBiomass,n_M_BC1_BC)+
     *                                 tr3Dsource(l,nAircraft,n_M_BC1_BC)+
     *                                 tr3Dsource(l,nRocket,n_M_BC1_BC)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OC>0)
     * EMIS_MASS(4) = EMIS_MASS(4) + ((tr3Dsource(l,nBiomass,n_M_OCC_OC)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OC)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OC)
     *                                )*1.d9 / AVOL)
!     enable dust as proxy for volcanic ash in direct_inject experiments
      if (n_M_DD1_DU>0)
     * EMIS_MASS(5) = EMIS_MASS(5) + ((tr3Dsource(l,nVolcanic,n_M_DD1_DU)
     *                                )*1.d9 / AVOL)
      if (n_M_DD2_DU>0)
     * EMIS_MASS(10) = EMIS_MASS(10) + ((tr3Dsource(l,nVolcanic,n_M_DD2_DU)
     *                                )*1.d9 / AVOL)
#ifdef TRACERS_AMP_M9
      if (n_M_OCC_OCM2>0)
     * EMIS_MASS(11) = EMIS_MASS(11) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCM2)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCM2)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCM2)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCM1>0)
     * EMIS_MASS(12) = EMIS_MASS(12) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCM1)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCM1)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCM1)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCM0>0)
     * EMIS_MASS(13) = EMIS_MASS(13) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCM0)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCM0)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCM0)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP1>0)
     * EMIS_MASS(14) = EMIS_MASS(14) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP1)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP1)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP1)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP2>0)
     * EMIS_MASS(15) = EMIS_MASS(15) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP2)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP2)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP2)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP3>0)
     * EMIS_MASS(16) = EMIS_MASS(16) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP3)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP3)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP3)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP4>0)
     * EMIS_MASS(17) = EMIS_MASS(17) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP4)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP4)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP4)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP5>0)
     * EMIS_MASS(18) = EMIS_MASS(18) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP5)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP5)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP5)
     *                                )*1.d9 / AVOL)
      if (n_M_OCC_OCP6>0)
     * EMIS_MASS(19) = EMIS_MASS(19) + ((tr3Dsource(l,nBiomass,n_M_OCC_OCP6)+
     *                                 tr3Dsource(l,nAircraft,n_M_OCC_OCP6)+
     *                                 tr3Dsource(l,nRocket,n_M_OCC_OCP6)
     *                                )*1.d9 / AVOL)
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */

       CALL SPCMASSES(AERO,GAS,SPCMASS)

!=========
! WARNING: EMIS_MASS is only used to modify number, the mass is already modified in ATURB.
!=========
       CALL MATRIX(AERO,GAS,EMIS_MASS,TSTEP,tl(l),RH,PRES,AQSO4RATE,WUP,FLAND(i,j),DT_AERO,PM)
c       CALL SIZE_PDFS(AERO,PDF1,PDF2)
       do n=1,nweights
         DIAM(i,j,l,n)=DP(n)
         DIAM_dry(i,j,l,n)=DP_DRY(n)
         NACTV(i,j,l,n)=NACTIV(n)
       enddo
 
       DO n=ntmAMPi,ntmAMPe
         nAMP=n-ntmAMPi+1
         select case(trname(n))
         case ('M_NO3','M_NH4','M_H2O')
      tr3Dsource(l,nThermo,n) =((AERO(AMP_AERO_MAP(nAMP)) *AVOL *1.d-9)
     *        -trm_col(l,n)) /dtsrc
         case default
           if(AMP_NUMB_MAP(nAMP).eq. 0) then
      tr3Dsource(l,nMicrophys,n) =((AERO(AMP_AERO_MAP(nAMP)) *AVOL *1.d-9)
     *        -trm_col(l,n)) /dtsrc 
           else
      tr3Dsource(l,nMicrophys,n) =((AERO(AMP_AERO_MAP(nAMP)) *AVOL)
     *        -trm_col(l,n)) /dtsrc
           endif   
         end select
       ENDDO

      tr3Dsource(l,nMicrophys,n_H2SO4)=((GAS(GAS_H2SO4)*AVOL *1.d-9)
     *        -trm_col(l,n_H2SO4))/dtsrc 
#ifndef TRACERS_AMP_M11 /* not */
#ifdef  TRACERS_SPECIAL_Shindell
      tr3Dsource(l,nThermo,n_HNO3)=((GAS(GAS_HNO3)*AVOL * 1.d-9)
     *        -trm_col(l,n_HNO3))/dtsrc
#endif
      tr3Dsource(l,nThermo,n_NH3)=((GAS(GAS_NH3)*AVOL *1.d-9)
     *        -trm_col(l,n_NH3))/dtsrc
#ifdef TRACERS_AMP_M9
      tr3Dsource(l,nOther,n_vbsGm2)=((GAS(GAS_OCM2)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGm2))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGm1)=((GAS(GAS_OCM1)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGm1))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGz)=((GAS(GAS_OCM0)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGz))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp1)=((GAS(GAS_OCP1)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp1))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp2)=((GAS(GAS_OCP2)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp2))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp3)=((GAS(GAS_OCP3)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp3))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp4)=((GAS(GAS_OCP4)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp4))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp5)=((GAS(GAS_OCP5)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp5))/dtsrc 
      tr3Dsource(l,nOther,n_vbsGp6)=((GAS(GAS_OCP6)*AVOL *1.d-9)
     *        -trm_col(l,n_vbsGp6))/dtsrc 
#endif  /* TRACERS_AMP_M9 */
#endif  /* not TRACERS_AMP_M11 */

c       DT_AERO(:,:) = DT_AERO(:,:) * dtsrc !DT_AERO [# or ug/m3/s] , taijs [kg m2/kg(air)], byMA [kg/m2]

      if (l.eq.1) then
c - 2d acc output
c      PM1  [ug/m3] - [kg/kg(air)]
        taijs(i,j,ijts_AMPe(1))=taijs(i,j,ijts_AMPe(1)) + PM(1)*1.d-9*rgas*tl(l)/pres 
c      PM2.5
        taijs(i,j,ijts_AMPe(2))=taijs(i,j,ijts_AMPe(2)) + PM(2)*1.d-9*rgas*tl(l)/pres 
        ampPM2p5(i,j) = PM(2)*1.d-9*rgas*tl(l)/pres 
c      PM10
        taijs(i,j,ijts_AMPe(3))=taijs(i,j,ijts_AMPe(3)) + PM(3)*1.d-9*rgas*tl(l)/pres 
        ampPM10(i,j)  = PM(3)*1.d-9*rgas*tl(l)/pres 
        endif

c Update physical properties per mode
       do n=ntmAMPi,ntmAMPe
         nAMP=n-ntmAMPi+1
c Diagnostic of Processes - Sources and Sincs - timestep included
          if(AMP_NUMB_MAP(nAMP).eq. 0) then  !taijs [kg/s] -> in acc [kg/m2*s]
            do m=1,7
              taijs(i,j,ijts_AMPp(m,n)) =taijs(i,j,ijts_AMPp(m,n)) +(DT_AERO(m+8,AMP_AERO_MAP(nAMP))* AVOL * 1.d-9)
              if (itcon_amp(m,n).gt.0) call inc_diagtcb(i,j,(DT_AERO(m+8,AMP_AERO_MAP(nAMP))* AVOL * 1.d-9),itcon_amp(m,n),n)
            end do
             
          else
            taijs(i,j,ijts_AMPp(1,n)) =taijs(i,j,ijts_AMPp(1,n))+(DT_AERO(2,AMP_AERO_MAP(nAMP))* AVOL)
              if (itcon_amp(1,n).gt.0) call inc_diagtcb(i,j,(DT_AERO(2,AMP_AERO_MAP(nAMP))* AVOL),itcon_amp(1,n),n)
            taijs(i,j,ijts_AMPp(2,n)) =taijs(i,j,ijts_AMPp(2,n))+(DT_AERO(3,AMP_AERO_MAP(nAMP))* AVOL)
              if (itcon_amp(2,n).gt.0) call inc_diagtcb(i,j,(DT_AERO(3,AMP_AERO_MAP(nAMP))* AVOL),itcon_amp(2,n),n)
            taijs(i,j,ijts_AMPp(3,n)) =taijs(i,j,ijts_AMPp(3,n))+(DT_AERO(1,AMP_AERO_MAP(nAMP))* AVOL)
              if (itcon_amp(3,n).gt.0) call inc_diagtcb(i,j,(DT_AERO(1,AMP_AERO_MAP(nAMP))* AVOL),itcon_amp(3,n),n)
            do m=4,7
              taijs(i,j,ijts_AMPp(m,n)) =taijs(i,j,ijts_AMPp(m,n))+(DT_AERO(m,AMP_AERO_MAP(nAMP))* AVOL)
              if (itcon_amp(m,n).gt.0) call inc_diagtcb(i,j,(DT_AERO(m,AMP_AERO_MAP(nAMP))* AVOL),itcon_amp(m,n),n)
            end do

          endif
       select case (trname(n)) !taijs [kg * m2/kg air] -> in acc [kg/kg air]
      CASE('N_AKK_1 ','N_ACC_1 ','N_DD1_1 ','N_DS1_1 ','N_DD2_1 ','N_DS2_1 ','N_SSA_1','N_SSC_1','N_OCC_1 ','N_BC1_1 ',
     *     'N_BC2_1 ','N_BC3_1 ','N_DBC_1 ','N_BOC_1 ','N_BCS_1 ','N_MXX_1 ','N_OCS_1 ')
c - 3d acc output
        taijls(i,j,l,ijlt_AMPm(1,n))=taijls(i,j,l,ijlt_AMPm(1,n)) + DIAM(i,j,l,AMP_MODES_MAP(nAMP))*AERO(AMP_AERO_MAP(nAMP))
        taijls(i,j,l,ijlt_AMPm(3,n))=taijls(i,j,l,ijlt_AMPm(3,n)) + DIAM_dry(i,j,l,AMP_MODES_MAP(nAMP))*AERO(AMP_AERO_MAP(nAMP))
        taijls(i,j,l,ijlt_AMPm(2,n))=taijls(i,j,l,ijlt_AMPm(2,n)) + (NACTV(i,j,l,AMP_MODES_MAP(nAMP))*AVOL*byMA(l))


c - 2d PRT Diagnostic
        if (itcon_AMPm(1,n) .gt.0) call inc_diagtcb(i,j,(DIAM(i,j,l,AMP_MODES_MAP(nAMP))*1d6),itcon_AMPm(1,n),n) 
        if (itcon_AMPm(3,n) .gt.0) call inc_diagtcb(i,j,(DIAM_dry(i,j,l,AMP_MODES_MAP(nAMP))*1d6),itcon_AMPm(3,n),n) 
        if (itcon_AMPm(2,n) .gt.0) call inc_diagtcb(i,j,NACTV(i,j,l,AMP_MODES_MAP(nAMP))*AVOL ,itcon_AMPm(2,n),n) 
       end select

      enddo !n
      ENDDO !l

      RETURN
      END SUBROUTINE MATRIX_DRV
c -----------------------------------------------------------------

      subroutine matrix_post
#ifdef CACHED_SUBDD /* currently, this routine is exclusively for subdd */
      use domain_decomp_atm, only : grid
      use subdd_mod, only : subdd_groups,subdd_type,subdd_ngroups,
     &                      inc_subdd,find_groups
      use tracer_com, only : ntmAMPi,ntmAMPe
      use OldTracer_mod, only : trname
      use AmpTracersMetadata_mod, only: amp_modes_map
      use amp_aerosol, only : diam
      use resolution, only : lm
      USE TRACER_COM, only: trm
      USE TRACER_COM, only: n_M_DD1_DU,n_M_DS1_DU,n_M_DD2_DU,n_M_DS2_DU,
     &                      n_M_DBC_DU,n_M_MXX_DU
      USE TRACER_COM, only: n_M_BC1_BC,n_M_BC2_BC,n_M_BC3_BC,n_M_DBC_BC,
     &                      n_M_BOC_BC,n_M_BCS_BC,n_M_MXX_BC
      USE TRACER_COM, only: n_M_NH4
      USE TRACER_COM, only: n_M_NO3
      USE TRACER_COM, only: n_M_OCC_OC,n_M_BOC_OC,n_M_MXX_OC
      USE TRACER_COM, only: n_M_AKK_SU,n_M_ACC_SU,n_M_DD1_SU,n_M_DS1_SU,
     &                      n_M_DD2_SU,n_M_DS2_SU,n_M_SSA_SU,n_M_OCC_SU,
     &                      n_M_BC1_SU,n_M_BC2_SU,n_M_BC3_SU,n_M_BOC_SU,
     &                      n_M_BCS_SU,n_M_DBC_SU,n_M_MXX_SU
      USE TRACER_COM, only: n_M_SSA_SS,n_M_SSC_SS,n_M_MXX_SS
      implicit none
      integer :: igrp,ngroups,grpids(subdd_ngroups),k,n,nAMP
      type(subdd_type), pointer :: subdd
      real*8, dimension(grid%i_strt:grid%i_stop,
     &                  grid%j_strt:grid%j_stop) :: sddarr2d
      real*8, dimension(grid%i_strt:grid%i_stop,
     &                  grid%j_strt:grid%j_stop,lm) :: sddarr3d

      call find_groups('taijlh',grpids,ngroups)
      do igrp=1,ngroups
        subdd => subdd_groups(grpids(igrp))
        do k=1,subdd%ndiags
          do n=ntmAMPi,ntmAMPe
            if (trim(subdd%name(k)) /= 'd'//trim(trname(n))) cycle
            nAMP=n-ntmAMPi+1
            sddarr3d(:,:,:)=diam(:,:,:,amp_modes_map(nAMP))
            call inc_subdd(subdd,k,sddarr3d)
          enddo ! n
        enddo ! k
      enddo ! igrp

! Tracer 2D I-J diags
      call find_groups('taijh',grpids,ngroups)
      do igrp=1,ngroups
      subdd => subdd_groups(grpids(igrp))
      do k=1,subdd%ndiags
      select case(trim(subdd%name(k)))
         case('ampDustload')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_DD1_DU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DS1_DU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DD2_DU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DS2_DU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DBC_DU),dim=3)
     *                  +sum(trm(:,:,:,n_M_MXX_DU),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampBCload')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_BC1_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_BC2_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_BC3_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_DBC_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_BOC_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_BCS_BC),dim=3)
     *                  +sum(trm(:,:,:,n_M_MXX_BC),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampNH4load')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_NH4),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampNO3load')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_NO3),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampOAload')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_OCC_OC),dim=3)
     *                  +sum(trm(:,:,:,n_M_BOC_OC),dim=3)
     *                  +sum(trm(:,:,:,n_M_MXX_OC),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampSO4load')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_AKK_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_ACC_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DD1_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DS1_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DD2_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DS2_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_SSA_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_OCC_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_BC1_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_BC2_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_BC3_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_BOC_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_BCS_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_DBC_SU),dim=3)
     *                  +sum(trm(:,:,:,n_M_MXX_SU),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 
         case('ampSSload')
         sddarr2d(:,:)= (sum(trm(:,:,:,n_M_SSA_SS),dim=3)
     *                  +sum(trm(:,:,:,n_M_SSC_SS),dim=3)
     *                  +sum(trm(:,:,:,n_M_MXX_SS),dim=3))
         call inc_subdd(subdd,k,sddarr2d) 

       end select
        
        enddo ! k
      enddo ! igrp
#endif
      end subroutine matrix_post


!=======================================================================
! Get AMP radius for tracer with index n at gridbox i,j,l
!=======================================================================
      real*8 function AMPtrradius(i,j,l,n) result(radius)
!=======================================================================
      use OldTracer_mod, only: trradius
      USE AmpTracersMetadata_mod, only: AMP_NUMB_MAP
      USE AmpTracersMetadata_mod, only: AMP_MODES_MAP
      USE AERO_SETUP, only: CONV_DPAM_TO_DGN
      USE AMP_AEROSOL, only: DIAM
      use TRACER_COM, only: ntmAMPi

      implicit none
      integer, intent(in) :: i,j,l,n
      integer :: nAMP

      nAMP=n-ntmAMPi+1
      if (AMP_MODES_MAP(nAMP).gt.0) then
        if(AMP_NUMB_MAP(nAMP).eq.0) then ! Mass
          radius=0.5*DIAM(i,j,l,AMP_MODES_MAP(nAMP))
        else                             ! Number
          radius=0.5*DIAM(i,j,l,AMP_MODES_MAP(nAMP))
     &          *CONV_DPAM_TO_DGN(AMP_MODES_MAP(nAMP))
        endif
      else
        radius=trradius(n)
      endif
!=======================================================================
      end function AMPtrradius
!=======================================================================

!=======================================================================
! Get AMP density for mode or tracer with index n at gridbox i,j,l
!=======================================================================
      real*8 function AMPtrdens(i,j,l,n,from_trm_col) result(density)
!=======================================================================
      USE OldTracer_mod, only: trpdens,trname
      USE AmpTracersMetadata_mod, only: AMP_MODES_MAP, AMP_trm_nm1,
     *  AMP_trm_nm2
      USE TRACER_COM, only: ntmAMPi, trm, trm_col

      IMPLICIT NONE
      integer, intent(in) :: i,j,l,n
      logical, intent(in) :: from_trm_col
      real*8, dimension(:), allocatable :: masses,volumes
      real*8 :: totmass,totvol,sulfratio
      integer :: x,nAMP

      nAMP=n-ntmAMPi+1
      if(AMP_MODES_MAP(nAMP) > 0) then ! for tracers in a mode
        allocate(masses(AMP_trm_nm1(nAMP):AMP_trm_nm2(nAMP)))
        allocate(volumes(AMP_trm_nm1(nAMP):AMP_trm_nm2(nAMP)))
        if (from_trm_col) then
          do x=AMP_trm_nm1(nAMP),AMP_trm_nm2(nAMP)
            masses(x)=trm_col(l,x)
            volumes(x)=trm_col(l,x)/trpdens(x)
          enddo
        else
          do x=AMP_trm_nm1(nAMP),AMP_trm_nm2(nAMP)
            masses(x)=trm(i,j,l,x)
            volumes(x)=trm(i,j,l,x)/trpdens(x)
          enddo
        endif
        totmass=sum(masses)
        totvol=sum(volumes)
        if (totvol > 0.d0) then
          density=totmass/totvol
        else
          density=trpdens(n) ! just a value to use when totvol is zero
        endif
        deallocate(masses)
        deallocate(volumes)
      else ! for tracers not belonging to a mode
        density=trpdens(n)
      endif
!=======================================================================
      END function AMPtrdens
!=======================================================================

!=======================================================================
! Get AMP molecular mass for mode or tracer with index n at gridbox i,j,l
!=======================================================================
      real*8 function AMPtrmass(i,j,l,n) result(trmass)
!=======================================================================
      USE OldTracer_mod, only: tr_mm,trname
      USE AmpTracersMetadata_mod, only: AMP_MODES_MAP, AMP_trm_nm1,
     *  AMP_trm_nm2
      USE TRACER_COM, only: ntmAMPi, trm

      IMPLICIT NONE
      integer, intent(in) :: i,j,l,n
      real*8, dimension(:), allocatable :: masses,moles
      real*8 :: totmass,totmol,sulfratio
      integer :: x,nAMP

      nAMP=n-ntmAMPi+1
      if(AMP_MODES_MAP(nAMP) > 0) then ! for tracers in a mode
        allocate(masses(AMP_trm_nm1(nAMP):AMP_trm_nm2(nAMP)))
        allocate(moles(AMP_trm_nm1(nAMP):AMP_trm_nm2(nAMP)))
        do x=AMP_trm_nm1(nAMP),AMP_trm_nm2(nAMP)
          masses(x)=trm(i,j,l,x)
          moles(x)=trm(i,j,l,x)/tr_mm(x)
        enddo
        totmass=sum(masses)
        totmol=sum(moles)
        if (totmol > 0.d0) then
          trmass=totmass/totmol
        else
          trmass=tr_mm(n) ! just a value to use when totmol is zero
        endif
        deallocate(masses)
        deallocate(moles)
      else ! for tracers not belonging to a mode
        trmass=tr_mm(n)
      endif
!=======================================================================
      END function AMPtrmass
!=======================================================================

      subroutine alloc_tracer_amp_com(grid)
!@SUM  To alllocate arrays whose sizes now need to be determined
!@+    at run-time
!@auth Susanne Bauer
      use domain_decomp_atm, only: dist_grid, getDomainBounds
      use resolution, only : lm
      use amp_aerosol
      use aero_config, only: nmodes

      IMPLICIT NONE

      type (dist_grid), intent(in) :: grid
      integer :: I_0,I_1,J_0,J_1
      INTEGER :: J_0H,J_1H, I_0H,I_1H
      logical :: init = .false.

      if(init)return
      init=.true.
    
      call getDomainBounds(grid)
      I_0=GRID%I_STRT
      I_1=GRID%I_STOP
      J_0=GRID%J_STRT
      J_1=GRID%J_STOP


      call getDomainBounds(grid, J_STRT_HALO=J_0H, J_STOP_HALO=J_1H)
      I_0H=GRID%I_STRT_HALO
      I_1H=GRID%I_STOP_HALO

! I,J,L
      allocate(  AQsulfRATE(LM,I_0:I_1,J_0:J_1)   )
! other dimensions
      allocate(  ampPM10(I_0H:I_1H,J_0H:J_1H)  )
      allocate(  ampPM2p5(I_0H:I_1H,J_0H:J_1H)  )
      allocate(  DIAM(I_0:I_1,J_0:J_1,LM,nmodes)  )
      allocate(  DIAM_dry(I_0:I_1,J_0:J_1,LM,nmodes)  )
      allocate(  NACTV(I_0:I_1,J_0:J_1,LM,nmodes) )

      NACTV    = 1.0D-30
      DIAM     = 1.0D-30
      DIAM_dry = 1.0D-30
      ampPM10  = 1.0D-30
      ampPM2p5 = 1.0D-30

      end subroutine alloc_tracer_amp_com

