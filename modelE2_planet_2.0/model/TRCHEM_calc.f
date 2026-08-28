#include "rundeck_opts.h"
      SUBROUTINE chemstep(maxL,I,J)
!@sum chemstep Calculate new concentrations after photolysis & chemistry
!@auth Drew Shindell (modelEifications by Greg Faluvegi)
!@calls rates,chem1,chem1prn
c
C**** GLOBAL parameters and variables:
C
      USE RAD_COM, only         : clim_interact_chem
      USE RESOLUTION, only      : ls1=>ls1_nominal
      USE RESOLUTION, only      : im,jm,lm
      USE ATM_COM, only         : ltropo
      use ATMCOL_COM, only: tl,ma,byma,update_qv,qv,update_qvmom,qvmom
      USE DOMAIN_DECOMP_ATM,only : grid,getDomainBounds,write_parallel
      USE TRDIAG_COM, only : taijls=>taijls_loc,jls_OHcon,jls_day
     &     ,jls_OxpT,jls_OxdT,jls_Oxp,jls_Oxd,jls_COp,jls_COd
     &     ,ijlt_OHvmr,ijlt_OHconc,ijlt_NOxd
     &     ,ijlt_HO2,ijlt_COp,ijlt_COd,ijlt_Oxd,ijlt_Oxp,ijlt_CH4d
     &     ,ijlt_OxpRO2,jls_ClOcon,jls_H2Ocon,jls_H2Ochem
      use GHGMOD, only: save_dQ_for_NINT
      use trdiag_com, only : ijlt_dQ,ijlt_dQoh,ijlt_dQo1d,ijlt_dQcl,
     &                       ijlt_dQsf3
#ifdef TRACERS_ACETONE
      USE TRDIAG_COM, only : jls_AcetP, jls_AcetD
      USE TRACER_COM, only : n_Acetone
#endif
      use OldTracer_mod, only: vol2mass, mass2vol
#ifdef TRACERS_dCO
      use tracers_dCO, only: R_17O_16O
      use tracers_dCO, only: R_18O_16O
#endif  /* TRACERS_dCO */
      USE TRACER_COM, only  : ntm_chem_beg, ntm_chem_end, ntm_chem,
#ifdef TRACERS_dCO
     &  n_d13Calke,n_d13CPAR,
     &  n_d17OPAN,n_d18OPAN,n_d13CPAN,
#endif  /* TRACERS_dCO */
     &  n_CH4,n_Paraffin,n_PAN,n_Isoprene,
     &  n_Terpenes,n_AlkylNit,n_Alkenes,n_N2O5,n_NOx,n_HO2NO2,
     &  n_isopp1g,n_isopp1a,n_isopp2g,n_isopp2a,n_apinp1g,
     &  n_apinp1a,n_apinp2g,n_apinp2a,n_Ox,n_HNO3,n_H2O2,n_CO,
     &  trm_col,NTM,n_N2O,n_ClOx,n_BrOx,n_HCl,n_HOCl,n_ClONO2,n_HBr,
     &  n_HOBr,n_BrONO2,n_CFC
#ifdef TRACERS_WATER
      use OldTracer_mod, only: tr_wd_type, nWater, tr_H2ObyCH4
      USE TRACER_COM, only: trmom_col 
#endif
#ifdef TRACERS_HETCHEM
      USE TRACER_COM, only: krate,n_N_d1,n_N_d2,n_N_d3
#endif
      USE TRCHEM_Shindell_COM, only: chemrate,photrate,cpd,
     &                   yCH3O2,yC2O3,yXO2,yXO2N,yRXPAR,yAldehyde,
     &                   yROR,nCH3O2,nC2O3,nXO2,nXO2N,nRXPAR,
     &                   nAldehyde,nROR,nn,dt2,dest,prod,
#ifdef TRACERS_dCO
     &                   ydCH317O2,ydCH318O2,yd13CH3O2,
     &                   ndCH317O2,ndCH318O2,nd13CH3O2,
     &                   ydC217O3,ydC218O3,yd13C2O3,
     &                   ndC217O3,ndC218O3,nd13C2O3,
     &                   yd17OROR,yd18OROR,yd13CROR,
     &                   nd17OROR,nd18OROR,nd13CROR,
     &                   yd17Oald,yd18Oald,yd13Cald,
     &                   nd17Oald,nd18Oald,nd13Cald,
     &                   yd13CXPAR,
     &                   nd13CXPAR,
#ifndef TRACERS_ACETONE /* NOT */
     &                   d17Oacetone,d18Oacetone,d13Cacetone,
#endif
#endif  /* TRACERS_dCO */
     &                   rr,nO1D,nOH,nNO,nHO2,nM,
     &                   nO3,nNO2,nNO3,prnrts,ijlprn,trchemname,
     &                   prnchg,y,nps,kps,nds,kds,n_rx,n_rj,
     &                   npnr,nnr,ndnr,kpnr,kdnr,nH2O,which_trop,
     &                   ks,kss,
     &                   minKG,rj,rrmono,rrbi,rrtri
     &                   ,SF3,ratioNs,ratioN2,rNO2frac,nO,nClO,nBrO
     &                   ,rNOfrac,rNOdenom,nOClO,nCl,nBr,OxlossbyH
     &                   ,nCl2,yCl2,SF2,nO2,MWabyMWw,yCl2O2
     &                   ,changeL
#ifndef TRACERS_ACETONE /* NOT */
     &                   ,Jacet,acetone
#endif
#if (defined TRACERS_ACETONE) && (defined TRACERS_dCO)
       use tracers_dCO, only: dacetone_fact
#endif
#ifdef TRACERS_AEROSOLS_SOA
       USE TRACERS_SOA, only: apartmolar,whichsoa,soa_apart,LM_soa
#endif  /* TRACERS_AEROSOLS_SOA */
      USE DIAG_COM, only : ftype,ntype
      USE ATM_COM, only : pmidl00
      use TRACER_COM, only: nn_CH4,  nn_N2O, nn_Ox,   nn_NOx, 
     &      nn_N2O5,   nn_HNO3,  nn_H2O2,  nn_CH3OOH,   nn_HCHO, 
     &      nn_HO2NO2, nn_CO,    nn_PAN,   nn_H2O17,             
     &      nn_Isoprene, nn_AlkylNit, nn_Alkenes, nn_Paraffin,   
     &      nn_Terpenes,
     &      nn_isopp1g,nn_isopp1a,nn_isopp2g,nn_isopp2a,         
     &      nn_apinp1g,nn_apinp1a,nn_apinp2g,nn_apinp2a,         
     &      nn_ClOx,   nn_BrOx,  nn_HCl,   nn_HOCl,   nn_ClONO2,  
     &      nn_HBr,    nn_HOBr,  nn_BrONO2,nn_CFC
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO
     &     ,nn_d13Calke,nn_d13CPAR
     &     ,nn_d17OPAN,nn_d18OPAN,nn_d13CPAN
     &     ,nn_dMe17OOH,nn_dMe18OOH,nn_d13MeOOH
     &     ,nn_dHCH17O,nn_dHCH18O,nn_dH13CHO
#endif  /* TRACERS_dCO */
     &     ,nn_dC17O,nn_dC18O,nn_d13CO
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
#ifdef TRACERS_ACETONE
     &     ,nn_Acetone
#endif

      USE DIAG_COM_RAD, only : j_h2och4
      use photolysis, only: zj
      use MODEL_COM, only: DTsrc
#ifdef SOLAR_ENERGETIC_PARTICLES
      use TRCHEM_Shindell_COM, only: IonProdRate
#endif /* SOLAR_ENERGETIC_PARTICLES */
c
      IMPLICIT NONE
c
C**** Local parameters and variables and arguments:
!@var changeL change due to chemistry (kg)
!@var I,J passed horizontal spatial indicies
!@var L,iter,Lz dummy loop variable
!@var maxL highest level with chemistry 
!@var maxT top of troposphere (or highest layer of chemistry in the
!@+ unlikely event that is lower. Note in that case,
!@var maxl highest level with chemistry, maxT top of troposphere
!@var qqqCH3O2,CH3O2loss,C2O3prod,
!@+   C2O3dest,XO2prod,XO2dest,XO2Nprod,XO2Ndest,RXPARprod,
!@+   RXPARdest,Aldehydeprod,Aldehydedest,RORprod,RORdest,total,
!@+   rnewval,dNOx,ratio,sumD,newD,ratioD,newP,ratioP,changeA,
!@+   sumP dummy temp variables
!@var sumN,sumC,sumH,sumB,sumO,sumA variables for O3 catalytic diags
!@var tempiter temp var for equilibrium calc iterations
!@var changeX temporary variable for equil calcs
!@var rMAbyM is airmass over air concentration
!@var dxbym2v is 1 over mass2volume
!@var sv_changeN2O N2O change without portion making N2 (for N cons)
!@var vClONO2, vBrONO2 temporary vars within N conservation
!@var changeH2O chemical change in H2O
!@var Oxcorr account for Ox change from within NOx partitioning
!@+   Not In Use.
!@var rNO3prod,rNO2prod,rNOprod to acct for dOx from NOx partitioning
!@var PRES local nominal pressure for regional Ox tracers
      INTEGER, INTENT(IN) :: maxL,I,J
      INTEGER :: L,iter,igas,maxT,Lz,it,n
      INTEGER :: J_0, J_1
      character(len=300) :: out_line
      logical            :: jay
      real*8, allocatable, dimension(:) :: rMAbyM,sv_changeN2O,
     & changeH2O,dQ,dQM,fraQ2,c2ml,conOH,conClO,conH2O,NprodOx_pos,
     & NprodOx_neg ! Oxcorr,
      real*8, allocatable, dimension(:) :: dQo1d,dQoh,dQcl,dQsf3
      real*8, dimension(LM) :: PRES ! for consistency with elsewhere, I keep this LM
!      real*8, parameter :: rCOplusO1D=1.d-9
      real*8, parameter :: chemtiny=1.d-12

      REAL*8 qqqCH3O2,CH3O2loss,
     & C2O3prod,C2O3dest,XO2prod,XO2dest,XO2Nprod,XO2Ndest,
     & RXPARprod,RXPARdest,Aldehydeprod,Aldehydedest,RORprod,RORdest,
     & total,rnewval,dNOx,ratio,sumD,newD,ratioD,newP,ratioP,
     & changeA,sumP,tempiter,sumC,sumN,sumH,sumB,sumO,sumA,
     & dxbym2v,changeX,vClONO2,vBrONO2,conc2mass,rNO3prod,rNO2prod,
     & rNOprod,changeAldehyde,rxnN2,rxnN3,rxnN4,NprodOx,NlossNOx,bytl,
     & diffCH3O2,tempAcet,prodCH3O2,dQMsum
      integer :: idx

#ifdef SOLAR_ENERGETIC_PARTICLES
      real*8 :: prodN4S,lossN4Sone,lossN4Stwo,lossN4Sthree,lossN4Ssum
      real*8 :: phaseOut
#endif /* SOLAR_ENERGETIC_PARTICLES */

      call getDomainBounds(grid, J_STRT    =J_0,  J_STOP    =J_1)

      jay = (J >= J_0 .and. J <= J_1) 

      allocate( rMAbyM(maxL) )
      allocate( sv_changeN2O(maxL) )
      allocate( changeH2O(maxL) )
      allocate( dQ(maxL) )
      if (save_dQ_for_NINT==1) then
        allocate( dQo1d(maxL) )
        allocate( dQoh(maxL) )
        allocate( dQcl(maxL) )
        allocate( dQsf3(maxL) )
      end if
      allocate( dQM(maxL) )
      allocate( fraQ2(maxL) )
      allocate( c2ml(maxL) )
      allocate( conOH(maxL) )
      allocate( conClO(maxL) )
      allocate( conH2O(maxL) )
      allocate( NprodOx_pos(maxL) )
      allocate( NprodOx_neg(maxL) ) 

      select case(which_trop)
      case(0); maxT=min(ltropo(I,J),maxL)
      case(1); maxT=min(ls1-1,maxL)
      case default; call stop_model('which_trop problem 1',255)
      end select 

      PRES(1:LM)=PMIDL00(1:LM)   !SIG(1:maxL)*(PSF-PTOP)+PTOP
      
      do L=1,maxL
        y(nCH3O2,L)   =    yCH3O2(L,I,J)
#ifdef TRACERS_dCO
        y(ndCH317O2,L)= ydCH317O2(L,I,J)
        y(ndCH318O2,L)= ydCH318O2(L,I,J)
        y(nd13CH3O2,L)= yd13CH3O2(L,I,J)
#endif  /* TRACERS_dCO */
        y(nC2O3,L)    =     yC2O3(L,I,J)
#ifdef TRACERS_dCO
        y(ndC217O3,L) =  ydC217O3(L,I,J)
        y(ndC218O3,L) =  ydC218O3(L,I,J)
        y(nd13C2O3,L) =  yd13C2O3(L,I,J)
#endif  /* TRACERS_dCO */
        y(nXO2,L)     =      yXO2(L,I,J)
        y(nXO2N,L)    =     yXO2N(L,I,J)
        y(nRXPAR,L)   =    yRXPAR(L,I,J)
#ifdef TRACERS_dCO
        y(nd13CXPAR,L)= yd13CXPAR(L,I,J)
#endif  /* TRACERS_dCO */
        y(nAldehyde,L)= yAldehyde(L,I,J)
#ifdef TRACERS_dCO
        y(nd17Oald,L) =  yd17Oald(L,I,J)
        y(nd18Oald,L) =  yd18Oald(L,I,J)
        y(nd13Cald,L) =  yd13Cald(L,I,J)
#endif  /* TRACERS_dCO */
        y(nROR,L)     =      yROR(L,I,J)
#ifdef TRACERS_dCO
        y(nd17OROR,L) =  yd17OROR(L,I,J)
        y(nd18OROR,L) =  yd18OROR(L,I,J)
        y(nd13CROR,L) =  yd13CROR(L,I,J)
#endif  /* TRACERS_dCO */
      end do
C
C Calculate reaction rates with present concentrations:
      call rates(maxL,I,J)
 
c chem1 call sample:
c (klist,l,numeL,nlist,ndlist,rate,change,multip)
c numeL=number of elements in reaction list nlist (1 or 2)
c change=dest or prod array
c multip=1(prod) or -1(dest)

c chemical destruction:
      call chem1(kdnr,maxL,2,n_rx,nn,ndnr,chemrate,dest,-1)
c chemical production:
      call chem1(kpnr,maxL,2,n_rx,nnr,npnr,chemrate,prod,1)
c photolytic destruction:
      call chem1(kds,maxL,1,n_rj,ks,nds,photrate,dest,-1)
c photolytic production:
      call chem1(kps,maxL,2,n_rj,kss,nps,photrate,prod,1)

c Add additional Cl from CFC photolysis + background :
      do L=1,maxL
        prod(nn_ClOx,L)=prod(nn_ClOx,L)
     &    +0.33d0*photrate(rj%CFC__Cl_O2,L)
     &    +7.5d-3*photrate(rj%N2O__M_O1D,L)
        prod(nn_BrOx,L)=prod(nn_BrOx,L)
     &    +5.55d-4*photrate(rj%CFC__Cl_O2,L)
     &    +5.2d-6*photrate(rj%N2O__M_O1D,L)
      end do

#ifdef SOLAR_ENERGETIC_PARTICLES
c Add effects of Solar Energetic Particles on nitrogen species.
c Each ion pair prodution makes 1.25 N, assumed to be 0.55 base state
c N(4S) and 0.7 N(2D). Based on CMIP6 recommendations in Matthes
c et al., GMD, 2017. We are only using protons case.
      do L=1,maxL
        if(tl(L).eq.0.d0) call stop_model('tl(L)=0',255)
        if(pres(L) > 1.d0)then
          phaseOut=0.d0
        elseif(pres(L) < 0.1d0)then
          phaseOut=1.d0
        else
          phaseOut=abs(log10(pres(L)))
        endif
        bytl=1.d0/tl(L)
        prodN4S=0.55d0*IonProdRate(L)
        lossN4Sone=y(nO2,L)*1.5d-11*exp(-3600.d0*bytl)   ! N(4S) + O2  -> NO + O
        lossN4Stwo=y(nNO,L)*2.1d-11*exp(100.d0*bytl)     ! N(4S) + NO  -> N2 + O
        lossN4Sthree=y(nNO2,L)*5.8d-12*exp(220.d0*bytl)  ! N(4S) + NO2 -> N2O + O
        lossN4Ssum=lossN4Sone+lossN4Stwo+lossN4Sthree
        if(lossN4Ssum.eq.0.d0) call stop_model('lossN4Ssum=0',255)
        prod(nn_NOx,L)=prod(nn_NOx,L)+ phaseOut*(
     &      prodN4S*lossN4Sone/lossN4Ssum + 0.7d0*IonProdRate(L) )
        dest(nn_NOx,L)=dest(nn_NOx,L)+ min( phaseOut*
     &      prodN4S*(lossN4Stwo+lossN4Sthree)/lossN4Ssum,
     &      0.1d0*y(nNO,L), 0.1d0*y(nNO2,L) )
        prod(nn_N2O,L)=prod(nn_N2O,L)+ phaseOut*
     &      prodN4S*lossN4Sthree/lossN4Ssum
        prod(nn_Ox,L)=prod(nn_Ox,L)+phaseOut*1.25d0*IonProdRate(L)
      end do
#endif  /* SOLAR_ENERGETIC_PARTICLES */

c Oxidation of Isoprene and Alkenes produces less than one
c HCHO, Alkenes, and CO per rxn, correct here following Houweling:
      do L=1,maxL
        prod(nn_CO,L)=prod(nn_CO,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_CO,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_ACETONE
     &    +photrate(rj%Acetone__CH3O2_CH3O2,L)
#endif
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
#if defined(TRACERS_dCO)
        prod(nn_dC17O,L)=prod(nn_dC17O,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_dC17O,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
        prod(nn_dC18O,L)=prod(nn_dC18O,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_dC18O,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
        prod(nn_d13CO,L)=prod(nn_d13CO,L)
     &    -0.63d0*chemrate(rrbi%d13Calke_O3__dH13CHO_d13CO,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
#elif defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO_bin_reprod
        prod(nn_dC17O,L)=prod(nn_dC17O,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_dC17O,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
        prod(nn_dC18O,L)=prod(nn_dC18O,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_dC18O,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
        prod(nn_d13CO,L)=prod(nn_d13CO,L)
     &    -0.63d0*chemrate(rrbi%Alkenes_O3__HCHO_d13CO,L)
     &    +0.36d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    +0.36d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif /* TRACERS_TERP */
#endif  /* TRACERS_dCO_bin_reprod */
#endif  /* TRACERS_dCO || TRACERS_dCOlite */

        prod(nn_HCHO,L)=prod(nn_HCHO,L)
     &    -0.36d0*chemrate(rrbi%Alkenes_O3__HCHO_CO,L)
#ifdef TRACERS_dCO
        prod(nn_dHCH17O,L)=prod(nn_dHCH17O,L)
     &    -0.36d0*chemrate(rrbi%Alkenes_O3__dHCH17O_CO,L)
        prod(nn_dHCH18O,L)=prod(nn_dHCH18O,L)
     &    -0.36d0*chemrate(rrbi%Alkenes_O3__dHCH18O_CO,L)
        prod(nn_dH13CHO,L)=prod(nn_dH13CHO,L)
     &    -0.36d0*chemrate(rrbi%d13Calke_O3__dH13CHO_d13CO,L)
#endif  /* TRACERS_dCO */

        prod(nn_HCHO,L)=prod(nn_HCHO,L)
     &    -0.39d0*chemrate(rrbi%Isoprene_OH__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.39d0*chemrate(rrbi%Terpenes_OH__HCHO_Alkenes,L)
#endif  /* TRACERS_TERP */
#ifdef TRACERS_dCO
        prod(nn_dHCH17O,L)=prod(nn_dHCH17O,L)
     &    -0.39d0*chemrate(rrbi%Isoprene_OH__dHCH17O_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.39d0*chemrate(rrbi%Terpenes_OH__dHCH17O_Alkenes,L)
#endif  /* TRACERS_TERP */
        prod(nn_dHCH18O,L)=prod(nn_dHCH18O,L)
     &    -0.39d0*chemrate(rrbi%Isoprene_OH__dHCH18O_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.39d0*chemrate(rrbi%Terpenes_OH__dHCH18O_Alkenes,L)
#endif  /* TRACERS_TERP */
        prod(nn_dH13CHO,L)=prod(nn_dH13CHO,L)
     &    -0.39d0*chemrate(rrbi%Isoprene_OH__dH13CHO_d13Calke,L)
#ifdef TRACERS_TERP
     &    -0.39d0*chemrate(rrbi%Terpenes_OH__dH13CHO_d13Calke,L)
#endif  /* TRACERS_TERP */
#endif  /* TRACERS_dCO */

        prod(nn_Alkenes,L)=prod(nn_Alkenes,L)
     &    -0.42d0*chemrate(rrbi%Isoprene_OH__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.42d0*chemrate(rrbi%Terpenes_OH__HCHO_Alkenes,L)
#endif  /* TRACERS_TERP */
#ifdef TRACERS_dCO
        prod(nn_d13Calke,L)=prod(nn_d13Calke,L)
     &    -0.42d0*chemrate(rrbi%Isoprene_OH__dH13CHO_d13Calke,L)
#ifdef TRACERS_TERP
     &    -0.42d0*chemrate(rrbi%Terpenes_OH__dH13CHO_d13Calke,L)
#endif  /* TRACERS_TERP */
#endif  /* TRACERS_dCO */

        prod(nn_HCHO,L)=prod(nn_HCHO,L)
     &    -0.10d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.10d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif  /* TRACERS_TERP */
#ifdef TRACERS_dCO
        prod(nn_dHCH17O,L)=prod(nn_dHCH17O,L)
     &    -0.10d0*chemrate(rrbi%Isoprene_O3__dHCH17O_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.10d0*chemrate(rrbi%Terpenes_O3__dHCH17O_Alkenes,L)
#endif  /* TRACERS_TERP */
        prod(nn_dHCH18O,L)=prod(nn_dHCH18O,L)
     &    -0.10d0*chemrate(rrbi%Isoprene_O3__dHCH18O_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.10d0*chemrate(rrbi%Terpenes_O3__dHCH18O_Alkenes,L)
#endif  /* TRACERS_TERP */
        prod(nn_dH13CHO,L)=prod(nn_dH13CHO,L)
     &    -0.10d0*chemrate(rrbi%Isoprene_O3__dH13CHO_d13Calke,L)
#ifdef TRACERS_TERP
     &    -0.10d0*chemrate(rrbi%Terpenes_O3__dH13CHO_d13Calke,L)
#endif  /* TRACERS_TERP */
#endif  /* TRACERS_dCO */

        prod(nn_Alkenes,L)=prod(nn_Alkenes,L)
     &    -0.45d0*chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L)
#ifdef TRACERS_TERP
     &    -0.45d0*chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif  /* TRACERS_TERP */
#ifdef TRACERS_dCO
        prod(nn_d13Calke,L)=prod(nn_d13Calke,L)
     &    -0.45d0*chemrate(rrbi%Isoprene_O3__dH13CHO_d13Calke,L)
#ifdef TRACERS_TERP
     &    -0.45d0*chemrate(rrbi%Terpenes_O3__dH13CHO_d13Calke,L)
#endif  /* TRACERS_TERP */
#endif  /* TRACERS_dCO */
#ifdef TRACERS_HETCHEM
        dest(nn_HNO3,l)=dest(nn_HNO3,l) -
     &       krate(l,1,1)*y(nn_HNO3,l)*dt2
#endif

c       Add parrafin prod term via isoprene and terpenes oxidation
        prod(nn_Paraffin,L)=prod(nn_Paraffin,L)
     &    +0.63d0*y(nn_Isoprene,L)
     &      *(rr(rrbi%Isoprene_OH__HCHO_Alkenes,L)*y(nOH,L)
     &       +rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)*y(nO3,L)
     &       )*dt2
#ifdef TRACERS_TERP
     &    +5.0d0*0.63d0*y(nn_Terpenes,L)
     &      *(rr(rrbi%Terpenes_OH__HCHO_Alkenes,L)*y(nOH,L)
     &       +rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &       )*dt2
#endif  /* TRACERS_TERP */

       ! Additional CH3OOH from C2O3+HO2:
       prod(nn_CH3OOH,L)=prod(nn_CH3OOH,L)
     &   +0.21d0*chemrate(rrbi%C2O3_HO2__HCHO_HO2,L)

#ifdef TRACERS_ACETONE
       ! The 0.35 factor below is an estimate based on the sub-species
       ! of Paraffin emissions (anthropogenic & biomass burning) and
       ! their yields as reported in literature. Arrived at by Kostas,
       ! Alexandra R. and Greg F. in emails and meetings, Summer 2021.
       ! The division by 3 Kostas said is "for Par->Acet # of carbons":
       prod(nn_Acetone,L)=prod(nn_Acetone,L)
     &   +y(nn_Paraffin,L)*0.35d0/3.d0*
     &    rr(rrbi%Paraffin_OH__HO2_M,L)*y(nOH,L)*dt2
#ifdef TRACERS_TERP
     &   +y(nn_Terpenes,L)
     &     *(0.12d0*rr(rrbi%Terpenes_OH__HCHO_Alkenes,L)*y(nOH,L)
     &     +0.12d0*rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &     )*dt2
#endif /* TRACERS_TERP */
#endif /* TRACERS_ACETONE */

#ifdef TRACERS_dCO
c       Add parrafin prod term via isoprene and terpenes oxidation
! divide by 3, to distribute carbon in HCHO, Alkenes, and Paraffin
        prod(nn_d13CPAR,L)=prod(nn_d13CPAR,L)
     &    +0.63d0*y(nn_Isoprene,L)
     &      *(rr(rrbi%Isoprene_OH__HCHO_Alkenes,L)*y(nOH,L)
     &       +rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)*y(nO3,L)
     &       )*dt2/3.d0
     &    +0.63d0*y(nn_Isoprene,L)
     &      *(rr(rrbi%Isoprene_OH__dH13CHO_d13Calke,L)*y(nOH,L)
     &       +rr(rrbi%Isoprene_O3__dH13CHO_d13Calke,L)*y(nO3,L)
     &       )*dt2/3.d0
     &    +0.63d0*y(nn_Isoprene,L)
     &      *(rr(rrbi%Isoprene_OH__dH13CHO_d13Calke,L)*y(nOH,L)
     &       +rr(rrbi%Isoprene_O3__dH13CHO_d13Calke,L)*y(nO3,L)
     &       )*dt2/3.d0
#ifdef TRACERS_TERP
     &    +5.0d0*0.63d0*y(nn_Terpenes,L)
     &      *(rr(rrbi%Terpenes_OH__HCHO_Alkenes,L)*y(nOH,L)
     &       +rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &       )*dt2/3.d0
     &    +5.0d0*0.63d0*y(nn_Terpenes,L)
     &      *(rr(rrbi%Terpenes_OH__dH13CHO_d13Calke,L)*y(nOH,L)
     &       +rr(rrbi%Terpenes_O3__dH13CHO_d13Calke,L)*y(nO3,L)
     &       )*dt2/3.d0
     &    +5.0d0*0.63d0*y(nn_Terpenes,L)
     &      *(rr(rrbi%Terpenes_OH__dH13CHO_d13Calke,L)*y(nOH,L)
     &       +rr(rrbi%Terpenes_O3__dH13CHO_d13Calke,L)*y(nO3,L)
     &       )*dt2/3.d0
#endif  /* TRACERS_TERP */
#endif  /* TRACERS_dCO */

      end do

#ifdef TRACERS_AEROSOLS_SOA
      call soa_apart ! calculate current apartmolar factors
#ifdef SOA_DIAGS
     &              (I,J)
#endif  /* SOA_DIAGS */
      do L=1,min(LM_soa,maxL)
        prod(nn_isopp1g,L)=prod(nn_isopp1g,L)+
     &    apartmolar(L,whichsoa(n_isopp1a))*
     &    (chemrate(rrbi%Isoprene_OH__HCHO_Alkenes,L)
     &    +chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L))
        prod(nn_isopp2g,L)=prod(nn_isopp2g,L)+
     &    apartmolar(L,whichsoa(n_isopp2a))*
     &    (chemrate(rrbi%Isoprene_OH__HCHO_Alkenes,L)
     &    +chemrate(rrbi%Isoprene_O3__HCHO_Alkenes,L))
#ifdef TRACERS_TERP
        prod(nn_apinp1g,L)=prod(nn_apinp1g,L)+
     &    apartmolar(L,whichsoa(n_apinp1a))*
     &    chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
        prod(nn_apinp2g,L)=prod(nn_apinp2g,L)+
     &    apartmolar(L,whichsoa(n_apinp2a))*
     &    chemrate(rrbi%Terpenes_O3__HCHO_Alkenes,L)
#endif  /* TRACERS_TERP */
      end do
#endif  /* TRACERS_AEROSOLS_SOA */

      do L=1,maxL
c Set CH3O2 values (concentration = production/specific loss):
        iter=1
        qqqCH3O2=(rr(rrbi%O1D_CH4__OH_CH3O2,L)*y(nO1D,L)
     &      +rr(rrbi%CH4_OH__H2O_CH3O2,L)*y(nOH,L))
     &    *y(nn_CH4,L)
     &    +rr(rrbi%CH3OOH_OH__CH3O2_H2O,L)*y(nn_CH3OOH,L)*y(nOH,L)
     &    +rr(rrbi%Cl_CH4__HCl_CH3O2,l)*y(nCl,L)
#ifdef TRACERS_ACETONE
        tempAcet=( zj(L,rj%Acetone__CH3O2_C2O3) +
     &  zj(L,rj%Acetone__CH3O2_CH3O2) )*y(nn_Acetone,L)
#else
        tempAcet=2.d0*Jacet(L)*acetone(L)
#endif
        prodCH3O2=qqqCH3O2+tempAcet
        tempiter=rr(rrbi%CH3O2_NO__HCHO_NO2,L)*y(nNO,L)
     &    +rr(rrbi%CH3O2_HO2__CH3OOH_O2,L)*y(nHO2,L)
     &    +rr(rrbi%ClO_CH3O2__Cl_HCHO,l)*y(nClO,l)
        do while(iter <= 7)
          CH3O2loss=tempiter
     &      +rr(rrbi%CH3O2_CH3O2__HCHO_HCHO,L)*y(nCH3O2,L)
          if(CH3O2loss > 1.d-7)then
            y(nCH3O2,L)=prodCH3O2/CH3O2loss
          else
            y(nCH3O2,L)=1.d-5
          end if
          iter=iter+1
        end do

c Conserve carbon wrt CH3O2 changes:
        diffCH3O2=y(nCH3O2,L)-yCH3O2(L,I,J)
        if(diffCH3O2 > tempAcet)then
c         reduce non-acetone source gases (CH4 and CH3OOH):
          dest(nn_CH4,L)=dest(nn_CH4,L)-(diffCH3O2-tempAcet)
     &      *(qqqCH3O2-rr(rrbi%CH3OOH_OH__CH3O2_H2O,L)*y(nn_CH3OOH,L)
     &      *y(nOH,L))/qqqCH3O2
          dest(nn_CH3OOH,L)=dest(nn_CH3OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%CH3OOH_OH__CH3O2_H2O,L)*y(nn_CH3OOH,L)
     &      *y(nOH,L))/qqqCH3O2
        else if(diffCH3O2 < tempAcet)then
c         increase non-acetone product gases:
          prod(nn_HCHO,L)=prod(nn_HCHO,L)-(diffCH3O2-tempAcet)
     &      *(CH3O2loss-rr(rrbi%CH3O2_HO2__CH3OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
          prod(nn_CH3OOH,L)=prod(nn_CH3OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%CH3O2_HO2__CH3OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
        end if
        yCH3O2(L,I,J)=y(nCH3O2,L)

#ifdef TRACERS_dCO
c Set dCH317O2 values (concentration = production/specific loss):
! ok to overwrite here qqqCH3O2,prodCH3O2,diffCH3O2,CH3O2loss,tempAcet
        iter=1
        qqqCH3O2=(rr(rrbi%O1D_CH4__OH_dCH317O2,L)*y(nO1D,L)
     &      +rr(rrbi%CH4_OH__H2O_dCH317O2,L)*y(nOH,L))
     &    *y(nn_CH4,L)
     &    +rr(rrbi%dMe17OOH_OH__dCH317O2_H2O,L)*y(nn_dMe17OOH,L)
     &      *y(nOH,L)
     &    +rr(rrbi%Cl_CH4__HCl_dCH317O2,l)*y(nCl,L)
#ifdef TRACERS_ACETONE
        tempAcet=( zj(L,rj%Acetone__CH3O2_C2O3) +
     &  zj(L,rj%Acetone__CH3O2_CH3O2) )*y(nn_Acetone,L)*dacetone_fact
#else
        tempAcet=2.d0*Jacet(L)*d17Oacetone(L)
#endif
        prodCH3O2=qqqCH3O2+tempAcet
        tempiter=rr(rrbi%dCH317O2_NO__dHCH17O_NO2,L)*y(nNO,L)
     &    +rr(rrbi%dCH317O2_HO2__dMe17OOH_O2,L)*y(nHO2,L)
     &    +rr(rrbi%ClO_dCH317O2__Cl_dHCH17O,l)*y(nClO,l)
        do while(iter <= 7)
          CH3O2loss=tempiter
     &      +0.5d0*(rr(rrbi%dCH317O2_CH3O2__dHCH17O_HCHO,L)
     &        +rr(rrbi%CH3O2_dCH317O2__HCHO_dHCH17O,L))*y(ndCH317O2,L)
          if(CH3O2loss > 1.d-7)then
            y(ndCH317O2,L)=prodCH3O2/CH3O2loss
          else
            y(ndCH317O2,L)=1.d-5
          end if
          iter=iter+1
        end do

c Conserve carbon wrt dCH317O2 changes:
        diffCH3O2=y(ndCH317O2,L)-ydCH317O2(L,I,J)
        if(diffCH3O2 > tempAcet)then
c         reduce non-acetone source gases (CH4 and dMe17OOH):
!          dest(nn_CH4,L)=dest(nn_CH4,L)-(diffCH3O2-tempAcet)
!     &      *(qqqCH3O2-rr(rrbi%dMe17OOH_OH__dCH317O2_H2O,L)*y(nn_dMe17OOH,L)
!     &      *y(nOH,L))/qqqCH3O2
          dest(nn_dMe17OOH,L)=dest(nn_dMe17OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%dMe17OOH_OH__dCH317O2_H2O,L)*y(nn_dMe17OOH,L)
     &      *y(nOH,L))/qqqCH3O2
        else if(diffCH3O2 < tempAcet)then
c         increase non-acetone product gases:
          prod(nn_dHCH17O,L)=prod(nn_dHCH17O,L)-(diffCH3O2-tempAcet)
     &      *(CH3O2loss-rr(rrbi%dCH317O2_HO2__dMe17OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
          prod(nn_dMe17OOH,L)=prod(nn_dMe17OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%dCH317O2_HO2__dMe17OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
        end if
        ydCH317O2(L,I,J)=y(ndCH317O2,L)

c Set dCH318O2 values (concentration = production/specific loss):
! ok to overwrite here qqqCH3O2,prodCH3O2,diffCH3O2,CH3O2loss
        iter=1
        qqqCH3O2=(rr(rrbi%O1D_CH4__OH_dCH318O2,L)*y(nO1D,L)
     &      +rr(rrbi%CH4_OH__H2O_dCH318O2,L)*y(nOH,L))
     &    *y(nn_CH4,L)
     &    +rr(rrbi%dMe18OOH_OH__dCH318O2_H2O,L)*y(nn_dMe18OOH,L)
     &      *y(nOH,L)
     &    +rr(rrbi%Cl_CH4__HCl_dCH318O2,l)*y(nCl,L)
#ifdef TRACERS_ACETONE
        tempAcet=( zj(L,rj%Acetone__CH3O2_C2O3) +
     &  zj(L,rj%Acetone__CH3O2_CH3O2) )*y(nn_Acetone,L)*dacetone_fact
#else
        tempAcet=2.d0*Jacet(L)*d18Oacetone(L)
#endif
        prodCH3O2=qqqCH3O2+tempAcet
        tempiter=rr(rrbi%dCH318O2_NO__dHCH18O_NO2,L)*y(nNO,L)
     &    +rr(rrbi%dCH318O2_HO2__dMe18OOH_O2,L)*y(nHO2,L)
     &    +rr(rrbi%ClO_dCH318O2__Cl_dHCH18O,l)*y(nClO,l)
        do while(iter <= 7)
          CH3O2loss=tempiter
     &      +0.5d0*(rr(rrbi%dCH318O2_CH3O2__dHCH18O_HCHO,L)
     &        +rr(rrbi%CH3O2_dCH318O2__HCHO_dHCH18O,L))*y(ndCH318O2,L)
          if(CH3O2loss > 1.d-7)then
            y(ndCH318O2,L)=prodCH3O2/CH3O2loss
          else
            y(ndCH318O2,L)=1.d-5
          end if
          iter=iter+1
        end do

c Conserve carbon wrt dCH318O2 changes:
        diffCH3O2=y(ndCH318O2,L)-ydCH318O2(L,I,J)
        if(diffCH3O2 > tempAcet)then
c         reduce non-acetone source gases (CH4 and dMe18OOH):
!          dest(nn_CH4,L)=dest(nn_CH4,L)-(diffCH3O2-tempAcet)
!     &      *(qqqCH3O2-rr(rrbi%dMe18OOH_OH__dCH318O2_H2O,L)*y(nn_dMe18OOH,L)
!     &      *y(nOH,L))/qqqCH3O2
          dest(nn_dMe18OOH,L)=dest(nn_dMe18OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%dMe18OOH_OH__dCH318O2_H2O,L)*y(nn_dMe18OOH,L)
     &      *y(nOH,L))/qqqCH3O2
        else if(diffCH3O2 < tempAcet)then
c         increase non-acetone product gases:
          prod(nn_dHCH18O,L)=prod(nn_dHCH18O,L)-(diffCH3O2-tempAcet)
     &      *(CH3O2loss-rr(rrbi%dCH318O2_HO2__dMe18OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
          prod(nn_dMe18OOH,L)=prod(nn_dMe18OOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%dCH318O2_HO2__dMe18OOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
        end if
        ydCH318O2(L,I,J)=y(ndCH318O2,L)

c Set d13CH3O2 values (concentration = production/specific loss):
! ok to overwrite here qqqCH3O2,prodCH3O2,diffCH3O2,CH3O2loss
        iter=1
        qqqCH3O2=(rr(rrbi%O1D_CH4__OH_d13CH3O2,L)*y(nO1D,L)
     &      +rr(rrbi%CH4_OH__H2O_d13CH3O2,L)*y(nOH,L))
     &    *y(nn_CH4,L)
     &    +rr(rrbi%d13MeOOH_OH__d13CH3O2_H2O,L)*y(nn_d13MeOOH,L)
     &      *y(nOH,L)
     &    +rr(rrbi%Cl_CH4__HCl_d13CH3O2,l)*y(nCl,L)
#ifdef TRACERS_ACETONE
        tempAcet=( zj(L,rj%Acetone__CH3O2_C2O3) +
     &  zj(L,rj%Acetone__CH3O2_CH3O2) )*y(nn_Acetone,L)*dacetone_fact
#else
        tempAcet=2.d0*Jacet(L)*d13Cacetone(L)
#endif
        prodCH3O2=qqqCH3O2+tempAcet
        tempiter=rr(rrbi%d13CH3O2_NO__dH13CHO_NO2,L)*y(nNO,L)
     &    +rr(rrbi%d13CH3O2_HO2__d13MeOOH_O2,L)*y(nHO2,L)
     &    +rr(rrbi%ClO_d13CH3O2__Cl_dH13CHO,l)*y(nClO,l)
        do while(iter <= 7)
          CH3O2loss=tempiter
     &      +0.5d0*(rr(rrbi%d13CH3O2_CH3O2__dH13CHO_HCHO,L)
     &        +rr(rrbi%CH3O2_d13CH3O2__HCHO_dH13CHO,L))*y(nd13CH3O2,L)
          if(CH3O2loss > 1.d-7)then
            y(nd13CH3O2,L)=prodCH3O2/CH3O2loss
          else
            y(nd13CH3O2,L)=1.d-5
          end if
          iter=iter+1
        end do

c Conserve carbon wrt d13CH3O2 changes:
        diffCH3O2=y(nd13CH3O2,L)-yd13CH3O2(L,I,J)
        if(diffCH3O2 > tempAcet)then
c         reduce non-acetone source gases (CH4 and d13MeOOH):
!          dest(nn_CH4,L)=dest(nn_CH4,L)-(diffCH3O2-tempAcet)
!     &      *(qqqCH3O2-rr(rrbi%d13MeOOH_OH__d13CH3O2_H2O,L)*y(nn_d13MeOOH,L)
!     &      *y(nOH,L))/qqqCH3O2
          dest(nn_d13MeOOH,L)=dest(nn_d13MeOOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%d13MeOOH_OH__d13CH3O2_H2O,L)*y(nn_d13MeOOH,L)
     &      *y(nOH,L))/qqqCH3O2
        else if(diffCH3O2 < tempAcet)then
c         increase non-acetone product gases:
          prod(nn_dH13CHO,L)=prod(nn_dH13CHO,L)-(diffCH3O2-tempAcet)
     &      *(CH3O2loss-rr(rrbi%d13CH3O2_HO2__d13MeOOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
          prod(nn_d13MeOOH,L)=prod(nn_d13MeOOH,L)-(diffCH3O2-tempAcet)
     &      *(rr(rrbi%d13CH3O2_HO2__d13MeOOH_O2,L)*y(nHO2,L))
     &      /CH3O2loss
        end if
        yd13CH3O2(L,I,J)=y(nd13CH3O2,L)
#endif  /* TRACERS_dCO */

c       Set value for C2O3:
        iter=1
        C2O3prod=rr(rrbi%Aldehyde_OH__C2O3_M,L)*yAldehyde(L,I,J)
     &      *y(nOH,L)
     &    +(rr(rrbi%PAN_M__C2O3_NO2,L)*y(nM,L)
     &      +zj(L,rj%PAN__C2O3_NO2))*y(nn_PAN,L)
     &    +0.15d0*rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)
     &      *y(nO3,L)*y(nn_Isoprene,L)
#ifdef TRACERS_TERP
     &    +0.15d0*rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &      *y(nn_Terpenes,L)
#endif  /* TRACERS_TERP */
        tempiter=rr(rrbi%C2O3_NO__HCHO_NO2,L)*y(nNO,L)
     &    +rr(rrtri%C2O3_NO2__PAN_M,L)*y(nNO2,L)
     &    +rr(rrbi%C2O3_HO2__HCHO_HO2,L)*y(nHO2,L)
        do while (iter <= 7)
          C2O3dest=tempiter
     &      +rr(rrbi%C2O3_C2O3__HCHO_HCHO,L)*y(nC2O3,L)
          if(C2O3dest > 1.d-7)then
            y(nC2O3,L)=(C2O3prod/C2O3dest)
          else
            y(nC2O3,L)=1.d0
          endif
          iter=iter+1
        end do
        yC2O3(L,I,J)=y(nC2O3,L)
#ifdef TRACERS_dCO
! ok to overwrite C2O3prod and C2O3dest
c       Set value for dC217O3:
        iter=1
        C2O3prod=rr(rrbi%d17Oald_OH__dC217O3_M,L)*yd17Oald(L,I,J)
     &      *y(nOH,L)
     &    +(rr(rrbi%d17OPAN_M__dC217O3_NO2,L)*y(nM,L)
     &      +zj(L,rj%d17OPAN__dC217O3_NO2))*y(nn_d17OPAN,L)
     &    +0.15d0*rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)
     &      *y(nO3,L)*y(nn_Isoprene,L)
#ifdef TRACERS_TERP
     &    +0.15d0*rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &      *y(nn_Terpenes,L)
#endif  /* TRACERS_TERP */
        tempiter=rr(rrbi%dC217O3_NO__dHCH17O_NO2,L)*y(nNO,L)
     &    +rr(rrtri%dC217O3_NO2__d17OPAN_M,L)*y(nNO2,L)
     &    +rr(rrbi%dC217O3_HO2__dHCH17O_HO2,L)*y(nHO2,L)
        do while (iter <= 7)
          C2O3dest=tempiter
     &      +rr(rrbi%dC217O3_dC217O3__dHCH17O_dHCH17O,L)*y(ndC217O3,L)
          if(C2O3dest > 1.d-7)then
            y(ndC217O3,L)=(C2O3prod/C2O3dest)
          else
            y(ndC217O3,L)=1.d0
          endif
          iter=iter+1
        end do
        ydC217O3(L,I,J)=y(ndC217O3,L)
c       Set value for dC218O3:
        iter=1
        C2O3prod=rr(rrbi%d18Oald_OH__dC218O3_M,L)*yd18Oald(L,I,J)
     &      *y(nOH,L)
     &    +(rr(rrbi%d18OPAN_M__dC218O3_NO2,L)*y(nM,L)
     &      +zj(L,rj%d18OPAN__dC218O3_NO2))*y(nn_d18OPAN,L)
     &    +0.15d0*rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)
     &      *y(nO3,L)*y(nn_Isoprene,L)
#ifdef TRACERS_TERP
     &    +0.15d0*rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &      *y(nn_Terpenes,L)
#endif  /* TRACERS_TERP */
        tempiter=rr(rrbi%dC218O3_NO__dHCH18O_NO2,L)*y(nNO,L)
     &    +rr(rrtri%dC218O3_NO2__d18OPAN_M,L)*y(nNO2,L)
     &    +rr(rrbi%dC218O3_HO2__dHCH18O_HO2,L)*y(nHO2,L)
        do while (iter <= 7)
          C2O3dest=tempiter
     &      +rr(rrbi%dC218O3_dC218O3__dHCH18O_dHCH18O,L)*y(ndC218O3,L)
          if(C2O3dest > 1.d-7)then
            y(ndC218O3,L)=(C2O3prod/C2O3dest)
          else
            y(ndC218O3,L)=1.d0
          endif
          iter=iter+1
        end do
        ydC218O3(L,I,J)=y(ndC218O3,L)
c       Set value for d13C2O3:
        iter=1
        C2O3prod=rr(rrbi%d13Cald_OH__d13C2O3_M,L)*yd13Cald(L,I,J)
     &      *y(nOH,L)
     &    +(rr(rrbi%d13CPAN_M__d13C2O3_NO2,L)*y(nM,L)
     &      +zj(L,rj%d13CPAN__d13C2O3_NO2))*y(nn_d13CPAN,L)
     &    +0.15d0*rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)
     &      *y(nO3,L)*y(nn_Isoprene,L)
#ifdef TRACERS_TERP
     &    +0.15d0*rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nO3,L)
     &      *y(nn_Terpenes,L)
#endif  /* TRACERS_TERP */
        tempiter=rr(rrbi%d13C2O3_NO__dH13CHO_NO2,L)*y(nNO,L)
     &    +rr(rrtri%d13C2O3_NO2__d13CPAN_M,L)*y(nNO2,L)
     &    +rr(rrbi%d13C2O3_HO2__dH13CHO_HO2,L)*y(nHO2,L)
        do while (iter <= 7)
          C2O3dest=tempiter
     &      +rr(rrbi%d13C2O3_d13C2O3__dH13CHO_dH13CHO,L)*y(nd13C2O3,L)
          if(C2O3dest > 1.d-7)then
            y(nd13C2O3,L)=(C2O3prod/C2O3dest)
          else
            y(nd13C2O3,L)=1.d0
          endif
          iter=iter+1
        end do
        yd13C2O3(L,I,J)=y(nd13C2O3,L)
#endif  /* TRACERS_dCO */

c       Set value for XO2:
! remember to update voc2nox if you update any of the following XO2 loss reactions
        iter=1
        XO2prod=zj(L,rj%Aldehyde__HCHO_CO)*yAldehyde(L,I,J)
     &    +y(nC2O3,L)*(rr(rrbi%C2O3_NO__HCHO_NO2,L)*y(nNO2,L)
     &      +rr(rrbi%C2O3_C2O3__HCHO_HCHO,L)*y(nC2O3,L)*2.d0
     &      +rr(rrbi%C2O3_HO2__HCHO_HO2,L)*y(nHO2,L))
     &    +rr(rrbi%Alkenes_NO3__HCHO_NO2,L)*y(nNO3,L)*y(nn_Alkenes,L)
     &      *0.91d0
     &    +rr(rrbi%ROR_M__Aldehyde_HO2,L)*y(nM,L)*yROR(L,I,J)*0.96d0
     &    +y(nOH,L)*(rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *0.87d0
     &    +rr(rrbi%Alkenes_OH__HCHO_HO2,L)*y(nn_Alkenes,L)
     &    +rr(rrbi%Isoprene_OH__HCHO_Alkenes,L)*y(nn_Isoprene,L)*0.85d0
#ifdef TRACERS_TERP
     &    +rr(rrbi%Terpenes_OH__HCHO_Alkenes,L)*y(nn_Terpenes,L)*0.85d0
#endif  /* TRACERS_TERP */
     &    +rr(rrbi%AlkylNit_OH__NO2_XO2,L)*y(nn_AlkylNit,L))
     &    +y(nO3,L)*(rr(rrbi%Alkenes_O3__HCHO_CO,L)*y(nn_Alkenes,L)
     &      *0.29d0
     &    +rr(rrbi%Isoprene_O3__HCHO_Alkenes,L)*y(nn_Isoprene,L)*0.18d0
#ifdef TRACERS_TERP
     &    +rr(rrbi%Terpenes_O3__HCHO_Alkenes,L)*y(nn_Terpenes,L)*0.18d0
#endif  /* TRACERS_TERP */
     &           )
        tempiter=rr(rrbi%XO2_NO__NO2_M,L)*y(nNO,L)
     &    +rr(rrbi%XO2_HO2__CH3OOH_O2,L)*y(nHO2,L)
        do while (iter <= 7)
          XO2dest=tempiter
     &      +rr(rrbi%XO2_XO2__M_M,L)*y(nXO2,L)
          if(XO2dest > 1.d-7.and.
     &       zj(L,rj%Aldehyde__HCHO_CO) > 1.d-6)then
            y(nXO2,L)=(XO2prod/XO2dest)
          else
            y(nXO2,L)=1.d0
          end if
          iter=iter+1
        end do
        yXO2(L,I,J)=y(nXO2,L)

c       Set value for XO2N:
        XO2Nprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.13d0
     &    +rr(rrbi%Alkenes_NO3__HCHO_NO2,L)*y(nNO3,L)*y(nn_Alkenes,L)
     &      *0.09d0
     &    +rr(rrbi%ROR_M__Aldehyde_HO2,L)*y(nM,L)*yROR(L,I,J)*0.04d0
     &    +rr(rrbi%Isoprene_OH__HCHO_Alkenes,L)*y(nn_Isoprene,L)*
     &      y(nOH,L)*0.15d0
#ifdef TRACERS_TERP
     &    +rr(rrbi%Terpenes_OH__HCHO_Alkenes,L)*y(nn_Terpenes,L)
     &      *y(nOH,L)*0.15d0
#endif  /* TRACERS_TERP */
        XO2Ndest=rr(rrbi%XO2N_HO2__CH3OOH_O2,L)*y(nHO2,L)
     &    +rr(rrbi%XO2N_NO__AlkylNit_M,L)*y(nNO,L)
        if(XO2Ndest > 1.d-7)then
          y(nXO2N,L)=(XO2Nprod/XO2Ndest)
        else
          y(nXO2N,L)=1.d0
        end if
        yXO2N(L,I,J)=y(nXO2N,L)

        ! ACCMIP diag:
        TAIJLS(I,J,L,ijlt_OxpRO2)=TAIJLS(I,J,L,ijlt_OxpRO2)
     &    +(y(nXO2,L)*y(nNO,L)*rr(rrbi%XO2_NO__NO2_M,L)
     &      +y(nXO2N,L)*y(nNO,L)*rr(rrbi%XO2N_NO__AlkylNit_M,L))*cpd

c       Set value for RXPAR:
        RXPARprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.11d0
     &    +rr(rrbi%Alkenes_OH__HCHO_HO2,L)*y(nn_Alkenes,L)*y(nOH,L)
     &    +rr(rrbi%ROR_M__Aldehyde_HO2,L)*y(nM,L)*yROR(L,I,J)*2.1d0
     &    +rr(rrbi%Alkenes_O3__HCHO_CO,L)*y(nn_Alkenes,L)*y(nO3,L)*0.9d0
     &    +rr(rrbi%Alkenes_NO3__HCHO_NO2,L)*y(nNO3,L)*y(nn_Alkenes,L)
        RXPARdest=rr(rrbi%Paraffin_RXPAR__M_M,L)*y(nn_Paraffin,L)
        if(RXPARdest > 0.d0)then
          y(nRXPAR,L)=(RXPARprod/RXPARdest)
        else
          y(nRXPAR,L)=1.d0
        end if
        yRXPAR(L,I,J)=y(nRXPAR,L)

#ifdef TRACERS_dCO
! ok to overwrite RXPARprod and RXPARdest here
c       Set value for d13CXPAR:
        RXPARprod=rr(rrbi%d13CPAR_OH__HO2_M,L)*y(nn_d13CPAR,L)
     &      *y(nOH,L)*0.11d0
     &    +rr(rrbi%d13Calke_OH__dH13CHO_HO2,L)*y(nn_d13Calke,L)*y(nOH,L)
     &    +rr(rrbi%d13CROR_M__d13Cald_HO2,L)*y(nM,L)*yd13CROR(L,I,J)
     &      *2.1d0
     &    +rr(rrbi%d13Calke_O3__dH13CHO_d13CO,L)*y(nn_d13Calke,L)
     &      *y(nO3,L)*0.9d0
     &    +rr(rrbi%d13Calke_NO3__dH13CHO_NO2,L)*y(nNO3,L)
     &      *y(nn_d13Calke,L)
        RXPARdest=rr(rrbi%d13CPAR_d13CXPAR__M_M,L)*y(nn_d13CPAR,L)
        if(RXPARdest > 0.d0)then
          y(nd13CXPAR,L)=(RXPARprod/RXPARdest)
        else
          y(nd13CXPAR,L)=1.d0
        end if
        yd13CXPAR(L,I,J)=y(nd13CXPAR,L)
#endif  /* TRACERS_dCO */

c       Set value for Aldehyde:
        Aldehydeprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.11d0
     &    +rr(rrbi%Alkenes_OH__HCHO_HO2,L)*y(nn_Alkenes,L)*y(nOH,L)
     &    +rr(rrbi%ROR_M__Aldehyde_HO2,L)*y(nM,L)*yROR(L,I,J)*1.1d0
     &    +rr(rrbi%Alkenes_O3__HCHO_CO,L)*y(nn_Alkenes,L)
     &      *y(nO3,L)*0.44d0
        Aldehydedest=rr(rrbi%Aldehyde_OH__C2O3_M,L)*y(nOH,L)
     &    +zj(L,rj%Aldehyde__HCHO_CO)
c       Check for equilibrium:
        if(Aldehydedest*y(nAldehyde,L)*dt2 < y(nAldehyde,L))then
          changeAldehyde=
     &    (Aldehydeprod-y(nAldehyde,L)*Aldehydedest)*dt2
          if(changeAldehyde > y(nAldehyde,L))
     &    changeAldehyde=y(nAldehyde,L)
          y(nAldehyde,L)=y(nAldehyde,L)+changeAldehyde
          if(y(nAldehyde,L) < 0.d0) y(nAldehyde,L)=1.d0
        else
          y(nAldehyde,L)=(Aldehydeprod/(Aldehydedest+0.5d-5))
        end if
        yAldehyde(L,I,J)=y(nAldehyde,L)

#ifdef TRACERS_dCO
c       Set value for d17Oald:
!ok to overwrite here Aldehydeprod,Aldehydedest,changeAldehyde
        Aldehydeprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.11d0*R_17O_16O
     &    +rr(rrbi%Alkenes_OH__HCHO_HO2,L)*y(nn_Alkenes,L)*y(nOH,L)
     &    +rr(rrbi%d17OROR_M__d17Oald_HO2,L)*y(nM,L)*yd17OROR(L,I,J)
     &      *1.1d0
     &    +rr(rrbi%Alkenes_O3__HCHO_CO,L)*y(nn_Alkenes,L)
     &      *y(nO3,L)*0.44d0
        Aldehydedest=rr(rrbi%d17Oald_OH__dC217O3_M,L)*y(nOH,L)
#ifndef TRACERS_dCO_bin_reprod
     &    +zj(L,rj%d17Oald__dHCH17O_CO)
     &    +zj(L,rj%d17Oald__HCHO_dC17O)
#endif  /* TRACERS_dCO_bin_reprod */
     &    +zj(L,rj%d17Oald__HCHO_CO)
c       Check for equilibrium:
        if(Aldehydedest*y(nd17Oald,L)*dt2 < y(nd17Oald,L))then
          changeAldehyde=
     &    (Aldehydeprod-y(nd17Oald,L)*Aldehydedest)*dt2
          if(changeAldehyde > y(nd17Oald,L))
     &    changeAldehyde=y(nd17Oald,L)
          y(nd17Oald,L)=y(nd17Oald,L)+changeAldehyde
          if(y(nd17Oald,L) < 0.d0) y(nd17Oald,L)=1.d0
        else
          y(nd17Oald,L)=(Aldehydeprod/(Aldehydedest+0.5d-5))
        end if
        yd17Oald(L,I,J)=y(nd17Oald,L)

c       Set value for d18Oald:
        Aldehydeprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.11d0*R_18O_16O
     &    +rr(rrbi%Alkenes_OH__HCHO_HO2,L)*y(nn_Alkenes,L)*y(nOH,L)
     &    +rr(rrbi%d18OROR_M__d18Oald_HO2,L)*y(nM,L)*yd18OROR(L,I,J)
     &      *1.1d0
     &    +rr(rrbi%Alkenes_O3__HCHO_CO,L)*y(nn_Alkenes,L)
     &      *y(nO3,L)*0.44d0
        Aldehydedest=rr(rrbi%d18Oald_OH__dC218O3_M,L)*y(nOH,L)
#ifndef TRACERS_dCO_bin_reprod
     &    +zj(L,rj%d18Oald__dHCH18O_CO)
     &    +zj(L,rj%d18Oald__HCHO_dC18O)
#endif  /* TRACERS_dCO_bin_reprod */
     &    +zj(L,rj%d18Oald__HCHO_CO)
c       Check for equilibrium:
        if(Aldehydedest*y(nd18Oald,L)*dt2 < y(nd18Oald,L))then
          changeAldehyde=
     &    (Aldehydeprod-y(nd18Oald,L)*Aldehydedest)*dt2
          if(changeAldehyde > y(nd18Oald,L))
     &    changeAldehyde=y(nd18Oald,L)
          y(nd18Oald,L)=y(nd18Oald,L)+changeAldehyde
          if(y(nd18Oald,L) < 0.d0) y(nd18Oald,L)=1.d0
        else
          y(nd18Oald,L)=(Aldehydeprod/(Aldehydedest+0.5d-5))
        end if
        yd18Oald(L,I,J)=y(nd18Oald,L)

c       Set value for d13Cald:
        Aldehydeprod=rr(rrbi%d13CPAR_OH__HO2_M,L)*y(nn_d13CPAR,L)
     &      *y(nOH,L)*0.11d0
     &    +rr(rrbi%d13Calke_OH__dH13CHO_HO2,L)*y(nn_d13Calke,L)*y(nOH,L)
     &    +rr(rrbi%d13CROR_M__d13Cald_HO2,L)*y(nM,L)*yd13CROR(L,I,J)
     &      *1.1d0
     &    +rr(rrbi%d13Calke_O3__dH13CHO_d13CO,L)*y(nn_d13Calke,L)
     &      *y(nO3,L)*0.44d0
        Aldehydedest=rr(rrbi%d13Cald_OH__d13C2O3_M,L)*y(nOH,L)
#ifndef TRACERS_dCO_bin_reprod
     &    +zj(L,rj%d13Cald__dH13CHO_CO)
     &    +zj(L,rj%d13Cald__HCHO_d13CO)
#endif  /* TRACERS_dCO_bin_reprod */
     &    +zj(L,rj%d13Cald__HCHO_CO)
c       Check for equilibrium:
        if(Aldehydedest*y(nd13Cald,L)*dt2 < y(nd13Cald,L))then
          changeAldehyde=
     &    (Aldehydeprod-y(nd13Cald,L)*Aldehydedest)*dt2
          if(changeAldehyde > y(nd13Cald,L))
     &    changeAldehyde=y(nd13Cald,L)
          y(nd13Cald,L)=y(nd13Cald,L)+changeAldehyde
          if(y(nd13Cald,L) < 0.d0) y(nd13Cald,L)=1.d0
        else
          y(nd13Cald,L)=(Aldehydeprod/(Aldehydedest+0.5d-5))
        end if
        yd13Cald(L,I,J)=y(nd13Cald,L)
#endif  /* TRACERS_dCO */

c       Set value for ROR:
        RORprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.76d0
        RORdest=(0.98d0*rr(rrbi%ROR_M__Aldehyde_HO2,L)
     &    +rr(rrbi%ROR_M__HO2_M,L))*y(nM,L)
        if(RORdest > 0.d0)then
          y(nROR,L)=(RORprod/RORdest)
        else
          y(nROR,L)=1.d0
        end if
        yROR(L,I,J)=y(nROR,L)

#ifdef TRACERS_dCO
! ok ot overwrite RORprod,RORdest
c       Set value for d17OROR:
        RORprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.76d0*R_17O_16O
        RORdest=(0.98d0*rr(rrbi%d17OROR_M__d17Oald_HO2,L)
     &    +rr(rrbi%d17OROR_M__HO2_M,L))*y(nM,L)
        if(RORdest > 0.d0)then
          y(nd17OROR,L)=(RORprod/RORdest)
        else
          y(nd17OROR,L)=1.d0
        end if
        yd17OROR(L,I,J)=y(nd17OROR,L)

c       Set value for d18OROR:
        RORprod=rr(rrbi%Paraffin_OH__HO2_M,L)*y(nn_Paraffin,L)
     &      *y(nOH,L)*0.76d0*R_18O_16O
        RORdest=(0.98d0*rr(rrbi%d18OROR_M__d18Oald_HO2,L)
     &    +rr(rrbi%d18OROR_M__HO2_M,L))*y(nM,L)
        if(RORdest > 0.d0)then
          y(nd18OROR,L)=(RORprod/RORdest)
        else
          y(nd18OROR,L)=1.d0
        end if
        yd18OROR(L,I,J)=y(nd18OROR,L)

c       Set value for d13CROR:
        RORprod=rr(rrbi%d13CPAR_OH__HO2_M,L)*y(nn_d13CPAR,L)
     &      *y(nOH,L)*0.76d0
        RORdest=(0.98d0*rr(rrbi%d13CROR_M__d13Cald_HO2,L)
     &    +rr(rrbi%d13CROR_M__HO2_M,L))*y(nM,L)
        if(RORdest > 0.d0)then
          y(nd13CROR,L)=(RORprod/RORdest)
        else
          y(nd13CROR,L)=1.d0
        end if
        yd13CROR(L,I,J)=y(nd13CROR,L)
#endif  /* TRACERS_dCO */
      end do  ! --------------------------------------

c If NOx in equil with N2O5, HO2NO2, or PAN, remove from changes:
      do L=1,maxL
        if(-dest(nn_N2O5,L) >= y(nn_N2O5,L) .or.
     &  chemrate(rrtri%NO3_NO2__N2O5_M,L) > y(nn_NOx,L))then
          dest(nn_NOx,L)=dest(nn_NOx,L)
     &      +2.d0*chemrate(rrtri%NO3_NO2__N2O5_M,L)
          prod(nn_NOx,L)=prod(nn_NOx,L)
     &      -2.d0*(chemrate(rrmono%N2O5_M__NO3_NO2,L)
     &      +photrate(rj%N2O5__NO3_NO2,L))
        endif
        if(-dest(nn_HO2NO2,L) >= y(nn_HO2NO2,L) .or.
     &  chemrate(rrtri%HO2_NO2__HO2NO2_M,L) > y(nn_NOx,L))then
          dest(nn_NOx,L)=dest(nn_NOx,L)
     &      +chemrate(rrtri%HO2_NO2__HO2NO2_M,L)
          prod(nn_NOx,L)=prod(nn_NOx,L)
     &      -(chemrate(rrbi%OH_HO2NO2__H2O_NO2,L)
     &        +chemrate(rrmono%HO2NO2_M__HO2_NO2,L)
     &        +photrate(rj%HO2NO2__HO2_NO2,L)
     &        +photrate(rj%HO2NO2__OH_NO3,L))
        endif
        if(-dest(nn_PAN,L) >= y(nn_PAN,L) .or.
     &  chemrate(rrtri%C2O3_NO2__PAN_M,L) > y(nn_NOx,L))then
          dest(nn_NOx,L)=dest(nn_NOx,L)
     &      +chemrate(rrtri%C2O3_NO2__PAN_M,L)
          prod(nn_NOx,L)=prod(nn_NOx,L)
     &      -(chemrate(rrbi%PAN_M__C2O3_NO2,L)
     &      +photrate(rj%PAN__C2O3_NO2,L))
        end if
        
c If BrOx in equil with HOBr or BrONO2, remove from changes:
        if(-dest(nn_HOBr,L) >= y(nn_HOBr,L).or.
     &  chemrate(rrbi%BrO_HO2__HOBr_O2,L) > 0.5d0*y(nn_BrOx,L))then
          dest(nn_BrOx,L)=dest(nn_BrOx,L)
     &      +chemrate(rrbi%BrO_HO2__HOBr_O2,L)
          prod(nn_BrOx,L)=prod(nn_BrOx,L)
     &      -photrate(rj%HOBr__Br_OH,L)
        endif
        if(-dest(nn_BrONO2,L) >= y(nn_BrONO2,L).or.
     &  chemrate(rrtri%BrO_NO2__BrONO2_M,L) > 0.5d0*y(nn_BrOx,L))then
          dest(nn_BrOx,L)=dest(nn_BrOx,L)
     &      +chemrate(rrtri%BrO_NO2__BrONO2_M,L)
          prod(nn_BrOx,L)=prod(nn_BrOx,L)
     &      -photrate(rj%BrONO2__BrO_NO2,L)
          dest(nn_NOx,L)=dest(nn_NOx,L)
     &      +chemrate(rrtri%BrO_NO2__BrONO2_M,L)
          prod(nn_NOx,L)=prod(nn_NOx,L)
     &      -photrate(rj%BrONO2__BrO_NO2,L)
        end if
        
c If ClOx in equil with HOCl or ClONO2, remove from changes:
        if(-dest(nn_HOCl,L) >= y(nn_HOCl,L) .or.
     &  chemrate(rrbi%ClO_HO2__HOCl_O2,L) > y(nn_ClOx,L))then
          dest(nn_ClOx,L)=dest(nn_ClOx,L)
     &      +chemrate(rrbi%ClO_HO2__HOCl_O2,L)
          prod(nn_ClOx,L)=prod(nn_ClOx,L)
     &      -(photrate(rj%HOCl__OH_Cl,L)
     &      +chemrate(rrbi%O_HOCl__OH_ClO,L))
        endif
        if(-dest(nn_ClONO2,L) >= y(nn_ClONO2,L) .or.
     &  chemrate(rrtri%ClO_NO2__ClONO2_M,L) > 0.8d0*y(nn_ClOx,L))then
          dest(nn_ClOx,L)=dest(nn_ClOx,L)
     &      +chemrate(rrtri%ClO_NO2__ClONO2_M,L)
          prod(nn_ClOx,L)=prod(nn_ClOx,L)
     &      -(photrate(rj%ClONO2__Cl_NO3,L)
     &      +chemrate(rrbi%ClONO2_O__ClO_NO3,L))
          dest(nn_NOx,L)=dest(nn_NOx,L)
     &      +chemrate(rrtri%ClO_NO2__ClONO2_M,L)
          prod(nn_NOx,L)=prod(nn_NOx,L)
     &      -(photrate(rj%ClONO2__Cl_NO3,L)
     &      +chemrate(rrbi%ClONO2_O__ClO_NO3,L))
        end if
      end do

c Calculate water vapor change AND APPLY TO MODEL Q VARIABLE:
      do L=1,maxL ! for a long time, this used to be stratosphere only loop...
        changeH2O(L)=(2.d0*y(nn_CH4,L)*
     &    (rr(rrbi%O1D_CH4__OH_CH3O2,L)*y(nO1D,L)
     &      +rr(rrbi%CH4_OH__H2O_CH3O2,L)*y(nOH,L)
     &      +rr(rrbi%Cl_CH4__HCl_CH3O2,L)*y(nCl,L))
     &      -SF3(L)*y(nH2O,L))*dt2
C       And apply that change here and accumulate a diagnostic:
C       --- y --- :
        y(nH2O,L)=y(nH2O,L)+changeH2O(L)
C       --- Q --- :
        dQ(L) = changeH2O(L)/(y(nM,L)*MWabyMWw)
        dQM(L) = dQ(L)*ma(L)
        if(clim_interact_chem > 0)then
          fraQ2(l)=(qv(L)+dQ(L))/qv(L)
          call update_qv(l,qv(L)+dQ(L))
C       -- Qmom --:
          if(changeH2O(L) < 0.)then
            call update_qvmom(L,qvmom(:,L)*fraQ2(L))
            if(fraQ2(l) <= 0.98)then
              write(out_line,*)'> 2% Q change in calc IJL,change='
     &        ,I,J,L,fraQ2(l)
              call write_parallel(trim(out_line),crit=.true.)
              call stop_model('big Q change in calc',255)
            end if
          end if
        end if
      end do

C     -- diags --:
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_H2Ochem,dQM)
      if(clim_interact_chem > 0)then
        dQMsum = sum(dQM(1:maxL))
        do it=1,ntype
          call inc_aj(i,j,it,j_h2och4,dQMsum*ftype(it,i,j))
        end do
        if (save_dQ_for_NINT==1) then
          do L=1,maxL
            dQo1d(L)=(2.d0*y(nn_CH4,L)*
     &       rr(rrbi%O1D_CH4__OH_CH3O2,L)*y(nO1D,L)
     &       )*dt2/(y(nM,L)*MWabyMWw)
            dQoh(L)=(2.d0*y(nn_CH4,L)*
     &       rr(rrbi%CH4_OH__H2O_CH3O2,L)*y(nOH,L)
     &       )*dt2/(y(nM,L)*MWabyMWw)
            dQcl(L)=(2.d0*y(nn_CH4,L)*
     &       rr(rrbi%Cl_CH4__HCl_CH3O2,L)*y(nCl,L)
     &       )*dt2/(y(nM,L)*MWabyMWw)
            dQsf3(L)=(-SF3(L)*y(nH2O,L)
     &       )*dt2/(y(nM,L)*MWabyMWw)
            taijls(i,j,L,ijlt_dQ)=taijls(i,j,L,ijlt_dQ)+dQ(L)
            taijls(i,j,L,ijlt_dQo1d)=taijls(i,j,L,ijlt_dQo1d)+dQo1d(L)
            taijls(i,j,L,ijlt_dQoh)=taijls(i,j,L,ijlt_dQoh)+dQoh(L)
            taijls(i,j,L,ijlt_dQcl)=taijls(i,j,L,ijlt_dQcl)+dQcl(L)
            taijls(i,j,L,ijlt_dQsf3)=taijls(i,j,L,ijlt_dQsf3)+dQsf3(L)
          end do
        end if
#ifdef TRACERS_WATER
C     -- water tracers --:
        do n=1,ntm
          select case(tr_wd_type(n))
          case(nWater)           ! water: add CH4-sourced water to tracers
            do L=1,maxL
              trm_col(L,n) = trm_col(L,n) + tr_H2ObyCH4(n)*dQM(L)
              if(changeH2O(L) < 0.) trmom_col(:,L,n) = trmom_col(:,L,n)
     *             *fraQ2(L)
            end do
          end select
        end do
#endif
      end if

C There was a section here in the code that altereded ozone change
C based on within-NOx partitioning. Drew said that arguments could be
C made to include such a section or not. But he notes that, once we
C made day and night N-chemistry more similar, this section was incomplete
C anyway because it was programmed when NO3 was 0 during the day.
C To see this (commented-out) code, check out the master branch from Nov 1,
C 2016.

c Calculate ozone change due to Cl2O2 cycling:
      do L=1,maxL
        if(yCl2O2(L,I,J) > 1d1)dest(nn_Ox,L)=dest(nn_Ox,L)
     &    -0.75d0*rr(rrbi%Cl_O3__ClO_O2,L)*y(nCl,L)*y(nO3,L)*dt2
     &      *yCl2O2(L,I,J)*1.5d9/y(nM,L)
      end do

! c Include oxidation of CO by O(1D)
!       do L=1,maxL
!         dest(nn_CO,L)=dest(nn_CO,L)-rCOplusO1D*y(nn_CO,L)*y(nO1D,L)*dt2
! #ifdef TRACERS_dCO
!         dest(nn_dC17O,L)=dest(nn_dC17O,L)
!      &                  -rdC17OplusO1D*y(nn_dC17O,L)*y(nO1D,L)*dt2
!         dest(nn_dC18O,L)=dest(nn_dC18O,L)
!      &                  -rdC18OplusO1D*y(nn_dC18O,L)*y(nO1D,L)*dt2
!         dest(nn_d13CO,L)=dest(nn_d13CO,L)
!      &                  -rd13COplusO1D*y(nn_d13CO,L)*y(nO1D,L)*dt2
! #endif  /* TRACERS_dCO */
!       end do

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c           Print chemistry diagnostics if desired :
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C             REACTION RATES, CHEMICAL CHANGES
c (chem1prn: argument before multip is index = number of call):
      
      if(prnrts .and. J==ijlprn(2) .and. I==ijlprn(1))then
        do igas=1,ntm_chem
          total=0.d0
          write(out_line,108)' Species: ',trchemname(igas)
          call write_parallel(trim(out_line),crit=jay)

          call chem1prn
     &    (kdnr,2,n_rx,nn,ndnr,chemrate,1,-1,igas,total,maxL,I,J,jay)

          if(igas == nn_NOx)then
            if(-dest(nn_HO2NO2,ijlprn(3)) >= y(nn_HO2NO2,ijlprn(3)) .or.
     &      chemrate(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'loss by reaction rrtri%HO2_NO2__HO2NO2_M removed',
     &          chemrate(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_N2O5,ijlprn(3)) >= y(nn_N2O5,ijlprn(3)) .or.
     &      chemrate(rrtri%NO3_NO2__N2O5_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'losses by reaction rrtri%NO3_NO2__N2O5_M removed',
     &          2.d0*chemrate(rrtri%NO3_NO2__N2O5_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_PAN,ijlprn(3)) >= y(nn_PAN,ijlprn(3)) .or.
     &      chemrate(rrtri%C2O3_NO2__PAN_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'losses by reaction rrtri%C2O3_NO2__PAN_M removed',
     &          chemrate(rrtri%C2O3_NO2__PAN_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            end if
          end if
          
          call chem1prn
     &    (kpnr,2,n_rx,nnr,npnr,chemrate,2,1,igas,total,maxL,I,J,jay)
     
          if(igas == nn_NOx)then
            if(-dest(nn_HO2NO2,ijlprn(3)) >= y(nn_HO2NO2,ijlprn(3)) .or.
     &      chemrate(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'gain by reactions destroying HO2NO2 removed  ',
     &          (rr(rrbi%OH_HO2NO2__H2O_NO2,ijlprn(3))*y(nOH,L)
     &           +rr(rrmono%HO2NO2_M__HO2_NO2,ijlprn(3))*y(nM,ijlprn(3))
     &            +zj(ijlprn(3),rj%HO2NO2__HO2_NO2)
     &            +zj(ijlprn(3),rj%HO2NO2__OH_NO3)
     &          )*y(nn_HO2NO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)     
            endif
            if(-dest(nn_N2O5,ijlprn(3)) >= y(nn_N2O5,ijlprn(3)).or.
     &      chemrate(rrtri%NO3_NO2__N2O5_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'gains by reaction rrmono%N2O5_M__NO3_NO2 removed',
     &          2.d0*chemrate(rrmono%N2O5_M__NO3_NO2,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_PAN,ijlprn(3)) >= y(nn_PAN,ijlprn(3)).or.
     &      chemrate(rrtri%C2O3_NO2__PAN_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)
     &          'gain by reaction rrbi%PAN_M__C2O3_NO2 removed',
     &          chemrate(rrbi%PAN_M__C2O3_NO2,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            end if
          end if

          call chem1prn
     &    (kds,1,n_rj,ks,nds,photrate,3,-1,igas,total,maxL,I,J,jay)
          call chem1prn
     &    (kps,2,n_rj,kss,nps,photrate,4,1,igas,total,maxL,I,J,jay)

! Commenting this goes along with reference commented section
! involving Oxcorr above:
!          if(igas == nn_Ox) then
!            write(out_line,110)'Ox change due to within NOx rxns  ',
!     &      -Oxcorr(ijlprn(3))
!            call write_parallel(trim(out_line),crit=jay)
!          end if

          if(igas == nn_NOx)then
            if(-dest(nn_N2O5,ijlprn(3)) >= y(nn_N2O5,ijlprn(3)) .or.
     &      chemrate(rrtri%NO3_NO2__N2O5_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)'gains by reaction 7'//
     &          ' (N2O5 photolysis) removed',
     &          zj(ijlprn(3),rj%N2O5__NO3_NO2)*
     &          y(nn_N2O5,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_N2O5,ijlprn(3)) >= y(nn_N2O5,ijlprn(3)) .or.
     &      chemrate(rrtri%NO3_NO2__N2O5_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)'net change due to N2O5 is ',
     &          2.d0*(y(nn_N2O5,ijlprn(3))
     &           -(rr(rrtri%NO3_NO2__N2O5_M,ijlprn(3))*y(nNO3,ijlprn(3))
     &                    *y(nNO2,ijlprn(3)))
     &            /(rr(rrmono%N2O5_M__NO3_NO2,ijlprn(3))*y(nM,ijlprn(3))
     &                   +zj(ijlprn(3),rj%N2O5__NO3_NO2)))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_HO2NO2,ijlprn(3)) >= y(nn_HO2NO2,ijlprn(3)) .or.
     &      chemrate(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)'gain by rxns 10 & 11 (HO2NO2'
     &          //' photolysis) removed',
     &          (zj(ijlprn(3),rj%HO2NO2__HO2_NO2)
     &            +zj(ijlprn(3),rj%HO2NO2__OH_NO3)
     &          )*y(nn_HO2NO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_HO2NO2,ijlprn(3)) >= y(nn_HO2NO2,ijlprn(3)) .or.
     &      chemrate(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)'net change due to HO2NO2 is ',
     &          y(nn_HO2NO2,ijlprn(3))
     &          -((rr(rrtri%HO2_NO2__HO2NO2_M,ijlprn(3))
     &          *y(nHO2,ijlprn(3))*y(nNO2,ijlprn(3)))
     &          /(rr(rrbi%OH_HO2NO2__H2O_NO2,ijlprn(3))*y(nOH,ijlprn(3))
     &           +rr(rrmono%HO2NO2_M__HO2_NO2,ijlprn(3))*y(nM,ijlprn(3))
     &            +zj(ijlprn(3),rj%HO2NO2__HO2_NO2)
     &            +zj(ijlprn(3),rj%HO2NO2__OH_NO3)))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_PAN,ijlprn(3)) >= y(nn_PAN,ijlprn(3)) .or.
     &      chemrate(rrtri%C2O3_NO2__PAN_M,ijlprn(3))>
     &      y(nn_NOx,ijlprn(3))) then
              write(out_line,110)'net change due to PAN is ',
     &          y(nn_PAN,ijlprn(3))
     &          -((rr(rrtri%C2O3_NO2__PAN_M,ijlprn(3))
     &          *y(nC2O3,ijlprn(3))*y(nNO2,ijlprn(3)))
     &          /(rr(rrbi%PAN_M__C2O3_NO2,ijlprn(3))*y(nM,ijlprn(3))
     &            +zj(ijlprn(3),rj%PAN__C2O3_NO2)))
              call write_parallel(trim(out_line),crit=jay)    
            end if
          end if
                
          if(igas == nn_Ox .or. igas == nn_NOx) total=
     &    100.d0*(dest(igas,ijlprn(3))+prod(igas,ijlprn(3)))
     &    /y(igas,ijlprn(3))
     
          if(igas == nn_BrOx)then
            if(-dest(nn_HOBr,ijlprn(3)) >= y(nn_HOBr,ijlprn(3)).or.
     &         chemrate(rrbi%BrO_HO2__HOBr_O2,ijlprn(3)) >
     &           0.5d0*y(nn_BrOx,ijlprn(3)))then
              write(out_line,110)
     &          'gain by rxns 24 (HOBr photolysis) removed',
     &         zj(ijlprn(3),rj%HOBr__Br_OH)*y(nn_HOBr,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'loss by rxn rrbi%BrO_HO2__HOBr_O2 removed',
     &          chemrate(rrbi%BrO_HO2__HOBr_O2,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            endif
            if(-dest(nn_BrONO2,ijlprn(3)) >= y(nn_BrONO2,ijlprn(3)) .or.
     &         chemrate(rrtri%BrO_NO2__BrONO2_M,ijlprn(3)) >
     &           0.5d0*y(nn_BrOx,ijlprn(3)))then
              write(out_line,110)
     &          'gain by rxns 23 (BrONO2 photolysis) removed',
     &          zj(ijlprn(3),rj%BrONO2__BrO_NO2)
     &          *y(nn_BrONO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'loss by rxn rrtri%BrO_NO2__BrONO2_M removed'
     &          ,chemrate(rrtri%BrO_NO2__BrONO2_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            end if
          end if
          
          if(igas == nn_NOx)then
            if(-dest(nn_BrONO2,ijlprn(3)) >= y(nn_BrONO2,ijlprn(3)) .or.
     &         chemrate(rrtri%BrO_NO2__BrONO2_M,ijlprn(3)) >
     &           0.5d0*y(nn_BrOx,ijlprn(3)))then
              write(out_line,110)
     &        'gain by rxns 23 (BrONO2 photolysis) removed'
     &        ,zj(ijlprn(3),rj%BrONO2__BrO_NO2)
     &        *y(nn_BrONO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'loss by rxn rrtri%BrO_NO2__BrONO2_M removed'
     &          ,chemrate(rrtri%BrO_NO2__BrONO2_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)     
            end if
          end if
          
          if(igas == nn_ClOx)then
            if(-dest(nn_HOCl,ijlprn(3)) >= y(nn_HOCl,ijlprn(3)) .or.
     &      chemrate(rrbi%ClO_HO2__HOCl_O2,ijlprn(3))>
     &      y(nn_ClOx,ijlprn(3)))then
              write(out_line,110)
     &          'gain by rxn 21 (HOCl photolysis) removed',
     &         zj(ijlprn(3),rj%HOCl__OH_Cl)*y(nn_HOCl,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'gain by rxn rrbi%O_HOCl__OH_ClO removed',
     &          chemrate(rrbi%O_HOCl__OH_ClO,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
                write(out_line,110)
     &          'loss by rxn rrbi%ClO_HO2__HOCl_O2 removed',
     &          chemrate(rrbi%ClO_HO2__HOCl_O2,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            endif 
            if(-dest(nn_ClONO2,ijlprn(3)) >= y(nn_ClONO2,ijlprn(3)) .or.
     &         chemrate(rrtri%ClO_ClO__Cl2O2_M,ijlprn(3)) >
     &           0.8d0*y(nn_ClOx,ijlprn(3)))then
              write(out_line,110)
     &          'gain by rxn 22 (ClONO2 photolysis) removed',
     &          zj(ijlprn(3),rj%ClONO2__Cl_NO3)
     &          *y(nn_ClONO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'gain by rxn rrbi%ClONO2_O__ClO_NO3 removed',
     &          chemrate(rrbi%ClONO2_O__ClO_NO3,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'loss by rxn rrtri%ClO_ClO__Cl2O2_M removed'
     &          ,chemrate(rrtri%ClO_ClO__Cl2O2_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            end if
          end if
        
          if(igas == nn_NOx)then
            if(-dest(nn_ClONO2,ijlprn(3)) >= y(nn_ClONO2,ijlprn(3)) .or.
     &         chemrate(rrtri%ClO_ClO__Cl2O2_M,ijlprn(3)) >
     &           0.8d0*y(nn_ClOx,ijlprn(3)))then
              write(out_line,110)
     &          'gain by rxn 22 (ClONO2 photolysis) removed',
     &          zj(ijlprn(3),rj%ClONO2__Cl_NO3)
     &          *y(nn_ClONO2,ijlprn(3))*dt2
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'gain by rxn rrbi%ClONO2_O__ClO_NO3 removed',
     &          chemrate(rrbi%ClONO2_O__ClO_NO3,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
              write(out_line,110)
     &          'loss by rxn rrtri%ClO_ClO__Cl2O2_M removed'
     &          ,chemrate(rrtri%ClO_ClO__Cl2O2_M,ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
            end if
          end if

          if(igas == nn_CH3OOH) then
            write(out_line,'(a48,a6,e10.3)')
     &        'production from XO2N + HO2 ','dy = ',
     &        y(nHO2,ijlprn(3))*y(nXO2N,ijlprn(3))
     &        *rr(rrbi%XO2N_HO2__CH3OOH_O2,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
#ifdef TRACERS_dCO
          if(igas == nn_dMe17OOH) then
            write(out_line,'(a48,a6,e10.3)')
     &        'production from XO2N + HO2 ','dy = ',
     &        y(nHO2,ijlprn(3))*y(nXO2N,ijlprn(3))
     &        *rr(rrbi%XO2N_HO2__dMe17OOH_O2,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
          if(igas == nn_dMe18OOH) then
            write(out_line,'(a48,a6,e10.3)')
     &        'production from XO2N + HO2 ','dy = ',
     &        y(nHO2,ijlprn(3))*y(nXO2N,ijlprn(3))
     &        *rr(rrbi%XO2N_HO2__dMe18OOH_O2,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
          if(igas == nn_d13MeOOH) then
            write(out_line,'(a48,a6,e10.3)')
     &        'production from XO2N + HO2 ','dy = ',
     &        y(nHO2,ijlprn(3))*y(nXO2N,ijlprn(3))
     &        *rr(rrbi%XO2N_HO2__d13MeOOH_O2,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
#endif  /* TRACERS_dCO */

#ifdef TRACERS_HETCHEM
          if(igas == nn_HNO3) then
            write(out_line,'(a48,a6,e10.3)')
     &      'destruction from HNO3 +dust ','dy = ',
     &      -y(nn_HNO3,ijlprn(3))*krate(ijlprn(3),1,1)*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
#endif
          if(igas == nn_Paraffin) then
            write(out_line,'(a48,a6,e10.3)')'destruction from RXPAR ',
     &      'dy = ',-y(nRXPAR,ijlprn(3))*y(nn_Paraffin,ijlprn(3))
     &        *rr(rrbi%Paraffin_RXPAR__M_M,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
#ifdef TRACERS_dCO
          if(igas == nn_d13CPAR) then
            write(out_line,'(a48,a6,e10.3)')'destruction from d13CXPAR',
     &      'dy = ',-y(nd13CXPAR,ijlprn(3))*y(nn_d13CPAR,ijlprn(3))
     &        *rr(rrbi%d13CPAR_d13CXPAR__M_M,ijlprn(3))*dt2
            call write_parallel(trim(out_line),crit=jay)
          end if
#endif  /* TRACERS_dCO */
          
          write(out_line,118) ' Total change in ',trchemname(igas),
     &    ' is ',total,' percent; dy= ',dest(igas,ijlprn(3))
     &    +prod(igas,ijlprn(3))
          call write_parallel(trim(out_line),crit=jay)
          write(out_line,*) ' '
          call write_parallel(trim(out_line),crit=jay)
        end do ! igas
      end if  ! chem diags
 108  format(a10,2x,a8)
 110  format(a68,e10.3)
 118  format(a17,a8,a4,f10.0,a14,e12.3)

      if(prnchg .and. J == ijlprn(2) .and. I == ijlprn(1)) then
        write(out_line,*)
     &  'Percentage ozone loss per cycle at I,J:',I,J
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a41,a56,a15)')
     &  '  L   ClOx    NOx     HOx     BrOx    Ox ',
     &  '   NO2+O     NO+O3     ClO+O     Cl+O3    NO2+hv    net',
     &  '    JO2+hv   JNO'     
        call write_parallel(trim(out_line),crit=jay)

        do Lz=maxL,LS1,-1
          sumC=rr(rrbi%Cl_O3__ClO_O2,Lz)*y(nCl,Lz)*y(nO3,Lz)
     &      +rr(rrbi%ClO_O__Cl_O2,Lz)*y(nClO,Lz)*y(nO,Lz)
     &      +rr(rrbi%ClO_O3__OClO_O2,Lz)*y(nClO,Lz)*y(nO3,Lz)
     &      +rr(rrbi%O_OClO__ClO_O2,Lz)*y(nOClO,Lz)*y(nO,Lz)
!     &      -zj(Lz,rj%ClO__Cl_O)*y(nClO,Lz)
          sumN=rr(rrbi%O3_NO__NO2_O2,Lz)*y(nNO,Lz)*y(nO3,Lz)
     &      +rr(rrbi%O_NO2__NO_O2,Lz)*y(nNO2,Lz)*y(nO,Lz)
     &      +rr(rrbi%NO2_O3__NO3_O2,Lz)*y(nNO2,Lz)*y(nO3,Lz)
     &      +rr(rrtri%NO_O__NO2_M,Lz)*y(nNO,Lz)*y(nO,Lz)
     &      -zj(Lz,rj%NO2__NO_O)*y(nNO2,Lz)
          sumH=rr(rrbi%OH_O3__HO2_O2,Lz)*y(nOH,Lz)*y(nO3,Lz)
     &      +rr(rrbi%HO2_O3__OH_O2,Lz)*y(nHO2,Lz)*y(nO3,Lz)
     &      +rr(rrbi%O_OH__O2_H,Lz)*y(nOH,Lz)*y(nO,Lz)
     &      +rr(rrbi%O_HO2__OH_O2,Lz)*y(nHO2,Lz)*y(nO,Lz)
          sumB=rr(rrbi%BrO_O__Br_O2,Lz)*y(nBrO,Lz)*y(nO,Lz)
     &      +rr(rrbi%Br_O3__BrO_O2,Lz)*y(nBr,Lz)*y(nO3,Lz)
!     &      -zj(Lz,rj%BrO__Br_O)*y(nBrO,Lz)
          sumO=2*rr(rrbi%O_O3__O2_O2,Lz)*y(nO,Lz)*y(nO3,Lz)
          sumA=sumC+sumN+sumH+sumB+sumO
          write(out_line,'(i3,1x,5(f7.2,1x),8(e9.2,1x))')
     &      Lz,100.d0*sumC/sumA,
     &      100.d0*sumN/sumA,100.d0*sumH/sumA,100.d0*sumB/sumA,
     &      100.d0*sumO/sumA,
     &      rr(rrbi%O_NO2__NO_O2,Lz)*y(nNO2,Lz)*y(nO,Lz),
     &      rr(rrbi%O3_NO__NO2_O2,Lz)*y(nNO,Lz)*y(nO3,Lz),
     &      rr(rrbi%ClO_O__Cl_O2,Lz)*y(nClO,Lz)*y(nO,Lz),
     &      rr(rrbi%Cl_O3__ClO_O2,Lz)*y(nCl,Lz)*y(nO3,Lz),
     &      zj(Lz,rj%NO2__NO_O)*y(nNO2,Lz),sumA,
     &      zj(Lz,rj%O2__O_O),SF2(Lz)
          call write_parallel(trim(out_line),crit=jay)
        end do
        write(out_line,*) ' '
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a35,3(2x,i2))')
     &  ' Total change by species at I, J, L',i,j,ijlprn(3)
        call write_parallel(trim(out_line),crit=jay)
      end if ! end of chemistry diagnostics ----------------------------

c Loops to calculate tracer changes:

      rMAbyM(1:maxL)=ma(1:maxL)/y(nM,1:maxL)

      do igas=1,ntm_chem ! TRACER LOOP -----------------
       idx = igas+ntm_chem_beg-1
       dxbym2v=vol2mass(idx)
       do L=1,maxL
         conc2mass=rMAbyM(L)*dxbym2v
         c2ml(l) = conc2mass
         changeL(L,idx)=
     &   (dest(igas,L)+prod(igas,L))*conc2mass
         ! ACCMIP diags:
         if(idx == n_CO)then
           TAIJLS(I,J,L,ijlt_COp)=TAIJLS(I,J,L,ijlt_COp)+prod(igas,L)
     *          *cpd/DTsrc
           TAIJLS(I,J,L,ijlt_COd)=TAIJLS(I,J,L,ijlt_COd)+dest(igas,L)
     *          *cpd/DTsrc
         else if(idx == n_Ox)then
           TAIJLS(I,J,L,ijlt_Oxp)=TAIJLS(I,J,L,ijlt_Oxp)+prod(igas,L)
     *          *cpd/DTsrc
           TAIJLS(I,J,L,ijlt_Oxd)=TAIJLS(I,J,L,ijlt_Oxd)+dest(igas,L)
     *          *cpd/DTsrc
         else if(idx==n_CH4)then
           ! destruction only
           TAIJLS(I,J,L,ijlt_CH4d)=
     &          TAIJLS(I,J,L,ijlt_CH4d)+dest(igas,L)*cpd/DTsrc
         else if(idx==n_NOx)then
           ! destruction only
           TAIJLS(I,J,L,ijlt_NOxd)=
     &          TAIJLS(I,J,L,ijlt_NOxd)+dest(igas,L)*cpd/DTsrc
         end if

c Set N2O5 to equilibrium when necessary (near ground,
c N2O5 is thermally unstable, has a very short lifetime):
         if(idx==n_N2O5.and.(-dest(igas,L) >= y(nn_N2O5,L)*0.75d0
     &    .or. chemrate(rrtri%NO3_NO2__N2O5_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%NO3_NO2__N2O5_M,L)*y(nNO3,L)*y(nNO2,L))/
     &       (rr(rrmono%N2O5_M__NO3_NO2,L)*y(nM,L)
     &         +zj(L,rj%N2O5__NO3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_N2O5,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve NOx with respect to N2O5:
         if(idx == n_NOx.and.(-dest(nn_N2O5,L) >= y(nn_N2O5,L)
     &   .or. chemrate(rrtri%NO3_NO2__N2O5_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%NO3_NO2__N2O5_M,L)*y(nNO3,L)*y(nNO2,L))/
     &       (rr(rrmono%N2O5_M__NO3_NO2,L)*y(nM,L)
     &         +zj(L,rj%N2O5__NO3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_N2O5,L))
           if(changeX > 0.33d0*y(nNO2,L))changeX=0.33d0*y(nNO2,L)
           changeL(L,idx)=
     &     changeL(L,idx)-changeX*conc2mass
         end if

c Set HO2NO2 to equil when necessary:
         if(idx == n_HO2NO2.and.(-dest(igas,L) >= y(nn_HO2NO2,L)
     &   .or. chemrate(rrtri%HO2_NO2__HO2NO2_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%HO2_NO2__HO2NO2_M,L)*y(nHO2,L)*y(nNO2,L))
     &       /(rr(rrbi%OH_HO2NO2__H2O_NO2,L)*y(nOH,L)
     &         +rr(rrmono%HO2NO2_M__HO2_NO2,L)*y(nM,L)
     &         +zj(L,rj%HO2NO2__HO2_NO2)
     &         +zj(L,rj%HO2NO2__OH_NO3)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_HO2NO2,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve NOx with respect to HO2NO2:
         if(idx == n_NOx.and.(-dest(nn_HO2NO2,L) >= y(nn_HO2NO2,L)
     &   .or. chemrate(rrtri%HO2_NO2__HO2NO2_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%HO2_NO2__HO2NO2_M,L)*y(nHO2,L)*y(nNO2,L))
     &       /(rr(rrbi%OH_HO2NO2__H2O_NO2,L)*y(nOH,L)
     &         +rr(rrmono%HO2NO2_M__HO2_NO2,L)*y(nM,L)
     &         +zj(L,rj%HO2NO2__HO2_NO2)
     &         +zj(L,rj%HO2NO2__OH_NO3)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_HO2NO2,L))
           if(changeX > 0.33d0*y(nNO2,L))changeX=0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)-
     &     changeX*conc2mass
         end if

c Set PAN to equilibrium when necessary (near ground,
c PAN is thermally unstable, has a very short lifetime):
         if(idx == n_PAN.and.(-dest(igas,L) >= y(nn_PAN,L).or.
     &   chemrate(rrtri%C2O3_NO2__PAN_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%C2O3_NO2__PAN_M,L)*y(nC2O3,L)*y(nNO2,L))/
     &       (rr(rrbi%PAN_M__C2O3_NO2,L)*y(nM,L)
     &         +zj(L,rj%PAN__C2O3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_PAN,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         endif

#ifdef TRACERS_dCO
         if(idx == n_d17OPAN.and.(-dest(igas,L) >= y(nn_d17OPAN,L).or.
     &   chemrate(rrtri%dC217O3_NO2__d17OPAN_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%dC217O3_NO2__d17OPAN_M,L)*y(ndC217O3,L)*
     &       y(nNO2,L))/(rr(rrbi%d17OPAN_M__dC217O3_NO2,L)*y(nM,L)
     &         +zj(L,rj%d17OPAN__dC217O3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_d17OPAN,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         endif

         if(idx == n_d18OPAN.and.(-dest(igas,L) >= y(nn_d18OPAN,L).or.
     &   chemrate(rrtri%dC218O3_NO2__d18OPAN_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%dC218O3_NO2__d18OPAN_M,L)*y(ndC218O3,L)*
     &       y(nNO2,L))/(rr(rrbi%d18OPAN_M__dC218O3_NO2,L)*y(nM,L)
     &         +zj(L,rj%d18OPAN__dC218O3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_d18OPAN,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         endif

         if(idx == n_d13CPAN.and.(-dest(igas,L) >= y(nn_d13CPAN,L).or.
     &   chemrate(rrtri%d13C2O3_NO2__d13CPAN_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%d13C2O3_NO2__d13CPAN_M,L)*y(nd13C2O3,L)*
     &       y(nNO2,L))/(rr(rrbi%d13CPAN_M__d13C2O3_NO2,L)*y(nM,L)
     &         +zj(L,rj%d13CPAN__d13C2O3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_d13CPAN,L))
           if(changeL(L,idx) > 0.33d0*y(nNO2,L))changeL(L,idx)=
     &     0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         endif
#endif  /* TRACERS_dCO */

c Conserve NOx with respect to PAN:
         if(idx == n_NOx.and.(-dest(nn_PAN,L) >= y(nn_PAN,L).or.
     &   chemrate(rrtri%C2O3_NO2__PAN_M,L) > y(nn_NOx,L)))then
           rnewval=(rr(rrtri%C2O3_NO2__PAN_M,L)*y(nC2O3,L)*y(nNO2,L))/
     &       (rr(rrbi%PAN_M__C2O3_NO2,L)*y(nM,L)
     &         +zj(L,rj%PAN__C2O3_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_PAN,L))
           if(changeX > 0.33d0*y(nNO2,L))changeX=0.33d0*y(nNO2,L)
           changeL(L,idx)=changeL(L,idx)-changeX*conc2mass
         end if

c Cacluate Cl2 amount to P/L:
         if((zj(L,rj%Cl2__Cl_Cl)
     &      +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nOH,L)) > 0.)then
           y(nCl2,L)=rr(rrbi%Cl_HOCl__Cl2_OH,L)*y(nn_HOCl,L)*y(nCl,L)/
     &       (zj(L,rj%Cl2__Cl_Cl)
     &       +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nOH,L)
     &       +chemtiny)
         else
           y(nCl2,L)=0.d0
         end if
         yCl2(L,I,J)=y(nCl2,L)

c Set HOBr to equilibrium when necessary:
         if(idx == n_HOBr.and.(-dest(igas,L) >= y(nn_HOBr,L).or.
     &     chemrate(rrbi%BrO_HO2__HOBr_O2,L) > 0.5d0*y(nn_BrOx,L)))then
           rnewval=(rr(rrbi%BrO_HO2__HOBr_O2,L)*y(nBrO,L)*y(nHO2,L))/
     &     (zj(L,rj%HOBr__Br_OH)+chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_HOBr,L))
           if(changeL(L,idx) > 0.5d0*y(nBrO,L))changeL(L,idx)=
     &     0.5d0*y(nBrO,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve BrOx with respect to HOBr:
         if(idx == n_BrOx.and.(-dest(nn_HOBr,L) >= y(nn_HOBr,L).or.
     &      chemrate(rrbi%BrO_HO2__HOBr_O2,L) > 0.5d0*y(nn_BrOx,L)))then
           rnewval=(rr(rrbi%BrO_HO2__HOBr_O2,L)*y(nBrO,L)*y(nHO2,L))/
     &     (zj(L,rj%HOBr__Br_OH)+chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_HOBr,L))
           if(changeX > 0.5d0*y(nBrO,L))changeX=0.5d0*y(nBrO,L)
           changeL(L,idx)=changeL(L,idx)-
     &     changeX*conc2mass
         end if

c Set BrONO2 to equilibrium when necessary:
         if(idx == n_BrONO2.and.(-dest(igas,L) >= y(nn_BrONO2,L).or.
     &      chemrate(rrtri%BrO_NO2__BrONO2_M,L)>0.5d0*y(nn_BrOx,L)))then
           rnewval=(rr(rrtri%BrO_NO2__BrONO2_M,L)*y(nBrO,L)*y(nNO2,L))
     &       /(zj(L,rj%BrONO2__BrO_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_BrONO2,L))
           if(changeL(L,idx) > 0.5d0*y(nBrO,L))changeL(L,idx)=
     &     0.5d0*y(nBrO,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve BrOx with respect to BrONO2:
         if(idx == n_BrOx.and.(-dest(nn_BrONO2,L) >= y(nn_BrONO2,L).or.
     &      chemrate(rrtri%BrO_NO2__BrONO2_M,L)>0.5d0*y(nn_BrOx,L)))then
           rnewval=(rr(rrtri%BrO_NO2__BrONO2_M,L)*y(nBrO,L)*y(nNO2,L))
     &       /(zj(L,rj%BrONO2__BrO_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_BrONO2,L))
           if(changeX > 0.5d0*y(nBrO,L))changeX=0.5d0*y(nBrO,L)
           changeL(L,idx)=changeL(L,idx)-changeX*
     &     conc2mass
         end if

c Conserve NOx with respect to BrONO2:
         if(idx == n_NOx.and.(-dest(nn_BrONO2,L) >= y(nn_BrONO2,L).or.
     &      chemrate(rrtri%BrO_NO2__BrONO2_M,L)>0.5d0*y(nn_BrOx,L)))then
           rnewval=(rr(rrtri%BrO_NO2__BrONO2_M,L)*y(nBrO,L)*y(nNO2,L))
     &       /(zj(L,rj%BrONO2__BrO_NO2)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_BrONO2,L))
           if(changeX > 0.5d0*y(nBrO,L))changeX=0.5d0*y(nBrO,L)
           changeL(L,idx)=changeL(L,idx)-changeX*
     &     conc2mass
         end if

c Set ClONO2 to equilibrium when necessary:
         if(idx == n_ClONO2.and.(-dest(igas,L) >= y(nn_ClONO2,L).or.
     &      chemrate(rrtri%ClO_NO2__ClONO2_M,L)>0.8d0*y(nn_ClOx,L)))then
           rnewval=(rr(rrtri%ClO_NO2__ClONO2_M,L)*y(nClO,L)*y(nNO2,L))
     &       /(zj(L,rj%ClONO2__Cl_NO3)
     &         +rr(rrbi%ClONO2_O__ClO_NO3,L)*y(nO,L)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_ClONO2,L))
           if(changeL(L,idx) > 0.3d0*y(nClO,L))changeL(L,idx)=
     &     0.3d0*y(nClO,L)
           if(-changeL(L,idx) > 0.8d0*y(nn_ClONO2,L))
     &     changeL(L,idx)=-0.8d0*y(nn_ClONO2,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve ClOx with respect to ClONO2:
         if(idx == n_ClOx.and.(-dest(nn_ClONO2,L) >= y(nn_ClONO2,L).or.
     &      chemrate(rrtri%ClO_NO2__ClONO2_M,L)>0.8d0*y(nn_ClOx,L)))then
           rnewval=(rr(rrtri%ClO_NO2__ClONO2_M,L)*y(nClO,L)*y(nNO2,L))
     &       /(zj(L,rj%ClONO2__Cl_NO3)
     &         +rr(rrbi%ClONO2_O__ClO_NO3,L)*y(nO,L)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_ClONO2,L))
           if(changeX > 0.3d0*y(nClO,L))changeX=0.3d0*y(nClO,L)
           if(-changeX > 0.8d0*y(nn_ClONO2,L))changeX=
     &     -0.8d0*y(nn_ClONO2,L)
           changeL(L,idx)=changeL(L,idx)-changeX*conc2mass
         end if

c Conserve NOx with respect to ClONO2:
         if(idx == n_NOx.and.(-dest(nn_ClONO2,L) >= y(nn_ClONO2,L).or.
     &      chemrate(rrtri%ClO_NO2__ClONO2_M,L)>0.8d0*y(nn_ClOx,L)))then
           rnewval=(rr(rrtri%ClO_NO2__ClONO2_M,L)*y(nClO,L)*y(nNO2,L))
     &       /(zj(L,rj%ClONO2__Cl_NO3)
     &         +rr(rrbi%ClONO2_O__ClO_NO3,L)*y(nO,L)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_ClONO2,L))
           if(changeX > 0.3d0*y(nClO,L))changeX=0.3d0*y(nClO,L)
           if(-changeX > 0.8d0*y(nn_ClONO2,L))changeX=
     &     -0.8d0*y(nn_ClONO2,L)
           changeL(L,idx)=changeL(L,idx)-changeX*conc2mass
         end if

c Set HOCl to equilibrium when necessary:
         if(idx == n_HOCl.and.(-dest(igas,L) >= y(nn_HOCl,L).or.
     &   chemrate(rrbi%ClO_HO2__HOCl_O2,L) > y(nn_ClOx,L)))then
           rnewval=(rr(rrbi%ClO_HO2__HOCl_O2,L)*y(nClO,L)*y(nHO2,L)
     &         +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nCl2,L)*y(nOH,L))
     &       /(zj(L,rj%HOCl__OH_Cl)
     &         +rr(rrbi%O_HOCl__OH_ClO,L)*y(nO,L)
     &         +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nCl2,L)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeL(L,idx)=(rnewval-y(nn_HOCl,L))
           if(changeL(L,idx) > 0.3d0*y(nClO,L))changeL(L,idx)=
     &     0.3d0*y(nClO,L)
           changeL(L,idx)=changeL(L,idx)*conc2mass
         end if

c Conserve ClOx with respect to HOCl:
         if(idx == n_ClOx.and.(-dest(nn_HOCl,L) >= y(nn_HOCl,L)
     &   .or. chemrate(rrbi%ClO_HO2__HOCl_O2,L) > y(nn_ClOx,L)))then
           rnewval=(rr(rrbi%ClO_HO2__HOCl_O2,L)*y(nClO,L)*y(nHO2,L)
     &         +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nCl2,L)*y(nOH,L))
     &       /(zj(L,rj%HOCl__OH_Cl)
     &         +rr(rrbi%O_HOCl__OH_ClO,L)*y(nO,L)
     &         +rr(rrbi%OH_Cl2__HOCl_Cl,L)*y(nCl2,L)
     &         +chemtiny)
           if(rnewval < 1.d0)rnewval=1.d0
           changeX=(rnewval-y(nn_HOCl,L))
           if(changeX > 0.3d0*y(nClO,L))changeX=0.3d0*y(nClO,L)
           changeL(L,idx)=changeL(L,idx)-changeX*conc2mass

         end if

       end do ! L

       if(idx == n_CO)then
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_COp,
     &        prod(igas,1:maxL)*c2ml(1:maxL))
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_COd,
     &        dest(igas,1:maxL)*c2ml(1:maxL))
#ifdef TRACERS_ACETONE
       else if(idx == n_Acetone)then
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_AcetP,
     &        prod(igas,1:maxL)*c2ml(1:maxL))
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_AcetD,
     &        dest(igas,1:maxL)*c2ml(1:maxL))
#endif /* TRACERS_ACETONE */
       else if(idx == n_Ox)then
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_Oxp ,
     &        prod(igas,1:maxL)*c2ml(1:maxL))
         call inc_tajls2_column(i,j,1,maxT,maxT,jls_OxpT,
     &        prod(igas,1:maxT)*c2ml(1:maxT))
         call inc_tajls2_column(i,j,1,maxL,maxL,jls_Oxd ,
     &        dest(igas,1:maxL)*c2ml(1:maxL))
         call inc_tajls2_column(i,j,1,maxT,maxT,jls_OxdT,
     &        dest(igas,1:maxT)*c2ml(1:maxT))
       end if

      end do  ! igas ! end of TRACER LOOP -----------------

c Separate N2O change for N cons, leave out N2O->N2+O fromm cons:
      sv_changeN2O(1:maxL)=
     &  -chemrate(rrbi%N2O_O1D__NO_NO,1:maxL)
     &    *rMAbyM(1:maxL)*vol2mass(n_N2O)

c Ensure nitrogen conservation,
c (since equilibration of short lived gases may alter this):

      if(prnchg .and. J == ijlprn(2) .and. I == ijlprn(1))then
        write(out_line,*)
     &  'changes (mass) before nitrogen conservation routine'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,*) 'NOx, N2O5, HO2NO2, HNO3, PAN, AlkylNit, N2O'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,*) 'ClONO2, BrONO2'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,*) changeL(ijlprn(3),n_NOx),
     &  changeL(ijlprn(3),n_N2O5),
     &  changeL(ijlprn(3),n_HO2NO2),changeL(ijlprn(3),n_HNO3),
     &  changeL(ijlprn(3),n_PAN),changeL(ijlprn(3),n_AlkylNit)
     &  ,changeL(ijlprn(3),n_N2O)
     &  ,changeL(ijlprn(3),n_ClONO2),changeL(ijlprn(3),n_BrONO2)
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,*)
     &  'N2O change w/o rxns forming N2',sv_changeN2O(ijlprn(3))
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_HETCHEM
        write(out_line,*) 'HNO3 loss on dust replaced for cons ',
     &  (krate(ijlprn(3),1,1)*y(nn_HNO3,ijlprn(3))*dt2)
     &  *rMAbyM(ijlprn(3))
        call write_parallel(trim(out_line),crit=jay)
#endif
      end if

      do L=1,maxL ! start big L-LOOP ---------------

c First check for nitrogen loss > 100% :
        if(-changeL(L,n_NOx) > trm_col(L,n_NOx))
     &  changeL(L,n_NOx)=minKG-trm_col(L,n_NOx)
        if(-changeL(L,n_N2O5) > trm_col(L,n_N2O5))
     &  changeL(L,n_N2O5)=minKG-trm_col(L,n_N2O5)
        if(-changeL(L,n_HO2NO2) > trm_col(L,n_HO2NO2))
     &  changeL(L,n_HO2NO2)=minKG-trm_col(L,n_HO2NO2)
        if(-changeL(L,n_HNO3) > trm_col(L,n_HNO3))
     &  changeL(L,n_HNO3)=minKG-trm_col(L,n_HNO3)
        if(-changeL(L,n_PAN) > trm_col(L,n_PAN))
     &  changeL(L,n_PAN)=minKG-trm_col(L,n_PAN)
#ifdef TRACERS_dCO
        if(-changeL(L,n_d17OPAN) > trm_col(L,n_d17OPAN))
     &  changeL(L,n_d17OPAN)=minKG-trm_col(L,n_d17OPAN)
        if(-changeL(L,n_d18OPAN) > trm_col(L,n_d18OPAN))
     &  changeL(L,n_d18OPAN)=minKG-trm_col(L,n_d18OPAN)
        if(-changeL(L,n_d13CPAN) > trm_col(L,n_d13CPAN))
     &  changeL(L,n_d13CPAN)=minKG-trm_col(L,n_d13CPAN)
#endif  /* TRACERS_dCO */
        if(-changeL(L,n_AlkylNit) > trm_col(L,n_AlkylNit))
     &  changeL(L,n_AlkylNit)=minKG-trm_col(L,n_AlkylNit)
        if(-changeL(L,n_ClONO2) > trm_col(L,n_ClONO2))
     &  changeL(L,n_ClONO2)=minKG-trm_col(L,n_ClONO2)
        if(-changeL(L,n_BrONO2) > trm_col(L,n_BrONO2))
     &  changeL(L,n_BrONO2)=minKG-trm_col(L,n_BrONO2)
#ifdef TRACERS_HETCHEM
        changeL(L,n_HNO3)=changeL(L,n_HNO3)+(krate(l,1,1)
     &  *y(nn_HNO3,l)*dt2)*rMAbyM(L)*vol2mass(n_HNO3)
!       if(prnchg .and. i == ijlprn(1) .and. j == ijlprn(2)) then
!         write(out_line,*)
!    &    changeL(L,n_HNO3),krate(l,1,1),y(nn_HNO3,l)
!         call write_parallel(trim(out_line),crit=jay)
!       endif   
#endif

c Next insure balance between dNOx and sum of dOthers:
        sumN=(2.d0*changeL(L,n_N2O5))*mass2vol(n_N2O5)+
     &  (changeL(L,n_HNO3))*mass2vol(n_HNO3)+
     &  (changeL(L,n_HO2NO2))*mass2vol(n_HO2NO2)+
     &  (changeL(L,n_PAN))*mass2vol(n_PAN)+
     &  (changeL(L,n_AlkylNit))*mass2vol(n_AlkylNit)

        sumN=sumN+
     &  changeL(L,n_ClONO2)*mass2vol(n_ClONO2)+
     &  changeL(L,n_BrONO2)*mass2vol(n_BrONO2)
        dNOx=changeL(L,n_NOx)*mass2vol(n_NOx)+
     &  2.d0*sv_changeN2O(L)*mass2vol(n_N2O)
        if (prnchg.and.
     &      J==ijlprn(2).and.I==ijlprn(1).and.L==ijlprn(3)) then
          write(out_line,*)
     &    'other N changes, dNOx (less prod fm N2O) = (molec) ',
     &    sumN,dNOx
          call write_parallel(trim(out_line),crit=jay)
        end if

        ratio=-sumN/dNOx

        if(ratio <= 0.999d0 .or. ratio >= 1.001d0) then
         if(dNOx > 0.d0)then ! NOx produced (net positive change)
          if (ratio > 1.d0)then
           sumD=0.d0
c          reduce N destruction to match NOx prodcution:
           if(changeL(L,n_N2O5) < 0.d0)   sumD=sumD+
     &     2.d0*changeL(L,n_N2O5)*mass2vol(n_N2O5)
           if(changeL(L,n_HO2NO2) < 0.d0) sumD=sumD+
     &     changeL(L,n_HO2NO2)*mass2vol(n_HO2NO2)
           if(changeL(L,n_HNO3) < 0.d0)   sumD=sumD+
     &     changeL(L,n_HNO3)*mass2vol(n_HNO3)
           if(changeL(L,n_PAN) < 0.d0)    sumD=sumD+
     &     changeL(L,n_PAN)*mass2vol(n_PAN)
           if(changeL(L,n_AlkylNit) < 0.d0)sumD=sumD+
     &     changeL(L,n_AlkylNit)*mass2vol(n_AlkylNit)
           if(changeL(L,n_ClONO2) < 0.d0)sumD=sumD+
     &     changeL(L,n_ClONO2)*mass2vol(n_ClONO2)
           if(changeL(L,n_BrONO2) < 0.d0)sumD=sumD+
     &     changeL(L,n_BrONO2)*mass2vol(n_BrONO2)
           newD=(sumN/ratio)+sumD-sumN
           ratioD=newD/sumD
           if(changeL(L,n_N2O5) < 0.d0)    changeL(L,n_N2O5)=
     &     changeL(L,n_N2O5)    *ratioD
           if(changeL(L,n_HO2NO2) < 0.d0)  changeL(L,n_HO2NO2)=
     &     changeL(L,n_HO2NO2)  *ratioD
           if(changeL(L,n_HNO3) < 0.d0)    changeL(L,n_HNO3)=
     &     changeL(L,n_HNO3)    *ratioD
           if(changeL(L,n_PAN) < 0.d0)     changeL(L,n_PAN)=
     &     changeL(L,n_PAN)     *ratioD
#ifdef TRACERS_dCO
           if(changeL(L,n_d17OPAN) < 0.d0) changeL(L,n_d17OPAN)=
     &     changeL(L,n_d17OPAN)     *ratioD
           if(changeL(L,n_d18OPAN) < 0.d0) changeL(L,n_d18OPAN)=
     &     changeL(L,n_d18OPAN)     *ratioD
           if(changeL(L,n_d13CPAN) < 0.d0) changeL(L,n_d13CPAN)=
     &     changeL(L,n_d13CPAN)     *ratioD
#endif  /* TRACERS_dCO */
           if(changeL(L,n_AlkylNit) < 0.d0)changeL(L,n_AlkylNit)=
     &     changeL(L,n_AlkylNit)*ratioD
           vClONO2=changeL(L,n_ClONO2)*(1.d0-ratioD)
           if(changeL(L,n_ClONO2) < 0.d0)changeL(L,n_ClONO2)=
     &     changeL(L,n_ClONO2)*ratioD
           changeL(L,n_ClOx)=changeL(L,n_ClOx)+vClONO2*
     &     (mass2vol(n_ClONO2)*vol2mass(n_ClOx)) !ensure Cl cons
           vBrONO2=changeL(L,n_BrONO2)*(1.d0-ratioD)
           if(changeL(L,n_BrONO2) < 0.d0)changeL(L,n_BrONO2)=
     &     changeL(L,n_BrONO2)*ratioD
           changeL(L,n_BrOx)=changeL(L,n_BrOx)+vBrONO2*
     &     (mass2vol(n_BrONO2)*vol2mass(n_BrOx)) !ensure Br cons
          end if

          if (ratio <= 1.d0 .and. ratio > 0.d0)then
c          reduce NOx production to match N loss:
           changeL(L,n_NOx)=changeL(L,n_NOx)*ratio
           changeL(L,n_NOx)=changeL(L,n_NOx)-
     &     2.d0*sv_changeN2O(L)*mass2vol(n_N2O)*vol2mass(n_NOx)
          end if

         else       ! NOx destroyed (net change is negative):

          if (ratio > 1.d0)then
           sumP=0.d0
c          reduce N production to match NOx loss:
           if(changeL(L,n_N2O5) > 0.d0)    sumP=sumP+
     &     2.d0*changeL(L,n_N2O5)*mass2vol(n_N2O5)
           if(changeL(L,n_HO2NO2) > 0.d0)  sumP=sumP+
     &     changeL(L,n_HO2NO2)*mass2vol(n_HO2NO2)
           if(changeL(L,n_HNO3) > 0.d0)    sumP=sumP+
     &     changeL(L,n_HNO3)*mass2vol(n_HNO3)
           if(changeL(L,n_PAN) > 0.d0)     sumP=sumP+
     &     changeL(L,n_PAN)*mass2vol(n_PAN)
           if(changeL(L,n_AlkylNit) > 0.d0)sumP=sumP+
     &     changeL(L,n_AlkylNit)*mass2vol(n_AlkylNit)
           if(changeL(L,n_ClONO2) > 0.d0)sumP=sumP+
     &      changeL(L,n_ClONO2)*mass2vol(n_ClONO2)
           if(changeL(L,n_BrONO2) > 0.d0)sumP=sumP+
     &      changeL(L,n_BrONO2)*mass2vol(n_BrONO2)
           newP=(sumN/ratio)+sumP-sumN
           if (sumP == 0.) then
             write(out_line,*)'SUMP = 0***', sumP, L,i,j,
     &       changeL(L,n_HNO3)*mass2vol(n_HNO3)
             call write_parallel(trim(out_line),crit=.true.)
           end if
           ratioP=newP/sumP
           if(changeL(L,n_N2O5) > 0.d0)    changeL(L,n_N2O5)=
     &     changeL(L,n_N2O5)*ratioP
           if(changeL(L,n_HO2NO2) > 0.d0)  changeL(L,n_HO2NO2)=
     &     changeL(L,n_HO2NO2)*ratioP
           if(changeL(L,n_HNO3) > 0.d0)    changeL(L,n_HNO3)=
     &     changeL(L,n_HNO3)*ratioP
           if(changeL(L,n_PAN) > 0.d0)     changeL(L,n_PAN)=
     &     changeL(L,n_PAN)*ratioP
#ifdef TRACERS_dCO
           if(changeL(L,n_d17OPAN) > 0.d0) changeL(L,n_d17OPAN)=
     &     changeL(L,n_d17OPAN)*ratioP
           if(changeL(L,n_d18OPAN) > 0.d0) changeL(L,n_d18OPAN)=
     &     changeL(L,n_d18OPAN)*ratioP
           if(changeL(L,n_d13CPAN) > 0.d0) changeL(L,n_d13CPAN)=
     &     changeL(L,n_d13CPAN)*ratioP
#endif  /* TRACERS_dCO */
           if(changeL(L,n_AlkylNit) > 0.d0)changeL(L,n_AlkylNit)=
     &     changeL(L,n_AlkylNit)*ratioP
           vClONO2=changeL(L,n_ClONO2)*(1.d0-ratioP)
           if(changeL(L,n_ClONO2) > 0.d0)changeL(L,n_ClONO2)=
     &     changeL(L,n_ClONO2)*ratioP
           changeL(L,n_ClOx)=changeL(L,n_ClOx)+vClONO2*
     &     (mass2vol(n_ClONO2)*vol2mass(n_ClOx)) !ensure Cl cons
           vBrONO2=changeL(L,n_BrONO2)*(1.d0-ratioP)
           if(changeL(L,n_BrONO2) > 0.d0)changeL(L,n_BrONO2)=
     &     changeL(L,n_BrONO2)*ratioP
           changeL(L,n_BrOx)=changeL(L,n_BrOx)+vBrONO2*
     &     (mass2vol(n_BrONO2)*vol2mass(n_BrOx)) !ensure Br cons
          end if

          if (ratio <= 1.d0 .and. ratio > 0.d0)then
c          reduce NOx destruction to match N production:
           changeL(L,n_NOx)=changeL(L,n_NOx)*ratio
           changeL(L,n_NOx)=changeL(L,n_NOx)-
     &     2.d0*sv_changeN2O(L)*mass2vol(n_N2O)*vol2mass(n_NOx)
          end if
         end if
#ifdef TRACERS_HETCHEM
         changeL(L,n_HNO3)=changeL(L,n_HNO3)-(krate(l,1,1)
     &   *y(nn_HNO3,l)*dt2)*rMAbyM(L)*vol2mass(n_HNO3)
#ifdef TRACERS_NITRATE
         changeL(L,n_N_d1)=changeL(L,n_N_d1)+(krate(l,2,1)
     &   *y(nn_HNO3,l)*dt2)*rMAbyM(L)*vol2mass(n_HNO3)
         changeL(L,n_N_d2)=changeL(L,n_N_d2)+(krate(l,3,1)
     &   *y(nn_HNO3,l)*dt2)*rMAbyM(L)*vol2mass(n_HNO3)
         changeL(L,n_N_d3)=changeL(L,n_N_d3)+(krate(l,4,1)
     &   *y(nn_HNO3,l)*dt2)*rMAbyM(L)*vol2mass(n_HNO3)
#endif  /* TRACERS_NITRATE */
#endif  /* TRACERS_HETCHEM */

        end if ! skipped section above if ratio very close to one

        if (prnchg.and.
     &      J==ijlprn(2).and.I==ijlprn(1).and.L==ijlprn(3)) then
          write(out_line,*) 'ratio for conservation =',ratio
          call write_parallel(trim(out_line),crit=jay)
        endif

c       Calculate NOx and Ox changes due to atomic nitrogen
c       produced by SRB photlysis (SF2 is NO + hv rate) :
        bytl=1.d0/tl(L)
c       rxnN1=3.8d-11*exp(85d0*bytl)*y(nOH,L)
        ! that's N+OH->NO+H, not in JPL (rates from IUPAC 1989)
        rxnN2=1.5d-11*exp(-3600.d0*bytl)*y(nO2,L) ! N+O2->NO+O
        rxnN3=5.8d-12*exp(220.d0*bytl)*y(nNO2,L)  ! N+NO2->N2O+O
        rxnN4=2.1d-11*exp(100.d0*bytl)*y(nNO,L)   ! N+NO->N2+O
        NprodOx=2.0d0*SF2(L)*y(nNO,L)*dt2
        NlossNOx=3.0d1*NprodOx*(rxnN3+rxnN4)/(rxnN2+rxnN3+rxnN4)
        changeL(L,n_NOx)=changeL(L,n_NOx)-NlossNOx
     &  *(rMAbyM(L))*vol2mass(n_NOx)
        conc2mass=rMAbyM(L)*vol2mass(n_Ox)
        changeL(L,n_Ox)=changeL(L,n_Ox)+NprodOx*conc2mass
        if(NprodOx <  0.) then ! necessary?
          NprodOx_pos(l) = 0.
          NprodOx_neg(l) = NprodOx*conc2mass
          ! ACCMIP diag:
          TAIJLS(I,J,L,ijlt_Oxd)=TAIJLS(I,J,L,ijlt_Oxd)+NprodOx
     *         *cpd/DTsrc
        else
          NprodOx_neg(l) = 0.
          NprodOx_pos(l) = NprodOx*conc2mass
          ! ACCMIP diag:
          TAIJLS(I,J,L,ijlt_Oxp)=TAIJLS(I,J,L,ijlt_Oxp)+NprodOx
     *         *cpd/DTsrc
        end if
        if (prnchg.and.
     &      J==ijlprn(2).and.I==ijlprn(1).and.L==ijlprn(3)) then
          write(out_line,*) 'NOx loss & Ox gain due to rxns  w/ N '
     &    ,NlossNOx,NprodOx
          call write_parallel(trim(out_line),crit=jay)
        end if

      end do ! end big L loop -----------------

c     In the stratosphere, calculate ozone change due to rxn with atomic H:
      if(prnchg.and.J==ijlprn(2).and.I==ijlprn(1)) then
        write(out_line,*) 'Ox loss due to rxns  w/ H : L, OxlossbyH(L)'
        call write_parallel(trim(out_line),crit=jay)
      end if
      do L=1,maxL
        if(OxlossbyH(L)<y(nn_Ox,L))dest(nn_Ox,L)=
     &  dest(nn_Ox,L)-OxlossbyH(L)
        if(prnchg.and.J==ijlprn(2).and.I==ijlprn(1)) then 
          write(out_line,'(i3,1X,E20.5)') L,OxlossbyH(L)
          call write_parallel(trim(out_line),crit=jay)
        end if
      end do
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_Oxd ,
     &     NprodOx_neg)
      call inc_tajls2_column(i,j,1,maxT,maxL,jls_OxdT,
     &     NprodOx_neg)
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_Oxp ,
     &     NprodOx_pos)
      call inc_tajls2_column(i,j,1,maxT,maxL,jls_OxpT,
     &     NprodOx_pos)

      ! We USED TO remove here some of the HNO3 formed heterogeneously,
      ! as it doesn't come back to the gas phase.

c Print chemical changes in a particular grid box if desired:
      if(prnchg .and. J==ijlprn(2) .and. I==ijlprn(1))then
        do igas=1,ntm_chem
          idx=igas+ntm_chem_beg-1
          changeA=changeL(ijlprn(3),idx)*y(nM,ijlprn(3))*mass2vol(idx)*
     &    byma(ijlprn(3))
          if(y(igas,ijlprn(3)) == 0.d0)then
            write(out_line,156) trchemname(igas),': ',changeA,
     &                          ' molecules;  y=0'
            call write_parallel(trim(out_line),crit=jay)
          else
            write(out_line,155)trchemname(igas),': ',changeA
     &      ,' molecules produced; ',
     &      (100.d0*changeA)/y(igas,ijlprn(3)),' percent of'
     &      ,y(igas,ijlprn(3)),'(',1.d9*y(igas,ijlprn(3))
     &      /y(nM,ijlprn(3)),' ppbv)'
            call write_parallel(trim(out_line),crit=jay)
          end if
        end do ! igas
        write(out_line,155) trchemname(nH2O),': ',
     &  changeH2O(ijlprn(3)),' molecules produced; ',
     &  (100*changeH2O(ijlprn(3)))/y(nH2O,ijlprn(3)),' percent of',
     &  y(nH2O,ijlprn(3)),'(',1.d6*y(nH2O,ijlprn(3))/y(nM,ijlprn(3)),
     &  ' ppmv)'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' CH3O2   :',yCH3O2(ijlprn(3),I,J),(yCH3O2(ijlprn(3),I,J)/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_dCO
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' dCH317O2:',ydCH317O2(ijlprn(3),I,J),(ydCH317O2(ijlprn(3),I,J)/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' dCH318O2:',ydCH318O2(ijlprn(3),I,J),(ydCH318O2(ijlprn(3),I,J)/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d13CH3O2:',yd13CH3O2(ijlprn(3),I,J),(yd13CH3O2(ijlprn(3),I,J)/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#endif  /* TRACERS_dCO */
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' C2O3    :',y(nC2O3,ijlprn(3)),(y(nC2O3,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_dCO
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' dC217O3 :',y(ndC217O3,ijlprn(3)),(y(ndC217O3,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' dC218O3 :',y(ndC218O3,ijlprn(3)),(y(ndC218O3,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d13C2O3 :',y(nd13C2O3,ijlprn(3)),(y(nd13C2O3,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#endif  /* TRACERS_dCO */
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' XO2     :',y(nXO2,ijlprn(3)),(y(nXO2,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,
     &  ' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' XO2N    :',y(nXO2N,ijlprn(3)),(y(nXO2N,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' RXPAR   :',y(nRXPAR,ijlprn(3)),(y(nRXPAR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_dCO
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d13CXPAR:',y(nd13CXPAR,ijlprn(3)),(y(nd13CXPAR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#endif  /* TRACERS_dCO */
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' Aldehyde:',y(nAldehyde,ijlprn(3)),(y(nAldehyde,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_dCO
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d17Oald :',y(nd17Oald,ijlprn(3)),(y(nd17Oald,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d18Oald :',y(nd18Oald,ijlprn(3)),(y(nd18Oald,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d13Cald :',y(nd13Cald,ijlprn(3)),(y(nd13Cald,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#endif  /* TRACERS_dCO */
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' ROR     :',y(nROR,ijlprn(3)),(y(nROR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#ifdef TRACERS_dCO
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d17OROR :',y(nd17OROR,ijlprn(3)),(y(nd17OROR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d18OROR :',y(nd18OROR,ijlprn(3)),(y(nd18OROR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        write(out_line,'(a10,58x,e13.3,6x,f10.3,a5)')
     &  ' d13CROR :',y(nd13CROR,ijlprn(3)),(y(nd13CROR,ijlprn(3))/
     &  y(nM,ijlprn(3)))*1.d9,' ppbv'
        call write_parallel(trim(out_line),crit=jay)
#endif  /* TRACERS_dCO */

      end if  ! end this section of chem diags 

C Tracer masses & slopes are updated in apply_tracer_3Dsource,
C so here, just saved in changeL:
      do igas=1,ntm_chem
       idx=igas+ntm_chem_beg-1
       do L=1,maxL
        ! In previous model versions changeL of > 1.d20 kg
        ! (changeL USED TO be in kg) would trigger setting of
        ! changeL to 0 and a print to fort.99. This went back
        ! to pre-modelE days. Let's try stopping the model for
        ! values > 1.d12 kg m-2 instead:
        if(changeL(L,idx) > 1.d12) then
          WRITE(out_line,*)
     &    'Large changeL in chemstep: I,J,L,idx,changeL'
     &    ,I,J,L,idx,changeL(L,idx)
          call write_parallel(trim(out_line),crit=.true.)
          call stop_model("Large changeL in chemstep.",255)
        end if
        if(-changeL(L,idx) > trm_col(L,idx)) THEN
          if(prnchg)then
            WRITE(out_line,*)
     &      'change > mass, so use 95%: I,J,L,igas,change'
     &      ,I,J,L,igas,changeL(L,idx)
            call write_parallel(trim(out_line),unit=99,crit=.true.)
          end if
          changeL(L,idx) = -0.95d0*trm_col(L,idx)
        end if
       end do    ! L
      end do     ! igas

C**** special diags not associated with a particular tracer

      DO L=1,maxL
        conOH(L) = 0.
        if (y(nOH,L) > 0.d0 .and. y(nOH,L) < 1.d20)then
          conOH(l) = y(nOH,L)
          ! ACCMIP diag:
          TAIJLS(I,J,L,ijlt_OHvmr)=TAIJLS(I,J,L,ijlt_OHvmr)+y(nOH,L)
     &                                             /y(nM,L)
          TAIJLS(I,J,L,ijlt_OHconc)=TAIJLS(I,J,L,ijlt_OHconc)+y(nOH,L)
        end if
        if (y(nHO2,L) > 0.d0 .and. y(nHO2,L) < 1.d20)
     &       TAIJLS(I,J,L,ijlt_HO2)=TAIJLS(I,J,L,ijlt_HO2)+y(nHO2,L)
        conClO(l) = 0.
        if (y(nClO,L) > 0.d0 .and. y(nClO,L) < 1.d20)
     &       conClO(l) = y(nClO,L)/y(nM,L)
        conH2O(l) = 0.
        if (y(nH2O,L) > 0.d0 .and. y(nH2O,L) < 1.d20)
     &       conH2O(l) = y(nH2O,L)/y(nM,L)
      END DO
      conOH = conOH*ma(1:maxL) ! for mass weighting
      conClO = conClO*ma(1:maxL) ! for mass weighting
      conH2O = conH2O*ma(1:maxL) ! for mass weighting
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_OHcon,conOH)
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_ClOcon,conClO)
      call inc_tajls2_column(i,j,1,maxL,maxL,jls_H2Ocon,conH2O)
      CALL INC_TAJLS2(I,J,1,jls_day,1.d0)

      deallocate( rMAbyM )
      deallocate( sv_changeN2O )
      deallocate( changeH2O )
      deallocate( dQ )
      deallocate( dQM )
      if (save_dQ_for_NINT==1) then
        deallocate( dQo1d )
        deallocate( dQoh )
        deallocate( dQcl )
        deallocate( dQsf3 )
      end if
      deallocate( fraQ2 )
      deallocate( c2ml )
      deallocate( conOH )
      deallocate( conClO )
      deallocate( conH2O )
      deallocate( NprodOx_pos )
      deallocate( NprodOx_neg )

 155  format(1x,a8,a2,e13.3,a21,f10.0,a11,2x,e13.3,3x,a1,f12.5,a6)
 156  format(1x,a8,a2,e13.3,a16)

      return
      end SUBROUTINE chemstep



      SUBROUTINE rates(maxL,I,J)
!@sum rates calculate reaction rates with present concentrations
!@auth Drew Shindell (modelEifications by Greg Faluvegi)
c
C**** GLOBAL parameters and variables:

      USE TRCHEM_Shindell_COM, only: n_rx,chemrate,photrate,rr,y,nn,dt2,
     &                          ks,ny,dest,prod,n_het,n_rj
      use photolysis, only: zj

      IMPLICIT NONE

C**** Local parameters and variables and arguments:
!@var kalt local dummy L-loop variable
!@var maxL passed highest chemistry level
!@var ireac,igas dummy loop variables
!@var I,J passed horizontal spatial indicies
      INTEGER :: kalt, ireac, igas, maxL
      INTEGER, INTENT(IN) :: I,J

C Set up rates:
      do kalt=1,maxL
        do ireac=1,n_rx-n_het       ! non-heterogeneous
          chemrate(ireac,kalt)=rr(ireac,kalt)*y(nn(1,ireac),kalt)*
     &    y(nn(2,ireac),kalt)*dt2
        end do
        do ireac=n_rx-n_het+1,n_rx    ! heterogeneous
          chemrate(ireac,kalt)=rr(ireac,kalt)*y(nn(1,ireac),kalt)*dt2
        end do
        do ireac=1,n_rj          ! photolysis
          photrate(ireac,kalt)=zj(kalt,ireac)*y(ks(ireac),kalt)*dt2
        end do

c Initialize change arrays:
        do igas=1,ny
          dest(igas,kalt)=0.d0
          prod(igas,kalt)=0.d0
        end do
      end do
      return
      end SUBROUTINE rates



      SUBROUTINE chem1(kdnr,maxL,numeL,n_rr,nn,npdnrs,rrate,proddest,
     &                 multip)
!@sum chem1 calculate chemical destruction/production
!@auth Drew Shindell (modelEifications by Greg Faluvegi)

C**** GLOBAL parameters and variables:

      USE TRCHEM_Shindell_COM, only:  p_1, nc, ny, numfam,nfam
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
      use OldTracer_mod, only: is_dCO_tracer
#endif  /* TRACERS_dCO || TRACERS_dCOlite */

      IMPLICIT NONE

C**** Local parameters and variables and arguments:
!@var maxL passed highest chemistry level
!@var numeL first index of nn array, 1 for single reactant (photolytic
!@+   destruction) 2 for all other cases, meaning either two reactants or
!@+   two products
!@var kdnr kdnr,kpnr,kds, or kps    passed from chemstep
!@var nn nn,nnr,ks, or kss          passed from chemstep
!@var npdnrs ndnr,npnr,nds, or nps    passed from chemstep.
!@+   npdnrs(ireac) gives reaction index number as found in JPLRX or JPLPH
!@var rrate rrate or photrate passed from chemstep
!@var proddest dest or prod             passed from chemstep
!@var multip -1 for destruction, +1 for production
!@var igas index of tracer, as defined in the trchemname array
!@var ireac index of reaction per tracer. Starts from 1 and increases
!@+   every time a tracer has a reaction. E.g.: tracer a has 3 destruction
!@+   reactions, and tracer b has 4; ireac is [123] for a and [4567] for b.
!@+   Production and destruction are tracked separately.
!@var i,dk,nl dummy variable
      INTEGER ireac,igas,i,dk,nl
      INTEGER, INTENT(IN)            :: maxL,numeL,n_rr,multip
      INTEGER, DIMENSION(nc)         :: kdnr
      INTEGER, DIMENSION(numeL,n_rr) :: nn ! automatic array
      INTEGER, DIMENSION(p_1*n_rr)   :: npdnrs
      REAL*8,  DIMENSION(n_rr,maxL)  :: rrate ! automatic array
      REAL*8,  DIMENSION(ny,maxL)    :: proddest ! automatic array
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
      logical :: is_dCO_reaction
#endif  /* TRACERS_dCO || TRACERS_dCOlite */

      ireac=0
      
c Reactive families:

      do igas=1,numfam
        dk=kdnr(igas+1)-kdnr(igas)
        if(dk >= 1) then
          do i=1,dk
            ireac=ireac+1
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
            if (is_dCO_reaction(ireac,n_rr,npdnrs)) then
              if (.not.is_dCO_tracer(igas)) cycle ! do not affect chemistry
            endif
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
            do nl=1,numeL
              if(nn(nl,npdnrs(ireac)) >= nfam(igas) .and. 
     &           nn(nl,npdnrs(ireac)) < nfam(igas+1))then
                proddest(igas,1:maxL)=
     &            proddest(igas,1:maxL)+
     &            multip*rrate(npdnrs(ireac),1:maxL)
c               Save change array for individual family elements:
                proddest(nn(nl,npdnrs(ireac)),1:maxL)=
     &            proddest(nn(nl,npdnrs(ireac)),1:maxL)+
     &            multip*rrate(npdnrs(ireac),1:maxL)
              end if
            end do ! numeL
          end do  ! i
        end if
      end do      ! igas

c Individual Species:

      do igas=numfam+1,nfam(1)-1
        dk=kdnr(igas+1)-kdnr(igas)
        if(dk >= 1) then
          do i=1,dk
            ireac=ireac+1
#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
            if (is_dCO_reaction(ireac,n_rr,npdnrs)) then
              if (.not.is_dCO_tracer(igas)) cycle ! do not affect chemistry
            endif
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
            proddest(igas,1:maxL)=
     &        proddest(igas,1:maxL)+
     &        multip*rrate(npdnrs(ireac),1:maxL)
          end do
        end if
      end do

      return
      end SUBROUTINE chem1



      SUBROUTINE chem1prn(kdnr,numeL,n_rr,nn,npdnrs,rrate,
     &                    index,multip,igas,total,maxL,I,J,jay)
!@sum chem1prn for printing out the chemical reactions
!@auth Drew Shindell (modelEifications by Greg Faluvegi)

C**** GLOBAL parameters and variables:

      USE DOMAIN_DECOMP_ATM, only : write_parallel
      USE TRCHEM_Shindell_COM, only: trchemname,ijlprn,nfam,nc,numfam,
     &                               y,p_1,ks,kss

      IMPLICIT NONE

C**** Local parameters and variables and arguments:
!@var kdnr kdnr,kpnr,kds, or kps from chemstep
!@var numeL first index of nn array
!@var nn nn,nnr,ks, or kss from chemstep
!@var npdnrs ndnr,npnr,nds, or nps from chemstep
!@var rrate rrate or photrate from chemstep
!@var index passed index to know which call this is... {1,2,3,4}
!@var multip 1 for production, -1 for destruction
!@var igas passed index for gas number
!@var total dummy summation
!@var maxL highest chemistry level
!@var I,J passed horizontal spatial indicies
!@var label character string for printing
!@var irec dummy loop variables
!@var per dummy temp variable
      INTEGER, INTENT(IN) :: igas,I,J,maxL,multip,index,numeL,n_rr
      INTEGER, DIMENSION(p_1*n_rr)   :: npdnrs
      INTEGER, DIMENSION(numeL,n_rr) :: nn ! automatic array
      INTEGER, DIMENSION(nc)         :: kdnr      
      INTEGER                        :: ireac
      character*17                   :: label
      character(len=300)             :: out_line
      logical                        :: jay
      REAL*8                         :: total,per
      REAL*8, DIMENSION(n_rr,maxL)   :: rrate ! automatic array

c FAMILIES ONLY:

      if(igas <= numfam) then
        if(kdnr(igas+1)-kdnr(igas) < 1) goto 200 ! return
        do ireac=kdnr(igas),kdnr(igas+1)-1
          if(index <= 2)then 
            label=' chem reaction # '
          else
            label=' phot reaction # '
          end if
          if(nn(1,npdnrs(ireac)) >= nfam(igas) .and. 
     &    nn(1,npdnrs(ireac)) < nfam(igas+1))then
            per=0.d0
            if(y(igas,ijlprn(3)) /= 0.d0) per=multip*100.d0*
     &      rrate(npdnrs(ireac),ijlprn(3))/y(igas,ijlprn(3))
            write(out_line,177) label,npdnrs(ireac),' percent change'
     &      //' from ',trchemname(nn(1,npdnrs(ireac))),' = ',per,
     &      ' dy=',multip*rrate(npdnrs(ireac),ijlprn(3))
            call write_parallel(trim(out_line),crit=jay)
            total=total+per
          end if
          if(numeL == 2)then
            if(nn(2,npdnrs(ireac)) >= nfam(igas) .and. 
     &      nn(2,npdnrs(ireac)) < nfam(igas+1))then
              per=0.d0
              if(y(igas,ijlprn(3)) /= 0.d0) per=multip*100.d0*
     &        rrate(npdnrs(ireac),ijlprn(3))/y(igas,ijlprn(3))
              write(out_line,177) label,npdnrs(ireac),' percent change'
     &        //' from ',trchemname(nn(2,npdnrs(ireac))),' = ',per,
     &        ' dy=',multip*rrate(npdnrs(ireac),ijlprn(3))
              call write_parallel(trim(out_line),crit=jay)
              total=total+per
            end if
          end if  
        end do 
        goto 200 ! return
      end if

c INDIVIDUAL SPECIES:

      if(kdnr(igas+1)-kdnr(igas) < 1)goto 200 ! return
      do ireac=kdnr(igas),kdnr(igas+1)-1
        if(index <= 2) then
          label=' chem reaction # '
        else
          label=' phot reaction # '
        end if
c       skip same reaction if written twice:
        if (ireac > 1) then
          if (npdnrs(ireac) == npdnrs(ireac-1)) CYCLE
        end if
        if(nn(1,npdnrs(ireac)) == igas)then
          per=0.d0
          if(y(igas,ijlprn(3)) /= 0.d0) per=100.d0*multip*
     &    rrate(npdnrs(ireac),ijlprn(3))/y(igas,ijlprn(3))
          write(out_line,106) label,npdnrs(ireac),' percent change = '
     &    ,per,' dy=',multip*rrate(npdnrs(ireac),ijlprn(3))
          call write_parallel(trim(out_line),crit=jay)
          total=total+per
        end if
        if(numeL == 2)then
          if(nn(2,npdnrs(ireac)) == igas)then
            per=0.d0
            if(y(igas,ijlprn(3)) /= 0.d0) per=100.d0*multip*
     &      rrate(npdnrs(ireac),ijlprn(3))/y(igas,ijlprn(3))
            write(out_line,106) label,npdnrs(ireac),' percent change = '
     &      ,per,' dy=',multip*rrate(npdnrs(ireac),ijlprn(3))
            call write_parallel(trim(out_line),crit=jay)
            total=total+per
          end if
        end if 
      end do
 106  format(a17,i3,a18,f10.0,a4,e12.3)
 177  format(a17,i3,a21,a8,a3,f10.0,a4,e12.3)

 200  CONTINUE

      return
      end SUBROUTINE chem1prn

#if defined(TRACERS_dCO) || defined(TRACERS_dCOlite)
      logical function is_dCO_reaction(ireac,n_rr,npdnrs)
!@sum is_dCO_reaction Returns .true. if reaction ireac involves dCO tracers,
!@+                   false otherwise
!@auth Kostas Tsigaridis

      use TRCHEM_Shindell_COM, only: p_1,n_bi_dCO,n_tri_dCO,
     &                               n_rj_dCO,rj,rrbi,rrtri,n_rj
      implicit none

      integer, intent(in) :: ireac,n_rr
      integer, dimension(p_1*n_rr), intent(in) :: npdnrs
!@var dCOrrbi_i First dCO bimolecular reaction in JPLRX
!@var dCOrrbi_e Last dCO bimolecular reaction in JPLRX
!@var dCOrrtri_i First dCO trimolecular reaction in JPLRX
!@var dCOrrtri_e Last dCO trimolecular reaction in JPLRX
      integer :: dCOrrbi_i,dCOrrbi_e,dCOrrtri_i,dCOrrtri_e,
     &           dCOrj_i,dCOrj_e

#ifdef TRACERS_dCO
      dCOrrbi_i=rrbi%O1D_CH4__OH_dCH317O2
      dCOrrbi_e=rrbi%Terpenes_NO3__HO2_d13Calke
#elif defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO_bin_reprod
      dCOrrbi_i=rrbi%dC17O_OH__HO2_O2
      dCOrrbi_e=rrbi%Alkenes_O3__HCHO_d13CO
#else
      dCOrrbi_i=rrbi%CH4_OH__dC17O_M
      dCOrrbi_e=rrbi%d13CO_OH__HO2_O2
#endif  /* TRACERS_dCO_bin_reprod */
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
      if (dCOrrbi_e-dCOrrbi_i+1 /= n_bi_dCO)
     &  call stop_model('ERROR: Check the first and last dCO '//
     &                  'bimolecular reactions', 255)

#ifdef TRACERS_dCO
      dCOrrtri_i=rrtri%dC217O3_NO2__d17OPAN_M
      dCOrrtri_e=rrtri%d13C2O3_NO2__d13CPAN_M
#elif defined(TRACERS_dCOlite)
      dCOrrtri_i=0
      dCOrrtri_e=dCOrrtri_i-1
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
      if (dCOrrtri_e-dCOrrtri_i+1 /= n_tri_dCO)
     &  call stop_model('ERROR: Check the first and last dCO '//
     &                  'trimolecular reactions', 255)

#ifdef TRACERS_dCO
      dCOrj_i=rj%dHCH17O__dC17O_H2
      dCOrj_e=rj%d13Cald__HCHO_CO
#elif defined(TRACERS_dCOlite)
#ifdef TRACERS_dCO_bin_reprod
      dCOrj_i=rj%HCHO__dC17O_H2
      dCOrj_e=rj%Aldehyde__HCHO_d13CO
#else
      dCOrj_i=0
      dCOrj_e=dCOrj_i-1
#endif  /* TRACERS_dCO_bin_reprod */
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
      if (dCOrj_e-dCOrj_i+1 /= n_rj_dCO)
     &  call stop_model('ERROR: Check the first and last dCO '//
     &                  'photolysis reactions', 255)

      is_dCO_reaction=.false.
      if (maxval(npdnrs)==n_rj) then ! photolysis
        if ((npdnrs(ireac) >= dCOrj_i).and.
     &      (npdnrs(ireac) <= dCOrj_e)) then
          is_dCO_reaction=.true.
        endif
      else                           ! thermal
        if ((npdnrs(ireac) >= dCOrrbi_i).and.
     &      (npdnrs(ireac) <= dCOrrbi_e)) then
          is_dCO_reaction=.true.
        endif
        if ((npdnrs(ireac) >= dCOrrtri_i).and.
     &      (npdnrs(ireac) <= dCOrrtri_e)) then
          is_dCO_reaction=.true.
        endif
      endif

      end function is_dCO_reaction
#endif  /* TRACERS_dCO || TRACERS_dCOlite */
