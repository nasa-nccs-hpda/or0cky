#include "rundeck_opts.h"

#ifdef TRACERS_SPECIAL_Shindell
      SUBROUTINE HETCDUST(i,j)
!
! Version 1.
!
!-----------------------------------------------------------------------
!   Computation of heterogeneous reaction rates on dust aerosol surfaces
!   Susanne E Bauer, 2003
!-----------------------------------------------------------------------
      USE RESOLUTION, only : lm
      use atmcol_com, only: tl   ! layer temperature (K)
      use atmcol_com, only: pl   ! layer pressure (mb)
      use atmcol_com, only: byma ! 1/ma

      USE TRACER_COM, only: trm_col, krate, rhet
      use TRACER_COM, only: n_Clay, n_Silt1, n_Silt2, n_Silt3, ntm_clay,
     &     ntm_sil1, ntm_sil2, ntm_sil3, n_soilDust
      USE CONSTANT,   only:  lhe       ! latent heat of evaporation at 0 C
      USE CONSTANT,   only:  pi, avog, byavog, gasc
      USE DOMAIN_DECOMP_ATM, only : am_i_root
      use SpecialFunctions_mod, only: erf
      use OldTracer_mod, only: trpdens
      use trdust_mod, only : imDust, nSubClays, subClayWeights,
     &     nDustBinsFull, radiusMinerals
      use trdust_drv, only : calcSubClayWeights
      use RunTimeControls_mod, only: tracers_dust, tracers_minerals
      IMPLICIT NONE
      integer, intent(in) :: i,j
!-----------------------------------------------------------------------
!       ... Dummy arguments
!-----------------------------------------------------------------------

      integer, parameter    :: ndtr = 7  ! # dust bins for heterogenous chem.
      REAL*8, DIMENSION(lm,ndtr,rhet) :: rxtnox
      REAL*8, DIMENSION(lm,ndtr) :: dusttx,dustnc
      integer, parameter :: nhcaero_dust = nSubClays * ntm_clay +
     &     ntm_sil1 + ntm_sil2 + ntm_sil3
!-----------------------------------------------------------------------
!       ... Look up variables
!-----------------------------------------------------------------------
      integer, parameter :: klo = 1000
      integer :: ip,imd,np1,np2,nh1,nh2
      real( kind=8 ) :: klook,phelp
      real( kind=8 ) :: look_p, look_t,hx,px,hp1,hp2
!-----------------------------------------------------------------------
!       ... Local variables
!-----------------------------------------------------------------------

      integer :: k, l, ii, il, jj, ll, nc, nd, ndc
      integer, parameter :: ktoa = 300
! 1-SO2
!@param alph  uptake coeff for HNO3,N2O5,NO3 (only the one for HNO3 used)
      real( kind=8 ), parameter :: alph(rhet)=(/ 0.0001d0, 0.001d0,
     &     0.003d0 /)
      real( kind=8 ), parameter :: mQ(rhet)=(/ 0.063d0, 0.108d0, 0.062d0
     &     /)
      real( kind=8 ), parameter :: xx    = 0.d0  !correction factor anisotropic movement
      real( kind=8 ), parameter :: Bolz  = 1.3807d-23 !Boltzmann kg m2/s2 K molec.
      real( kind=8 ), parameter :: Mgas  = 28.97d0 /1000.d0 ! Molekular Gewicht Luft
      real( kind=8 ), parameter :: Diaq  = 4.5d-10      ! m Molecul Diameter
C**** functions
      real*8 :: QSAT,temp

      real( kind=8 ) :: Kn(rhet), Mdc(rhet), Kdj(rhet)
      real( kind=8 ) :: lamb(rhet), wrk(rhet),VSP(rhet)
      real( kind=8 ) :: lsig0,drada,dn,Roh

      logical, save  :: enteredb = .false.
      real( kind=8 ), save, dimension(ktoa) :: rada
      real( kind=8 ), save     :: lookS(11,klo,rhet),Rrange,md_look(klo)

!-----------------------------------------------------------------
!     Dust variables
!-----------------------------------------------------------------

!@var ntix_dust  index for mapping advected dust bins onto ndtr dust bins in
!@+     heterogeneous chemistry
      integer, dimension( nhcaero_dust ) :: ntix_dust
!@var ntm_array array for size of dust bins
      integer :: ntm_array( ndtr-nSubClays+1 )
!@var dradi  dust radii used in heterogeneous chemistry [m]
      real( kind=8 ), dimension( nDustBinsFull ) :: dradi
!@var rop  dust particle density [kg/m^3]
      real( kind=8 ), dimension( ndtr ) :: rop
!@var wttr_dust  weighting array for mass in dust bins
      real( kind=8 ), dimension( nhcaero_dust ) :: wttr_dust

!-----------------------------------------------------------------
!    1000 Intervals for Radius = 0.01ym ->10ym
!-----------------------------------------------------------------
!      Integration of radius:  0.01 ym to 10 ym
       rada(1) = 0.01d-6    ! smallest radius
       drada   = 0.1d-6     ! delta radius


      if (.not. enteredb) then
      enteredb = .true.
      if (am_i_root())
     &  PRINT*, 'CALCULATING LOOK UP TABLE FOR HETEROGENEOUS CHEMISTY'
      DO ii   = 2, ktoa
      rada(ii) = rada(ii-1) + drada
      END DO

c      enteredb = .true.

      lsig0 = LOG(2.d0)

!-----------------------------------------------------------------
!     HNO3 + DUSTM =>    Dust Aerosol Reaction
!-----------------------------------------------------------------


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!                         LOOK UP TABLE
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       md_look(1) = 1.d-10       ! smallest md
       Rrange=5.d-8

       DO ii   = 2, klo
       md_look(ii) = md_look(ii-1) +  Rrange
       END DO

      DO il  = 1,1 ! rhet  ! no loop over rhet, only HNO3 uptake
      wrk(il)=  (mQ(il) + Mgas) / mQ(il)

      DO ip  = 1, 11  !pressure from 1000 to 0 hPa

      look_p=max(0.001d0,1.1d0-ip*0.1d0)         !atmosphere (minimum is 1 hPa)
      look_t=max(210.d0,288.d0*(look_p/1.d0)**((1.40d0-1d0)/1.40d0))
      look_p=look_p * 100000.d0              ! pressure in Pa

       Roh    = look_p / look_t / 287.d0
C Molecular diffusion coefficient for a trace gas in air [ m/s ]
       Mdc(il)  = 3.d0 / (8.d0* Avog * Roh * (Diaq**2))
       Mdc(il)  = Mdc(il) * SQRT( ((gasc*look_t*Mgas)/(2.d0*pi))
     &      *wrk(il))
C thermal velocity of a trace gas molecule [m/s2]
       VSP(il)  = SQRT((8.d0 * Bolz * look_t)/(Pi * mQ(il) * byAvog))
C lamb  mean free pathway  [m]
       lamb(il)   = 3.d0 * Mdc(il)/VSP(il)
C Loop over radius

      DO imd  = 1,klo
       lookS(ip,imd,il) = 0.d0
      DO k = 1,ktoa-1
C Knudsen Number
       Kn(il)= lamb(il) / rada(k) ! Radius in [m]
C Mass Transfer Coefficient
       Kdj(il) =(4.d0 * pi * rada(k) * Mdc(il)) / (1.d0 + Kn(il) * (xx +
     &      4d0 *(1.d0- alph(il))/(3.d0 *alph(il))))
C Number distribution
       dn=abs(erf(log( rada(k)/md_look(imd)) / lsig0
     .       /sqrt(2.0d0))-erf(log(rada(k+1)/
     .       md_look(imd)) / lsig0 /sqrt(2.0d0)))/2.d0
C Net removal rate [s-1]
       lookS(ip,imd,il)= lookS(ip,imd,il) + Kdj(il)  * dn
      END DO              !radius loop
      END DO              !median diameter loop
      END DO              !pressure loop
      END DO              !reaction loop

      ENDIF


c--------------------------------------------------------------

c  Or use online dust

      if ( tracers_dust .or. tracers_minerals ) then

      ntm_array = (/ nSubClays*ntm_clay, ntm_sil1, ntm_sil2, ntm_sil3 /)

      ntix_dust = (/ ( ( ii, ii = n_SoilDust,n_SoilDust+ntm_clay-1 ), jj
     &     = 1,nSubClays), ( ii, ii = n_SoilDust+ntm_clay,n_SoilDust+
     &     ntm_clay+ntm_sil1+ntm_sil2+ntm_sil3-1 ) /)

      dradi = (/ ( radiusMinerals( ii ), ii=1,nDustBinsFull ) /) * 1.d-6 ![um]->[m]

      if ( tracers_dust ) then
        rop = (/ ( trpdens( n_clay ), ii = 1,nSubClays ), trpdens(
     &       n_silt1 ), trpdens( n_silt2 ), trpdens( n_silt3 ) /)

      else

c for dust with mineral speciation set dust tracer density for clay
c (1-4) and silt size classes (5-7) to some representative value to be
c used in calculations
        rop = (/ ( 2.5d3, ii = 1,nSubClays ), 2.65d3, 2.65d3, 2.65d3 /)

      end if

      if ( tracers_minerals .or. imDust >= 4 ) call calcSubClayWeights

      wttr_dust = (/ ( ( subClayWeights( jj, ii ), ii = 1,ntm_clay ), jj
     &     = 1,nSubClays ), ( 1.d0, ii = 1,ntm_sil1+ntm_sil2+ntm_sil3 )
     &     /)

c sum over all dust types in a size bin to get total dust for each size
c bin
      dusttx = 0.d0
      do nd = 1,nSubClays
        do nc = 1,ntm_clay
          ndc = nc + ntm_clay * (nd - 1)
          do l  = 1,lm
            dusttx( l, nd ) = dusttx( l, nd ) + wttr_dust( ndc ) *
     &           trm_col( l, ntix_dust( ndc ) ) * byMA( l )
          end do                ! l
        end do                  ! nc
      end do                    ! nd
      ndc = ntm_array( 1 ) + 1
      do nd = nSubClays+1,ndtr
        do nc = 1,ntm_array(nd-nSubClays+1)
          do l  = 1,lm
            dusttx( l, nd ) = dusttx( l, nd ) + wttr_dust( ndc ) *
     &           trm_col( l, ntix_dust( ndc ) ) * byMA( l )
          end do                ! l
          ndc = ndc + 1
        end do                  ! nc
      end do                    ! nd

c--------------------------------------------------------------
c--------------------------------------------------------------

c INTERPOLATION FROM LOOK UP TABLES

C Net removal rates [s-1]
        krate(:,:,:) = 0.d0
        rxtnox(:,:,:)=0.d0
      DO il = 1,1 !rhet ! Loop over het reactions
      DO nd = 1,ndtr    ! Loop over dust tracers
      DO l  = 1,lm

       if(dusttx(l,nd).GT.0.d0) then
c number concentration
        dustnc(l,nd) = dusttx(l,nd)/pi*0.75d0/rop(nd)/Dradi(nd)**3
       if(dustnc(l,nd).GT.0.d0) then
c pressure
        phelp = Min (99999d0, pl(l)*100d0)
c pressure interpolation
        np1=min(11,1+nint((10.d0-phelp/10000.d0)-0.499d0))  !pressure
        np1=max(1,np1)
        np2=min(11,np1+1)
c radii interpolation
        nh1=max( 1, nint( (dradi( nd ) / Rrange)+0.499d0 ) ) !median diameter
        nh1=min(klo,nh1)
        nh2=min(klo,nh1+1)
        px=((11d0-np1)*10000.d0- phelp)/10000.d0
        hx=((nh1*Rrange+md_look(1)) - dradi( nd )) / Rrange
        hp1=px*lookS(np2,nh1,il)+(1.d0-px)*lookS(np1,nh1,il)
        hp2=px*lookS(np2,nh2,il)+(1.d0-px)*lookS(np1,nh2,il)
        klook=hx*hp1+(1.d0-hx)*hp2

        if  (dustnc(l,nd).gt.1000.d0.and.dustnc(l,nd).lt.(1.d30))then
        rxtnox(l,nd,il) = klook* dustnc(l,nd)
     .              / (287.054d0 * tl(l) / (pl(l)*100.d0))
        else
        rxtnox(l,nd,il) = 0.d0
        endif

        else
        rxtnox(l,nd,il) = 0.d0
        endif
        ENDIF

      ENDDO ! l
      ENDDO ! nd

      DO nd = 1,ndtr-1  !1,ndtr
        krate(:,1,il) = krate(:,1,il) + rxtnox(:,nd,il) ! Total HNO3 loss on all dust types
      ENDDO
      do nd = 1,nSubClays
        krate( :, 2, il ) = krate( :, 2, il ) + rxtnox( :, nd, il ) ! Total formation on all clays
      end do
        krate(:,3,il) = rxtnox(:,nSubClays+1,il) ! Formation on silt1
        krate(:,4,il) = rxtnox(:,nSubClays+2,il) ! Formation on silt2
!        krate(:,5,il) = rxtnox(:,nSubClays+3,il) ! Formation on silt3
      ENDDO ! il

      end if                    ! tracers_dust .or. tracers_minerals

      return
      end subroutine
#endif


#if (defined TRACERS_AEROSOLS_Koch)  || (defined TRACERS_TOMAS)
      SUBROUTINE SULFDUST(i,j)
!
! Version 1.   (version 2 needs to be written... without integration over ndr)
!
!-----------------------------------------------------------------------
!   Computation of heterogeneous reaction rates on dust aerosol surfaces
!   Susanne E Bauer, 2003
!-----------------------------------------------------------------------
      USE RESOLUTION, only : lm
      use atmcol_com, only: tl   ! layer temperature (K)
      use atmcol_com, only: rhl  ! layer relative humidity (0-1)
      use atmcol_com, only: pl   ! layer pressure (mb)
      use atmcol_com, only: byma ! 1/ma

      USE TRACER_COM, only: trm_col, rxts, rhet
      use TRACER_COM, only: n_Clay, n_Silt1, n_Silt2, n_Silt3, ntm_clay,
     &     ntm_sil1, ntm_sil2, ntm_sil3, n_SoilDust
      use TRACER_COM, only: rxts1, rxts2, rxts3, rxts4
      USE CONSTANT,   only:  lhe       ! latent heat of evaporation at 0 C
      USE CONSTANT,   only:  pi, avog, byavog, gasc
      USE DOMAIN_DECOMP_ATM, only : am_i_root
      use SpecialFunctions_mod, only: erf
      use OldTracer_mod, only: trpdens
      use trdust_mod, only : imDust, nSubClays, subClayWeights,
     &     nDustBinsFull, radiusMinerals
      use trdust_drv, only : calcSubClayWeights
      use RunTimeControls_mod, only: tracers_dust, tracers_minerals
      IMPLICIT NONE
      integer, intent(in) :: i,j
!-----------------------------------------------------------------------
!       ... Dummy arguments
!-----------------------------------------------------------------------
      integer, parameter     :: ndtr = 7  ! # dust bins for sulfate on dust
      REAL*8, DIMENSION(lm,ndtr) :: rxt,dusttx,dustnc
      integer, parameter :: nhcaero_dust = nSubClays * ntm_clay +
     &     ntm_sil1 + ntm_sil2 + ntm_sil3
!-----------------------------------------------------------------------
!       ... Look up variables
!-----------------------------------------------------------------------
      integer, parameter :: klo = 1000
      integer :: ip,imd,np1,np2,nh1,nh2
      real( kind=8 ) :: klook,phelp
      real( kind=8 ) :: look_p, look_t,hx,px,hp1,hp2
!-----------------------------------------------------------------------
!       ... Local variables
!-----------------------------------------------------------------------
      integer :: k, l, ii, jj, ll, nc, nd, ndc
      integer, parameter :: ktoa = 300
! 1-SO2
c      real, parameter :: alph1  = 0.0001 !uptake coeff of Rossi EPFL (independent of humidity)
      real( kind=8 ), parameter :: alph1  = 0.000001d0 !uptake coeff for SO2: RH < 60 %
      real( kind=8 ), parameter :: alph2  = 0.0001d0    !uptake coeff for SO2: RH > 60 %
      real( kind=8 ), parameter :: mQ1    = 64.d0/1000.d0    ! kg/mol SO2
      real( kind=8 ), parameter :: xx    = 0.d0  !correction factor anisotropic movement
      real( kind=8 ), parameter :: Bolz  = 1.3807d-23 !Boltzmann kg m2/s2 K molec.
      real( kind=8 ), parameter :: Mgas  = 28.97d0 /1000.d0 ! Molekular Gewicht Luft
      real( kind=8 ), parameter :: Diaq  = 4.5d-10      ! m Molecul Diameter
C**** functions
      real*8 :: QSAT,temp

      real( kind=8 ) :: Kn(rhet), Mdc(rhet), Kdj(2)
      real( kind=8 ) :: lamb(rhet), wrk(rhet),VSP(rhet)
      real( kind=8 ) :: lsig0,drada,dn,Roh!,temp

      logical, save             :: entereda = .false.
      real( kind=8 ), save, dimension(ktoa) :: rada
      real( kind=8 ), save                :: look(11,klo,2),Rrange
     &     ,md_look(klo)

!-----------------------------------------------------------------
!     Dust variables
!-----------------------------------------------------------------

!@var ntix_dust  index for mapping advected dust bins onto ndtr dust bins in
!@+     heterogeneous chemistry
      integer, dimension( nhcaero_dust ) :: ntix_dust
!@var ntm_array array for size of dust bins
      integer :: ntm_array( ndtr-nSubClays+1 )
!@var dradi  dust radii for sulfate uptake [m]
      real( kind=8 ), dimension( nDustBinsFull ) :: dradi
!@var rop  dust particle density [kg/m^3]
      real( kind=8 ), dimension( ndtr ) :: rop
!@var wttr_dust weighting array for mass in dust bins
      real( kind=8 ), dimension( nhcaero_dust ) :: wttr_dust


      if (.not. entereda) then

!-----------------------------------------------------------------
!    1000 Intervals for Radius = 0.01ym ->10ym
!-----------------------------------------------------------------
!      Integration of radius:  0.01 ym to 10 ym
       rada(1) = 0.01d-6    ! smallest radius
       drada   = 0.1d-6     ! delta radius


      entereda = .true.
      if (am_i_root())
     &  PRINT*, 'CALCULATING LOOK UP TABLE FOR HETEROGENEOUS CHEMISTY'
      DO ii   = 2, ktoa
      rada(ii) = rada(ii-1) + drada
      END DO

c      entereda = .true.

      lsig0 = LOG(2.d0)

!-----------------------------------------------------------------
!     SO2 + DUSTM =>    Dust Aerosol Reaction
!-----------------------------------------------------------------

      wrk(1)=  (mQ1 + Mgas) / mQ1

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!                         LOOK UP TABLE
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       md_look(1) = 1.d-10       ! smallest md
       Rrange=5.d-8

       DO ii   = 2, klo
       md_look(ii) = md_look(ii-1) +  Rrange
       END DO

      DO ip  = 1, 11  !pressure from 1000 to 0 hPa

      look_p=max(0.001d0,1.1d0-ip*0.1d0)         !atmosphere (minimum is 1 hPa)
      look_t=max(210.d0,288.d0*(look_p/1.d0)**((1.40d0-1d0)/1.40d0))
      look_p=look_p * 100000.d0              ! pressure in Pa

       Roh    = look_p / look_t / 287.d0
C Molecular diffusion coefficient for a trace gas in air [ m/s ]
       Mdc(1)  = 3.d0 / (8.d0* Avog * Roh * (Diaq**2))
       Mdc(1)  = Mdc(1) * SQRT( ((gasc*look_t*Mgas)/(2.d0*pi))*wrk(1))

C thermal velocity of a trace gas molecule [m/s2]
       VSP(1)  = SQRT((8.d0 * Bolz * look_t)/(Pi * mQ1 * byAvog))

C lamb  mean free pathway  [m]
       lamb(1)   = 3.d0 * Mdc(1)/VSP(1)

C Loop over radius

      DO imd  = 1,klo
       look(ip,imd,:) = 0.d0

      DO k = 1,ktoa-1
C Knudsen Number
       Kn(1)= lamb(1) / rada(k) ! Radius in [m]

C Mass Transfer Coefficient
c RH < 60 %
       Kdj(1) =(4.d0 * pi * rada(k) * Mdc(1))
     .      / (1.d0 + Kn(1) * (xx + 4d0 *(1.d0- alph1)/(3.d0 *alph1)))
c RH > 60 %
       Kdj(2) =(4.d0 * pi * rada(k) * Mdc(1))
     .      / (1.d0 + Kn(1) * (xx + 4d0 *(1.d0- alph2)/(3.d0 *alph2)))

C Number distribution
       dn=abs(erf(log( rada(k)/md_look(imd)) / lsig0
     .       /sqrt(2.0d0))-erf(log(rada(k+1)/
     .       md_look(imd)) / lsig0 /sqrt(2.0d0)))/2.d0

C Net removal rate [s-1]

       look(ip,imd,1)= look(ip,imd,1) + Kdj(1)  * dn
       look(ip,imd,2)= look(ip,imd,2) + Kdj(2)  * dn

      END DO              !radius loop
      END DO              !median diameter loop
      END DO              !pressure loop

      ENDIF


c--------------------------------------------------------------

c  Or use online dust

      if ( tracers_dust .or. tracers_minerals ) then

      ntm_array = (/ nSubClays*ntm_clay, ntm_sil1, ntm_sil2, ntm_sil3 /)

      ntix_dust = (/ ( ( ii, ii = n_SoilDust,n_SoilDust+ntm_clay-1 ), jj
     &     = 1,nSubClays), ( ii, ii = n_SoilDust+ntm_clay,n_SoilDust+
     &     ntm_clay+ntm_sil1+ntm_sil2+ntm_sil3-1 ) /)

      dradi = (/ ( radiusMinerals( ii ), ii=1,nDustBinsFull ) /) * 1.d-6 ![um]->[m]

      if ( tracers_dust ) then
        rop = (/ ( trpdens( n_clay ), ii = 1,nSubClays ), trpdens(
     &       n_silt1 ), trpdens( n_silt2 ), trpdens( n_silt3 ) /)

      else

c for dust with mineral speciation set dust tracer density for clay
c (1-4) and silt size classes (5-7) to some representative value to be
c used in calculations
        rop = (/ ( 2.5d3, ii = 1,nSubClays ), 2.65d3, 2.65d3, 2.65d3 /)

      end if

      if ( tracers_minerals .or. imDust >= 4 ) call calcSubClayWeights

      wttr_dust = (/ ( ( subClayWeights( jj, ii ), ii = 1,ntm_clay ), jj
     &     = 1,nSubClays ), ( 1.d0, ii = 1,ntm_sil1+ntm_sil2+ntm_sil3 )
     &     /)

c sum over all dust types in a size bin to get total dust for each size
c bin
      dusttx = 0.d0
      do nd = 1,nSubClays
        do nc = 1,ntm_clay
          ndc = nc + ntm_clay * (nd - 1)
          do l  = 1,lm
            dusttx( l, nd ) = dusttx( l, nd ) + wttr_dust( ndc ) *
     &           trm_col( l, ntix_dust( ndc ) ) * byMA( l )
          end do                ! l
        end do                  ! nc
      end do                    ! nd
      ndc = ntm_array( 1 ) + 1
      do nd = nSubClays+1,ndtr
        do nc = 1,ntm_array(nd-nSubClays+1)
          do l  = 1,lm
            dusttx( l, nd ) = dusttx( l, nd ) + wttr_dust( ndc ) *
     &           trm_col( l, ntix_dust( ndc ) ) * byMA( l )
          end do                ! l
          ndc = ndc + 1
        end do                  ! nc
      end do                    ! nd

c--------------------------------------------------------------
c--------------------------------------------------------------

c INTERPOLATION FROM LOOK UP TABLES

C Net removal rate for SO2 [s-1]

      DO nd = 1,ndtr    ! Loop over dust tracers
      DO l  = 1,lm

c number concentration
        dustnc(l,nd) = dusttx(l,nd)/pi*0.75d0/rop(nd)/Dradi(nd)**3
        if(dustnc(l,nd).GT.0.d0) then
c pressure
        phelp = Min (99999d0, pl(l)*100d0)
c compute relative humidity
        IF(rhl(l).LT.0.6d0) ll = 1
        IF(rhl(l).GE.0.6d0) ll = 2
c pressure interpolation
        np1=min(11,1+nint((10.d0-phelp/10000.d0)-0.499d0))  !pressure
        np1=max(1,np1)
        np2=min(11,np1+1)
c radii interpolation
        nh1=max( 1, nint( (dradi( nd ) / Rrange)+0.499d0 ) )      !median diameter
        nh1=min(klo,nh1)
        nh2=min(klo,nh1+1)
        px=((11d0-np1)*10000.d0- phelp)/10000.d0
        hx=((nh1*Rrange+md_look(1)) - dradi( nd )) / Rrange
        hp1=px*look(np2,nh1,ll)+(1.d0-px)*look(np1,nh1,ll)
        hp2=px*look(np2,nh2,ll)+(1.d0-px)*look(np1,nh2,ll)
        klook=hx*hp1+(1.d0-hx)*hp2

        if  (dustnc(l,nd).gt.1000.d0.and.dustnc(l,nd).lt.(1.d30)) then
c        if  (dustnc(i,j,l,nd).gt.1000.)
          rxt(l,nd) = klook* dustnc(l,nd)
     .              / (287.054d0 * tl(l) / (pl(l)*100.d0))
        else
          rxt(l,nd) = 0.d0
        endif

        else
        rxt(l,nd) = 0.d0
      endif
      ENDDO ! l
      ENDDO ! nd

      rxts(:) = 0.d0
      rxts1(: ) = 0.d0

      DO nd = 1,ndtr-1  !1,ndtr
        rxts(:) = rxts(:) + rxt(:,nd)
      ENDDO
      do nd = 1,nSubClays
        rxts1(: ) = rxts1(: ) + rxt(:, nd )
      end do
      rxts2(:) = rxt(:,5)
      rxts3(:) = rxt(:,6)
!        rxts4(:) = rxt(:,8)

      end if                    ! tracers_dust .or. tracers_minerals

      return
      end subroutine sulfdust
#endif
