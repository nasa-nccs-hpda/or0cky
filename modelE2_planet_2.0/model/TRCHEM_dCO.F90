#include "rundeck_opts.h"
!=======================================================================
module tracers_dCO
!=======================================================================
implicit none

!@param R_17O_16O Ratio of d17O over total O in the atmosphere. Divided
!@+               by 2.d0, since one O17 is assumed per molecule of O2.
!@param R_18O_16O Ratio of d18O over total O in the atmosphere. Divided
!@+               by 2.d0, since one O18 is assumed per molecule of O2.
!@param dacetone_fact factor to multiply with acetone (scaled) conc
!@param dalke_IC_fact factor to multiply with Alkenes initial conc
!@param dPAR_IC_fact factor to multiply with Paraffin initial conc
!@param dPAN_IC_fact factor to multiply with PAN initial conc
!@param dMeOOH_IC_fact factor to multiply with MeOOH initial conc
!@param dHCHO_IC_fact factor to multiply with HCHO initial conc
!@param dC17O_IC_fact factor to multiply with CO_IC file
!@param dC18O_IC_fact factor to multiply with CO_IC file
!@param d13CO_IC_fact factor to multiply with CO_IC file

!@param dCO_fact Derived type that holds all isotopic factors to multiply
!@+              with the various reactions and concentrations

!-----------------------------------------------------------------------
! Main type with all relevant factos
!-----------------------------------------------------------------------
type dCO_factors
! If the variable name is a species, multiplying the species concentration
! with the corresponding factor will provide the concentration of the
! isotopically-labeled species.
! If the variable name is a reaction or a flux, multiplying the reaction
! or flux rate with the corresponding factor will provide the reaction
! rate for the reaction with the corresponding isotopically-labeled species.
  real(8) :: d13CH4
  real(8) :: d13Cisop
  real(8) :: d13Calk
  real(8) :: d13Cpar
  real(8) :: d13Cterp

  real(8) :: dC17O_airc
  real(8) :: dC18O_airc
  real(8) :: d13CO_airc
  real(8), dimension(:), allocatable :: dC17O_emis
  real(8), dimension(:), allocatable :: dC18O_emis
  real(8), dimension(:), allocatable :: d13CO_emis

  real(8) :: CH4_OH__dC17O_M
  real(8) :: Isoprene_OH__dC17O_M
  real(8) :: Isoprene_O3__dC17O_M
  real(8) :: Isoprene_NO3__dC17O_M
  real(8) :: Alkenes_OH__dC17O_M
  real(8) :: Alkenes_O3__dC17O_M
  real(8) :: Alkenes_NO3__dC17O_M
  real(8) :: Paraffin_OH__dC17O_M
  real(8) :: Terpenes_OH__dC17O_M
  real(8) :: Terpenes_O3__dC17O_M
  real(8) :: Terpenes_NO3__dC17O_M

  real(8) :: CH4_OH__dC18O_M
  real(8) :: Isoprene_OH__dC18O_M
  real(8) :: Isoprene_O3__dC18O_M
  real(8) :: Isoprene_NO3__dC18O_M
  real(8) :: Alkenes_OH__dC18O_M
  real(8) :: Alkenes_O3__dC18O_M
  real(8) :: Alkenes_NO3__dC18O_M
  real(8) :: Paraffin_OH__dC18O_M
  real(8) :: Terpenes_OH__dC18O_M
  real(8) :: Terpenes_O3__dC18O_M
  real(8) :: Terpenes_NO3__dC18O_M

  real(8) :: CH4_OH__d13CO_M
  real(8) :: Isoprene_OH__d13CO_M
  real(8) :: Isoprene_O3__d13CO_M
  real(8) :: Isoprene_NO3__d13CO_M
  real(8) :: Alkenes_OH__d13CO_M
  real(8) :: Alkenes_O3__d13CO_M
  real(8) :: Alkenes_NO3__d13CO_M
  real(8) :: Paraffin_OH__d13CO_M
  real(8) :: Terpenes_OH__d13CO_M
  real(8) :: Terpenes_O3__d13CO_M
  real(8) :: Terpenes_NO3__d13CO_M

  real(8) :: dC17O_OH__HO2_O2
  real(8) :: dC18O_OH__HO2_O2
  real(8) :: d13CO_OH__HO2_O2
end type dCO_factors
type(dCO_factors), protected :: dCO_fact

!-----------------------------------------------------------------------
! options that force binary reproducibility
!-----------------------------------------------------------------------
#ifdef TRACERS_dCO_bin_reprod

real(8), parameter :: R_17O_16O=1.d0
real(8), parameter :: R_18O_16O=1.d0
#ifdef TRACERS_dCO
real(8), parameter :: dacetone_fact=1.d0
real(8), parameter :: dalke_IC_fact=1.d0
real(8), parameter :: dPAR_IC_fact=1.d0
real(8), parameter :: dPAN_IC_fact=1.d0
real(8), parameter :: dMeOOH_IC_fact=1.d0
real(8), parameter :: dHCHO_IC_fact=1.d0
#endif  /* TRACERS_dCO */
real(8), parameter :: dC17O_IC_fact=1.d0
real(8), parameter :: dC18O_IC_fact=1.d0
real(8), parameter :: d13CO_IC_fact=1.d0

!-----------------------------------------------------------------------
! options for production
!-----------------------------------------------------------------------
#else

#ifdef TRACERS_dCO
You should not even compile, not implemented yet.
#endif  /* TRACERS_dCO */
real(8), parameter :: R_17O_16O=0.000377421d0 ! wikipedia
real(8), parameter :: R_18O_16O=0.002 ! wikipedia

! Vienna Standard Mean Ocean Water (VSMOW) value, from Wikipedia.
real(8), parameter :: dC17O_IC_fact=379.9d-6
! Vienna Standard Mean Ocean Water (VSMOW) value, from Wikipedia.
real(8), parameter :: dC18O_IC_fact=2005.20d-6
! Vienna Pee Dee Belemnite (VPDB) value, from Wikipedia.
real(8), parameter :: d13CO_IC_fact=0.0112372d0

#endif  /* TRACERS_dCO_bin_reprod */

!=======================================================================
contains
!=======================================================================

!-----------------------------------------------------------------------
subroutine dCO_init
!-----------------------------------------------------------------------
  use OldTracer_mod, only: trname
  use TRACER_COM, only: ntm
  use TRACER_COM, only: ntsurfsrc
  use OldTracer_mod, only: nBBsources
  use Dictionary_mod, only: sync_param
  implicit none
  real(8) :: dCO_airc_e
  real(8), dimension(:), allocatable :: dCO_emis_e
  integer :: nsrc,n,ns

! emissions factors, as provided from the rundeck
  do n=1,ntm
    nsrc=ntsurfsrc(n)+nBBsources(n)
    if (nsrc == 0) cycle

    select case (trname(n))

    case ('dC17O')
      call sync_param(trim(trname(n))//'_airc_e',dCO_airc_e)
      dCO_fact%dC17O_airc=dCO_iso_sig(dC17O_IC_fact, dCO_airc_e)

      allocate(dCO_emis_e(nsrc))
      allocate(dCO_fact%dC17O_emis(nsrc))
      call sync_param(trim(trname(n))//'_emis_e', &
                      dCO_emis_e, nsrc)
      do ns=1,nsrc
        dCO_fact%dC17O_emis(ns)=dCO_iso_sig(dC17O_IC_fact, dCO_emis_e(ns))
      enddo
      deallocate(dCO_emis_e)

    case ('dC18O')
      call sync_param(trim(trname(n))//'_airc_e',dCO_airc_e)
      dCO_fact%dC18O_airc=dCO_iso_sig(dC18O_IC_fact, dCO_airc_e)

      allocate(dCO_emis_e(nsrc))
      allocate(dCO_fact%dC18O_emis(nsrc))
      call sync_param(trim(trname(n))//'_emis_e', &
                      dCO_emis_e, nsrc)
      do ns=1,nsrc
        dCO_fact%dC18O_emis(ns)=dCO_iso_sig(dC18O_IC_fact, dCO_emis_e(ns))
      enddo
      deallocate(dCO_emis_e)

    case ('d13CO')
      call sync_param(trim(trname(n))//'_airc_e',dCO_airc_e)
      dCO_fact%d13CO_airc=dCO_iso_sig(d13CO_IC_fact, dCO_airc_e)

      allocate(dCO_emis_e(nsrc))
      allocate(dCO_fact%d13CO_emis(nsrc))
      call sync_param(trim(trname(n))//'_emis_e', &
                      dCO_emis_e, nsrc)
      do ns=1,nsrc
        dCO_fact%d13CO_emis(ns)=dCO_iso_sig(d13CO_IC_fact, dCO_emis_e(ns))
      enddo
      deallocate(dCO_emis_e)

    end select

  enddo

! Mean parent HCs factors. Set those before setting out reaction factors.
dCO_fact%d13CH4=dCO_iso_sig(d13CO_IC_fact, -47.d0) ! Park et al., 2015, table S1; Lowe et al., 1991; 1997; Mak et al., 2000; Quay et al., 1999
dCO_fact%d13Cisop=dCO_iso_sig(d13CO_IC_fact, -32.2d0) ! Park et al., 2015, table S1
dCO_fact%d13Calk=dCO_iso_sig(d13CO_IC_fact, -32.2d0) ! Park et al., 2015, table S1
dCO_fact%d13Cpar=dCO_iso_sig(d13CO_IC_fact, -32.2d0) ! Park et al., 2015, table S1
dCO_fact%d13Cterp=dCO_iso_sig(d13CO_IC_fact, -32.2d0) ! Park et al., 2015, table S1

! dC17O formation yields from parent HCs
dCO_fact%CH4_OH__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_OH__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_O3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_NO3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_OH__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_O3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_NO3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Paraffin_OH__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_OH__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_O3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_NO3__dC17O_M=R_17O_16O*dCO_iso_sig(dC17O_IC_fact, 0.d0) ! guess

! dC18O formation yields from parent HCs
dCO_fact%CH4_OH__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! Brenninkmeijer and Rockmann (1997)
dCO_fact%Isoprene_OH__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_O3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_NO3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_OH__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_O3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_NO3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Paraffin_OH__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_OH__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_O3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_NO3__dC18O_M=R_18O_16O*dCO_iso_sig(dC18O_IC_fact, 0.d0) ! guess

! d13CO formation yields from parent HCs
dCO_fact%CH4_OH__d13CO_M=dCO_fact%d13CH4*dCO_KIE(-3.9d0) ! from Park et al., 2015, table S1; Saueressig et al. (2001)
dCO_fact%Isoprene_OH__d13CO_M=dCO_fact%d13Cisop*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_O3__d13CO_M=dCO_fact%d13Cisop*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Isoprene_NO3__d13CO_M=dCO_fact%d13Cisop*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_OH__d13CO_M=dCO_fact%d13Calk*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_O3__d13CO_M=dCO_fact%d13Calk*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Alkenes_NO3__d13CO_M=dCO_fact%d13Calk*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Paraffin_OH__d13CO_M=dCO_fact%d13Cpar*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_OH__d13CO_M=dCO_fact%d13Cterp*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_O3__d13CO_M=dCO_fact%d13Cterp*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess
dCO_fact%Terpenes_NO3__d13CO_M=dCO_fact%d13Cterp*dCO_iso_sig(d13CO_IC_fact, 0.d0) ! guess

! destruction via CO+OH
dCO_fact%dC17O_OH__HO2_O2=dCO_KIE(0.d0) ! Feilberg et al., 2005
!dCO_fact%dC17O_OH__HO2_O2=dCO_KIE(4.7d0) ! Gromov et al., 2010
dCO_fact%dC18O_OH__HO2_O2=dCO_KIE(-15.d0) ! Feilberg et al., 2005
!dCO_fact%dC18O_OH__HO2_O2=dCO_KIE(-9.4d0) ! Gromov et al., 2010
dCO_fact%d13CO_OH__HO2_O2=dCO_KIE(11.d0) ! Feilberg et al., 2005
!dCO_fact%d13CO_OH__HO2_O2=dCO_KIE(6.5d0) ! Gromov et al., 2010

end subroutine dCO_init
!-----------------------------------------------------------------------
real(8) function dCO_KIE(sig) result(byKIE)
!-----------------------------------------------------------------------
! Kinetic Isotope Effect (KIE)
! https://en.wikipedia.org/wiki/Kinetic_isotope_effect
! The kinetic isotope effect (KIE) is the change in the reaction rate of
! a chemical reaction when one of the atoms in the reactants is replaced
! by one of its isotopes. Formally, it is the ratio of rate constants
! for the reactions involving the light (kL) and the heavy (kH)
! isotopically substituted reactants:
! KIE=k_L/k_H
! This change in rate results from heavier isotopologues having a lower
! velocity/mobility and an increased stability from the higher
! dissociation energies when compared to the compounds containing
! lighter isotopes. The study of kinetic isotope effects can help the
! elucidation of the reaction mechanism of certain chemical reactions.
! (Feilberg et al., 2005), (Park et al., 2015): ε = (KIE - 1)*1000‰.
!
! This function returns 1.d0/KIE, so in order to get k_H one should
! multiply the function result with k_L.
!-----------------------------------------------------------------------
  implicit none
  real(8), intent(in) :: sig

#ifdef TRACERS_dCO_bin_reprod
  byKIE=1.d0
#else
  byKIE=1.d0/(1.d0+sig*1.d-3)
#endif  /* TRACERS_dCO_bin_reprod */

end function dCO_KIE
!-----------------------------------------------------------------------
real(8) function dCO_iso_sig(Rstandard, delta) result(Rsample)
!-----------------------------------------------------------------------
! Isotopic Signature
! https://en.wikipedia.org/wiki/Δ13C
! In geochemistry, paleoclimatology and paleoceanography δ13C
! (pronounced "delta c thirteen") is an isotopic signature, a measure of
! the ratio of stable isotopes 13C:12C, reported in parts per thousand
! (per mil, ‰). The standard is an established reference material, such
! as the Vienna Pee Dee Belemnite (VPDB) for 13C and the Vienna Standard
! Mean Ocean Water (VSMOW) for 18O.
! δ13C=(Rsample/Rstandard-1)*1000‰
! Where R=[13C]/[12C] of the sample and the standard, respectively.
!
! This function returns Rsample, so in order to get the concentration of
! the heavy isotope one should multiply the function result with the
! concentration of the light isotope.
!-----------------------------------------------------------------------
  implicit none
  real(8), intent(in) :: Rstandard
  real(8), intent(in) :: delta

#ifdef TRACERS_dCO_bin_reprod
  Rsample=1.d0
#else
  Rsample=Rstandard*(1.d0+delta*1.d-3)
#endif  /* TRACERS_dCO_bin_reprod */

end function dCO_iso_sig
!-----------------------------------------------------------------------

!=======================================================================
end module tracers_dCO
!=======================================================================
