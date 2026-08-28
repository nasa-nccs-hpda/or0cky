#include "rundeck_opts.h"

#ifdef TRACERS_ON
!@sum  TRACERS: generic tracer routines used for all tracers
!@+    Routines included:
!@+      Generic diags: set_generic_tracer_diags
!@+      Apply previously set sources: apply_tracer_sources
!@+      Radioactive Decay: tdecay
!@+      Gravitaional Settling: trgrav
!@+      Check routine: checktr
!@auth Jean Lerner/Gavin Schmidt
      MODULE apply3d
!@sum apply3d is used simply so that I can get optional arguments
!@+   to work. If anyone can some up with something neater, let me know.
      USE TRACER_COM, only : trm_col,trmom_col
      USE RESOLUTION, only: lm
      USE MODEL_COM, only : dtsrc
#ifndef SKIP_TRACER_SRCS
      USE FLUXES, only : tr3Dsource
#endif
      USE TRDIAG_COM, only : jls_3Dsource,itcon_3Dsrc
     *     ,ijts_3Dsource,taijs=>taijs_loc

      IMPLICIT NONE

      CONTAINS

      SUBROUTINE apply_tracer_3Dsource(i,j, ns , n , momlog )
      USE CONSTANT, only : UNDEF_VAL
!@sum apply_tracer_3Dsource adds 3D sources to tracers
!@auth Jean Lerner/Gavin Schmidt
!@var MOM true (default) if moments are to be modified
      logical, optional, intent(in) :: momlog
      integer, intent(in) :: i,j,n,ns
      real*8 fred,dtrm(lm),eps

      logical :: domom
      integer najl,l,naij,nn

C**** Ensure that this is a valid tracer and source
      if (n.eq.0 .or. ns.eq.0) then
         return
      end if
#ifndef SKIP_TRACER_SRCS
C**** parse options
      domom=.true.
      if( present(momlog) ) then
        domom=momlog
      end if
C**** This is tracer independent coding designed to work for all
C**** 3D sources.
C**** Modify tracer amount, moments, and diagnostics
      najl = jls_3Dsource(ns,n)
      naij = ijts_3Dsource(ns,n)

      eps = tiny(trm_col(1,n))
      fred = UNDEF_VAL
      dtrm = UNDEF_VAL
      do l=1,lm
        dtrm(l) = tr3Dsource(l,ns,n)*dtsrc
C**** calculate fractional loss and update tracer mass
        fred = max(0.d0,1.+min(0.d0,dtrm(l))/(trm_col(l,n)+eps))
        trm_col(l,n) = trm_col(l,n)+dtrm(l)
        if(fred.le.1d-16) trm_col(l,n) = 0.
        if(domom .and. fred.lt.1.) then
          trmom_col(:,l,n) = trmom_col(:,l,n)*fred
        endif
        if (naij.gt.0) then
          taijs(i,j,naij) = taijs(i,j,naij) + dtrm(l)
        end if
      enddo ! l

      if(jls_3Dsource(ns,n) > 0) then
        call inc_tajls2_column(i,j,1,lm,lm,najl,dtrm)
      endif
#endif

      if (itcon_3Dsrc(ns,n).gt.0)
     &  call DIAGTCA_1pt(itcon_3Dsrc(ns,n),n,i,j)

C****
      RETURN
      END SUBROUTINE apply_tracer_3Dsource

      end module apply3d

      SUBROUTINE set_generic_tracer_diags
!@sum set_generic_tracer_diags init trace gas attributes and diagnostics
!@auth J. Lerner
!@calls sync_param
      use OldTracer_mod, only: trName, tr_wd_TYPE, mass2vol
      use OldTracer_mod, only: dowetdep, dodrydep, trradius, ntrocn
      use OldTracer_mod, only: ntm_power, nwater, src_dist_index
      use OldTracer_mod, only: to_volume_MixRat
      use OldTracer_mod, only: to_conc
      USE CONSTANT, only: mair
      USE MODEL_COM, only: dtsrc
      USE FLUXES, only : nisurf,atmice
      USE DIAG_COM, only: ia_src,ia_srf,ia_12hr,ir_log2,ir_0_71
      USE TRACER_COM, only: ntm, n_Water, nPart
      USE TRDIAG_COM
      use Dictionary_mod
      USE DOMAIN_DECOMP_ATM, only: AM_I_ROOT
      implicit none
      integer :: l,k,n,nd
      character*10, DIMENSION(NTM) :: CMR,CMRWT
      logical :: T=.TRUE. , F=.FALSE.
      character*50 :: unit_string

      atmice%taijn => taijn_loc
#ifdef TRACERS_WATER
      allocate(atmice%do_accum(ntm))
      do n=1,ntm
        atmice%do_accum(n) =
     &       (tr_wd_TYPE(n).eq.nWater .or. tr_wd_TYPE(n).eq.nPART)
      enddo
#endif

C**** Get factor to convert from mass to volume mr if necessary
      do n=1,ntm
        if (to_volume_MixRat(n) .eq.1) then
          MMR_to_VMR(n) = mass2vol(n)
          cmr(n) = 'V/V air'
        else
          MMR_to_VMR(n) = 1.d0
          cmr(n) = 'kg/kg air'
        endif
#ifdef TRACERS_WATER
        if (to_per_mil(n) .eq.1) then
          cmrwt(n) = 'per mil'
          cmr(n) = 'per mil'     ! this overrides to_volume_ratio
          MMR_to_VMR(n) = 1.
        else
          cmrwt(n) = 'kg/m^2'
        end if
#endif
      end do
C****
C**** TAJLN(J,L,KQ,N)  (SUM OVER LONGITUDE AND TIME OF)
C****
C**** jlq_power Exponent associated with a physical process
C****      (for printing tracers).   (ntm_power+jlq_power)
C**** jls_power Exponent associated with a source/sink (for printing)
C**** Defaults for JLN
      scale_jlq(:) = 1./DTsrc
      jgrid_jlq(:) = 1
      ia_jlq(:) = ia_src

C**** Tracer concentration
      do n=1,ntm
        k = 1        ! <<<<< Be sure to do this
        jlnt_conc = k
        sname_jln(k,n) = trim(trname(n))//'_CONCENTRATION'
        lname_jln(k,n) = trim(trname(n))//' CONCENTRATION'
        jlq_power(k) = 0.
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k),cmr(n))
        scale_jlq(k) = 1.d0
        scale_jln(n) = MMR_to_VMR(n)
#ifdef TRACERS_WATER
        if (to_per_mil(n).gt.0) units_jln(k,n) = unit_string(0,cmr(n))
#endif
C**** Tracer mass
        k = k + 1
        jlnt_mass = k
        sname_jln(k,n) = trim(trname(n))//'_MASS'
        lname_jln(k,n) = trim(trname(n))//' MASS'
        jlq_power(k) = 4

#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) ||\
    (defined TRACERS_TOMAS) || (defined TRACERS_AEROSOLS_SEASALT)
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k)+13
     *       ,'kg')
#else
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k)
     *       ,'kg/m^2')
#endif
        scale_jlq(k) = 1.d0
#ifdef TRACERS_WATER
C****   TRACER CONCENTRATION IN CLOUD WATER
        k = k + 1
        jlnt_cldh2o = k
        sname_jln(k,n) = trim(trname(n))//'_WM_CONC'
        lname_jln(k,n) = trim(trname(n))//' CLOUD WATER CONCENTRATION'
        jlq_power(k) = 4
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k)
     *       ,'kg/kg water')
        scale_jlq(k) = 1.d0
        if (to_per_mil(n).gt.0) units_jln(k,n) = unit_string(0,cmr(n))
#endif
C**** Physical processes affecting tracers
C****   F (TOTAL NORTHWARD TRANSPORT OF TRACER MASS)  (kg)
        k = k + 1
        jlnt_nt_tot = k
        sname_jln(k,n) = 'tr_nt_tot_'//trname(n)
        lname_jln(k,n) = 'TOTAL NORTHWARD TRANSPORT OF '//
     &     trim(trname(n))//' MASS'
        jlq_power(k) = 11
        jgrid_jlq(k) = 2
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k),'kg/s')
C****   STM/SM (MEAN MERIDIONAL N.T. OF TRACER MASS)  (kg)
        k = k + 1
        jlnt_nt_mm = k
        sname_jln(k,n) = 'tr_nt_mm_'//trname(n)
        lname_jln(k,n) = 'NORTHWARD TRANS. OF '//
     &     trim(trname(n))//' MASS BY MERIDIONAL CIRC.'
        jlq_power(k) = 10
        jgrid_jlq(k) = 2
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k),'kg/s')
C****   F (TOTAL VERTICAL TRANSPORT OF TRACER MASS)  (kg)
        k = k + 1
        jlnt_vt_tot = k
        sname_jln(k,n) = 'tr_vt_tot_'//trname(n)
        lname_jln(k,n) = 'TOTAL VERTICAL TRANSPORT OF '//
     &     trim(trname(n))//' MASS'
        jlq_power(k) = 11
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k),'kg/s')
C****   STM/SM (MEAN MERIDIONAL V.T. OF TRACER MASS)  (kg)
        k = k + 1
        jlnt_vt_mm = k
        sname_jln(k,n) = 'tr_vt_mm_'//trname(n)
        lname_jln(k,n) = 'VERTICAL TRANS. OF '//
     &     trim(trname(n))//' MASS BY MERIDIONAL CIRC.'
        jlq_power(k) = 10.
        units_jln(k,n) = unit_string(ntm_power(n)+jlq_power(k),'kg/s')
C****   TMBAR-TM (CHANGE OF TRACER MASS BY MOIST CONVEC)(kg)
        k = k + 1
        jlnt_mc = k
        sname_jln(k,n) = 'tr_mc_'//trname(n)
        lname_jln(k,n) = 'CHANGE OF '//
     &     trim(trname(n))//' MASS BY MOIST CONVECTION'
        jlq_power(k) = 10
        units_jln(k,n) = unit_string(ntm_power(n),'kg/kg/s')
C****   TMBAR-TM (CHANGE OF TRACER MASS BY Large-scale CONDENSE)  (kg)
        k = k + 1
        jlnt_lscond = k
        sname_jln(k,n) = 'tr_lscond'//trname(n)
        lname_jln(k,n) ='CHANGE OF '//
     &     trim(trname(n))//' MASS BY LARGE-SCALE CONDENSE'
        jlq_power(k) = 10.
        units_jln(k,n) = unit_string(ntm_power(n),'kg/kg/s')
C****   TMBAR-TM (CHANGE OF TRACER MASS BY DRY CONVEC)  (kg)
        k = k + 1
        jlnt_turb = k
        sname_jln(k,n) = 'tr_turb_'//trname(n)
        lname_jln(k,n) = 'CHANGE OF '//
     &     trim(trname(n))//' MASS BY TURBULENCE/DRY CONVECTION'
        jlq_power(k) = 10
        units_jln(k,n) = unit_string(ntm_power(n),'kg/kg/s')


        if (k.gt. ktajl) then
           if (AM_I_ROOT()) write (6,*)
     &          'tjl_defs: Increase ktajl=',ktajl,' to at least ',k
           call stop_model('ktajl too small',255)
        end if
      end do

C**** CONTENTS OF TAIJLN(I,J,LM,N)  (SUM OVER TIME OF)
C****        TML (M*M * KG TRACER/KG AIR)
C**** Set defaults that are true for all tracers and layers
      do n=1,ntm
        ijtm_power(n) = ntm_power(n)+4 !n for integrated mass
        ijtc_power(n) = ntm_power(n)+1 !n for concentration
#ifdef TRACERS_WATER
        if (to_per_mil(n) .eq.1) ijtc_power(n) = 0
#endif
      end do
C**** Tracer concentrations (TAIJLN)
      do n=1,ntm
        write(sname_ijt(n),'(a)') trim(TRNAME(n))
        write(lname_ijt(n),'(a)') trim(TRNAME(n))
        if (to_conc(n).eq.1) then   ! diag in kg/m3
          units_ijt(n) = unit_string(ijtc_power(n),'kg/m3')
          scale_ijt(n) = 10.**(-ijtc_power(n))
        else ! mixing ratio
          units_ijt(n) = unit_string(ijtc_power(n),cmr(n))
          scale_ijt(n) = MMR_to_VMR(n)*10.**(-ijtc_power(n))
        end if
      end do

C**** AIJN
C****     1  TM (SUM OVER ALL LAYERS) (M*M * KG TRACER/KG AIR)
C****     2  TRS (SURFACE TRACER CONC.) (M*M * KG TRACER/KG AIR)
C****     3  TM (SUM OVER ALL LAYERS) (M*M * KG TRACER)

      do n=1,ntm
C**** Summation of mass over all layers
      k = 0        ! <<<<< Be sure to do this
      if (src_dist_index(n)<=1) then
        k = k+1
        tij_mass = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_Total_Mass'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//' Total Mass'
        units_tij(k,n) = unit_string(ijtm_power(n),'kg/m^2')
        scale_tij(k,n) = 10.**(-ijtm_power(n))
C**** Average concentration over layers
        k = k+1
        tij_conc = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_Average'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//' Average'
        units_tij(k,n) = unit_string(ijtc_power(n),cmr(n))
        scale_tij(k,n) = MMR_to_VMR(n)*10.**(-ijtc_power(n))
#ifdef TRACERS_WATER
        if (to_per_mil(n) .eq.1) then
          denom_tij(k,n)=n_Water
          scale_tij(k,n)=1.
        endif
#endif
C**** Surface concentration
        k = k+1
        tij_surf = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_At_Surface'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//' At Surface'
        units_tij(k,n) = unit_string(ijtc_power(n),cmr(n))
        scale_tij(k,n)=MMR_to_VMR(n)*10.**(-ijtc_power(n))/
     *                 REAL(NIsurf,KIND=8)
#ifdef TRACERS_WATER
        if (to_per_mil(n) .eq.1) then
          denom_tij(k,n)=n_Water
          scale_tij(k,n)=1.
        endif
#endif
C**** Surface concentration by volume (units kg/m^3)
        k = k+1
        tij_surfbv = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *     '_byVol_At_Surface'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' By Vol At Surface'
        units_tij(k,n) = unit_string(ijtc_power(n),'kg/m^3')
        scale_tij(k,n)=MMR_to_VMR(n)*10.**(-ijtc_power(n))/
     *                 REAL(NIsurf,KIND=8)
      endif ! if (src_dist_index(n)<=1) then
C**** Tropopause flux Diagnostics
        k = k+1
        tij_strop = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *     '_StratTropflux'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Flux Tropopause'
        units_tij(k,n) = unit_string(ijtc_power(n),'kg/m^2/s')
        scale_tij(k,n)=10.**(-ijtc_power(n))
#ifdef TRACERS_WATER
C**** the following diagnostics are set assuming that the particular
C**** tracer exists in water.
C**** Tracers in precipitation (=Wet deposition)
      k = k+1
      tij_prec = k
      if (dowetdep(n)) then
        if (to_per_mil(n) .eq.1) then
          write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_in_prec'
          write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *         ' in Precip'
          units_tij(k,n)=cmrwt(n)
          scale_tij(k,n)=1.
          denom_tij(k,n)=n_Water
        else
          write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_wet_dep'
          write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *         ' Wet Deposition'
          units_tij(k,n)=unit_string(ijtc_power(n)-5,trim(cmrwt(n))
     *         //'/s')
          scale_tij(k,n)=10.**(-ijtc_power(n)+5)/dtsrc
        end if
      end if
C**** Tracers in evaporation
      if (src_dist_index(n)<=1) then
        k = k+1
        tij_evap = k
        if (tr_wd_type(n).eq.nWater) then
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_in_evap'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' in Evaporation'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(ijtc_power(n),cmrwt(n))
          scale_tij(k,n)=1.
          denom_tij(k,n)=n_Water
        else
          units_tij(k,n)=unit_string(ijtc_power(n),trim(cmrwt(n))//'/s')
          scale_tij(k,n)=10.**(-ijtc_power(n))/dtsrc
        end if
#ifdef TRACERS_SPECIAL_O18
C**** Tracers at sea surface
        k = k+1
        tij_owiso = k
        write(sname_tij(k,n),'(a,i2)')
     &        trim(TRNAME(n))//'_Sea_Surface'
        write(lname_tij(k,n),'(a,i2)')
     &        trim(TRNAME(n))//' at Sea Surface'
        !Convert to permil if specified:
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=cmrwt(n)
          denom_tij(k,n)=n_Water
          scale_tij(k,n)=1.
        else
          units_tij(k,n)='kg/kg fresh water'
          !Scale quantity to account for extra surface flux time steps:
          scale_tij(k,n)=1.d0/REAL(NIsurf,KIND=8)
          !Set special denominator name in order to use ocean fraction:
          dname_tij(k,n) = 'ocnfrac'
        endif
#endif
        endif !if (tr_wd_type(n).eq.nWater)
C**** Tracers in river runoff (two versions - for inflow and outflow)
        k = k+1
        tij_rvr = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_rvr'
        lname_tij(k,n) = trim(TRNAME(n))//' in River Inflow'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
        k = k+1
        tij_rvro = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_rvro'
        lname_tij(k,n) = trim(TRNAME(n))//' in River Outflow'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracers in iceberg runoff 
        k = k+1
        tij_icb = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_icb'
        lname_tij(k,n) = trim(TRNAME(n))//' in Iceberg Inflow'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracers in sea ice
        k = k+1
        atmice%tij_seaice = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_ice'
        lname_tij(k,n) = trim(TRNAME(n))//' in Sea Ice'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
          denom_tij(k,n)=n_Water
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,cmrwt(n))
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
          dname_tij(k,n) = 'oicefr'
c        denom_tij(k,n)=n_Water ! if kg/kg units for non-water-isotopes
        end if
C**** Tracers conc. in ground component (ie. water or ice surfaces)
        k = k+1
        tij_grnd = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_at_Grnd'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' at Ground'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
          denom_tij(k,n)=n_Water
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg wat')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)/REAL(NIsurf,KIND=8)
        end if
C**** Tracers conc. in lakes (layer 1)
        k = k+1
        tij_lk1 = k
        sname_tij(k,n) = trim(TRNAME(n))//'_Lake1'
        lname_tij(k,n) = trim(TRNAME(n))//' Lakes layer 1'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg wat')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracers conc. in lakes (layer 2)
        k = k+1
        tij_lk2 = k
        sname_tij(k,n) = trim(TRNAME(n))//'_Lake2'
        lname_tij(k,n) = trim(TRNAME(n))//' Lakes layer 2'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg wat')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracers conc. in soil water
        k = k+1
        tij_soil = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_Soil'
        lname_tij(k,n) = trim(TRNAME(n))//' Soil Water'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg wat')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracers conc. in land snow water
        k = k+1
        tij_snow = k
        sname_tij(k,n) = trim(TRNAME(n))//'_in_Snow'
        lname_tij(k,n) = trim(TRNAME(n))//' Land Snow Water'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)+3,'kg/kg wat')
          scale_tij(k,n)=10.**(-ijtc_power(n)-3)
        end if
        denom_tij(k,n)=n_Water
C**** Tracer ice-ocean flux
        k = k+1
        atmice%tij_icocflx = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_ic_oc_flx'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Ice-Ocean Flux'
        if (to_per_mil(n) .eq.1) then
          units_tij(k,n)=unit_string(0,cmrwt(n))
          denom_tij(k,n)=n_Water
          scale_tij(k,n)=1.
        else
          units_tij(k,n)=unit_string(ijtc_power(n)-5,'kg/m^2/s')
          scale_tij(k,n)=10.**(-ijtc_power(n)+5)/DTsrc
        end if
C**** Tracers integrated E-W atmospheric flux
        k = k+1
        tij_uflx = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_uflx'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' E-W Atmos Flux'
        units_tij(k,n)=unit_string(ijtc_power(n)+10,'kg/s')
        scale_tij(k,n)=10.**(-ijtc_power(n)-10)/DTsrc
C**** Tracers integrated N-S atmospheric flux
        k = k+1
        tij_vflx = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_vflx'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' N-S Atmos Flux'
        units_tij(k,n)=unit_string(ijtc_power(n)+10,'kg/s')
        scale_tij(k,n)=10.**(-ijtc_power(n)-10)/DTsrc
C**** Tracers integrated E-W sea ice flux
        k = k+1
        atmice%tij_tusi = k
        sname_tij(k,n) = trim(TRNAME(n))//'_tusi'
        lname_tij(k,n) = trim(TRNAME(n))//' E-W Ice Flux'
        units_tij(k,n) = unit_string(ntrocn(n),'kg/s')
        scale_tij(k,n) = (10.**(-ntrocn(n)))/DTsrc
C**** Tracers integrated N-S sea ice flux
        k = k+1
        atmice%tij_tvsi = k
        sname_tij(k,n) = trim(TRNAME(n))//'_tvsi'
        lname_tij(k,n) = trim(TRNAME(n))//' N-S Ice Flux'
        units_tij(k,n) = unit_string(ntrocn(n),'kg/s')
        scale_tij(k,n) = (10.**(-ntrocn(n)))/DTsrc
      endif ! if (src_dist_index(n)<=1) 
#endif /* TRACERS_WATER */
#ifdef TRACERS_DRYDEP
C**** Tracers dry deposition flux.

      if (dodrydep(n)) then
        k = k+1
        tij_drydep = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_dry_dep'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Dry Deposition'
        units_tij(k,n)=unit_string(ijtc_power(n)-5,'kg/m^2/s')
        scale_tij(k,n)=10.**(-ijtc_power(n)+5)/DTsrc

        k = k+1
        tij_vd = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_vd'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0

#ifdef TRACERS_DRYDEP_DIAG
        k = k+1
        tij_vd_for = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_vd_for'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity at forests'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0

        k = k+1
        tij_vd_cro = k
        write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_vd_cro'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity at crops'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0

        k = k+1
        tij_vd_gra = k
        write(sname_tij(k,n),'(a,i2)')
     *       trim(TRNAME(n))//'_vd_gra'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity at grasslands'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0
 
        k = k+1
        tij_vd_shr = k
        write(sname_tij(k,n),'(a,i2)')
     *        trim(TRNAME(n))//'_vd_shr'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity at shrubs'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0

        k = k+1
        tij_vd_bar = k
        write(sname_tij(k,n),'(a,i2)')
     *        trim(TRNAME(n))//'_vd_bar'
        write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Deposition velocity at bare surfaces (land ITYPE)'
        units_tij(k,n)='cm/s'
        scale_tij(k,n)=100.d0

        if (tr_wd_TYPE(n) /= nPART) then
         k = k+1
         tij_egstom1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_egstom'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective stomatal conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egcut_dry1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_egcut_dry'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective dry cuticular conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_eggro_dry1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_eggro_dry'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective dry ground conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egcut_wet1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_egcut_wet'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective wet cuticular conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_eggro_wet1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_eggro_wet'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective wet ground conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0


         k = k+1
         tij_egcut_snow1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_egcut_snow'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective snow cuticular conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_eggro_snow1 = k
         write(sname_tij(k,n),'(a,i2)') trim(TRNAME(n))//'_eggro_snow'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' Effective snow ground conductance'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egs_for = k
         write(sname_tij(k,n),'(a,i2)')
     *       trim(TRNAME(n))//'_egstom_for'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' effective stomatal conductance at forests'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egs_cro = k
         write(sname_tij(k,n),'(a,i2)')
     *       trim(TRNAME(n))//'_egstom_cro'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' effective stomatal conductance at crops'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egs_gra = k
         write(sname_tij(k,n),'(a,i2)')
     *        trim(TRNAME(n))//'_egstom_gra'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' effective stomatal conductance at grass'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         k = k+1
         tij_egs_shr = k
         write(sname_tij(k,n),'(a,i2)')
     *        trim(TRNAME(n))//'_egstom_shr'
         write(lname_tij(k,n),'(a,i2)') trim(TRNAME(n))//
     *       ' effective stomatal conductance at shrub'
         units_tij(k,n)='cm/s'
         scale_tij(k,n)=100.d0

         end if
#endif
      end if
#endif

      if (k .gt. ktaij) then
        if (AM_I_ROOT()) write (6,*)
     &   'tij_defs: Increase ktaij=',ktaij,' to at least ',k
        call stop_model('ktaij too small',255)
      end if

      end do !ntm

c
c Collect denominator short names for later use
c
      do n=1,ntm
      do k=1,ktaij
        nd = denom_tij(k,n)
        if(nd.gt.0) dname_tij(k,n) = sname_tij(k,nd)
      enddo
      enddo

      RETURN
      END SUBROUTINE set_generic_tracer_diags

      SUBROUTINE sum_prescribed_tracer_2Dsources(dtstep)
!@sum apply_tracer_2Dsource adds surface sources to tracers
!@auth Jean Lerner/Gavin Schmidt
      USE GEOM, only : imaxj
      USE QUSDEF, only : mz,mzz
      USE TRACER_COM, only : NTM,ntsurfsrc
#ifdef TRACERS_TOMAS
     &     ,n_ASO4,n_ANACL,n_AECOB,n_AOCOB,n_ADUST,n_SO2
#endif
#ifndef SKIP_TRACER_SRCS
      USE FLUXES, only : trsource
#endif
      USE FLUXES, only : trflux1,atmsrf
      USE TRDIAG_COM, only : taijs=>taijs_loc
      USE TRDIAG_COM, only : ijts_source,jls_source,itcon_surf
      USE DOMAIN_DECOMP_ATM, ONLY : GRID, getDomainBounds
      IMPLICIT NONE
      REAL*8, INTENT(IN) :: dtstep
      INTEGER n,ns,naij,najl,j,i
      REAL*8, DIMENSION(grid%I_STRT_HALO:grid%I_STOP_HALO
     *     ,grid%J_STRT_HALO:grid%J_STOP_HALO) :: dtracer

      INTEGER :: J_0, J_1, I_0, I_1
      INTEGER ntsurf !same as ntsurfsrc

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      
C**** This is tracer independent coding designed to work for all
C**** surface sources.
C**** Note that tracer flux is added to first layer either implicitly
C**** in ATURB or explicitly in 'apply_fluxes_to_atm' call in SURFACE.

      do n=1,ntm
        trflux1(:,:,n) = 0.
        ntsurf = ntsurfsrc(n) 
#ifdef TRACERS_TOMAS

! Overwrite with first bin 
       if(n.ge.n_ASO4(1).and.n.lt.n_ANACL(1))
     .               ntsurf=ntsurfsrc(n_SO2) !so4
       if(n.ge.n_AECOB(1).and.n.lt.n_AOCOB(1))
     .               ntsurf=ntsurfsrc(n_AECOB(1)) !ecob
       if(n.ge.n_AOCOB(1).and.n.lt.n_ADUST(1))
     .               ntsurf=ntsurfsrc(n_AOCOB(1)) !ocob + ocil
#endif
#ifndef SKIP_TRACER_SRCS
        do ns=1,ntsurf
C**** diagnostics
          naij = ijts_source(ns,n)
          IF (naij > 0) THEN
          taijs(:,:,naij) = taijs(:,:,naij) + trsource(:,:,ns,n)*dtstep
          ENDIF
          DO J=J_0,J_1
            do i=i_0,imaxj(j)
              dtracer(i,j)=trsource(i,j,ns,n)*dtstep
            end do
          end do
          najl = jls_source(ns,n)
          IF (najl > 0) THEN
            DO J=J_0,J_1
              DO I=I_0,imaxj(j)
                call inc_tajls2(i,j,1,najl,dtracer(i,j))
              END DO
            END  DO
          END IF
          if (itcon_surf(ns,n).gt.0)
     *         call DIAGTCB(dtracer,itcon_surf(ns,n),n)
C**** trflux1 is total flux into first layer
          trflux1(:,:,n) = trflux1(:,:,n)+trsource(:,:,ns,n)
        end do
#endif
        atmsrf%trflux_prescr(n,:,:) = trflux1(:,:,n)
      end do

#ifdef TRACERS_TOMAS
#ifdef ALT_EMISS_COAG
! The subgridcoag_drv_2d call (which adjusts trflux_prescr) has been
! moved from SURFACE in order to avoid "double-counting" trflux_prescr.
! trflux_prescr currently affects the interactive surface fluxes, but
! subgridcoag_drv_2d does not account for this.   Application of the
! full coagulation increment to trflux_prescr BEFORE the interactive
! surface fluxes is most consistent with the current model structure.
! Other possible routes not taken:
! (1) Inclusion of coagulation tendency terms within the interactive
!     surface flux calculation (complicated).
! (2) Modification of subgridcoag_drv_2d to only see the part of
!     trflux_prescr not consumed by downward interactive fluxes
!     (may not reflect the original intent).
C**** Apply subgrid coagulation for freshly emitted particles.
      call subgridcoag_drv_2D(dtstep)
#endif
#endif

      RETURN
      END SUBROUTINE sum_prescribed_tracer_2Dsources

      SUBROUTINE set_strattroptracer_diag(dtstep)
!@sum safe Tracer Fluxes at the Tropopause
!@auth Susanne Bauer
      USE GEOM, only : imaxj,byaxyp
      USE TRACER_COM, only : NTM
      USE TRDIAG_COM, only : taijn=>taijn_loc
      USE TRDIAG_COM, only : TSCF3D=>tscf3d_loc
      USE TRDIAG_COM, only : tij_strop
      USE ATM_COM,    only : LTROPO
      USE DOMAIN_DECOMP_ATM, ONLY : GRID, getDomainBounds

      IMPLICIT NONE
      INTEGER n,j,i
      REAL*8, INTENT(IN) :: dtstep
      INTEGER :: J_0, J_1, I_0, I_1

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP

#ifndef SKIP_TRACER_DIAGS
      do j=j_0,j_1
      do i=i_0,imaxj(j)
      do n=1,ntm
      taijn(i,j,tij_strop,n) = taijn(i,j,tij_strop,n)
     &         + TSCF3D(i,j,ltropo(i,j),n)*byaxyp(i,j)/dtstep

      end do ! tracer n
      enddo
      enddo
#endif /*SKIP_TRACER_DIAGS*/


      RETURN
      END SUBROUTINE set_strattroptracer_diag

      SUBROUTINE apply_tracer_2Dsource(dtstep)
!@sum apply_tracer_2Dsource adds surface sources to tracers
!@auth Jean Lerner/Gavin Schmidt
      USE GEOM, only : imaxj
      USE QUSDEF, only : mz,mzz
      USE TRACER_COM, only : NTM,trm,trmom
      USE FLUXES, only : trflux1,atmsrf
      USE DOMAIN_DECOMP_ATM, ONLY : GRID, getDomainBounds
      use oldtracer_mod, only: src_dist_index
      IMPLICIT NONE
      REAL*8, INTENT(IN) :: dtstep
      INTEGER n,i,j
      REAL*8 ftr1,dewflux,tinyReal8

      INTEGER :: J_0, J_1, I_0, I_1

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      
      tinyReal8=tiny(dewflux)

C**** This is tracer independent coding designed to work for all
C**** surface sources.
C**** Note that tracer flux is added to first layer either implicitly
C**** in ATURB or explicitly in 'apply_fluxes_to_atm' call in SURFACE.

      do n=1,ntm

       do j=j_0,j_1
C**** modify vertical moments (only from non-interactive sources)
c this is disabled until vertical moments are also modified
c during the vertical transport between layer 1 and the others
c        trmom( mz,:,j,1,n) = trmom( mz,:,j,1,n)-1.5*trflux_prescr(:,j,n)
c     *       *dtstep)
c        trmom(mzz,:,j,1,n) = trmom(mzz,:,j,1,n)+0.5*trflux_prescr(:,j,n)
c     *       *dtstep

C**** Add prescribed and interactive sources
c        trflux1(:,j,n) = trflux1(:,j,n)+atmsrf%trsrfflx(n,:,j)
        trflux1(:,j,n) =
     &        atmsrf%trflux_prescr(n,:,j)+atmsrf%trsrfflx(n,:,j)
       end do

C**** Technically speaking the vertical moments should be modified here
C**** as well. But for consistency with water vapour we only modify
C**** moments for dew.
       if (src_dist_index(n)==0) then
        do j=J_0,J_1
          do i=i_0,imaxj(j)
            dewflux=-atmsrf%trsrfflx(n,i,j)*dtstep
            ! The previous criteria were: 
            ! if(atmsrf%trsrfflx(n,i,j).lt.0 .and. trm(i,j,1,n).gt.0)then
            ! The new "dewflux > tinyReal8" takes care of the first of those,
            ! plus a saftey margin so we don't divide by an exceedingly small
            ! number. The max() in the denominator of ftr1 already prevents
            ! a negative ftr1, but I leave in the trm criterion just so we
            ! don't alter moments when trm is negative (as before).
            if ( dewflux > tinyReal8 .and. trm(i,j,1,n) > 0.) then
              ftr1=dewflux/max(dewflux,trm(i,j,1,n))
              trmom(:,i,j,1,n)=trmom(:,i,j,1,n)*(1.-ftr1)
            end if
          end do
        end do
       endif
      end do
C****
      RETURN
      END SUBROUTINE apply_tracer_2Dsource

      SUBROUTINE TDECAY(i,j)
!@sum TDECAY decays radioactive tracers every source time step
!@auth Gavin Schmidt/Jean Lerner
      USE RESOLUTION, only: im,jm,lm
      USE MODEL_COM, only : itime,dtsrc
#ifndef SKIP_TRACER_SRCS
      USE FLUXES, only : tr3Dsource
      USE apply3d, only: apply_tracer_3Dsource
#endif
      use OldTracer_mod, only: itime_tr0, trname, trdecay
      use TRACER_COM, only: nChemistry
      USE TRACER_COM, only : NTM
     &     ,trm_col,trmom_col,n_Pb210, n_Rn222
#ifdef TRACERS_WATER
     *     ,trwm
      USE SEAICE_COM, only : si_atm,si_ocn
#ifndef TRACERS_ATM_ONLY
      USE LAKES_COM, only : trlake
      USE LANDICE_COM, only : trlndi,trsnowli
      USE GHY_COM, only : tr_w_ij,tr_wsn_ij
#endif
#endif
      USE TRDIAG_COM, only : jls_decay,itcon_decay
      USE OldTracer_mod, only: tr_mm
      IMPLICIT NONE
      integer, intent(in) :: i,j
!
      real*8, dimension(ntm) :: expdec
      real*8, dimension(lm) :: told
      logical, save :: ifirst=.true.
      integer n,najl,l

      if (ifirst) then
        expdec = 1.
        ifirst = .false.
      end if

      do n=1,ntm
        if (trdecay(n).gt.0. .and. itime.ge.itime_tr0(n)) then
          expdec(n)=exp(-trdecay(n)*dtsrc)
C**** Atmospheric decay
          told(:)=trm_col(:,n)

#ifdef TRACERS_WATER
     *               +trwm(i,j,:,n)
          trwm(i,j,:,n)   = expdec(n)*trwm(i,j,:,n)
#endif
#ifndef SKIP_TRACER_SRCS
          if (trname(n) .eq. "Rn222" .and. n_Pb210.gt.0) then
            tr3Dsource(:,nChemistry,n_Pb210)=trm_col(:,n)
     &        *(1-expdec(n))*tr_mm(n_Pb210)/tr_mm(n_Rn222)/dtsrc
            call apply_tracer_3Dsource(i,j,nChemistry,n_Pb210) !radioactive decay of Rn222
          end if
#endif

          trm_col(:,n)    = expdec(n)*trm_col(:,n)
          trmom_col(:,:,n)= expdec(n)*trmom_col(:,:,n)

#ifdef TRACERS_WATER
C**** Note that ocean tracers are dealt with by separate ocean code.
C**** Decay sea ice tracers
#ifndef TRACERS_ATM_ONLY
          si_atm%trsi(n,:,i,j)   = expdec(n)*si_atm%trsi(n,:,i,j)
#endif
          if(si_atm%grid%im_world .ne. si_ocn%grid%im_world) then
            call stop_model(
     &           'TDECAY: tracers in sea ice are no longer on the '//
     &           'atm. grid - please move the next line',255)
          endif
#ifndef TRACERS_ATM_ONLY
          si_ocn%trsi(n,:,i,j)   = expdec(n)*si_ocn%trsi(n,:,i,j)
C**** ...lake tracers
          trlake(n,:,i,j) = expdec(n)*trlake(n,:,i,j)
C**** ...land surface tracers
          tr_w_ij(n,:,:,i,j) = expdec(n)*tr_w_ij(n,:,:,i,j)
          tr_wsn_ij(n,:,:,i,j)= expdec(n)*tr_wsn_ij(n,:,:,i,j)
          trsnowli(n,i,j,:) = expdec(n)*trsnowli(n,i,j,:)
          trlndi(n,i,j,:)   = expdec(n)*trlndi(n,i,j,:)
#endif
#endif
C**** atmospheric diagnostics
          najl = jls_decay(n)
          do l=1,lm
            call inc_tajls2(i,j,l,najl,(trm_col(l,n)
#ifdef TRACERS_WATER
     *           +trwm(i,j,l,n)
#endif
     *           -told(l)))
          enddo

          call DIAGTCA_1pt(itcon_decay(n),n,i,j)
        end if
      end do
C****
      return
      end subroutine tdecay


      SUBROUTINE TRGRAV(i,j)
!@sum TRGRAV gravitationally settles particular tracers
!@auth Gavin Schmidt/Reha Cakmur
      USE CONSTANT, only : visc_air
      USE RESOLUTION, only: im,jm,lm
      USE MODEL_COM, only : itime,dtsrc
      use atmcol_com, only : tl,airden=>rhotvl,zl,rhl
      USE SOMTQ_COM, only : mz,mzz,mzx,myz,zmoms
      use OldTracer_mod, only: trradius, itime_tr0, 
     *            trname, trpdens,hygro_oma
      USE TRACER_COM, only : NTM,trm_col,trmom_col
#ifdef TRACERS_AMP
      USE TRACER_COM, only : ntmAMPi,ntmAMPe
#endif
#ifdef TRACERS_TOMAS
      USE TRACER_COM, only : nbins,n_ASO4
      USE CONSTANT,   only : pi 
#endif
      USE TRDIAG_COM, only : jls_grav
      IMPLICIT NONE
      integer, intent(in) :: i,j
!
      real*8 :: stokevdt,fgrfluxd,vgs,tr_radius,tr_dens
      real*8, dimension(lm) :: told,visc,gbygz
      real*8 :: fluxd, fluxu
      integer n,najl,l
#ifdef TRACERS_TOMAS
      integer binnum,k
!@var vs : gravitational settling velocity at each bin (m s-1)
      real*8, dimension(lm,NBINS) :: vs !gravitational settling velocity (m s-1)
!@var Dp_gr : particle diameter (m)
      real*8 Dp_gr(nbins)         
      real*8 density_gr(nbins)  !density (kg/m3) of current size bin
      real*8 mp                 !particle mass (kg)
      real*8 mu                 !air viscosity (kg/m s)
#endif
#ifdef TRACERS_AMP
      real*8 :: AMPtrradius,AMPtrdens
#endif

C**** Calculate some tracer independent arrays      
C**** air density + relative humidity (wrt water) + air viscosity
      do l=1,lm
        visc(l)=visc_air(tl(l))
        if (l.eq.1) then
          gbygz(l)=0.
        else
          !gbygz(l)=grav/(gz(i,j,l)-gz(i,j,l-1))
          gbygz(l)=1d0/(zl(l)-zl(l-1))
        end if
      end do

C**** Gravitational settling
      do n=1,ntm
        if (trradius(n).gt.0. .and. itime.ge.itime_tr0(n)) then

          fluxd=0.
          do l=lm,1,-1          ! loop down

C*** save original tracer mass
            told(l)=trm_col(l,n)
C**** set incoming flux from previous level
            fluxu=fluxd

C**** set particle properties

#ifdef TRACERS_AMP
            if (n.ge.ntmAMPi.and.n.le.ntmAMPe) then
              tr_dens =AMPtrdens(i,j,l,n,.true.)
              tr_radius=AMPtrradius(i,j,l,n)
            endif
#else
            tr_dens = trpdens(n)
C**** Hydroscopic growth following Ghan and Zaveri, JGR (2007)
            call modal_aero_kohler(trradius(n)*1e6,hygro_oma(n)
     *             ,rhl(l),tr_radius,1)
            tr_radius=tr_radius*1e-6

#endif

#ifndef TRACERS_TOMAS
C**** calculate stokes velocity 
            stokevdt=dtsrc*vgs(airden(l),tr_radius
     *           ,tr_dens,visc(l))
#else 
       
            if(n.lt.n_ASO4(1))then
!     no size resolved aerosol tracer (e.g. NH4)
              stokevdt=dtsrc*vgs(airden(l),tr_radius
     *             ,tr_dens,visc(l))
          
            elseif(n.ge.n_ASO4(1)) then
              if(n.eq.n_ASO4(1))THEN
C     02/20/2012 - TOMAS trgrav is modified to be able to reproduce the model output
                call dep_getdp_from_column_trm(i,j,l,Dp_gr,density_gr) 
                do k=1,nbins
C     APR 2015 - FIX vs with slip correction factor (use vgs now)
             
cyhl              vs(I,J,L,k)=density_gr(k)*(Dp_gr(k)**2)*grav
cyhl     *             /18.d0/visc(i,j,l) 

                  vs(L,k) = 
     *                 vgs(airden(l),Dp_gr(k)/2.,density_gr(k),
     *                 visc(l)) 
                enddo
              endif
              binnum=mod(N-n_ASO4(1)+1,NBINS)
              if (binnum.eq.0) binnum=NBINS
              stokevdt=dtsrc*vs(l,binnum) !grav. settling velocity for TOMAS model
            endif               !size-resolved aerosols

#endif
C**** Calculate height differences using geopotential
C**** Next line causes problems in high vertical resolution models. Limit it for now:
C****           fgrfluxd=stokevdt*gbygz(i,j,l) 
            fgrfluxd=min(stokevdt*gbygz(l),1.d0) 
            fluxd = trm_col(l,n)*fgrfluxd ! total flux down
            trm_col(l,n) = trm_col(l,n)*(1.-fgrfluxd)+fluxu
            if (1.-fgrfluxd.le.1d-16) trm_col(l,n) = fluxu
            trmom_col(zmoms,l,n)=trmom_col(zmoms,l,n)*(1.-fgrfluxd)
          end do
          najl = jls_grav(n)
          IF (najl > 0) THEN
            do l=1,lm
              call inc_tajls2(i,j,l,najl,trm_col(l,n)-told(l))
            enddo
          END IF
        end if
      end do

C****

      return
      end subroutine trgrav


      REAL*8 FUNCTION slipc(airden,tr_radius)
!@ returns the slip correction factor
      USE CONSTANT, only : pi,avog,rt2,mair,grav
      IMPLICIT NONE
      real*8, intent(in) :: airden,tr_radius
      real*8, parameter  :: dair=3.65d-10 !m diameter of air molecule
      real*8, parameter  :: s1=1.257d0, s2=0.4d0, s3=1.1d0
      real*8             :: frpath,wmf
C**** wmf is the additional velocity if the particle size is small compared
C**** to the mean free path of the air; important in the stratosphere
      frpath=1.d-3*mair/(pi*rt2*avog*airden*(dair)**2)
      wmf=frpath/tr_radius*(s1+s2*exp(-s3*tr_radius/frpath))
      slipc = 1.d0+wmf
      return
      end function slipc

      REAL*8 FUNCTION vgs(airden,tr_radius,tr_dens,visc)
!@sum vgs returns settling velocity for tracers (m/s)
!@auth Gavin Schmidt/Reha Cakmur
!@ edited by Olivia Clifton (separated slip correction factor into slipc)
!@ and removed hydration (now calculating hydration of oma separately to be consistent w/ radiation
!@ matrix aerosols already hydrated
      USE CONSTANT, only : by3,avog,grav
      IMPLICIT NONE
      real*8, intent(in) ::  airden,tr_radius,tr_dens,visc
      real*8 :: temp,slipc


C**** calculate stokes velocity
      vgs=2.*grav*tr_dens*tr_radius**2/(9.*visc)

C**** apply slip correction factor 
      temp=slipc(airden,tr_radius)
      vgs=temp*vgs
C****
      return
      end function vgs

#endif  /* TRACERS_ON */

      subroutine checktr(subr)
!@sum  CHECKTR Checks whether atmos tracer variables are reasonable
!@vers 2013/03/26
!@auth Gavin Schmidt
#ifdef TRACERS_ON
      USE CONSTANT, only : teeny
      USE RESOLUTION, only: im,jm,lm
      USE ATM_COM, only : q,qcl,qci
      USE GEOM, only : imaxj
      USE SOMTQ_COM, only : qmom
      USE ATM_COM, only : MA
      USE FLUXES, only : atmocn,atmice,atmgla,atmlnd
      use OldTracer_mod, only: trname, t_qlimit
      USE TRACER_COM, only: ntm, trmom, trm, nmom
#ifdef TRACERS_WATER
      USE TRACER_COM, only: trwm
#endif
      USE DOMAIN_DECOMP_ATM, ONLY: GRID, getDomainBounds, AM_I_ROOT
      IMPLICIT NONE
      LOGICAL QCHECKT
      INTEGER I,J,L,N,m, imax,jmax,lmax
      REAL*8 relerr, errmax,errsc,tmax,amax,qmax,wmax,twmax,qmomax(nmom)
     *     ,tmomax(nmom)
#ifdef TRACERS_WATER
      real*8 :: qc
#endif

!@var SUBR identifies where CHECK was called from
      CHARACTER*6, INTENT(IN) :: SUBR
      INTEGER :: J_0, J_1, nj, I_0,I_1
      call getDomainBounds(GRID, J_STRT=J_0, J_STOP=J_1)
      I_0 = grid%I_STRT
      I_1 = grid%I_STOP
      nj = J_1 - J_0 + 1

      !CALL CHECK4(gtracer(1,1,1,J_0),NTM,4,IM,nJ,SUBR,'GTRACE')
      CALL CHECK3(atmocn%gtracer(1,1,J_0),NTM,IM,nJ,SUBR,'GTRACO')
      CALL CHECK3(atmice%gtracer(1,1,J_0),NTM,IM,nJ,SUBR,'GTRACI')
      CALL CHECK3(atmgla%gtracer(1,1,J_0),NTM,IM,nJ,SUBR,'GTRACG')
      CALL CHECK3(atmlnd%gtracer(1,1,J_0),NTM,IM,nJ,SUBR,'GTRACE')
      do n=1,NTM
        CALL CHECK4(trmom(:,:,J_0:J_1,:,n),NMOM,IM,nJ,LM,SUBR,
     *       'X'//trname(n))
        CALL CHECK3(trm(:,J_0:J_1,:,n),IM,nJ,LM,SUBR,trname(n))
#ifdef TRACERS_WATER
        CALL CHECK3(trwm(:,J_0:J_1,:,n),IM,nJ,LM,SUBR,'QCL'//trname(n))
#endif

C**** check for negative tracer amounts (if t_qlimit is set)
        if (t_qlimit(n)) then
          QCHECKT=.false.
          do l=1,lm
          do j=j_0,j_1
          do i=i_0,imaxj(j)
            if (trm(i,j,l,n).lt.0) then
              if (AM_I_ROOT())
     *         write(6,*) "Negative mass for ",trname(n),i,j,l,trm(i,j,l
     *             ,n)," after ",SUBR,"."
              QCHECKT=.true.
            end if
          end do
          end do
          end do
          if (QCHECKT)
     &         call stop_model("CHECKTR: Negative tracer amount",255)
        end if

C**** check whether air mass is conserved

        if (trname(n).eq.'Air') then
          errmax = 0. ; lmax=1 ; imax=I_0 ; jmax=J_0; tmax=0. ; amax=0.
          do l=1,lm
          do j=j_0,j_1
          do i=i_0,imaxj(j)
            relerr=abs(trm(i,j,l,n)-ma(l,i,j))/ma(l,i,j)
            if (relerr.gt.errmax) then
              lmax=l ; imax=i ; jmax=j ; errmax=relerr
              tmax=trm(i,j,l,n) ; amax=ma(l,i,j)
            end if
          end do
          end do
          end do
          print*,"Relative error in air mass after ",trim(subr),":",imax
     *         ,jmax,lmax,errmax,tmax,amax
        end if

#ifdef TRACERS_WATER
        if (trname(n).eq.'Water') then
          errmax = 0. ; lmax=1 ; imax=I_0 ; jmax=J_0
          tmax=0. ; twmax=0. ; qmax=0. ; wmax=0.
          tmomax = 0. ; qmomax = 0. 
          do l=1,lm
          do j=j_0,j_1
          do i=i_0,imaxj(j)
            errsc=(q(i,j,l)+sum(abs(qmom(:,i,j,l))))*ma(l,i,j)
            if (errsc.eq.0.) errsc=1.
            relerr=abs(trm(i,j,l,n)-q(i,j,l)*ma(l,i,j))/errsc
            qc = qcl(i,j,l)+qci(i,j,l) ! add liquid and ice to compare to trwm
            if (qc.gt.0 .and. trwm(i,j,l,n).gt.1.) relerr
     *           =max(relerr,(trwm(i,j,l,n)-qc*ma(l,i,j))
     *           /(qc*ma(l,i,j)))
            if ((qc.eq.0 .and.trwm(i,j,l,n).gt.1) .or. 
     *           (qc.gt.teeny .and.trwm(i,j,l,n).eq.0))
     *           print*,"Condensate water mismatch: ",subr,i,j,l,
     *           trwm(i,j,l,n),qc*ma(l,i,j)
            do m=1,nmom
              relerr=max(relerr,(trmom(m,i,j,l,n)-qmom(m,i,j,l)*ma(l,i,j
     *             ))/errsc)
            end do
            if (relerr.gt.errmax) then
              lmax=l ; imax=i ; jmax=j ; errmax=relerr
              tmax=trm(i,j,l,n) ; qmax=q(i,j,l)*ma(l,i,j)
              twmax=trwm(i,j,l,n) ; wmax=qc*ma(l,i,j)
              tmomax(:)=trmom(:,i,j,l,n)
              qmomax(:)=qmom(:,i,j,l)*ma(l,i,j)
            end if
          end do
          end do
          end do
          print*,"Relative error in water mass after ",trim(subr),":"
     *         ,imax,jmax,lmax,errmax,tmax,qmax,twmax,wmax,(tmomax(m)
     *         ,qmomax(m),m=1,nmom) 
        end if
#endif
      end do
#endif
      return
      end subroutine checktr

      subroutine tracerIO(fid, action)
!@sum tracerIO() provides a generic interface for IO actions
!@+   on the full list of tracers.
!@auth T. Clune
      use ParallelIo_mod
      use pario, only : read_data,defvar,write_data
      use domain_decomp_atm, only : grid
      USE Dictionary_mod
      USE TRACER_COM, only: ntm, TRmom, TRM
      USE TRACER_COM, only: ntm, nmom
#ifdef TRACERS_PHOTOLYSIS
      use photolysis, only: mostRecentNonZeroAlbedo
#endif  /* TRACERS_PHOTOLYSIS */
#ifdef TRACERS_SPECIAL_Shindell
      USE TRCHEM_Shindell_COM, only: pHOx,pNOx,pOx,yCH3O2,yC2O3,
     &yROR,yXO2,yAldehyde,yXO2N,yRXPAR,pNO3
#ifdef TRACERS_dCO
     &,ydCH317O2,ydCH318O2,yd13CH3O2
     &,ydC217O3,ydC218O3,yd13C2O3
     &,yd17OROR,yd18OROR,yd13CROR
     &,yd17Oald,yd18Oald,yd13Cald
     &,yd13CXPAR
#endif  /* TRACERS_dCO */
     &,pClOx,pClx,pOClOx,pBrOx,yCl2,yCl2O2
#ifdef INTERACTIVE_WETLANDS_CH4 
      use TRACER_SOURCES, only: day_ncep,DRA_ch4,sum_ncep,PRS_ch4,
     & HRA_ch4,iday_ncep,i0_ncep,iHch4,iDch4,i0ch4,first_ncep,first_mod,
     & avg_model,avg_ncep
#endif
#endif /* TRACERS_SPECIAL_Shindell */
#ifdef BC_ALB
      USE AEROSOL_SOURCES, only : snosiz
#endif  /* BC_ALB */
      use trdiag_com, only: trcSurfMixR_acc,trcSurfByVol_acc
#ifdef TRACERS_ACETONE
      use trdiag_com, only: trcSurfByVol
#endif
#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      USE fluxes,ONLY : pprec,pevap
      USE trdust_mod,ONLY : hbaij,ricntd
      use trdust_drv, only: def_rsf_trdust
      use trdust_drv, only: new_io_trdust
#endif
#ifdef TRACERS_WATER
      USE TRACER_COM, only: trwm
#endif
#if (defined CUBED_SPHERE) || (defined TRACERS_VOLCEXP) ||\
    (defined PRESC_BB_INJ)
      USE TRACER_COM, only: daily_z
#endif
      use OldTracer_mod, only: trName
      use model_com, only : ioread,iowrite

#ifdef TRACERS_AMP
      use amp_aerosol, only : diam,nactv
#endif

      implicit none

      integer, intent(in) :: fid
      character(len=*), intent(in) :: action

      type (ParallelIo) :: handle
      character(len=:), allocatable :: ijldims
#ifdef TRACERS_SPECIAL_Shindell
      character(len=:), allocatable :: cijdims
#endif
      integer :: n

      ijldims='(dist_im,dist_jm,lm)' 
#ifdef TRACERS_SPECIAL_Shindell
      cijdims='(topLevelOfChemistry,dist_im,dist_jm)'
#endif
      handle = ParallelIo(grid, fid)

      do n=1,NTM
        call doVar(handle,action,trm(:,:,:,n),
     &        'trm_'//trim(trname(n))//ijldims)
        call doVar(handle,action,trmom(:,:,:,:,n),
     &       'trmom_'//trim(trname(n))//'(nmom,dist_im,dist_jm,lm)',
     &       jdim=3)
#ifdef TRACERS_WATER
        call doVar(handle,action,trwm(:,:,:,n),
     &       'trwm_'//trim(trname(n))//ijldims)
#endif
      enddo

#if (defined CUBED_SPHERE) || (defined TRACERS_VOLCEXP) ||\
    (defined PRESC_BB_INJ)
c daily_z is currently only needed for CS
      call doVar(handle,action,daily_z,'daily_z'//ijldims)
#endif

#ifdef TRACERS_SPECIAL_Shindell

      handle = ParallelIo(grid, fid, 'TRACERS_SPECIAL_Shindell')

      call doVar(handle,action,pHOx,'pHOx'//cijdims,jdim=3)
      call doVar(handle,action,pNOx,'pNOx'//cijdims,jdim=3)
      call doVar(handle,action,pNO3,'pNO3'//cijdims,jdim=3)
      call doVar(handle,action,pOx ,'pOx'//cijdims,jdim=3)
      call doVar(handle,action,yCH3O2,'yCH3O2'//cijdims,jdim=3)
#ifdef TRACERS_dCO
      call doVar(handle,action,ydCH317O2,'ydCH317O2'//cijdims,jdim=3)
      call doVar(handle,action,ydCH318O2,'ydCH318O2'//cijdims,jdim=3)
      call doVar(handle,action,yd13CH3O2,'yd13CH3O2'//cijdims,jdim=3)
#endif  /* TRACERS_dCO */
      call doVar(handle,action,yC2O3,'yC2O3'//cijdims,jdim=3)
#ifdef TRACERS_dCO
      call doVar(handle,action,ydC217O3,'ydC217O3'//cijdims,jdim=3)
      call doVar(handle,action,ydC218O3,'ydC218O3'//cijdims,jdim=3)
      call doVar(handle,action,yd13C2O3,'yd13C2O3'//cijdims,jdim=3)
#endif  /* TRACERS_dCO */
      call doVar(handle,action,yROR,'yROR'//cijdims,jdim=3)
#ifdef TRACERS_dCO
      call doVar(handle,action,yd17OROR,'yd17OROR'//cijdims,jdim=3)
      call doVar(handle,action,yd18OROR,'yd18OROR'//cijdims,jdim=3)
      call doVar(handle,action,yd13CROR,'yd13CROR'//cijdims,jdim=3)
#endif  /* TRACERS_dCO */
      call doVar(handle,action,yXO2,'yXO2'//cijdims,jdim=3)
      call doVar(handle,action,yXO2N,'yXO2N'//cijdims,jdim=3)
      call doVar(handle,action,yAldehyde,'yAldehyde'//cijdims,jdim=3)
#ifdef TRACERS_dCO
      call doVar(handle,action,yd17Oald,'yd17Oald'//cijdims,jdim=3)
      call doVar(handle,action,yd18Oald,'yd18Oald'//cijdims,jdim=3)
      call doVar(handle,action,yd13Cald,'yd13Cald'//cijdims,jdim=3)
#endif  /* TRACERS_dCO */
      call doVar(handle,action,yRXPAR,'yRXPAR'//cijdims,jdim=3)
#ifdef TRACERS_dCO
      call doVar(handle,action,yd13CXPAR,'yd13CXPAR'//cijdims,jdim=3)
#endif  /* TRACERS_dCO */
      call doVar(handle,action,pClOx,'pClOx'//cijdims,jdim=3)
      call doVar(handle,action,pClx,'pClx'//cijdims,jdim=3)
      call doVar(handle,action,pOClOx,'pOClOx'//cijdims,jdim=3)
      call doVar(handle,action,pBrOx,'pBrOx'//cijdims,jdim=3)
      call doVar(handle,action,yCl2,'yCl2'//cijdims,jdim=3)
      call doVar(handle,action,yCl2O2,'yCl2O2'//cijdims,jdim=3)

#ifdef INTERACTIVE_WETLANDS_CH4 
      handle = ParallelIo(grid, fid, 'INTERACTIVE_WETLANDS_CH4')

      call doVar(handle,action,day_ncep,
     &     'day_ncep(dist_im,dist_jm,max_days,nra_ncep)')
      call doVar(handle,action,dra_ch4,
     &     'dra_ch4(dist_im,dist_jm,max_days,nra_ch4)')
      call doVar(handle,action,sum_ncep,
     &     'sum_ncep(dist_im,dist_jm,nra_ncep)')
      call doVar(handle,action,prs_ch4,
     &     'prs_ch4(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,HRA_ch4,
     &     'HRA_ch4(dist_im,dist_jm,maxHR_ch4,nra_ch4)')
      call doVar(handle,action,i0ch4,
     &     'i0ch4(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,iDch4,
     &     'iDch4(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,iHch4,
     &     'iHch4(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,first_mod,
     &     'first_mod(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,avg_model,
     &     'avg_model(dist_im,dist_jm,nra_ch4)')
      call doVar(handle,action,avg_ncep, 
     &     'avg_ncep(dist_im,dist_jm,nra_ncep)')
      select case (action)
        case('define')
          call defvar(grid,fid,iday_ncep,'iday_ncep(nra_ncep)')
          call defvar(grid,fid,i0_ncep,'i0_ncep(nra_ncep)')
          call defvar(grid,fid,first_ncep,'first_ncep(nra_ncep)')
        case('read_dist')
          call read_data(grid,fid,'iday_ncep',iday_ncep,
     &       bcast_all=.true.)
          call read_data(grid,fid,'i0_ncep',i0_ncep,
     &       bcast_all=.true.)
          call read_data(grid,fid,'first_ncep',first_ncep,
     &       bcast_all=.true.)
        case ('write_dist')
          call write_data(grid,fid,'iday_ncep',iday_ncep)
          call write_data(grid,fid,'i0_ncep',i0_ncep)
          call write_data(grid,fid,'first_ncep',first_ncep)
      end select
!        call doVar(handle,action,iday_ncep,'iday_ncep(nra_ncep)')
!        call doVar(handle,action,i0_ncep,'i0_ncep(nra_ncep)')
!        call doVar(handle,action,first_ncep,'first_ncep(nra_ncep)')
#endif /* INTERACTIVE_WETLANDS_CH4 */
#endif /* TRACERS_SPECIAL_Shindell */

      call doVar(handle,action,trcSurfMixR_acc
     &     ,'trcSurfMixR_acc(dist_im,dist_jm,ntm)')
      call doVar(handle,action,trcSurfByVol_acc
     &     ,'trcSurfByVol_acc(dist_im,dist_jm,ntm)')
#ifdef TRACERS_ACETONE
      call doVar(handle,action,trcSurfByVol
     &     ,'trcSurfByVol(dist_im,dist_jm,ntm)')
#endif

#ifdef TRACERS_PHOTOLYSIS
      handle = ParallelIo(grid, fid,'TRACERS_PHOTOLYSIS')
      call doVar(handle,action,mostRecentNonZeroAlbedo,
     & 'mostRecentNonZeroAlbedo(dist_im,dist_jm)')
#endif  /* TRACERS_PHOTOLYSIS */

#if (defined TRACERS_DUST) || (defined TRACERS_MINERALS)
      handle = ParallelIo(grid, fid,
     &   'TRACERS_DUST||TRACERS_MINERALS')
      call doVar(handle,action,hbaij,'hbaij(dist_im,dist_jm)')
      call doVar(handle,action,ricntd,'ricntd(dist_im,dist_jm)')
      call doVar(handle,action,pprec,'pprec(dist_im,dist_jm)')
      call doVar(handle,action,pevap,'pevap(dist_im,dist_jm)')

      select case (action)
      case ('define')
         call def_rsf_trdust(fid)
      case ('read_dist')
         call new_io_trdust(fid,ioread)
      case ('write_dist')
         call new_io_trdust(fid,iowrite)
      end select

#endif

#ifdef BC_ALB
      handle = ParallelIo(grid, fid,'BC_ALB')
      call doVar(handle,action,snosiz,'snosiz(dist_im,dist_jm)')
#endif  /* BC_ALB */

#ifdef TRACERS_AMP
      ! restartability hack until matrix code refactored to
      ! re-diagnose these qtys on demand
      handle = ParallelIo(grid, fid,'TRACERS_AMP')
      call doVar(handle,action,diam,
     &     'amp_diam(dist_im,dist_jm,lm,nmodes)')
      call doVar(handle,action,nactv,
     &     'amp_nactv(dist_im,dist_jm,lm,nmodes)')
#endif

      return
      end subroutine tracerIo
     

#ifdef CACHED_SUBDD
      subroutine tijh_defs(arr,nmax,decl_count)
! 2D tracer outputs (model horizontal grid).
! Each tracer output must be declared separately (no bundling).
      use model_com, only : dtsrc,nday
      use subdd_mod, only : info_type, sched_rad, reduc_max
      use OldTracer_mod, only: trname
#ifdef TRACERS_WATER 
      use OldTracer_mod, only : nWater, tr_wd_type
#endif
      use tracer_com, only : ntm
      use OldTracer_mod, only : to_volume_MixRat
      use trdiag_com, only : save_dry_aod
      use radpar, only: nraero_aod=>NTRACE
      use rad_com, only: ntrix_aod,nraero_rf,ntrix_rf,diag_fc
      use RunTimeControls_mod, only: tracers_amp, tracers_tomas
! info_type_ is a homemade structure constructor for older compilers
      use subdd_mod, only : info_type_
      implicit none
      integer :: nmax,decl_count,isub_ntrix
      type(info_type) :: arr(nmax)
! types of aods to be saved
! The name will be any combination of {,TRNAME}{as,cs,dry}{,a}aod
      character(len=10), dimension(3) :: ssky=(/'as ','cs ','dry'/),
     &                                lsky=(/'All-sky  ','Clear-sky',
     &                                       'Dry aeros'/)
      character(len=10), dimension(2) :: sabs=(/' ','a'/),
     &                                labs=(/'          ','absorption'/)
      character(len=13), dimension(3) ::
     &                              sfrc=(/'swftoa','swfsfc','lwf   '/),
     &         lfrc=(/'shortwave_toa','shortwave_sfc','longwave     '/)
      character(len=10) :: spcname
! types of PM/tracer surface amounts to be saved
! The name will be any combination of PM{2p5,10}{l1,s}{m,c}
!                            I.E. {species}{location}{units}
! and similar format for any tracer: trname(){l1,s}{m,c}.
! In practice did not include the l1c (L=1 cocentration) case
      character(len=20), dimension(2) :: 
     &   ssiz=(/'2p5','10 '/), lsiz=(/'PM2.5','PM10 '/),
     &   sloc=(/'l1','s '/),   lloc=(/'L=1    ','Surface'/),
     &   sunt=(/'m','c'/),
     &   lunt=(/'Mass Mixing Ratio','Concentration    '/),
     &   uunt=(/'kg species / kg air','kg m-3             '/)
      character*80 :: unitString,unitString2
      integer :: s,a,n,f,u,l,p

      decl_count = 0

! Optical Depths

      do s=1,size(ssky)
      if (ssky(s).eq.'dry' .and. save_dry_aod.eq.0) cycle
      do a=1,size(sabs)
      isub_ntrix=0
      do n=1,nraero_aod+1 ! +1 for total
        if (n<=nraero_aod) then
          spcname = trim(trname(ntrix_aod(n)))
          if (trim(spcname).eq.'Clay') then
            isub_ntrix=isub_ntrix+1
            write(spcname,'(a4,i1)') trim(spcname), isub_ntrix
          endif
        else
          spcname = ''
        endif
        arr(next()) = info_type_(
     &    sname = trim(spcname)//trim(ssky(s))//trim(sabs(a))//'aod',
     &    lname = trim(spcname)//' '//trim(lsky(s))//' '//
     &            trim(labs(a))//' aerosol optical depth',
     &    units = '-',
     &    sched = sched_rad
     &       )
      enddo ! n
      enddo ! a
      enddo ! s

! Forcing

      do f=1,size(sfrc)
      do n=1,nraero_rf
        if (diag_fc==2) then
          spcname = trim(trname(ntrix_rf(n)))
        else if (diag_fc==1) then
          if (tracers_amp) then
            spcname='AMP'
          elseif (tracers_tomas) then
            spcname='TOMAS'
          else
            spcname='OMA'
          endif
        endif
        arr(next()) = info_type_(
     &    sname = trim(sfrc(f))//'_'//trim(spcname),
     &    lname = trim(spcname)//' '//trim(lfrc(f))//' forcing',
     &    units = 'W m-2',
     &    sched = sched_rad
     &       )
      enddo ! n
      enddo ! f

! Surface Tracer Amount

      do n=1,ntm
        ! L=1 and surface mixing ratios:
        u=1
        if (to_volume_MixRat(n) == 1) then
          unitString='mole species / mole air'
          unitString2='Volume Mixing Ratio'
        else
          unitString=trim(uunt(u))
          unitString2=trim(lunt(u))
        endif 
        do l=1,size(sloc) 
          arr(next()) = info_type_(
     &    sname = trim(trname(n))//trim(sloc(l))//trim(sunt(u)),
     &    lname=
     &    trim(trname(n))//' '//trim(lloc(l))//' '//trim(unitString2),
     &    units = trim(unitString)
     &    )
        end do
        ! surface concentrations:
        u=2
        l=2
        arr(next()) = info_type_(
     &  sname = trim(trname(n))//trim(sloc(l))//trim(sunt(u)),
     &  lname=trim(trname(n))//' '//trim(lloc(l))//' '//trim(lunt(u)),
     &  units = trim(uunt(u))
     &  )
      end do ! ntm

! Tracer Load (column mass)

      do n=1,ntm
        arr(next()) = info_type_(
     &  sname = trim(trname(n))//'load',
     &  lname = trim(trname(n))//' Column Mass',
     &  units = 'kg m-2'
     &  )
      end do ! ntm

#ifdef TRACERS_AMP
      arr(next()) = info_type_(
     &  sname = 'ampBCload',
     &  lname = 'BC Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampDustload',
     &  lname = 'Dust Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampNH4load',
     &  lname = 'NH4 Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampNO3load',
     &  lname = 'NO3 Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampOAload',
     &  lname = 'OA Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampSO4load',
     &  lname = 'SO4 Column Mass',
     &  units = 'kg m-2'
     &  )
      arr(next()) = info_type_(
     &  sname = 'ampSSload',
     &  lname = 'SS Column Mass',
     &  units = 'kg m-2'
     &  )
#endif

! Surface Particulate Matter Amount

      do p=1,size(ssiz)
        ! L=1 and surface mixing ratios (always mass):
        u=1
        do l=1,size(sloc)
          arr(next()) = info_type_(
     &    sname = 'PM'//trim(ssiz(p))//trim(sloc(l))//trim(sunt(u)),
     &    lname = 
     &     trim(lsiz(p))//' '//trim(lloc(l))//' '//trim(lunt(u)),
     &    units = trim(uunt(u))
     &    )
        end do 
        ! surface concentrations:
        u=2
        l=2
        arr(next()) = info_type_(
     &  sname = 'PM'//trim(ssiz(p))//trim(sloc(l))//trim(sunt(u)),
     &  lname =
     &   trim(lsiz(p))//' '//trim(lloc(l))//' '//trim(lunt(u)),
     &  units = trim(uunt(u))
     &  )
      end do ! p (PM size)

#ifdef TRACERS_SPECIAL_Shindell
      arr(next()) = info_type_(
     &  sname = 'MRNO2l1', ! because not a tracer
     &  lname = 'L=1 NO2 mixing ratio',
     &  units = 'mole species / mole air'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'MRNOl1', ! because not a tracer
     &  lname = 'L=1 NO mixing ratio',
     &  units = 'mole species / mole air'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'MRO3l1max', ! because not a tracer
     &  lname = 'Maximum Daily L=1 O3 mixing ratio',
     &  units = 'mole species / mole air',
     &  reduc = reduc_max
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'O3col', ! not "load", to contrast with tracers
     &  lname = 'O3 Column Mass',
     &  units = 'kg m-2'
     &  )
#endif

#ifdef TRACERS_WATER
! Water tracer/isotope precipitation 
      do n=1,ntm
        if (tr_wd_type(n).eq.nWater) then !Is it a water (isotope) tracer?
          !Set 'subdd' variable/object meta-data:
          arr(next()) = info_type_(
     &      sname = trim(trname(n))//'_in_prec',
     &      lname = trim(trname(n))//' in Precip',
     &      units = 'kg/m^2/s',
     &      scale = 1./dtsrc !kg/m2 -> kg/m2/s
     &      )
        end if
      end do !ntm

! Water tracer/isotope evaporation
      do n=1,ntm
        if (tr_wd_type(n).eq.nWater) then !Is it a water (isotope) tracer?
          !Set 'subdd' variable/object meta-data:
          arr(next()) = info_type_(
     &      sname = trim(trname(n))//'_in_evap',
     &      lname = trim(trname(n))//' in Evap',
     &      units = 'kg/m^2/s',
     &      scale = 1./dtsrc !kg/m2 -> kg/m2/s
     &      )
        end if
      end do !ntm
#endif

      return
      contains
      integer function next()
      decl_count = decl_count + 1
      next = decl_count
      end function next
      end subroutine tijh_defs


      subroutine tijlh_defs(cp_mode,arr,nmax,decl_count)
! 3D tracer outputs (model horizontal grid and model layers
! or constant pressure levels).
! Each tracer output must be declared separately (no bundling).
      use model_com, only : dtsrc,nday
      use subdd_mod, only : info_type, sched_rad
! info_type_ is a homemade structure constructor for older compilers
      use subdd_mod, only : info_type_
      use tracer_com, only : ntm
      use OldTracer_mod, only: trname
      use radpar, only: nraero_aod=>NTRACE
      use rad_com, only: ntrix_aod
      use OldTracer_mod, only : to_volume_MixRat
      use trdiag_com, only : save_dry_aod
      implicit none
      integer :: nmax,decl_count
      integer :: n,isub_ntrix
      character*80 :: unitString
      type(info_type) :: arr(nmax)
! types of aods to be saved
! The name will be any combination of {,TRNAME}{as,cs,dry}{,a}aod3d
      character(len=10), dimension(3) :: ssky=(/'as ','cs ','dry'/),
     &                                lsky=(/'All-sky  ','Clear-sky',
     &                                       'Dry aeros'/)
      character(len=10), dimension(2) :: sabs=(/' ','a'/),
     &                               labs=(/'          ','absorption'/),
     &                               lcoef=(/'extinction','absorption'/)
      character(len=10) :: spcname
      integer :: s,a
!@var cp_mode whether defs are for a constant-pressure category.
!@+   Use this flag to skip defs that are not desired/relevant for CP.
      logical :: cp_mode
      logical :: layer_mode
      character(len=2) :: sfx

      layer_mode = .not. cp_mode
      if(cp_mode) then
        sfx = 'cp'
      else
        sfx = ''
      end if

      decl_count = 0

      ! First, diagnostics available for all tracers:
      do n=1,ntm
        ! 3D mixing ratios (SUBDD string is just tracer name, possibly
        ! with appendage if constant-pressure):
        if (to_volume_MixRat(n) == 1) then
          unitString='mole species / mole air'
        else
          unitString='kg species / kg air'
        endif
        arr(next()) = info_type_(
     &    sname = trim(trname(n))//trim(sfx),
     &    lname = trim(trname(n))//' mixing ratio',
     &    units = trim(unitString)
     &    )
      end do ! tracers loop

      ! Some aerosol diags only on model levels (not constant pressures):
      if(layer_mode) then

#ifdef TRACERS_AMP
        ! AMP aerosol diameters
        do n=1,ntm
          spcname=trim(trname(n))
          if((spcname(1:2) == 'N_') .and. (spcname(6:7) == '_1'))then
            arr(next()) = info_type_(
     &       sname = 'd'//trim(trname(n)),
     &       lname = trim(trname(n))//' mass mean diameter',
     &       units = 'm'
     &       )
          end if
        end do
#endif  /* TRACERS_AMP */

        ! 3d AOD
        do s=1,size(ssky)
          if (ssky(s).eq.'dry' .and. save_dry_aod.eq.0) cycle
          do a=1,size(sabs)
            isub_ntrix=0
            do n=1,nraero_aod+1 ! +1 for total
              if (n<=nraero_aod) then
                spcname = trim(trname(ntrix_aod(n)))
                if (trim(spcname).eq.'Clay') then
                  isub_ntrix=isub_ntrix+1
                  write(spcname,'(a4,i1)') trim(spcname), isub_ntrix
                endif
              else
                spcname = ''
              end if
              arr(next()) = info_type_(
     &          sname = trim(spcname)//
     &                  trim(ssky(s))//trim(sabs(a))//'aod3d',
     &          lname = trim(spcname)//' '//trim(lsky(s))//' '//
     &                  trim(labs(a))//' aerosol optical depth',
     &          units = '-',
     &          sched = sched_rad
     &          )
              arr(next()) = info_type_(
     &          sname = trim(spcname)//
     &                  trim(ssky(s))//trim(sabs(a))//'bcoef3d',
     &          lname = trim(spcname)//' '//trim(lsky(s))//' '//
     &                  trim(lcoef(a))//' coefficient',
     &          units = 'm-1',
     &          sched = sched_rad
     &          )
            end do ! n
          end do ! a
        end do ! s

      end if ! layer_mode

      ! Specialty tracer diags on model or constant pressure levels:

#ifdef TRACERS_SPECIAL_Shindell
      arr(next()) = info_type_(
     &  sname = 'MRNO2'//trim(sfx), ! because not a tracer
     &  lname = 'NO2 mixing ratio',
     &  units = 'mole species / mole air'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'MRNO'//trim(sfx), ! because not a tracer
     &  lname = 'NO mixing ratio',
     &  units = 'mole species / mole air'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'MRO3'//trim(sfx), ! because not a tracer
     &  lname = 'O3 mixing ratio',
     &  units = 'mole species / mole air'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'OH_conc'//trim(sfx), ! because not a tracer
     &  lname = 'OH concentration',
     &  units = 'molecules cm-3'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'HO2_conc'//trim(sfx), ! because not a tracer
     &  lname = 'HO2 concentration',
     &  units = 'molecules cm-3'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'JO1D'//trim(sfx), ! because not a tracer
     &  lname = 'O3-->O1D+O2 photolysis rate',
     &  units = 's-1'
     &  )
C
      arr(next()) = info_type_(
     &  sname = 'JNO2'//trim(sfx), ! because not a tracer
     &  lname = 'NO2-->NO+O photolysis rate',
     &  units = 's-1'
     &  )
#endif /* TRACERS_SPECIAL_Shindell */

      return
      contains
      integer function next()
      decl_count = decl_count + 1
      next = decl_count
      end function next
      end subroutine tijlh_defs


      subroutine accumCachedTracerSUBDDs

      use domain_decomp_atm, only : grid
      use resolution, only: LM
      use atm_com, only    : byma
      use tracer_com, only : ntm,trm
      use OldTracer_mod, only : mass2vol
      use OldTracer_mod, only: trname, pm10fact, pm2p5fact
      use OldTracer_mod, only : to_volume_MixRat
      use trdiag_com, only : trcsurf,trcSurfByVol
      use subdd_mod, only : subdd_groups,subdd_type,subdd_ngroups
     &     ,inc_subdd,find_groups, LmaxSUBDD
      USE TRDIAG_COM, only : taijn=>taijn_loc
#ifdef TRACERS_AMP
      use AMP_AEROSOL, only: ampPM2p5, ampPM10
#endif
      implicit none
      integer :: igrp,ngroups,grpids(subdd_ngroups)
      type(subdd_type), pointer :: subdd
      integer :: L, n, k
      integer :: layer_or_cp, Ltop
      character(len=16) :: vname,grpname
      real*8, dimension(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo) :: sddarr2d
      real*8, dimension(grid%i_strt_halo:grid%i_stop_halo,
     &                  grid%j_strt_halo:grid%j_stop_halo,
     &                  LM                               ) :: sddarr3d
      real*8 :: convert

      ! Standard tracer 3D diags on model levels or constant
      ! pressure levels:
      do layer_or_cp=1,2 ! model layers and constant-pressure categories
        if(layer_or_cp==1) then ! model layers
          grpname = 'taijlh'
          Ltop=LmaxSUBDD
        else                    ! CP levels
          grpname = 'taijph'
          Ltop=LM
        end if
        call find_groups(trim(grpname),grpids,ngroups)
        do igrp=1,ngroups
          subdd => subdd_groups(grpids(igrp))
          do k=1,subdd%ndiags
            vname=trim(subdd%name(k))
            if(layer_or_cp==2) then ! remove trailing cp from name
              vname = vname(1:len_trim(vname)-2)
            end if
            ntm_loop: do n=1,ntm
              ! tracer 3D mixing ratios (SUBDD names are just tracer
              ! name or {trname}cp ):
              if(trim(trname(n)) == trim(vname)) then
                if (to_volume_MixRat(n) == 1) then
                  convert=mass2vol(n)
                else
                  convert=1.d0
                end if
                do L=1,Ltop
                  sddarr3d(:,:,L) =
     &            trm(:,:,L,n)*convert*byma(L,:,:)
                end do
                call inc_subdd(subdd,k,sddarr3d)
                exit ntm_loop
              end if
            end do ntm_loop
          end do ! k
        end do ! igroup
      end do ! layer_or_cp

      ! Tracer 2D I-J diags
      call find_groups('taijh',grpids,ngroups)
      do igrp=1,ngroups
      subdd => subdd_groups(grpids(igrp))
      diag_loop: do k=1,subdd%ndiags
        ntm_loop3: do n=1,ntm

          ! tracer surface mixing ratios:
          if(trim(trname(n))//'sm'.eq.trim(subdd%name(k))) then
            if (to_volume_MixRat(n) == 1) then
              sddarr2d(:,:)=trcsurf(:,:,n)*mass2vol(n)
            else
              sddarr2d(:,:)=trcsurf(:,:,n)
            endif
            call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop
          end if

          ! tracer surface concentrations:
          if(trim(trname(n))//'sc'.eq.trim(subdd%name(k))) then
            sddarr2d(:,:)=trcSurfByVol(:,:,n)
            call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop
          end if

          ! tracer L=1 mixing ratios:
          if(trim(trname(n))//'l1m'.eq.trim(subdd%name(k))) then
            if (to_volume_MixRat(n) == 1) then
              sddarr2d(:,:)=
     &          trm(:,:,1,n)*mass2vol(n)*byma(1,:,:)
            else
              sddarr2d(:,:)=trm(:,:,1,n)*byma(1,:,:)
            endif
            call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop
          end if

          ! tracer column load:
          if(trim(trname(n))//'load'.eq.trim(subdd%name(k))) then
            sddarr2d(:,:)=sum(trm(:,:,:,n),dim=3)
            call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop
          end if

        enddo ntm_loop3

! Particulate matter to be treated differently for mass-based 
! aerosols or not:
#ifdef TRACERS_TOMAS
        select case(trim(subdd%name(k)))
        case('PM2p5sm','PM2p5l1m','PM2p5sc',
     &    'PM10sm','PM10l1m','PM10sc')
          call tomas_pm_subdd_accum(subdd,k,trim(subdd%name(k)))
          cycle diag_loop
        end select
#elif (defined TRACERS_AMP)
        select case(trim(subdd%name(k)))
        ! L=1 PM2.5 mass mixing ratio:
         case('PM2p5l1m')
         sddarr2d(:,:)= ampPM2p5(:,:)     ! kg/kg air
         call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        ! L=1 PM10 mass mixing ratio:
         case('PM10l1m')
         sddarr2d(:,:)= ampPM10(:,:)      ! kg/kg air
         call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        end select
#else
        select case(trim(subdd%name(k)))
        ! surface PM2.5 mass mixing ratio:
        case('PM2p5sm')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm2p5fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm2p5fact(n)*trcsurf(:,:,n)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop 

        ! L=1 PM2.5 mass mixing ratio:
        case('PM2p5l1m')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm2p5fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm2p5fact(n)*
     &            trm(:,:,1,n)*byma(1,:,:)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        ! surface PM2.5 concentration:
        case('PM2p5sc')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm2p5fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm2p5fact(n)*trcSurfByVol(:,:,n)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        ! surface PM10 mass mixing ratio:
        case('PM10sm')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm10fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm10fact(n)*trcsurf(:,:,n)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop  

        ! L=1 PM10 mass mixing ratio:
        case('PM10l1m')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm10fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm10fact(n)*
     &            trm(:,:,1,n)*byma(1,:,:)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        ! surface PM10 concentration:
        case('PM10sc')
          sddarr2d(:,:)=0.d0
          do n=1,ntm
            if(pm10fact(n)/=0.)
     &      sddarr2d(:,:)=sddarr2d(:,:)+pm10fact(n)*trcSurfByVol(:,:,n)
          end do
          call inc_subdd(subdd,k,sddarr2d) ; cycle diag_loop

        end select
#endif /* -not- TRACERS_TOMAS, TRACERS_AMP sections */

      enddo diag_loop
      enddo ! igroup

      end subroutine accumCachedTracerSUBDDs
#endif /* CACHED_SUBDD */

#if (defined TRACERS_SPECIAL_Shindell) || (defined TRACERS_AEROSOLS_Koch) ||\
    (defined TRACERS_AMP) || (defined TRACERS_TOMAS) ||\
    (defined TRACERS_GASEXCH_GCC)

      subroutine get_3d_tracer
     & (nTracer,nIndex,fileName,year,xday,phi,
     &  stream3d,tracer_src,streamS3d)
!@sum  get_3d_tracer to define the 3D source of tracers from aircraft
!@+                  and rockets
!@auth Drew Shindell? / Greg Faluvegi / Jean Learner
      use RESOLUTION, only : im,jm,lm
      use model_com, only: itime, master_yr
      use domain_decomp_atm, only: GRID,getDomainBounds,write_parallel
      use constant, only: bygrav
      use OldTracer_mod, only: itime_tr0, trname, scale_aircraft
      use OldTracer_mod, only: set_first_aircraft, first_aircraft
      use TRACER_COM, only: ntm_chem_beg,ntm_chem_end,nAircraft
      use OldTracer_mod, only: set_first_rocket, first_rocket
      use TRACER_COM, only: nRocket
      use TRACER_COM, only: emiss_over_model_top_at_LM
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || \
    (defined TRACERS_TOMAS)
      use TRACER_COM, only: aer_int_yr
      use TRACER_COM, only: SO2_int_yr
      use TRACER_COM, only: NH3_int_yr
      use TRACER_COM, only: BC_int_yr
      use TRACER_COM, only: OC_int_yr
#endif
      use Dictionary_mod, only: get_param
      use RAD_COM, only: o3_yr
      use timestream_mod, only : read_stream, timestream, init_stream,
     &                           getname_firstfile
      use pario, only: par_open,par_close,get_dimlen,read_data

      IMPLICIT NONE
 
!@param LM3d the number of layers of 3d data read from file
      INTEGER :: LM3d
!@var fileName the name of the 3d source file for this tracer
      character(len=*), intent(IN) :: fileName
!@var nTracer the index of the tracer in current call in ntm arrays
!@+   for example n_NOx or n_M_BC1_BC
      integer, intent(IN) :: year,xday,nTracer
      integer :: xyear
      real*8, dimension(GRID%I_STRT_HALO:GRID%I_STOP_HALO,
     &                  GRID%J_STRT_HALO:GRID%J_STOP_HALO,LM),
     &     intent(IN) :: phi
!@var tracer_src 3D source of tracer from 3d source (on model levels)
      real*8, dimension(GRID%I_STRT:GRID%I_STOP,
     &                  GRID%J_STRT:GRID%J_STOP,LM), intent(out)
     &     :: tracer_src

      integer :: fileUnit 
      integer L,i,j,k,LL
      interface
        real*8 function get_src_fact(n,is_bb,OA_not_OC)
          integer, intent(in) :: n
          logical, intent(in) :: is_bb
          logical, intent(in), optional :: OA_not_OC
        end function get_src_fact
      end interface

!@var src holds the tracer source returned from actual reading routine
      real*8, dimension(:,:,:), allocatable :: src
!@var scaling the scaling for this source if it was requested. For NOW
!+ only two-dimensional
      real*8, dimension(GRID%I_STRT:GRID%I_STOP
     *     ,GRID%J_STRT:GRID%J_STOP):: scaling
!@var zmod approx. geometric height at model layer(m), phi/grav
      real*8, dimension(LM) :: zmod
!@var z3d midpoint heights of 3d emission layers (km)
      real*8, dimension(:), allocatable :: z3d
      integer :: J_1, J_0, I_0, I_1, do_ppm
      integer :: copy_master_yr, cyclic_yr
      ! note the stream3d is passed before init_stream is called for it
!@var nIndex equals to either nAircraft or nRocket, to control functionality
      integer, intent(in) :: nIndex
      type(timestream), intent(in) :: stream3d
      type(timestream), intent(in), optional :: streamS3d
      character(len=200) :: fname1
      integer :: fid

! Read the file and interpolate each day

      tracer_src = 0.d0

      if (nTracer == 0) then
        call stop_model("nTracer undefined in get_3d_tracer",255)
      end if

      if (itime < itime_tr0(nTracer)) return ! returns w/o doing reading

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1)

      ! Determine year of emissions to use and whether the timestream
      ! should be cyclic or not.  (Actual year 'year' has been passed
      ! in. Allow override of this if, say, {master,o3,aer_int}_yr 
      ! non-zero):
      call get_param('master_yr',copy_master_yr,default=0)
      cyclic_yr=copy_master_yr
#ifdef TRACERS_SPECIAL_Shindell
      if ((nTracer>=ntm_chem_beg).and.(nTracer<=ntm_chem_end)) then
        call get_param('o3_yr',cyclic_yr,default=copy_master_yr)
        select case (trname(nTracer))
        case ('NOx')
          call get_param('NOx_yr',cyclic_yr,default=cyclic_yr)
        case ('CO')
          call get_param('CO_yr',cyclic_yr,default=cyclic_yr)
        case ('Alkenes', 'Paraffin')
          call get_param('VOC_yr',cyclic_yr,default=cyclic_yr)
        end select
      else
#endif
#if (defined TRACERS_AEROSOLS_Koch) || (defined TRACERS_AMP) || \
    (defined TRACERS_TOMAS)
        call get_param('aer_int_yr',cyclic_yr,default=copy_master_yr)
        select case (trname(nTracer))
        case ('SO2', 'SO4', 'M_ACC_SU', 'M_AKK_SU', 'ASO4__01')
          call get_param('SO2_int_yr',cyclic_yr,default=cyclic_yr)
        case ('NH3')
          call get_param('NH3_int_yr',cyclic_yr,default=cyclic_yr)
        case ('BCII', 'BCB', 'M_BC1_BC', 'M_BOC_BC', 'AECOB_01')
          call get_param('BC_int_yr',cyclic_yr,default=cyclic_yr)
        case ('OCII', 'OCB', 'M_OCC_OC', 'M_BOC_OC', 'AOCOB_01',
     &        'vbsAm2', 'vbsAm1', 'vbsAz', 'vbsAp1', 'vbsAp2',
     &        'vbsAp3', 'vbsAp4', 'vbsAp5', 'vbsAp6')
          call get_param('OC_int_yr',cyclic_yr,default=cyclic_yr)
        end select
#endif
#ifdef TRACERS_SPECIAL_Shindell
      end if
#endif
      xyear=year
      if (cyclic_yr > 0) xyear=cyclic_yr

      ! Monthly sources are interpolated to the current day
      ! Units are kg m-2 s-1, so no conversion is necessary:

      if (nIndex==nAircraft) then
        if(first_aircraft(nTracer)) then
          call set_first_aircraft(nTracer, .false.)
          call get_param('nc_emis_use_ppm_interp',do_ppm,default=1)
          if (do_ppm==1) then
            call init_stream(grid,stream3d,fileName,
     &      trim(trname(nTracer)), 0d0, 1d30, 'ppm',
     &      xyear, xday, cyclic = (cyclic_yr > 0) )
          else
            call init_stream(grid,stream3d,fileName,
     &      trim(trname(nTracer)), 0d0, 1d30, 'linm2m',
     &      xyear, xday, cyclic = (cyclic_yr > 0) )
          end if
          if(scale_aircraft(nTracer))then
            call init_stream(grid,streamS3d,trim(fileName)//'_scale',
     &      'scale', -1d10, 1d10, 'linm2m',
     &      xyear, xday, cyclic = (cyclic_yr > 0) )
          end if
        end if
      else if (nIndex==nRocket) then
        if(first_rocket(nTracer)) then
          call set_first_rocket(nTracer, .false.)
          call get_param('nc_emis_use_ppm_interp',do_ppm,default=1)
          if (do_ppm==1) then
            call init_stream(grid,stream3d,fileName,
     &      trim(trname(nTracer)), 0d0, 1d30, 'ppm',
     &      xyear, xday, cyclic = (cyclic_yr > 0) )
          else
            call init_stream(grid,stream3d,fileName,
     &      trim(trname(nTracer)), 0d0, 1d30, 'linm2m',
     &      xyear, xday, cyclic = (cyclic_yr > 0) )
          end if
        end if
      end if

      if (.not.allocated(src)) then
        call getname_firstfile(stream3d,fname1)
        fid=par_open(grid,trim(fname1),'read')
        LM3d=get_dimlen(grid,fid,'lev')
        allocate(z3d(LM3d))
        allocate(src(I_0:I_1,J_0:J_1,LM3d))
        call read_data(grid,fid,'lev',z3d,bcast_all=.true.)
        call par_close(grid,fid)
      endif

      ! update source:
      call read_stream(grid,stream3d,xyear,xday,src)

      ! update and apply potential scaling:
      if (nIndex==nAircraft) then
        if(scale_aircraft(nTracer))then
          call read_stream(grid,streamS3d,xyear,xday,scaling)
          do LL=1,LM3d
            src(:,:,LL)=src(:,:,LL)*scaling(:,:)
          end do
        end if
      end if

      ! Place 3d sources onto model levels:

      do j=J_0,J_1
        do i=I_0,I_1
          zmod(:)=phi(i,j,:)*bygrav*1.d-3 ! km
          do LL=1,LM3d
            if (src(i,j,LL) > 0.d0) then
              loop_L: do L=1,LM
                if (z3d(LL) <= zmod(L)) then
                  tracer_src(i,j,L)=tracer_src(i,j,L)+src(i,j,LL)
                  exit loop_L
                end if
                if (L==LM) then
                  if (emiss_over_model_top_at_LM) then
                    tracer_src(i,j,L)=tracer_src(i,j,L)+src(i,j,LL)
                    exit loop_L
                  else
                    call stop_model("get_3d_tracer level problem",255)
                  endif ! allow use of emissions overflow?
                endif ! hit model top?
              end do loop_L
            end if ! is there a source?
          end do ! LL 3d levels
        end do ! I
      end do ! J

      tracer_src(I_0:I_1,J_0:J_1,:) =
     &     tracer_src(I_0:I_1,J_0:J_1,:)*get_src_fact(nTracer,.false.)

      return
      end subroutine get_3d_tracer

#endif /* defined TRACERS_SPECIAL_Shindell or Koch/AMP/TOMAS aerosols */

!=======================================================================

#if defined(TRACERS_AEROSOLS_Koch) || defined(TRACERS_AMP) || \
    defined(TRACERS_TOMAS) || defined(TRACERS_PASSIVE)
      subroutine calc_and_apply_expo_decay(i,j,time,tri,tre)
!@sum Calculate and apply the exponetial decay of a tracer, and
!@+   (optionally) form a product.
!@auth Kostas Tsigaridis

      use TimeConstants_mod, only: SECONDS_PER_DAY
      use MODEL_COM,  only: dtsrc
      use TRACER_COM, only: trm_col
      use TRACER_COM, only: nChemistry
      use OldTracer_mod, only: tr_mm
      use FLUXES, only: tr3dsource
      use apply3d, only: apply_tracer_3Dsource

      implicit none
!@var i Longitude index
!@var j Latitude index
      integer, intent(in) :: i,j
!@var time Timescale of exponetial decay [days]
      real*8, intent(in) :: time
!@var tri Index of the tracer that ages (i for initial tracer)
      integer, intent(in) :: tri
!@var tre Index of the tracer that forms (e for end tracer)
      integer, intent(in), optional :: tre
!@var fact Factor of source tracer that would be affected by the decay
      real*8 :: fact

      fact=(1.d0-exp(-dtsrc/(time*SECONDS_PER_DAY)))/dtsrc

      tr3Dsource(:,nChemistry,tri)=-fact*trm_col(:,tri)
      call apply_tracer_3Dsource(i,j,nChemistry,tri)

      if (present(tre)) then
        tr3Dsource(:,nChemistry,tre)=-tr3Dsource(:,nChemistry,tri)
     &                               *(tr_mm(tre)/tr_mm(tri))
        call apply_tracer_3Dsource(i,j,nChemistry,tre)
      endif

      end subroutine calc_and_apply_expo_decay
#endif  /* Koch/AMP/TOMAS/PASSIVE aerosols */

!=======================================================================

!-----------------------------------------------------------------------
      subroutine modal_aero_kohler(rdry_in, hygro, s, rwet_out, im )

! Code inspired by CESM, Steve Ghan and Rahul Zaveri JGR 2007
! calculates equlibrium radius r of haze droplets as function of
! dry particle mass and relative humidity s using kohler solution
! given in pruppacher and klett (eqn 6-35)

! for multiple aerosol types, assumes an internal mixture of aerosols

      implicit none

! arguments
      integer :: im         ! number of grid points to be processed
      real*8 :: rdry_in(im)    ! aerosol dry radius (m) !!ATTN OEC & SEB THINK THIS SHOULD BE MICROMETERS
      real*8 :: hygro(im)      ! aerosol volume-mean hygroscopicity (--)
      real*8 :: s(im)          ! relative humidity (1 = saturated)
      real*8 :: rwet_out(im)   ! aerosol wet radius (m) !! ATTN OEC & SEB THINK THIS SHOULD BE MICROMETERS

! local variables
      integer, parameter :: imax=200
      integer :: i, n, nsol

      real*8 :: a, b
      real*8 :: p40(imax),p41(imax),p42(imax),p43(imax) ! coefficients of polynomial
      real*8 :: p30(imax),p31(imax),p32(imax) ! coefficients of polynomial
      real*8 :: p
      real*8 :: r3, r4
      real*8 :: r(im)         ! wet radius (microns)
      real*8 :: rdry(imax)    ! radius of dry particle (microns)
      real*8 :: ss            ! relative humidity (1 = saturated)
      real*8 :: slog(imax)    ! log relative humidity
      real*8 :: vol(imax)     ! total volume of particle (microns**3)
      real*8 :: xi, xr

      complex*8 :: cx4(4,imax),cx3(3,imax)

      real*8, parameter :: eps = 1.e-4
      real*8, parameter :: mw = 18.d0
      real*8, parameter :: pi = 3.14159d0
      real*8, parameter :: rhow = 1.d0
      real*8, parameter :: surften = 76.d0
      real*8, parameter :: tair = 273.d0
      real*8, parameter :: third = 1.d0/3.d0
      real*8, parameter :: ugascon = 8.3e7


!     effect of organics on surface tension is neglected
      a=2.e4*mw*surften/(ugascon*tair*rhow)

      do i=1,im
           rdry(i) = rdry_in(i)
           vol(i) = rdry(i)**3          ! vol is r**3, not volume
           b = vol(i)*hygro(i)

!          quartic
           ss=min(s(i),1.d0-eps)
           ss=max(ss,1.e-10)
           slog(i)=log(ss)
           p43(i)=-a/slog(i)
           p42(i)=0.d0
           p41(i)=b/slog(i)-vol(i)
           p40(i)=a*vol(i)/slog(i)
!          cubic for rh=1
           p32(i)=0.d0
           p31(i)=-b/a
           p30(i)=-vol(i)
      end do

       do 100 i=1,im

        if(vol(i).le.1.e-12)then
           r(i)=rdry(i)
           go to 100
        endif

        p=abs(p31(i))/(rdry(i)*rdry(i))
        if(p.lt.eps)then
!          approximate solution for small particles
           r(i)=rdry(i)*(1.d0+p*third/(1.d0-slog(i)*rdry(i)/a))
        else
           call makoh_quartic(cx4(1,i),p43(i),p42(i),p41(i),p40(i),1)
!          find smallest real(r8) solution
           r(i)=1000.d0*rdry(i)
           nsol=0
           do n=1,4
              xr=real(cx4(n,i))
              xi=aimag(cx4(n,i))
              if(abs(xi).gt.abs(xr)*eps) cycle
              if(xr.gt.r(i)) cycle
              if(xr.lt.rdry(i)*(1.d0-eps)) cycle
              if(xr.ne.xr) cycle
              r(i)=xr
              nsol=n
           end do
           if(nsol.eq.0)then
              r(i)=rdry(i)
           endif
        endif

        if(s(i).gt.1.d0-eps)then
!          save quartic solution at s=1-eps
           r4=r(i)
!          cubic for rh=1
           p=abs(p31(i))/(rdry(i)*rdry(i))
           if(p.lt.eps)then
              r(i)=rdry(i)*(1.d0+p*third)
           else
              call makoh_cubic(cx3,p32,p31,p30,im)
!             find smallest real(r8) solution
              r(i)=1000.d0*rdry(i)
              nsol=0
              do n=1,3
                 xr=real(cx3(n,i))
                 xi=aimag(cx3(n,i))
                 if(abs(xi).gt.abs(xr)*eps) cycle
                 if(xr.gt.r(i)) cycle
                 if(xr.lt.rdry(i)*(1.d0-eps)) cycle
                 if(xr.ne.xr) cycle
                 r(i)=xr
                 nsol=n
              end do
              if(nsol.eq.0)then
                 r(i)=rdry(i)
              endif
           endif
           r3=r(i)
!          now interpolate between quartic, cubic solutions
           r(i)=(r4*(1.d0-s(i))+r3*(s(i)-1.d0+eps))/eps
        endif

  100 continue

! bound and convert from microns to m
      do i=1,im
         r(i) = min(r(i),30.d0) ! upper bound based on 1 day lifetime
         rwet_out(i) = r(i)
      end do

      return
      end subroutine modal_aero_kohler

!-----------------------------------------------------------------------
      subroutine makoh_cubic( cx, p2, p1, p0, im )
!
!     solves  x**3 + p2 x**2 + p1 x + p0 = 0
!     where p0, p1, p2 are real
!
      integer, parameter :: imx=200
      integer :: im
      real*8 :: p0(imx), p1(imx), p2(imx)
      complex*8 :: cx(3,imx)

      integer :: i
      real*8 :: eps, q(imx), r(imx), sqrt3, third
      complex*8 :: ci, cq, crad(imx), cw, cwsq, cy(imx), cz(imx)

      save eps
      data eps/1.e-20/

      third=1.d0/3.d0
      ci=cmplx(0.d0,1d0)
      sqrt3=sqrt(3.d0)
      cw=0.5d0*(-1+ci*sqrt3)
      cwsq=0.5d0*(-1-ci*sqrt3)

      do i=1,im
      if(p1(i).eq.0.d0)then
!        completely insoluble particle
         cx(1,i)=(-p0(i))**third
         cx(2,i)=cx(1,i)
         cx(3,i)=cx(1,i)
      else
         q(i)=p1(i)/3.d0
         r(i)=p0(i)/2.d0
         crad(i)=r(i)*r(i)+q(i)*q(i)*q(i)
         crad(i)=sqrt(crad(i))

         cy(i)=r(i)-crad(i)
         if (abs(cy(i)).gt.eps) cy(i)=cy(i)**third
         cq=q(i)
         cz(i)=-cq/cy(i)

         cx(1,i)=-cy(i)-cz(i)
         cx(2,i)=-cw*cy(i)-cwsq*cz(i)
         cx(3,i)=-cwsq*cy(i)-cw*cz(i)
      endif
      enddo

      return
      end subroutine makoh_cubic


!-----------------------------------------------------------------------
      subroutine makoh_quartic( cx, p3, p2, p1, p0, im )

!     solves x**4 + p3 x**3 + p2 x**2 + p1 x + p0 = 0
!     where p0, p1, p2, p3 are real
!
      integer, parameter :: imx=200
      integer :: im
      real*8 :: p0(imx), p1(imx), p2(imx), p3(imx)
      complex*8 :: cx(4,imx)

      integer :: i
      real*8 :: third, q(imx), r(imx)
      complex*8 :: cb(imx), cb0(imx), cb1(imx),crad(imx), cy(imx), czero


      czero=cmplx(0.0d0,0.0d0)
      third=1.d0/3.d0

      do 10 i=1,im

      q(i)=-p2(i)*p2(i)/36.d0+(p3(i)*p1(i)-4*p0(i))/12.d0
      r(i)=-(p2(i)/6)**3+p2(i)*(p3(i)*p1(i)-4*p0(i))/48.d0
     &   +(4*p0(i)*p2(i)-p0(i)*p3(i)*p3(i)-p1(i)*p1(i))/16

      crad(i)=r(i)*r(i)+q(i)*q(i)*q(i)
      crad(i)=sqrt(crad(i))

      cb(i)=r(i)-crad(i)
      if(cb(i).eq.czero)then
!        insoluble particle
         cx(1,i)=(-p1(i))**third
         cx(2,i)=cx(1,i)
         cx(3,i)=cx(1,i)
         cx(4,i)=cx(1,i)
      else
         cb(i)=cb(i)**third

         cy(i)=-cb(i)+q(i)/cb(i)+p2(i)/6

         cb0(i)=sqrt(cy(i)*cy(i)-p0(i))
         cb1(i)=(p3(i)*cy(i)-p1(i))/(2*cb0(i))

         cb(i)=p3(i)/2+cb1(i)
         crad(i)=cb(i)*cb(i)-4*(cy(i)+cb0(i))
         crad(i)=sqrt(crad(i))
         cx(1,i)=(-cb(i)+crad(i))/2.d0
         cx(2,i)=(-cb(i)-crad(i))/2.d0

         cb(i)=p3(i)/2-cb1(i)
         crad(i)=cb(i)*cb(i)-4*(cy(i)-cb0(i))
         crad(i)=sqrt(crad(i))
         cx(3,i)=(-cb(i)+crad(i))/2.d0
         cx(4,i)=(-cb(i)-crad(i))/2.d0
      endif
   10 continue

      return
      end subroutine makoh_quartic

