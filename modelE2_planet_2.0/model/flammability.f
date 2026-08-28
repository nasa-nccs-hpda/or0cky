#include "rundeck_opts.h"

      subroutine calc_flammability(t,p,r,v,flammability,burnt_area,
     &                             fearth_axyp)
!@sum calculated the flammability of vegetation based on model
!@+ variables for temperature, precipitation, relative humidity,
!@+ and an index of vegetation density.
!
!@auth Keren Mezuman and Greg Faluvegi in part based on work by Olga Pechony
!@var T the surface air temperature passed in Kelvin for current I,J
!@var P the precipitation rate passed in mm/day for current I,J
!@var R the relative humidity passed as fraction for current I,J
!@var V vegetative density (unitless 0 to ~1) for current I,J
!@var burned_area area burned in a grid box (m^2)
!@var Z component of the Goff-Gratch saturation vapor pressure equation
!@var tsbyt = reciprocal of the temperature times ts
!@var flammability the flammability returned for current I,J
!@param a,b,s,d,f,h,ts,cr coefficients for the parameterization
!@+ e.g. from Goff-Gratch saturation vapor pressure equation
!
      use constant, only: tf!freezing point of water at 1 atm [K]

      implicit none

      real*8, parameter :: a=-7.90298d0,d=11.344d0,c=-1.3816d-7,
     & b=5.02808,f=8.1328d-3,h=-3.49149d0,ts=tf+100.d0,cr=-2.d0
      real*8, intent(in) :: t,p,r,v,fearth_axyp,burnt_area
      real*8, intent(out) :: flammability
      real*8 :: z,tsbyt,ba_frac

      tsbyt=ts/t

      z= a*(tsbyt-1.d0) + b*log10(tsbyt) + 
     &   c*(10.d0**(d*(1.d0-tsbyt))-1.d0) +
     &   f*(10.d0**(h*(tsbyt-1.d0))-1.d0)

      ba_frac = min(burnt_area/fearth_axyp,1.d0)
      flammability=min( (10.d0**z*(1.d0-r)) * exp(cr*p) * v *
     &                  (1.d0-ba_frac) , 1.d0)

      return
      end subroutine calc_flammability


      subroutine prec_running_average(p,avg,iH,iD,i0,first,HRA,DRA,PRS)
!@sum prec_running_average keeps a running average of the model
!@+ precipitation variable for use in the flammability model.
!@+ In practice, this does hourly and daily running averages and
!@+ uses those to get the period-long running average, to avoid
!@+ saving a huge array.
!@auth Greg Faluvegi
      use flammability_com, only: nday=>nday_prec,nmax=>maxHR_prec
 
      implicit none
      
!@var p model variable which will be used in the average (precip)
!@var temp just for holding current day average for use in avg_prec
!@var nmax number of accumulations in one day (for checking)
!@var bynmax reciprocal of nmax
!@var bynday reciprocal of nday
!@var i0 see i0fl(i,j) (day in period marker)
!@var iH see iHfl(i,j) (hour in day index)
!@var iD see iDfl(i,j) (day in period index)
!@var first see first_prec(i,j) whether in first period
!@var HRA see HRAfl(time,i,j) (hourly average)
!@var DRA see DRAfl(time,i,j) (daily average)
!@var PRS see PRSfl(i,j) (period running sum)
!@var avg see ravg_prec(i,j) the running average prec returned
      real*8, intent(IN) :: p
      real*8, dimension(nday) :: DRA
      real*8, dimension(nmax) :: HRA
      real*8 :: temp, bynmax, PRS, avg, iH, iD, i0, bynday
      logical :: first
      integer :: n

      if(nint(iH) < 0 .or. nint(iH) > nmax) then
        write(6,*) 'iH maxHR_prec=',iH,nint(iH),nmax
        call stop_model('iHfl or maxHR_prec problem',255)
      endif
      bynmax=1.d0/real(nmax)
      bynday=1.d0/real(nday)
      iH = iH + 1.d0
      HRA(nint(iH)) = p
      ! do no more, unless it is the end of the day:

      if(nint(iH) == nmax) then ! end of "day":
        iH = 0.d0
        if(first) then ! first averaging period only
          iD = iD + 1.d0
          do n=1,nmax
            DRA(nint(iD)) = DRA(nint(iD)) + HRA(n)
          end do
          DRA(nint(iD)) = DRA(nint(iD))*bynmax
          if(nint(iD) == nday) then ! end first period
            PRS = 0.d0
            do n=1,nday
              PRS = PRS + DRA(n)
            end do
            avg = PRS * bynday
            first=.false.
            iD=0.d0
            i0=0.d0
          end if
        else ! not first averaging period: update the running average
          i0 = i0 + 1.d0 ! move marker
          if(nint(i0) == nday+1) i0=1.d0 ! reset marker
          temp=0.d0
          do n=1,nmax
            temp = temp + HRA(n)
          end do
          temp = temp * bynmax ! i.e. today's average
          PRS = PRS - DRA(nint(i0))
          DRA(nint(i0)) = temp
          PRS = PRS + DRA(nint(i0))
          avg = PRS * bynday
        end if
      end if

      end subroutine prec_running_average

      subroutine step_ba(burnt_area,RH1,wsurf,saveFireCount,
     &                    pvt,fearth_axyp,i,j)
!@sum step_ba calculates the burnt area of the model at every time step
!@+ burnt area variable for use in the flammability model.
!@auth Keren Mezuman 
      use ent_const, only : N_COVERTYPES
      use ent_pfts, only: ent_cover_names
      use MathematicalConstants_mod, only: PI
      use TimeConstants_mod, only: SECONDS_PER_DAY,SECONDS_PER_YEAR
      use model_com, only : DTsrc
      use diag_com, only: ij_a_tree,ij_a_shrub,ij_a_grass,aij=>aij_loc
      use flammability_com, only: tuneBurntArea
 
      implicit none
      
      real*8 :: inst_FC
      real*8 :: RHlow,RHup,CRH,Cm,Cb,Lb,Hb,gW,g0,up,a,tau,N_steps
      real*8 :: new_burnt_area,previous_BA,pvt_area
      real*8, Dimension(N_COVERTYPES), intent(INOUT) :: burnt_area 
      real*8, intent(IN) :: RH1,wsurf,saveFireCount,fearth_axyp
      real*8, Dimension(N_COVERTYPES), intent(IN) :: pvt 
      real*8 :: umax
      integer :: nv
      integer, intent(in) :: i,j
      !if there are no fires to create new BA and no old BA to recover
      if ((saveFireCount <= 0.d0) .AND. (sum(burnt_area) <= 0.d0)) 
     &  return
!@var RHlow lower bound of RH for fire spread (fraction)
!@var RHup upper bound of RH for fire spread (fraction)
!@var CRH response of fuel combustibility to real-time 
!@+   climate conditions (unitless)
!@var Cb root zone soil wetness (unitless)
!@var g0 dependance of fire spread perpendicular to wind 
!@var tau average fire duration (seconds/fire)
      !+direction (unitless)
      RHlow=0.3d0
      RHup=0.7d0
      Cb=0.5d0
      g0=0.05d0
      tau=SECONDS_PER_DAY
      N_steps=SECONDS_PER_DAY/DTsrc

!@var saveFireCount fire count rate (fire/m2/s)
!@var DTSRC source time step (s) 
!@var inst_FC number of fires in a time step in 
      !+ a square meter (#fires/m^2)
      inst_FC = saveFireCount*DTsrc
      if(RH1 <= RHlow) then
        CRH=1.d0
      else if ((RH1 > RHlow) .AND. (RH1 < RHup)) then
        CRH = (RHup - RH1) / (RHup - RHlow)
      else
        CRH = 0.d0
      end if
!@var Cm dependence of downwind on fuel wetness (Li et al. 2018)
      Cm=Cb*CRH
!@var Lb length-to-breadth ratio (unitless)
!@var wsurf surface wind velocity (m/s)
      Lb=1. + 10. * (1. - (EXP(-0.06 * wsurf)))
!@var Hb head-to- back ratio (unitless)
      Hb= (Lb + (Lb**2 - 1.)**0.5) / (Lb - (Lb**2 - 1.)**0.5)
!@var the dependence of fire spread on wind speed
      gW=2*Lb/(1.+1./Hb)*g0
      do nv=1,N_COVERTYPES
        !if there are no fires to create new BA and no old BA to recover
        if ((saveFireCount <= 0.d0) .AND. (burnt_area(nv) <= 0.d0)) 
     &    cycle
        !umax: average maximum fire spread rate [m/s]
        select case(ent_cover_names(nv))
          case('arid_shrub','cold_shrub')
            umax=0.17d0
          case('c3_grass_ann','c3_grass_arct','c3_grass_per','c4_grass')
            umax=0.2d0
          case('cold_br_late','drought_br','ever_br_late')
            umax=0.11d0
          case('decid_nd','ever_nd_late')
            umax=0.15d0
          case default
            umax=0.d0
        end select
!@var up fire spread rate in the downwind direction (m/s)
        up=umax*Cm*gW
        if (Lb == 0.d0) then
          a=0.d0!(m)
        else
!@var a average fire spread area (m^2)
          a = PI * up**2 * tau**2 / (4*Lb) * (1+1/Hb)**2 / N_steps
        endif
        pvt_area = pvt(nv) * fearth_axyp
        new_burnt_area = inst_FC * a * pvt_area * tuneBurntArea
        burnt_area(nv) =  min(new_burnt_area , pvt_area) 
        select case(ent_cover_names(nv))
          case('arid_shrub','cold_shrub')
            aij(i,j,ij_a_shrub) = aij(i,j,ij_a_shrub) + a 
          case('c3_grass_ann','c3_grass_arct','c3_grass_per','c4_grass')
            aij(i,j,ij_a_grass) = aij(i,j,ij_a_grass) + a 
          case('cold_br_late','decid_nd','drought_br',
     &         'ever_br_late','ever_nd_late')
            aij(i,j,ij_a_tree) = aij(i,j,ij_a_tree) + a 
          case default
        end select
      end do

      end subroutine step_ba 

      subroutine lai_running_average(lai,avg,iH,iD,i0,first,HRA,DRA,PRS)
!@sum lai_running_average keeps a running average of the Ent leaf area 
!@+ index variable for use in the flammability model.
!@+ In practice, this does hourly and daily running averages and
!@+ uses those to get the period-long running average, to avoid
!@+ saving a huge array.
!@auth Greg Faluvegi
      use flammability_com, only: nday=>nday_lai,nmax=>maxHR_lai

      implicit none

!@var lai model variable which will be used in the average
!@var temp just for holding current day average for use in avg_prec
!@var nmax number of accumulations in one day (for checking)
!@var bynmax reciprocal of nmax
!@var bynday reciprocal of nday
!@var i0 see i0lai(i,j) (day in period marker)
!@var iH see iHlai(i,j) (hour in day index)
!@var iD see iDlai(i,j) (day in period index)
!@var first see first_lai(i,j) whether in first period
!@var HRA see HRAlai(time,i,j) (hourly average)
!@var DRA see DRAlai(time,i,j) (daily average)
!@var PRS see PRSlai(i,j) (period running sum)
!@var avg see ravg_lai(i,j) the running average prec returned
      real*8, intent(IN) :: lai
      real*8, dimension(nday) :: DRA
      real*8, dimension(nmax) :: HRA
      real*8 :: temp, bynmax, PRS, avg, iH, iD, i0, bynday
      logical :: first
      integer :: n
      
      if(nint(iH) < 0 .or. nint(iH) > nmax) then
        write(6,*) 'iH maxHR_lai=',iH,nint(iH),nmax
        call stop_model('iHlai or maxHR_lai problem',255)
      endif
      bynmax=1.d0/real(nmax)
      bynday=1.d0/real(nday)
      iH = iH + 1.d0
      HRA(nint(iH)) = lai
      ! do no more, unless it is the end of the day:

      if(nint(iH) == nmax) then ! end of "day":
        iH = 0.d0
        if(first) then ! first averaging period only
          iD = iD + 1.d0
          do n=1,nmax
            DRA(nint(iD)) = DRA(nint(iD)) + HRA(n)
          end do
          DRA(nint(iD)) = DRA(nint(iD))*bynmax
          if(nint(iD) == nday) then ! end first period
            PRS = 0.d0
            do n=1,nday
              PRS = PRS + DRA(n)
            end do
            avg = PRS * bynday
            first=.false.
            iD=0.d0
            i0=0.d0
          end if
        else ! not first averaging period: update the running average
          i0 = i0 + 1.d0 ! move marker
          if(nint(i0) == nday+1) i0=1.d0 ! reset marker
          temp=0.d0
          do n=1,nmax
            temp = temp + HRA(n)
          end do
          temp = temp * bynmax ! i.e. today's average
          PRS = PRS - DRA(nint(i0))
          DRA(nint(i0)) = temp
          PRS = PRS + DRA(nint(i0))
          avg = PRS * bynday
        end if
      end if

      end subroutine lai_running_average


      subroutine calculate_fire_count(i,j)
!@sum calculate_fire_count calculated the #fires rate for the
!@+ dynamic biomass burning sources.
!@auth Greg Faluvegi based on direction from Olga Pechony
!@+ later modified by Keren Mezuman
      use model_com, only : DTsrc
      use TimeConstants_mod, only: INT_MONTHS_PER_YEAR,DAYS_PER_YEAR,
     & SECONDS_PER_DAY
      use flammability_com, only: tuneFireCount
      use flammability_com, only: tuneCtoGlightning
      use flammability_com, only: mfcc,flammability,saveFireCount
      use diag_com, only: aij=>aij_loc
      use lightning, only : CG_DENS 
      use flammability_com, only: anthropogenic_fire_model
      use flammability_com, only: populationDensity, nonSuppressFrac
      use diag_com, only: ij_cgign,ij_humanign
      implicit none

      integer, intent(in) :: i,j
      real*8 :: CtoG, humanIgn, conv,
     & monthPerSecond,yearsPerSecond
!@var CtoG local copy of cloud-to-ground lightning strikes
!@var humanIgn the human-induced fire ignition rate (before 
!@+ supression in units of #/m2/sec)
!@var nonSuppressFrac the fraction of fire ignitions not supressed 

      yearsPerSecond=1.d0/(SECONDS_PER_DAY*DAYS_PER_YEAR)
      monthPerSecond=real(INT_MONTHS_PER_YEAR,kind=8)*yearsPerSecond

      if (anthropogenic_fire_model) then
! Anthropogenic/lightning fire model ignition/supression, based on Olga's  
! document: "Anthropogenic ignitions and supression.docx" Nov 2012.
! First, the lightning-induced portion, where CtoG is the total 
! cloud-to-ground lightning flashes per m2 per second, so that the
! fire count will also be in units of #fire. Starting with 
! CG_DENS in flashes/m2/s:
        CtoG=CG_DENS(i,j)*tuneCtoGlightning ! #/m2/s

! Human ingition portion: The population density units are humans/km2, 
! and the formula then puts the human ingition rate in #/km2/month. Thus
! we need a conversion factor (conv) to go to #/m2/s:
! 1.d-6 = 1km/1000m * 1km/1000m 
        conv=1.d-6*monthPerSecond !(km^2 * M)/(m^2 * s)
        humanIgn=conv*0.2d0*populationDensity(i,j)**(0.4d0) ! #/m2/s

! Putting that all together to get the fire count rate (fire/m2/s):
! change variable name of saveFireCount to something better
        saveFireCount(i,j)=tuneFireCount*
     &    flammability(i,j)*(CtoG+humanIgn)*nonSuppressFrac(i,j)

! Save a daignostic for the portion that is human-caused. (1.0-this) is the
! portion that is lightning-caused, so no reason to save that. Also save the
! fire count:
        aij(i,j,ij_humanign)=aij(i,j,ij_humanign)+humanIgn
        aij(i,j,ij_cgign)=aij(i,j,ij_cgign)+CtoG
      else
! Ubiquitous fire model. Note on units:
! flammability*mfcc*yearsPerSecond = [#fire/m2]
! mfcc*yearsPerSecond includes DTsrc in it
! yearsPerSecond = [yr/s]
! saveFireCount = [#fire/m2]
        saveFireCount(i,j)=flammability(i,j)*mfcc*yearsPerSecond
      endif

      end subroutine calculate_fire_count

      subroutine dynamic_biomass_burning(n,ns)
!@sum dynamic_biomass_burning fills in the surface source ns for
!@+ tracer n with biomass burning based on flammability, offline
!@+ correlations of emissions with observed fire counts by vegetation
!@+ type, and the GCM's online vegetation. For now, this is mapped
!@+ onto the traditional VDATA( ) types, even if Ent is on.
!@auth Greg Faluvegi based on direction from Olga Pechony, Igor A.

      use ent_const, only : N_COVERTYPES
      use domain_decomp_atm,only: grid, getDomainBounds
      use flammability_com, only: flammability,
     & saveFireCount,EPFCByVegType
#ifdef TRACERS_ON
      use tracer_com, only: sfc_src
#endif  /* TRACERS_ON */
      use ghy_com, only: fearth
      use ent_com, only: entcells
      use ent_mod, only: ent_get_exports
     &                   ,n_covertypes !YKIM-temp hack

      implicit none
   
      integer :: J_0, J_1, I_0, I_1, i, j, nv
      integer, intent(in) :: n,ns
!@var emisPerFire emission per fire count, generally kg/m2/fire
      real*8 :: emisPerFire
!@var pvt fraction vegetation type for N_COVERTYPES (fraction)
!@var N_COVERTYPES = N_PFT + N_SOILCOV + N_OTHER = 16 + 2 + 0
      real*8, dimension(N_COVERTYPES):: pvt, EPFBVT

      call getDomainBounds(grid,J_STRT=J_0,J_STOP=J_1,
     &                          I_STRT=I_0,I_STOP=I_1)

!@var EPFCByVegType emission factor (kg/#fire)
      EPFBVT(:)=EPFCByVegType(:,n)

#ifdef TRACERS_ON
      do j=J_0,J_1
        do i=I_0,I_1
          if(fearth(i,j)<=0.d0) cycle ! no emissions when there is no earth

          ! Obtain the vegetation types in the box:
          call ent_get_exports(entcells(i,j),vegetation_fractions=pvt)
          ! construct emisPerFire from EPFCByVegType:
          emisPerFire = sum(pvt(:)*fearth(i,j)*EPFBVT(:)) ! [kg/#fire]
          ! saveFireCount = [#fire/m2/sec] gridbox
          sfc_src(i,j,n,ns) = emisPerFire*saveFireCount(i,j)
        end do ! i
      end do   ! j
#endif  /* TRACERS_ON */
    
      end subroutine dynamic_biomass_burning

      subroutine readFlamPopDens(xyear,xday)
!@sum reads 2D human population density for flammability purposes
!@auth Greg Faluvegi
!@+ Later modified by Keren Mezuman 
      use domain_decomp_atm, only: grid
      use timestream_mod, only : init_stream,read_stream
      use model_com, only: master_yr
      use flammability_com, only: populationDensity,popDensStream,
     &                            firstPopDensStream
      use geom, only : lon_to_i,lat_to_j
      use domain_decomp_atm, only: getDomainBounds
      use flammability_com, only: nonSuppressFrac
      use diag_com, only: ij_nsuppress,aij=>aij_loc
 
      implicit none

      integer, intent(IN) :: xyear, xday

      integer :: J_0, J_1, I_0, I_1, j, i

      call getDomainBounds(grid, J_STRT=J_0, J_STOP=J_1)
      call getDomainBounds(grid, I_STRT=I_0, I_STOP=I_1)

      if(firstPopDensStream) then
        firstPopDensStream=.false.
        call init_stream(grid,popDensStream,'FLAMPOPDEN',
     &   'populationDensity',0d0,1d10,'linm2m',xyear,xday,
     &   cyclic=master_yr>0)
      end if
      call read_stream(grid,popDensStream,xyear,xday,populationDensity)
!calculate default fns 
!loop over i,j 
!in the loop if in the regions overwrite 
            ! Fraction not supressed by humans (unitless):
      do j=J_0,J_1
        do i=I_0,I_1
          !Temperate North America (USA) 
          if ((j>lat_to_j(29.d0)) .AND. (j<lat_to_j(49.d0)) .AND.
     &      (i>lon_to_i(-128.75d0)) .AND. (i<lon_to_i(-66.25d0))) then
            nonSuppressFrac(i,j)=0.2d0*exp(-0.05*populationDensity(i,j))
          !Middle East 
          else if ((j>lat_to_j(21.d0)) .AND. (j<lat_to_j(37.d0)) .AND.
     &      (i>lon_to_i(-18.75d0)) .AND. (i<lon_to_i(28.75d0))) then
            nonSuppressFrac(i,j)=0.2d0*exp(-0.05*populationDensity(i,j))
          else if ((j>lat_to_j(15.d0)) .AND. (j<lat_to_j(43.d0)) .AND.
     &      (i>lon_to_i(26.25d0)) .AND. (i<lon_to_i(71.25d0))) then
            nonSuppressFrac(i,j)=0.2d0*exp(-0.05*populationDensity(i,j))
          !Northern Hemisphere Africa 
          else if ((j>lat_to_j(-1.d0)) .AND. (j<lat_to_j(23.d0)) .AND.
     &      (i>lon_to_i(-21.25d0)) .AND. (i<lon_to_i(51.25d0))) then
            nonSuppressFrac(i,j)=1.d0
          !Southern Hemisphere Africa 
          else if ((j>lat_to_j(-37.d0)) .AND. (j<lat_to_j(1.d0)) .AND.
     &      (i>lon_to_i(0.d0)) .AND. (i<lon_to_i(53.75d0))) then
            nonSuppressFrac(i,j)=1.d0
          else
            nonSuppressFrac(i,j)=
     &       0.05d0+0.9d0*exp(-0.05*populationDensity(i,j))
          end if
          aij(i,j,ij_nsuppress)=aij(i,j,ij_nsuppress)+
     &                          nonSuppressFrac(i,j)
        enddo
      enddo

!     endif ! fbsa format or not

      return
      end subroutine readFlamPopDens
