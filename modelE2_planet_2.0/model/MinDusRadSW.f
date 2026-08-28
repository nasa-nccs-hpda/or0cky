#include "rundeck_opts.h"

      module MinDusRadSW

! Author: Vincenzo Obiso (FEB-MAR-APR 2022)

      use trdust_mod, only: bbz=>nSubClays
      use RAD_COM,    only: ddz=>nraero_dust
      use RESOLUTION, only: hhz=>LM

      implicit none

      save

      integer :: intsiz = 0

      integer, parameter :: eez = 6, ssz = 9
#if defined(TRACERS_MINERALS)
      integer :: kkz, llv, nnz
      integer, parameter :: llz = 20, vvz = 48
#if defined(HOMMIX_MINERALS)
      integer, parameter :: ccz = 1
#endif
#if defined(EXTMIX_MINERALS)
      integer, parameter :: ccz = 3
#endif
#if defined(INTMIX_MINERALS)
#if !defined(TRACERS_DUST_Silt4) && !defined(TRACERS_DUST_Silt5)
      integer, parameter :: ssx = 7
#elif defined(TRACERS_DUST_Silt4) && !defined(TRACERS_DUST_Silt5)
      integer, parameter :: ssx = 8
#elif defined(TRACERS_DUST_Silt4) && defined(TRACERS_DUST_Silt5)
      integer, parameter :: ssx = 9
#endif
#endif
#endif

      real*4, dimension(ssz) :: reff
      real*4, dimension(eez,ssz) :: qext, qsca, asym
#if defined(TRACERS_MINERALS)
      real*4, allocatable, dimension(:) :: llrv, ngrd, kgrd
      real*4, allocatable, dimension(:,:,:,:) :: qerv, qsrv, qgrv

      real*8, allocatable, dimension(:,:,:,:) :: qesw, qssw, qgsw

      real*8, dimension(llz) :: llsw = (/0.300, 0.325, 0.350, 0.375,    &
     &       0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575,    &
     &       0.600, 0.625, 0.650, 0.675, 0.700, 0.800, 0.900, 1.000/)
      real*8, dimension(llz) :: nhom = (/1.600, 1.595, 1.590, 1.585,    &
     &       1.580, 1.576, 1.572, 1.568, 1.564, 1.561, 1.558, 1.555,    &
     &       1.553, 1.551, 1.550, 1.548, 1.544, 1.540, 1.535, 1.530/)
      real*8, dimension(llz) :: khom = (/8.700, 7.000, 5.800, 4.600,    &
     &       3.600, 2.900, 2.400, 2.000, 1.800, 1.600, 1.400, 1.200,    &
     &       1.100, 0.900, 0.800, 0.700, 0.700, 0.800, 1.000, 2.000/)*  &
     &       1.0e-03
      real*8, dimension(llz) :: niox = (/1.682, 1.814, 1.988, 2.187,    &
     &       2.396, 2.601, 2.786, 2.936, 3.036, 3.071, 3.041, 2.978,    &
     &       2.917, 2.877, 2.849, 2.828, 2.808, 2.744, 2.704, 2.683/)
      real*8, dimension(llz) :: kiox = (/0.258, 1.319, 2.096, 2.615,    &
     &       2.903, 2.987, 2.894, 2.651, 2.284, 1.822, 1.322, 0.906,    &
     &       0.707, 0.745, 0.898, 1.040, 1.123, 1.036, 0.772, 0.982/)*  &
     &       1.0e-01
      real*8, dimension(llz) :: nhos = (/1.531, 1.522, 1.515, 1.511,    &
     &       1.508, 1.506, 1.505, 1.504, 1.503, 1.502, 1.501, 1.499,    &
     &       1.498, 1.496, 1.495, 1.494, 1.493, 1.492, 1.492, 1.488/)
      real*8, dimension(llz) :: khos = (/13.52, 13.10, 12.47, 11.67,    &
     &       10.74, 9.705, 8.607, 7.481, 6.362, 5.286, 4.330, 3.660,    &
     &       3.450, 3.694, 4.149, 4.569, 4.857, 5.125, 5.086, 6.218/)*  &
     &       1.0e-04
      real*8, dimension(llz) :: nacc = (/1.535, 1.530, 1.529, 1.530,    &
     &       1.532, 1.536, 1.540, 1.544, 1.546, 1.546, 1.544, 1.540,    &
     &       1.537, 1.534, 1.532, 1.531, 1.530, 1.527, 1.526, 1.521/)
      real*8, dimension(llz) :: kacc = (/6.832, 6.470, 6.202, 5.973,    &
     &       5.731, 5.430, 5.037, 4.534, 3.924, 3.238, 2.550, 1.989,    &
     &       1.645, 1.480, 1.385, 1.312, 1.257, 1.195, 1.115, 0.701/)*  &
     &       1.0e-03
      real*8, dimension(llz) :: kpar = (/3.444, 3.389, 3.405, 3.473,    &
     &       3.575, 3.691, 3.802, 3.889, 3.933, 3.916, 3.817, 3.619,    &
     &       3.304, 2.898, 2.488, 2.158, 1.936, 1.716, 1.587, 0.239/)*  &
     &       1.0e+01

      real*8 :: diox = 4.770, dhos = 2.590
#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
      real*8, dimension(llz,ccz) :: nnsw, kksw
      real*8, dimension(ccz,ssz) :: qevi, qsvi, qgvi
#endif
#if defined(INTMIX_MINERALS)
      real*8, dimension(hhz,llz,ssx) :: nnsw, kksw
      real*8, dimension(hhz,ssx) :: qevi, qsvi, qgvi
#endif
#endif

      contains

      subroutine IniOptPar

      implicit none

      include 'netcdf.inc'

      ! input

      ! output

      ! local
      integer :: v, z
      integer :: sta, nid
      integer, dimension(4) :: one
      integer, dimension(2) :: two
      integer, dimension(4) :: vid
#if defined(TRACERS_MINERALS)
      integer, dimension(4) :: fou
      integer, dimension(3) :: did
      integer, dimension(6) :: xid
      real*8 :: epsi = 0.0025
#endif

      ! start here

      sta = nf_open('lutdust', nf_nowrite, nid)

      sta = nf_inq_varid(nid, 'REFF', vid(1))
      sta = nf_inq_varid(nid, 'QEXT', vid(2))
      sta = nf_inq_varid(nid, 'QSCA', vid(3))
      sta = nf_inq_varid(nid, 'ASYM', vid(4))

      one = (/1, 1, 1, 1/)
      two = (/eez, ssz/)

      sta = nf_get_vara_real(nid, vid(1), one(:1), ssz, reff)
      sta = nf_get_vara_real(nid, vid(2), one(:2), two, qext)
      sta = nf_get_vara_real(nid, vid(3), one(:2), two, qsca)
      sta = nf_get_vara_real(nid, vid(4), one(:2), two, asym)

      sta = nf_close(nid)

#if defined(TRACERS_MINERALS)
      sta = nf_open('lutmine', nf_nowrite, nid)

      sta = nf_inq_dimid(nid, 'wvln', did(1))
      sta = nf_inq_dimid(nid, 'reix', did(2))
      sta = nf_inq_dimid(nid, 'imix', did(3))
      sta = nf_inq_dimlen(nid, did(1), llv)
      sta = nf_inq_dimlen(nid, did(2), nnz)
      sta = nf_inq_dimlen(nid, did(3), kkz)

      allocate( llrv(llv), ngrd(nnz), kgrd(kkz) )
      allocate( qerv(kkz,nnz,llv,ssz),                                  &
     &          qsrv(kkz,nnz,llv,ssz),                                  &
     &          qgrv(kkz,nnz,llv,ssz) )

      fou = (/kkz, nnz, llv, ssz/)

      sta = nf_inq_varid(nid, 'LLRV', xid(1))
      sta = nf_inq_varid(nid, 'NGRD', xid(2))
      sta = nf_inq_varid(nid, 'KGRD', xid(3))
      sta = nf_inq_varid(nid, 'QERV', xid(4))
      sta = nf_inq_varid(nid, 'QSRV', xid(5))
      sta = nf_inq_varid(nid, 'QGRV', xid(6))
      sta = nf_get_vara_real(nid, xid(1), one(:1), llv, llrv)
      sta = nf_get_vara_real(nid, xid(2), one(:1), nnz, ngrd)
      sta = nf_get_vara_real(nid, xid(3), one(:1), kkz, kgrd)
      sta = nf_get_vara_real(nid, xid(4), one(:4), fou, qerv)
      sta = nf_get_vara_real(nid, xid(5), one(:4), fou, qsrv)
      sta = nf_get_vara_real(nid, xid(6), one(:4), fou, qgrv)

      sta = nf_close(nid)

      allocate( qesw(kkz,nnz,llz,ssz),                                  &
     &          qssw(kkz,nnz,llz,ssz),                                  &
     &          qgsw(kkz,nnz,llz,ssz) )

      do z = 1, llz
       do v = 1, llv
        if (llrv(v)-epsi .le. llsw(z) .and.                             &
     &                        llsw(z) .lt. llrv(v)+epsi) then
         qesw(:,:,z,:) = qerv(:,:,v,:)
         qssw(:,:,z,:) = qsrv(:,:,v,:)
         qgsw(:,:,z,:) = qgrv(:,:,v,:)
        endif
       enddo
      enddo

#if defined(HOMMIX_MINERALS)
      nnsw(:,1) = nhom
      kksw(:,1) = khom

      call AveVisPar
#endif
#if defined(EXTMIX_MINERALS)
      nnsw(:,1) = niox
      kksw(:,1) = kiox
      nnsw(:,2) = nhos
      kksw(:,2) = khos
      nnsw(:,3) = nacc
      kksw(:,3) = kacc

      call AveVisPar
#endif
#endif

      end subroutine IniOptPar

      subroutine CalRadPro(dext,dsca,dasy)

      use RAD_COM,       only: du1=>nr_soildust,                        &
     &                         trx=>ntrix_aod
      use RADPAR,        only: rads=>TRRDRY,                            &
     &                         dens=>TRADEN,                            &
     &                         mass=>TRACER
#if defined(TRACERS_MINERALS)
      use trdust_mod,    only: fiox=>frIronOxideInAggregate
#endif
      use OldTracer_mod, only: tname=>trname

      implicit none

      ! input

      ! output
      real*8, dimension(hhz,eez,ddz), intent(out) :: dext, dsca, dasy

      ! local
      integer :: d, e, h, s, x
      integer :: du2
      integer, dimension(1) :: bmn
      real*8 :: qtok, gxkg, qtau
      character*8 :: xname
      character*4 :: rname
!#if defined(TRACERS_DUST)
!      character*4 :: DusNam
!#endif
#if defined(TRACERS_MINERALS)
#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
      integer :: c
#endif
#if defined(INTMIX_MINERALS)
      integer :: l
      real*8 :: fhos, ciox, chos, cacc, xiox, yiox
      real*8, dimension(hhz,ssx) :: miox, mhos
#endif
#endif

      ! start here

      du2 = du1 + ddz - 1

#if defined(TRACERS_MINERALS) && defined(INTMIX_MINERALS)
      fhos = 1.0 - fiox

      miox = 0.0
      mhos = 0.0

      do d = du1, du2

       xname = tname(trx(d))

       rname = xname(1:4)
       select case (rname)
        case('Clay')
         bmn = minloc(abs(rads(du1:du1+bbz-1)-rads(d)))
         s = bmn(1)
        case('Sil1')
         s = 5
        case('Sil2')
         s = 6
        case('Sil3')
         s = 7
#if defined(TRACERS_DUST_Silt4)
        case('Sil4')
         s = 8
#endif
#if defined(TRACERS_DUST_Silt5)
        case('Sil5')
         s = 9
#endif
       end select

       rname = xname(5:8)
       select case (rname)
        case('Hema')
         ciox = 1.0
         chos = 0.0
         cacc = 0.0
        case('Illi','Kaol','Smec','Calc','Quar','Feld','Gyps')
         ciox = 0.0
         chos = 1.0
         cacc = 0.0
        case('IlHe','KaHe','SmHe','CaHe','QuHe','FeHe','GyHe')
         ciox = 0.0
         chos = 0.0
         cacc = 1.0
       end select

       miox(:,s) = miox(:,s) + ciox*mass(1:hhz,d)                       &
     &                       + cacc*mass(1:hhz,d)*fiox
       mhos(:,s) = mhos(:,s) + chos*mass(1:hhz,d)                       &
     &                       + cacc*mass(1:hhz,d)*fhos

      enddo

      do s = 1, ssx
       do l = 1, llz
        do h = 1, hhz
         if (miox(h,s) + mhos(h,s) .gt. 0.0) then
          xiox = miox(h,s)/(miox(h,s) + mhos(h,s))
         else
          xiox = 0.0
         endif
         yiox = (xiox/diox)/(xiox/diox + (1.0-xiox)/dhos)
         nnsw(h,l,s) = nhos(l) + (niox(l) - nhos(l))*yiox
         kksw(h,l,s) = khos(l)*(1.0 + kpar(l)*xiox                      &
     &                              + (kpar(l)**2/2.0)*xiox**2          &
     &                              + (kpar(l)**3/6.0)*xiox**3)
        enddo
       enddo
      enddo

      call AveVisPar
#endif

      do d = du1, du2

       x = d - du1 + 1

       xname = tname(trx(d))

#if defined(TRACERS_DUST)
       rname = DusNam(trim(xname))
#endif
#if defined(TRACERS_MINERALS)
       rname = xname(1:4)
#endif
       select case (rname)
        case('Clay')
         bmn = minloc(abs(rads(du1:du1+bbz-1)-rads(d)))
         s = bmn(1)
        case('Sil1')
         s = 5
        case('Sil2')
         s = 6
        case('Sil3')
         s = 7
#if defined(TRACERS_DUST_Silt4)
        case('Sil4')
         s = 8
#endif
#if defined(TRACERS_DUST_Silt5)
        case('Sil5')
         s = 9
#endif
       end select

#if defined(TRACERS_MINERALS)
#if defined(HOMMIX_MINERALS)
       c = 1
#endif
#if defined(EXTMIX_MINERALS)
       rname = xname(5:8)
       select case (rname)
        case('Hema')
         c = 1
        case('Illi','Kaol','Smec','Calc','Quar','Feld','Gyps')
         c = 2
        case('IlHe','KaHe','SmHe','CaHe','QuHe','FeHe','GyHe')
         c = 3
       end select
#endif
#endif

       if (intsiz .eq. 0) then
        qtok = 3.0/(4.0*dens(d)*reff(s))
       else if (intsiz .eq. 1) then
        qtok = 3.0/(4.0*dens(d)*rads(d))
       endif
       gxkg = 1.0e+03

       do e = 1, eez
        do h = 1, hhz
         qtau = qtok*mass(h,d)*gxkg
         dext(h,e,x) = qext(e,s)*qtau
         dsca(h,e,x) = qsca(e,s)*qtau
         dasy(h,e,x) = asym(e,s)
        enddo
       enddo
#if defined(TRACERS_MINERALS)
       do h = 1, hhz
        qtau = qtok*mass(h,d)*gxkg
#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
        dext(h,6,x) = qevi(c,s)*qtau
        dsca(h,6,x) = qsvi(c,s)*qtau
        dasy(h,6,x) = qgvi(c,s)
#endif
#if defined(INTMIX_MINERALS)
        dext(h,6,x) = qevi(h,s)*qtau
        dsca(h,6,x) = qsvi(h,s)*qtau
        dasy(h,6,x) = qgvi(h,s)
#endif
       enddo
#endif

      enddo

      end subroutine CalRadPro

#if defined(TRACERS_MINERALS)
      subroutine AveVisPar

      implicit none

      ! input

      ! output

      ! local
      integer :: k, l, n, s, v
      integer :: nn1, nn2, kk1, kk2
      real*8 :: nnii, kkii, qeii, qsii, qgii
      real*8 :: prod, dndk, wg11, wg12, wg21, wg22
      real*8, dimension(vvz) :: vlrx, fxrx
      real*8, dimension(vvz) :: qerx, qsrx, qgrx, ferx, fsrx, fgrx
      real*8, dimension(vvz-1) :: wfix, weix, wsix, wgix
#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
      integer :: c
      real*8, dimension(llz,ccz,ssz) :: qenk, qsnk, qgnk
#endif
#if defined(INTMIX_MINERALS)
      integer :: h
      real*8, dimension(hhz,llz,ssx) :: qenk, qsnk, qgnk
#endif

      ! start here

      vlrx = (/0.300, 0.310, 0.320, 0.330, 0.340, 0.350, 0.360, 0.370,  &
     &         0.380, 0.390, 0.400, 0.410, 0.420, 0.430, 0.440, 0.450,  &
     &         0.460, 0.470, 0.480, 0.490, 0.500, 0.510, 0.520, 0.530,  &
     &         0.540, 0.550, 0.560, 0.570, 0.580, 0.590, 0.600, 0.610,  &
     &         0.620, 0.630, 0.640, 0.650, 0.660, 0.670, 0.680, 0.690,  &
     &         0.700, 0.710, 0.720, 0.730, 0.740, 0.750, 0.760, 0.770/)

      fxrx = (/0.514, 0.689, 0.830, 1.059, 1.074, 1.093, 1.068, 1.181,  &
     &         1.120, 1.098, 1.429, 1.751, 1.747, 1.639, 1.810, 2.006,  &
     &         2.066, 2.033, 2.074, 1.950, 1.942, 1.882, 1.833, 1.842,  &
     &         1.783, 1.725, 1.695, 1.712, 1.715, 1.700, 1.666, 1.635,  &
     &         1.602, 1.570, 1.544, 1.511, 1.486, 1.456, 1.427, 1.402,  &
     &         1.369, 1.344, 1.314, 1.290, 1.260, 1.235, 1.211, 1.185/)*&
     &         1.0e+03

#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
      do s = 1, ssz
       do c = 1, ccz
        do l = 1, llz

         nnii = max(nnsw(l,c), ngrd(1))
         nnii = min(nnii, ngrd(nnz))
         kkii = max(kksw(l,c), kgrd(1))
         kkii = min(kkii, kgrd(kkz))
#endif
#if defined(INTMIX_MINERALS)
      do s = 1, ssx
       do l = 1, llz
        do h = 1, hhz

         nnii = max(nnsw(h,l,s), ngrd(1))
         nnii = min(nnii, ngrd(nnz))
         kkii = max(kksw(h,l,s), kgrd(1))
         kkii = min(kkii, kgrd(kkz))
#endif

         do n = 1, nnz-1
          prod = (ngrd(n) - nnii)*(ngrd(n+1) - nnii)
          if (prod .le. 0.0) then
           nn1 = n
           nn2 = n + 1
          endif
         enddo
         do k = 1, kkz-1
          prod = (kgrd(k) - kkii)*(kgrd(k+1) - kkii)
          if (prod .le. 0.0) then
           kk1 = k
           kk2 = k + 1
          endif
         enddo

         dndk = (ngrd(nn2) - ngrd(nn1))*(kgrd(kk2) - kgrd(kk1))
         wg11 = (ngrd(nn2) - nnii)*(kgrd(kk2) - kkii)/dndk
         wg12 = (ngrd(nn2) - nnii)*(kkii - kgrd(kk1))/dndk
         wg21 = (nnii - ngrd(nn1))*(kgrd(kk2) - kkii)/dndk
         wg22 = (nnii - ngrd(nn1))*(kkii - kgrd(kk1))/dndk

         qeii = qesw(kk1,nn1,l,s)*wg11 + qesw(kk2,nn1,l,s)*wg12         &
     &        + qesw(kk1,nn2,l,s)*wg21 + qesw(kk2,nn2,l,s)*wg22
         qsii = qssw(kk1,nn1,l,s)*wg11 + qssw(kk2,nn1,l,s)*wg12         &
     &        + qssw(kk1,nn2,l,s)*wg21 + qssw(kk2,nn2,l,s)*wg22
         qgii = qgsw(kk1,nn1,l,s)*wg11 + qgsw(kk2,nn1,l,s)*wg12         &
     &        + qgsw(kk1,nn2,l,s)*wg21 + qgsw(kk2,nn2,l,s)*wg22

#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
         qenk(l,c,s) = qeii
         qsnk(l,c,s) = qsii
         qgnk(l,c,s) = qgii
#endif
#if defined(INTMIX_MINERALS)
         qenk(h,l,s) = qeii
         qsnk(h,l,s) = qsii
         qgnk(h,l,s) = qgii
#endif

        enddo
       enddo
      enddo

      wfix = (fxrx(1:vvz-1) + fxrx(2:vvz))*0.010/2.0

#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
      do s = 1, ssz
       do c = 1, ccz
        do v = 1, vvz
         call spline(llsw,qenk(:,c,s),llz,vlrx(v),qerx(v),1.0,1.0,1)
         call spline(llsw,qsnk(:,c,s),llz,vlrx(v),qsrx(v),1.0,1.0,1)
         call spline(llsw,qgnk(:,c,s),llz,vlrx(v),qgrx(v),1.0,1.0,1)
        enddo
#endif
#if defined(INTMIX_MINERALS)
      do s = 1, ssx
       do h = 1, hhz
        do v = 1, vvz
         call spline(llsw,qenk(h,:,s),llz,vlrx(v),qerx(v),1.0,1.0,1)
         call spline(llsw,qsnk(h,:,s),llz,vlrx(v),qsrx(v),1.0,1.0,1)
         call spline(llsw,qgnk(h,:,s),llz,vlrx(v),qgrx(v),1.0,1.0,1)
        enddo
#endif
        ferx = qerx*fxrx
        fsrx = qsrx*fxrx
        fgrx = qgrx*qsrx*fxrx
        weix = (ferx(1:vvz-1) + ferx(2:vvz))*0.010/2.0
        wsix = (fsrx(1:vvz-1) + fsrx(2:vvz))*0.010/2.0
        wgix = (fgrx(1:vvz-1) + fgrx(2:vvz))*0.010/2.0
#if defined(HOMMIX_MINERALS) || defined(EXTMIX_MINERALS)
        qevi(c,s) = sum(weix)/sum(wfix)
        qsvi(c,s) = sum(wsix)/sum(wfix)
        qgvi(c,s) = sum(wgix)/sum(wsix)
       enddo
      enddo
#endif
#if defined(INTMIX_MINERALS)
        qevi(h,s) = sum(weix)/sum(wfix)
        qsvi(h,s) = sum(wsix)/sum(wfix)
        qgvi(h,s) = sum(wgix)/sum(wsix)
       enddo
      enddo
#endif

      end subroutine AveVisPar
#endif

#if defined(TRACERS_DUST)
      function DusNam(iname) result(oname)
      character(len=*), intent(in) :: iname
      character*4 :: oname
      if (iname .eq. 'Silt1') then
       oname = 'Sil1'
      else if (iname .eq. 'Silt2') then
       oname = 'Sil2'
      else if (iname .eq. 'Silt3') then
       oname = 'Sil3'
      else if (iname .eq. 'Silt4') then
       oname = 'Sil4'
      else if (iname .eq. 'Silt5') then
       oname = 'Sil5'
      else
       oname = iname
      endif
      end function
#endif

      end module MinDusRadSW
