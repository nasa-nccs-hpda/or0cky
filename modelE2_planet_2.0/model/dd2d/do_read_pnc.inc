      integer :: fid
      character(len=*) :: varname
      type(dist_grid), intent(in) :: grid
      logical, intent(in), optional :: bcast_all
      integer, intent(in), optional :: record,record1
      integer :: vid,unlim_did,rc,ierr,nd,ndf,rd
      integer :: idim
      integer, dimension(7) :: srt4,cnt4
      integer*8, dimension(7) :: srt,cnt
      integer, dimension(7) :: dids
      integer*8 :: nrecs8,dlen
      logical :: var_has_unlimdim
      real*8 :: arr_dum
      logical :: bcast_all_
      bcast_all_ = .false.
      if(present(bcast_all)) bcast_all_ = bcast_all
      rc = nfmpi_inq_varid(fid,trim(varname),vid)
      if(rc.ne.nf_noerr) then
        if(grid%am_i_globalroot) write(6,*) 'variable ',
     &       trim(varname),' not found in input file - skipping'
        return
      endif
      nd = size(shape(arr))
      cnt4(:) = 1 ! ok?
      srt4(:) = 1
      if(nd.gt.0) then
        cnt4(1:nd) = shape(arr)
        srt4(1:nd) = 1
      endif
      if(.not.bcast_all_) then ! for all-proc shape-checking
        call broadcast(nd)
        call broadcast(srt4)
        call broadcast(cnt4)
      endif
      srt = srt4
      cnt = cnt4
      rc = nfmpi_inq_varndims(fid,vid,ndf)
      rc = nfmpi_inq_vardimid(fid,vid,dids)
      rc = nfmpi_inq_unlimdim(fid,unlim_did)
      if(ndf.gt.0) then
        var_has_unlimdim = dids(ndf).eq.unlim_did
      else
        var_has_unlimdim = .false.
      endif
      if(var_has_unlimdim) then
        rd = ndf
        if(present(record) .or. present(record1)) then
          nrecs8 = 0
          rc = nfmpi_inq_dimlen(fid,unlim_did,nrecs8)
        endif
        if(present(record)) then
          if(record.le.0 .or. nrecs8.lt.record) then
            if(grid%am_i_globalroot) write(6,*)
     &           'error in record dim spec. for variable ',trim(varname)
            call stoprc(0,1)
          endif
          srt(rd) = record
          cnt(rd) = 1
          ndf = ndf - 1
        elseif(present(record1)) then
          if(record1.le.0 .or. nrecs8.lt.record1) then
            if(grid%am_i_globalroot) write(6,*)
     &           'error in record dim spec. for variable ',trim(varname)
            call stoprc(0,1)
          endif
          srt(rd) = record1
        endif
      else
        rd = -1
        if(present(record) .or. present(record1)) then
          if(grid%am_i_globalroot) write(6,*)
     &         'error in record dim spec. for variable ',trim(varname)
          call stoprc(0,1)
        endif
      endif

      ! shape checking (on all processes even if not bcast_all)
      if(nd .ne. ndf) then
        if(grid%am_i_globalroot) then
          write(6,*) 'shape mismatch for variable ',trim(varname)
        endif
        call stoprc(0,1)
      endif
      do idim=1,ndf
        rc = nfmpi_inq_dimlen(fid,dids(idim),dlen)
        if(dlen .eq. cnt(idim)) cycle
        if(idim.eq.rd) then
          if(dlen.lt.cnt(idim)+srt(idim)-1) then
            if(grid%am_i_globalroot)
     &           write(6,*)
     &          'warning: only performing partial read of variable '//
     &           trim(varname)//' along the unlimited dimension'
            cnt(idim) = dlen-srt(idim)+1
          endif
        else
          if(grid%am_i_globalroot) then
            write(6,*) 'shape mismatch for variable ',trim(varname)
          endif
          call stoprc(0,1)
        endif
      enddo

      if(bcast_all_ .or. grid%am_i_globalroot) then
        rc = nfmpi_get_vara_double_all(fid,vid,srt,cnt,arr)
      else
        srt(:) = 1
        cnt(:) = 0
        rc = nfmpi_get_vara_double_all(fid,vid,srt,cnt,arr_dum)
      endif
