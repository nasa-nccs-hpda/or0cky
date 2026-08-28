      integer :: fid
      character(len=*) :: varname
      type(dist_grid), intent(in) :: grid
      logical, intent(in), optional :: bcast_all
      integer, intent(in), optional :: record,record1
      integer :: vid,unlim_did,rc,rc2,ierr,nd,rd,arrsize,shp(7)

      integer, dimension(7) :: srt,cnt,dids
      integer :: nrecs,dlen,idim,ndf
      logical :: var_has_unlimdim,partial_read

      if(grid%am_i_globalroot) then
        rc = nf_inq_varid(fid,trim(varname),vid)
        if(rc.ne.nf_noerr) write(6,*) 'variable ',
     &       trim(varname),' not found in input file - skipping'
      endif
#ifndef SERIAL_MODE
      call mpi_bcast(rc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
#endif
      if(rc.ne.nf_noerr) return

      if(grid%am_i_globalroot) then
        rc2 = 1
        nd = size(shape(arr))
        cnt(:) = 1
        srt(:) = 1
        if(nd.gt.0) then
          cnt(1:nd) = shape(arr)
          srt(1:nd) = 1
        endif
        rc = nf_inq_varndims(fid,vid,ndf)
        rc = nf_inq_vardimid(fid,vid,dids)
        rc = nf_inq_unlimdim(fid,unlim_did)
        if(ndf.gt.0) then
          var_has_unlimdim = dids(ndf).eq.unlim_did
        else
          var_has_unlimdim = .false.
        endif
        if(var_has_unlimdim) then
          rd = ndf
          if(present(record) .or. present(record1)) then
            nrecs = 0
            rc = nf_inq_dimlen(fid,unlim_did,nrecs)
          endif
          if(present(record)) then
            if(record.le.0 .or. nrecs.lt.record) then
              write(6,*)
     &           'error in record dim spec. for variable ',trim(varname)
              rc2 = 0
            else ! non-error case
              srt(rd) = record
              cnt(rd) = 1
              ndf = ndf - 1
            endif
          elseif(present(record1)) then
            if(record1.le.0 .or. nrecs.lt.record1) then
              write(6,*)
     &           'error in record dim spec. for variable ',trim(varname)
              rc2 = 0
            else ! non-error case
              srt(rd) = record1
            endif
          endif
        else
          rd = -1
          if(present(record) .or. present(record1)) then
            write(6,*)
     &           'error in record dim spec. for variable ',trim(varname)
            rc2 = 0
          endif
        endif

      endif ! am i root
      call stoprc(rc2,1)


      ! shape checking
      if(grid%am_i_globalroot) then
        rc2 = 1
        if(nd .ne. ndf) then
          write(6,*) 'shape mismatch for variable ',trim(varname)
          rc2 = 0
        else
          do idim=1,ndf
            rc = nf_inq_dimlen(fid,dids(idim),dlen)
            if(dlen .eq. cnt(idim)) cycle
            if(idim.eq.rd) then
              if(dlen.lt.cnt(idim)+srt(idim)-1) then
                write(6,*)
     &           'warning: only performing partial read of variable '//
     &             trim(varname)//' along the unlimited dimension'
                cnt(idim) = dlen-srt(idim)+1
              endif
            else
              write(6,*) 'shape mismatch for variable ',trim(varname)
              rc2 = 0; exit
            endif
          enddo
        endif
      endif ! am i root
      call stoprc(rc2,1)

      if(grid%am_i_globalroot) then
        if(ndf.eq.0) then
          rc = nf_get_var_double(fid,vid,arr)
        else
          rc = nf_get_vara_double(fid,vid,srt,cnt,arr)
        endif
      endif
#ifndef SERIAL_MODE
      if(present(bcast_all)) then
        if(bcast_all) then
          nd = size(shape(arr))
          if(nd.eq.0) then
            arrsize = 1
          else
            shp(1:nd) = shape(arr)
            arrsize = product(shp(1:nd))
          endif
          call mpi_bcast(arr,arrsize,
     &         MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
        endif
      endif
#endif
