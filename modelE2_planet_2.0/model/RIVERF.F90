! Use #define TOPO_DIRECTED_RIVER_FLOW to disable the version of this subroutine
! which requires prescribed river flow directions.
!
       Subroutine RIVERF
!@sum  RIVERF transports lake water above the sill to cells with lower surface water altitude
!@auth Gary L. Russell
!@ver  2018/03/30

!**** HLAKE  = initial lake depth (m) from Z file = min (HLAKE, HLAKE_MIN)
!**** FLAKE  = current FLAKE
!**** DLAKE  = current lake depth (m)
!**** ZATM/G = atmospheric topography (m) from Z file
!**** dY     = horizontal edge (m) between primary cells (IU) and (ID)
!**** dX     = distance between primary cell centers (IU) and (ID)
!**** dt     = time step (s)
!****
!**** ZSILL  = sill altitude (m) between two adjacent primary cells = max [ZATM/G(IU), ZATM/G(ID)]
!**** HABOVE = water height difference (m) above sill = max [DLAKE(I) - HLAKE(I) + ZATM/G(I) - ZSILL, 0]
!**** dH     = HABOVE(IU) - HABOVE(ID)
!**** RFTUNE = tunable coefficient (s) is related to river flow drag
!****
!**** MFLOW  = river flow (kg) from primary cell IU into cell ID = RFTUNE dH^2 GRAV RHOW FLAKE dt dY / dX
!**** If MFLOW < 0, it is ignored until IU and ID are switched.  It is computed for each of 4 edges; no diagonals.  
!**** dH dY FLAKE  = cross sectional area (m^2) of river flow at edge
!**** dH GRAV / dX = acceleration (m/s^2) of river flow
!**** RHOW = 1000  = conversion (kg/m^3) from volume to mass

      Use CONSTANT,  Only: RHOW, SHW, GRAV,byGRAV, TF, TEENY
      Use RESOLUTION,Only: IM,JM
      Use MODEL_COM, Only: DTSRC
      Use ATM_COM,   Only: ZATMO
      Use GEOM,      Only: DXP,DYP,DXV,DYV, AXYP,byAXYP, IMAXJ, byIM
      Use DIAG_COM,  Only: AIJ=>AIJ_LOC, IJ_MRVR,IJ_ERVR, IJ_MRVRO,IJ_ERVRO, IJ_F0OC,IJ_FWOC, IJ_RVRFLO, &
                           JREG, J_RVRD,J_ERVR, ITLAKE,ITLKICE,ITOCEAN,ITOICE
      Use FLUXES,    Only: AtmOcn,FOCEAN
      Use LAKES,     Only: RFTUNE=>RIVER_FAC
      Use LAKES_COM, Only: MWL,GML, FLAKE,HLAKE=>DLAKE0,DLAKE,GLAKE,TLAKE,MLDLK
      Use SEAICE_COM,Only: LakeIce=>SI_Atm
      Use DOMAIN_DECOMP_ATM,Only: HALO_UPDATE, GRID, GetDomainBounds
      Use DOMAIN_DECOMP_1D, Only: hasSouthPole, hasNorthPole
      Use TimerPackage_Mod, Only: StartTimer=>Start,StopTimer=>Stop
#ifdef TRACERS_WATER
      Use TRDIAG_COM,Only: TAIJN=>TAIJN_LOC, TIJ_RVR,TIJ_RVRO
      Use LAKES_COM, Only: NTM,TRLAKE
#endif
#ifdef SCM
      Use SCM_COM, Only: SCMopt,SCMin
#endif

      Implicit None
      Real*8,Parameter :: FLAKENONE = .03125  !  artificial lake fraction for MWL transport when FLAKE = 0
      Integer :: I1H,I1,IN,INH, J1H,J1,JN,JNH, I,J, IU,JU, ID,JD,KD, JR
      Logical :: QNP,QSP, QLL,QCS
      Real*8  :: dY,dX, ZSILL,dH,MFLOW,EFLOW,dMLDLK,GML1
      Real*8,Dimension(Grid%I_Strt_Halo:Grid%I_Stop_Halo,Grid%J_Strt_Halo:Grid%J_Stop_Halo) :: MFLOWOUT,EFLOWOUT
      Real*8,Dimension(:,:),Pointer :: RSI,GTEMP,GTEMPR,MFLOWIN,EFLOWIN
#ifdef TRACERS_WATER
      Real*8,Dimension(NTM) :: TRLAKE
      Real*8,Dimension(Grid%I_Strt_Halo:Grid%I_Stop_Halo,Grid%J_Strt_Halo:Grid%J_Stop_Halo) :: MFLOWOUTTR
      Real*8,Dimension(:,:),Pointer :: GTRACER,MFLOWINTR
#endif

!**** MWL (kg)  = Lake Water Mass in cell, defined even when FLAKE = 0
!**** GML (J)   = liquid Lake enthalpy
!**** TLAKE (C) = LAKE Temperature of layer 1
!**** MFLOW(kg) = Mass FLOW Out of cell or Into cell
!**** EFLOW (J) = static Energy FLOW Out of cell and Into cell
!**** HLK1 (J)  = Heat content of mixed layer 1 (J/m^2)
!**** MLDLK (m) = Mixed Layer Depth of lake layer 1
!**** GTRACER   = tracer concentration of layer 1  
!****
!**** If 0 = FOCEAN    :  MWL = MWL - MFLOWOUT + MFLOWIN     OCEAN = OCEAN
!**** If 0 < FOCEAN < 1:  MWL = MWL - MFLOWOUT               OCEAN = OCEAN + MFLOWIN
!**** If     FOCEAN = 1:  MWL = MWL                          OCEAN = OCEAN + MFLOWIN

      Call StartTimer ('RIVERF()')
      Call GetDomainBounds (GRID, J_STRT=J1, J_STOP=JN, J_STRT_HALO=J1H, J_STOP_HALO=JNH)
      QSP = HasSouthPole (GRID)
      QNP = HasNorthPole (GRID)
      I1  = GRID%I_STRT  ;  IN = GRID%I_STOP  ;  I1H = GRID%I_STRT_HALO  ;  INH = GRID%I_STOP_HALO
      QLL = I1 .eq. I1H  ;  If (QLL) Then     ;  J1H = Max (J1H,1)       ;  JNH = Min (JNH,JM)  ;  Endif
      QCS = I1 .ne. I1H 

      RSI     => LakeIce%RSI
      GTEMP   => AtmOcn%GTEMP
      GTEMPR  => AtmOcn%GTEMPR
      MFLOWIN => AtmOcn%FLOWO   ;  MFLOWIN(:,:) = 0  ;  MFLOWOUT(:,:) = 0
      EFLOWIN => AtmOcn%EFLOWO  ;  EFLOWIN(:,:) = 0  ;  EFLOWOUT(:,:) = 0
#ifdef TRACERS_WATER
      MFLOWINTR => AtmOcn%TRFLOWO  ;  MFLOWINTR(:,:) = 0  ;  MFLOWOUTTR(:,:) = 0
      GTRACER   => AtmOcn%GTRACER
#endif

!     Call HALO_UPDATE (GRID,FOCEAN)
      Call HALO_UPDATE (GRID, HLAKE)
      Call HALO_UPDATE (GRID, FLAKE)
      Call HALO_UPDATE (GRID,   MWL)
      Call HALO_UPDATE (GRID, MLDLK)
      Call HALO_UPDATE (GRID, TLAKE)
#ifdef TRACERS_WATER
      Call HALO_UPDATE (GRID, GTRACER, jdim=3)
      Call HALO_UPDATE (GRID, TRLAKE(:,1,:,:), jdim=3)
#endif

!**** At LL poles, copy data from I=1 to all longitudes
      If (QSP) Then  ;  FLAKE(2:IM,1 ) = FLAKE(1,1 )
                          MWL(2:IM,1 ) =   MWL(1,1 )
                        TLAKE(2:IM,1 ) = TLAKE(1,1 )  ;  EndIf
      If (QNP) Then  ;  FLAKE(2:IM,JM) = FLAKE(1,JM)
                          MWL(2:IM,JM) =   MWL(1,JM)
                        TLAKE(2:IM,JM) = TLAKE(1,JM)  ;  EndIf
#ifdef TRACERS_WATER
      If (QSP) Then  ;  Do I=2,IM  ;  GTRACER(:,I,1 ) =  GTRACER(:,1,1 )
                                     TRLAKE(:,:,I,1 ) = TRLAKE(:,:,1,1 )  ;  EndDo  ;  EndIf
      If (QNP) Then  ;  Do I=2,IM  ;  GTRACER(:,I,JM) =  GTRACER(:,1,JM)
                                     TRLAKE(:,:,I,JM) = TRLAKE(:,:,1,JM)  ;  EndDo  ;  EndIf
#endif

!**** Compute present lake depth DLAKE (m)
!**** When FLAKE >< 0, ficticious lake fraction is assumed to be .03125 in order to compute lake depth
      Do J=J1H,JNH  ;  Do I=I1H,INH
         If (FLAKE(I,J) > 0) &
            Then  ;  DLAKE(I,J) = MWL(I,J) / (RHOW*AXYP(I,J)*FLAKE(I,J) + TEENY)
            Else  ;  DLAKE(I,J) = MWL(I,J) / (RHOW*AXYP(I,J)*FLAKENONE)  ;  EndIf  ;  EndDo  ;  EndDo

!****
!**** IU,JU and ID,JD loop over all primary cells in an MPI Processor Elements's (PE's) domain and its halo ring.
!**** Consequently, each cell in a PE's computational domain are affected by all of the upstream and downstream fluxes.
!**** If both upstream and downstream cells are outside a PE's computational domain, they are cycled out of the KD loop.
!****
      Do 30 JU=J1H,JNH  ;  Do 20 IU=I1H,INH
         If (FOCEAN(IU,JU) == 1) GoTo 20
         Do 10 KD=1,4  !  adjacent downstream grid cells: east, west, north, south
            If (KD==1) Then  ;  If (IU > IN .or. JU < J1 .or. JU > JN) GoTo 10                     !  programmed for LL only
                                ID = IU+1  ;  JD = JU  ;  dY = DYP(JU)  ;  dX = DXP(JU)  ;  EndIf  !  not programmed for CS
            If (KD==2) Then  ;  If (IU < I1 .or. JU < J1 .or. JU > JN) GoTo 10
                                ID = IU-1  ;  JD = JU  ;  dY = DYP(JU)  ;  dX = DXP(JU)  ;  EndIf
            If (KD==3) Then  ;  If (JU > JN .or. IU < I1 .or. IU > IN .or. QLL.and.JU==JM) GoTo 10
                                JD = JU+1  ;  ID = IU  ;  dY = DXV(JD)  ;  dX = DYV(JD)  ;  EndIf
            If (KD==4) Then  ;  If (JU < J1 .or. IU > IN .or. IU < I1 .or. QLL.and.JU==1) GoTo 10
                                JD = JU-1  ;  ID = IU  ;  dY = DXV(JU)  ;  dX = DYV(JU)  ;  EndIf
            If (QLL) Then  ;  If (ID <  1) ID = IM
                              If (ID > IM) ID = 1  ;  EndIf
            ZSILL = Max(ZATMO(IU,JU),ZATMO(ID,JD)) * byGRAV
            dH = Max(DLAKE(IU,JU) - HLAKE(IU,JU) + ZATMO(IU,JU)*byGRAV - ZSILL, 0d0) &
               - Max(DLAKE(ID,JD) - HLAKE(ID,JD) + ZATMO(ID,JD)*byGRAV - ZSILL, 0d0)
            If (dH <= 0) GoTo 10
            If (FLAKE(IU,JU) > 0) &
               Then  ;  MFLOW = Min (RFTUNE * RHOW * FLAKE(IU,JU) * dH**2 * GRAV * DTSRC * dY / dX, &
                                     .125 * RHOW * FLAKE(IU,JU) * MLDLK(IU,JU) * AXYP(IU,JU))
               Else  ;  MFLOW = Min (RFTUNE * RHOW * FLAKENONE * dH**2 * GRAV * DTSRC * dY / dX, &
                                     .125 * MWL(IU,JU))  ;  EndIf
            EFLOW = MFLOW * SHW * TLAKE(IU,JU)
            MFLOWOUT(IU,JU) = MFLOWOUT(IU,JU) + MFLOW
            EFLOWOUT(IU,JU) = EFLOWOUT(IU,JU) + EFLOW
            MFLOWIN (ID,JD) = MFLOWIN (ID,JD) + MFLOW
            EFLOWIN (ID,JD) = EFLOWIN (ID,JD) + EFLOW  !  + MFLOW * (ZATMO(IU,JU) - ZATMO(ID,JD))
#ifdef TRACERS_WATER
            If (FLAKE(IU,JU) > 0) &
               Then  ;  MFLOWOUTTR(:,IU,JU) = MFLOWOUTTR(:,IU,JU) + MFLOW *  GTRACER(:,IU,JU)
                        MFLOWINTR (:,ID,JD) = MFLOWINTR (:,ID,JD) + MFLOW *  GTRACER(:,IU,JU)
               Else  ;  MFLOWOUTTR(:,IU,JU) = MFLOWOUTTR(:,IU,JU) + MFLOW * TRLAKE(:,1,IU,JU) / (MWL(IU,JU)+TEENY)
                        MFLOWINTR (:,ID,JD) = MFLOWINTR (:,ID,JD) + MFLOW * TRLAKE(:,1,IU,JU) / (MWL(IU,JU)+TEENY)  ;  EndIf
#endif
   10 Continue  !  End KD
   20 Continue  !  End IU
   30 Continue  !  End JU

!****
!**** Apply upstream and downstream river fluxes to cells in a PE's computational domain
!****
      Do J=J1,JN
         If (QSP) Then  ;  MFLOWOUT(1,1 ) = Sum(MFLOWOUT(:,1 )) * byIM
                           EFLOWOUT(1,1 ) = Sum(EFLOWOUT(:,1 )) * byIM
                           MFLOWIN (1,1 ) = Sum(MFLOWIN (:,1 )) * byIM
                           EFLOWIN (1,1 ) = Sum(EFLOWIN (:,1 )) * byIM  ;  EndIf
         If (QNP) Then  ;  MFLOWOUT(1,JM) = Sum(MFLOWOUT(:,JM)) * byIM
                           EFLOWOUT(1,JM) = Sum(EFLOWOUT(:,JM)) * byIM
                           MFLOWIN (1,JM) = Sum(MFLOWIN (:,JM)) * byIM
                           EFLOWIN (1,JM) = Sum(EFLOWIN (:,JM)) * byIM  ;  EndIf
#ifdef TRACERS_WATER
         If (QSP) Then  ;  Do N=1,NTM  ;  MFLOWOUTTR(N,1,1 ) = Sum(MFLOWOUTTR(N,:,1 )) * byIM
                                          MFLOWINTR (N,1,1 ) = Sum(MFLOWINTR (N,:,1 )) * byIM  ;  EndDo  ;  EndIf
         If (QNP) Then  ;  Do N=1,NTM  ;  MFLOWOUTTR(N,1,JM) = Sum(MFLOWOUTTR(N,:,JM)) * byIM
                                          MFLOWINTR (N,1,JM) = Sum(MFLOWINTR (N,:,JM)) * byIM  ;  EndDo  ;  EndIf
#endif

         Do I=I1,IMAXJ(J)
            AIJ(I,J,IJ_MRVRO) = AIJ(I,J,IJ_MRVRO) + MFLOWOUT(I,J)
            AIJ(I,J,IJ_ERVRO) = AIJ(I,J,IJ_ERVRO) + EFLOWOUT(I,J)
            AIJ(I,J,IJ_MRVR)  = AIJ(I,J,IJ_MRVR)  + MFLOWIN (I,J)
            AIJ(I,J,IJ_ERVR)  = AIJ(I,J,IJ_ERVR)  + EFLOWIN (I,J)
            JR = JREG(I,J)
            Call INC_AREG (I,J,JR,J_RVRD, (MFLOWIN(I,J)-MFLOWOUT(I,J))*byAXYP(I,J))
            Call INC_AREG (I,J,JR,J_ERVR, (EFLOWIN(I,J)-EFLOWOUT(I,J))*byAXYP(I,J))
#ifdef TRACERS_WATER
            TAIJN(I,J,TIJ_RVRO,:) = TAIJN(I,J,TIJ_RVRO,:) + MFLOWOUTTR(:) * byAXYP(I,J)
            TAIJN(I,J,TIJ_RVR ,:) = TAIJN(I,J,TIJ_RVR ,:) + MFLOWINTR (:) * byAXYP(I,J)
#endif
#ifdef TRACERS_OBIO_RIVERS
!!!         AIJ(I,J,IJ_RVRFLO) = AIJ(I,J,IJ_RVRFLO) + MFLOWIN(I,J)
#endif

!****       Apply river flow to continental reservoirs
            If (FOCEAN(I,J) == 0) Then
               MWL(I,J) = MWL(I,J) + (MFLOWIN(I,J) - MFLOWOUT(I,J)) 
               GML(I,J) = GML(I,J) + (EFLOWIN(I,J) - EFLOWOUT(I,J)) 
#ifdef TRACERS_WATER
               TRLAKE(:,1,I,J) = TRLAKE(:,1,I,J) + (MFLOWINTR(:,I,J) - MFLOWOUTTR(:,I,J))
#endif
               Call INC_AJ (I,J,ITLAKE ,J_RVRD, (MFLOWIN(I,J)-MFLOWOUT(I,J))*byAXYP(I,J)*(1-RSI(I,J)))
               Call INC_AJ (I,J,ITLAKE ,J_ERVR, (EFLOWIN(I,J)-EFLOWOUT(I,J))*byAXYP(I,J)*(1-RSI(I,J)))
               Call INC_AJ (I,J,ITLKICE,J_RVRD, (MFLOWIN(I,J)-MFLOWOUT(I,J))*byAXYP(I,J)*   RSI(I,J) )
               Call INC_AJ (I,J,ITLKICE,J_ERVR, (EFLOWIN(I,J)-EFLOWOUT(I,J))*byAXYP(I,J)*   RSI(I,J) )
               If (FLAKE(I,J) > 0) &
                  Then  ;  dMLDLK = (MFLOWIN(I,J) - MFLOWOUT(I,J)) / (RHOW*FLAKE(I,J)*AXYP(I,J))
                           GML1   = TLAKE(I,J) * MLDLK(I,J) * (SHW*RHOW*FLAKE(I,J)*AXYP(I,J))
                           MLDLK(I,J) = MLDLK(I,J) + dMLDLK
                           TLAKE(I,J) = (GML1 + (EFLOWIN(I,J) - EFLOWOUT(I,J))) / (MLDLK(I,J)*SHW*RHOW*FLAKE(I,J)*AXYP(I,J))
                           GTEMP(I,J) = TLAKE(I,J)
                          GTEMPR(I,J) = TLAKE(I,J) + TF
                           DLAKE(I,J) = MWL(I,J) / (RHOW*FLAKE(I,J)*AXYP(I,J))
                           GLAKE(I,J) = GML(I,J) / (FLAKE(I,J)*AXYP(I,J))
                           AtmOcn%MLHC(I,J) = SHW * RHOW * MLDLK(I,J)
#ifdef TRACERS_WATER
                           GTRACER(:,I,J) = TRLAKE(:,1,I,J) / (MLDLK(I,J) * RHOW * FLAKE(I,J) * AXYP(I,J))
#endif
#ifdef SCM
                           If (SCMopt%Tskin) Then
                               GTEMP(I,J)  = SCMin%Tskin - TF
                               GTEMPR(I,J) = SCMin%Tskin  ;  EndIf
#endif

                  Else  ;  TLAKE(I,J) = GML(I,J) / (SHW*MWL(I,J) + TEENY)  
                           DLAKE(I,J) = 0
                           GLAKE(I,J) = 0  ;  EndIf  ;  EndIf  !  If FLAKE > 0  !  If FOCEAN = 0

!****       Update lake arrays when 0 < FOCEAN < 1
            If (0 < FOCEAN(I,J) .and. FOCEAN(I,J) < 1) Then
               MWL(I,J) = MWL(I,J) - MFLOWOUT(I,J) 
               GML(I,J) = GML(I,J) - EFLOWOUT(I,J) 
#ifdef TRACERS_WATER
               TRLAKE(:,1,I,J) = TRLAKE(:,1,I,J) - MFLOWOUTTR(:,I,J)
#endif
               Endif

!****       Eliminate tiny lakes  
            If (FOCEAN(I,J) < 1 .and. MWL(I,J) < 1d-6) Then
               MFLOWIN(I,J) = MFLOWIN(I,J) + MWL(I,J)  ;  MWL(I,J) = 0
               EFLOWIN(I,J) = EFLOWIN(I,J) + GML(I,J)  ;  GML(I,J) = 0
#ifdef TRACERS_WATER
               MFLOWINTR(:,I,J) = MFLOWINTR(:,I,J) + (TRLAKE(:,1,I,J)+TRLAKE(:,2,I,J))  ;  TRLAKE(:,1:2,I,J) = 0
#endif
               EndIf

!****       Apply river flow to ocean arrays
            If (FOCEAN(I,J) > 0) Then
               AIJ(I,J,IJ_FWOC) = AIJ(I,J,IJ_FWOC) + MFLOWIN(I,J) * byAXYP(I,J)
               AIJ(I,J,IJ_F0OC) = AIJ(I,J,IJ_F0OC) + EFLOWIN(I,J) * byAXYP(I,J)
               Call INC_AJ (I,J,ITOCEAN,J_RVRD, MFLOWIN(I,J)*byAXYP(I,J)*(1-RSI(I,J)))
               Call INC_AJ (I,J,ITOCEAN,J_ERVR, EFLOWIN(I,J)*byAXYP(I,J)*(1-RSI(I,J)))
               Call INC_AJ (I,J,ITOICE ,J_RVRD, MFLOWIN(I,J)*byAXYP(I,J)*   RSI(I,J) )
               Call INC_AJ (I,J,ITOICE ,J_ERVR, EFLOWIN(I,J)*byAXYP(I,J)*   RSI(I,J) )
               MFLOWIN(I,J) = MFLOWIN(I,J) / (FOCEAN(I,J)*AXYP(I,J))  !  convert (kg) to (kg/m^2 over ocean fraction)
               EFLOWIN(I,J) = EFLOWIN(I,J) / (FOCEAN(I,J)*AXYP(I,J))
#ifdef TRACERS_WATER
               MFLOWINTR(:,I,J) = MFLOWINTR(:,I,J) / (FOCEAN(I,J)*AXYP(I,J))
#endif
               EndIf  ;  EndDo  ;  EndDo  !  If FOCEAN > 0  !  Do I  !  Do J

      Call PRINTLK ('RV')
      Call StopTimer ('RIVERF()')
      EndSubroutine RIVERF
