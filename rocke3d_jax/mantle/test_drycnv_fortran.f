! Minimal Fortran test for DRYCNV module (dry_convection_mixing)
      PROGRAM TEST_DRYCNV_FORTRAN
      INTEGER, PARAMETER :: IM = 2, JM = 2, LM = 3
      REAL*8, DIMENSION(IM, JM, LM) :: T, Q, PK, PDSIG
      REAL*8, DIMENSION(IM, JM, LM) :: T_OUT, Q_OUT
      REAL*8, DIMENSION(IM, JM, LM) :: TV
      REAL*8 :: PKMS, TVMS, QMS, RDP, THM, QM, term1, term2
      INTEGER :: I, J, L
      REAL*8, PARAMETER :: DELTX = 0.608D0

C Initialize inputs with UNSTABLE profile (T decreases with L)
C Shape: (I, J, L) = (2, 2, 3)
      DO I = 1, IM
        DO J = 1, JM
          T(I, J, 1) = 300.0D0
          T(I, J, 2) = 290.0D0
          T(I, J, 3) = 280.0D0
          Q(I, J, 1) = 0.01D0
          Q(I, J, 2) = 0.02D0
          Q(I, J, 3) = 0.03D0
          PK(I, J, 1) = 1.0D0
          PK(I, J, 2) = 1.0D0
          PK(I, J, 3) = 1.0D0
          PDSIG(I, J, 1) = 1.0D0
          PDSIG(I, J, 2) = 1.0D0
          PDSIG(I, J, 3) = 1.0D0
        END DO
      END DO

C Compute dry convection mixing (matching JAX logic with lax.scan)
      T_OUT = T
      Q_OUT = Q
      TV = T_OUT * (1.0D0 + Q_OUT * DELTX)

      DO L = 1, LM - 1
        DO J = 1, JM
          DO I = 1, IM
            IF (TV(I, J, L) > TV(I, J, L+1)) THEN
              term1 = PK(I, J, L) * PDSIG(I, J, L)
              term2 = PK(I, J, L+1) * PDSIG(I, J, L+1)
              PKMS = term1 + term2
              term1 = TV(I, J, L) * PK(I, J, L) * PDSIG(I, J, L)
              term2 = TV(I, J, L+1) * PK(I, J, L+1) * PDSIG(I, J, L+1)
              TVMS = term1 + term2
              term1 = Q_OUT(I, J, L) * PDSIG(I, J, L)
              term2 = Q_OUT(I, J, L+1) * PDSIG(I, J, L+1)
              QMS = term1 + term2
              RDP = 1.0D0 / (PDSIG(I, J, L) + PDSIG(I, J, L+1))
              THM = TVMS / (PKMS * (1.0D0 + QMS * RDP * DELTX))
              QM = QMS * RDP
              T_OUT(I, J, L) = THM
              T_OUT(I, J, L+1) = THM
              Q_OUT(I, J, L) = QM
              Q_OUT(I, J, L+1) = QM
            END IF
          END DO
        END DO
C Update TV after each layer (matching JAX logic)
        DO I = 1, IM
          DO J = 1, JM
            TV(I,J,L) = T_OUT(I,J,L)*(1.0D0+Q_OUT(I,J,L)*DELTX)
            TV(I,J,L+1) = T_OUT(I,J,L+1)*(1.0D0+Q_OUT(I,J,L+1)*DELTX)
          END DO
        END DO
      END DO

C Save outputs to file (flatten in (I, J, L) order)
      OPEN(UNIT=10, FILE='test_drycnv_fortran_output.txt')
      WRITE(10, *) 'T:'
      DO I = 1, IM
        DO J = 1, JM
          DO L = 1, LM
            WRITE(10, *) T_OUT(I, J, L)
          END DO
        END DO
      END DO
      WRITE(10, *) 'Q:'
      DO I = 1, IM
        DO J = 1, JM
          DO L = 1, LM
            WRITE(10, *) Q_OUT(I, J, L)
          END DO
        END DO
      END DO
      CLOSE(10)
      PRINT *, 'Output saved to test_drycnv_fortran_output.txt'
      END PROGRAM TEST_DRYCNV_FORTRAN
