"""D94: FFT72 (FFT / FFTI / DOCALC), AVRX ported to numpy.

Ports (Fortran line numbers in model/FFT72.f and model/ATMDYN.f of modelE2_planet_2.0):
  FFT0   (tables C,S)            FFT72.f:35-52   -> make_tables()
  DOCALC                         FFT72.f:306-450 -> _docalc()   (statements machine-translated, see below)
  FFT                            FFT72.f:59-107  -> fft72()
  FFTI                           FFT72.f:109-250 -> ffti72()
  AVRX (per-row truncation)      ATMDYN.f:1327-1415 -> avrx_rows(); one-time tables (DRAT,NMIN,BYSN)
                                  ATMDYN.f:1367-1387 -> avrx_tables()

The straight-line statement blocks of DOCALC and FFTI (about 200 statements) were machine-translated
from FFT72.f (Fortran NAME(i,j) -> Python NAME[i,j], same operator order, no re-association) by
scratchpad gen_fft.py / compose_avrx.py; the loops are hand-written in the same order.  All arrays
are laid out (index, row) so that one call transforms many rows at once; every operation is an
elementwise IEEE-754 double operation, so the result is bit-for-bit what a sequential scalar
evaluation in the Fortran statement order gives (the real build uses ifort -O2 -fp-model strict
-assume protect_parens, no FMA).

Besides the faithful radix port there are fft72_np()/ffti72_np(), a real FFT (numpy rfft/irfft)
with the same normalisation, used only to document the tolerance of that alternative.
Tables C,S can be (a) the recorded ones (bitwise tests) or (b) make_tables() (numpy cos).
"""
import numpy as np

KM = 72
IMH = KM // 2
RT2 = 1.4142135623730950
RT3 = 1.7320508075688772
PI = 3.1415926535897932
TWOPI = 2.0 * PI
BYKM = 1.0 / KM
BYKMH = 2.0 / KM


def make_tables():
    """FFT0: C(0:72), S(0:72) (numpy cos of the same arguments; libm may differ by 1 ulp)."""
    C = np.zeros(KM + 1)
    S = np.zeros(KM + 1)
    for N in range(0, KM // 4 + 1):
        C[N] = np.cos(TWOPI * N / float(KM))
        S[KM // 4 - N] = C[N]
        C[KM // 2 - N] = -C[N]
        S[KM // 4 + N] = C[N]
        C[KM // 2 + N] = -C[N]
        S[3 * KM // 4 - N] = -C[N]
        C[KM - N] = C[N]
        S[3 * KM // 4 + N] = -C[N]
    return C, S


def _docalc(F, C, S):
    """F: (KM+1, R) 1-based rows.  Returns dict of the intermediate arrays (index, R)."""
    R = F.shape[1]
    z = lambda *sh: np.zeros(sh + (R,))
    C240 = z(25); C241 = z(25); S241 = z(25)
    C8 = z(9, 5); S8 = z(9, 5)
    C41 = z(10); C42 = z(10); C43 = z(10); C44 = z(10)
    S41 = z(10); S42 = z(10); S43 = z(10); S44 = z(10)
    C21 = z(19); C22 = z(19); S21 = z(19); S22 = z(19)
    for IQ in range(1, 25):
        C240[IQ] = (F[IQ] + F[IQ + 24]) + F[IQ + 48]
        C241[IQ] = (F[IQ] * C[IQ] + F[IQ + 24] * C[IQ + 24]) + F[IQ + 48] * C[IQ + 48]
        S241[IQ] = (F[IQ] * S[IQ] + F[IQ + 24] * S[IQ + 24]) + F[IQ + 48] * S[IQ + 48]
    for IQ in range(1, 9):
        C8[IQ, 0] = (C240[IQ] + C240[IQ + 8]) + C240[IQ + 16]
        C8[IQ, 1] = (C241[IQ] + C241[IQ + 8]) + C241[IQ + 16]
        S8[IQ, 1] = (S241[IQ] + S241[IQ + 8]) + S241[IQ + 16]
    C8[1,2]=(C241[1]-S241[17])*S[15]+(S241[9]-C241[9])*S[9]+(S241[1]-C241[17])*S[3]
    C8[2,2]=(C241[2]-C241[10])*S[12]+(S241[2]+S241[10])*S[6]-S241[18]
    C8[3,2]=(C241[3]+S241[3])*S[9]-(C241[11]+S241[19])*S[15]+(C241[19]+S241[11])*S[3]
    C8[4,2]=(C241[4]+C241[20])*S[6]+(S241[4]-S241[20])*S[12]-C241[12]
    C8[5,2]=(C241[5]-S241[13])*S[3]+(C241[21]-S241[21])*S[9]-(C241[13]-S241[5])*S[15]
    C8[6,2]=(C241[22]-C241[14])*S[12]-(S241[22]+S241[14])*S[6]+S241[6]
    C8[7,2]=(C241[23]+S241[7])*S[15]-(S241[15]+C241[15])*S[9]-(S241[23]+C241[7])*S[3]
    C8[8,2]=C241[24]-(C241[8]+C241[16])*S[6]+(S241[8]-S241[16])*S[12]
    C8[1,3]=C240[1]*S[15]-C240[9]*S[9]-C240[17]*S[3]
    C8[2,3]=(C240[2]-C240[10])*S[12]
    C8[3,3]=C240[3]*S[9]-C240[11]*S[15]+C240[19]*S[3]
    C8[4,3]=(C240[4]+C240[20])*S[6]-C240[12]
    C8[5,3]=C240[5]*S[3]-C240[13]*S[15]+C240[21]*S[9]
    C8[6,3]=(C240[22]-C240[14])*S[12]
    C8[7,3]=C240[23]*S[15]-C240[15]*S[9]-C240[7]*S[3]
    C8[8,3]=C240[24]-(C240[8]+C240[16])*S[6]
    C8[1,4]=(C241[1]+S241[17])*S[15]-(S241[9]+C241[9])*S[9]-(S241[1]+C241[17])*S[3]
    C8[2,4]=(C241[2]-C241[10])*S[12]-(S241[2]+S241[10])*S[6]+S241[18]
    C8[3,4]=(C241[3]-S241[3])*S[9]-(C241[11]-S241[19])*S[15]+(C241[19]-S241[11])*S[3]
    C8[4,4]=(C241[4]+C241[20])*S[6]-(S241[4]-S241[20])*S[12]-C241[12]
    C8[5,4]=(C241[5]+S241[13])*S[3]+(C241[21]+S241[21])*S[9]-(C241[13]+S241[5])*S[15]
    C8[6,4]=(C241[22]-C241[14])*S[12]+(S241[22]+S241[14])*S[6]-S241[6]
    C8[7,4]=(C241[23]-S241[7])*S[15]+(S241[15]-C241[15])*S[9]+(S241[23]-C241[7])*S[3]
    C8[8,4]=C241[24]-(C241[8]+C241[16])*S[6]-(S241[8]-S241[16])*S[12]
    S8[1,2]=(C241[1]+S241[17])*S[3]+(C241[9]+S241[9])*S[9]-(C241[17]+S241[1])*S[15]
    S8[2,2]=(C241[10]+C241[2])*S[6]+(S241[10]-S241[2])*S[12]-C241[18]
    S8[3,2]=(C241[3]-S241[3])*S[9]+(S241[11]-C241[19])*S[15]+(C241[11]-S241[19])*S[3]
    S8[4,2]=(C241[4]-C241[20])*S[12]-(S241[4]+S241[20])*S[6]+S241[12]
    S8[5,2]=(C241[5]+S241[13])*S[15]-(S241[21]+C241[21])*S[9]-(C241[13]+S241[5])*S[3]
    S8[6,2]=C241[6]-(C241[14]+C241[22])*S[6]+(S241[14]-S241[22])*S[12]
    S8[7,2]=(C241[7]-S241[23])*S[15]+(S241[15]-C241[15])*S[9]-(C241[23]-S241[7])*S[3]
    S8[8,2]=(C241[8]-C241[16])*S[12]+(S241[8]+S241[16])*S[6]-S241[24]
    S8[1,3]=C240[1]*S[3]+C240[9]*S[9]-C240[17]*S[15]
    S8[2,3]=(C240[10]+C240[2])*S[6]-C240[18]
    S8[3,3]=C240[3]*S[9]-C240[19]*S[15]+C240[11]*S[3]
    S8[4,3]=(C240[4]-C240[20])*S[12]
    S8[5,3]=C240[5]*S[15]-C240[21]*S[9]-C240[13]*S[3]
    S8[6,3]=C240[6]-(C240[14]+C240[22])*S[6]
    S8[7,3]=C240[7]*S[15]-C240[15]*S[9]-C240[23]*S[3]
    S8[8,3]=(C240[8]-C240[16])*S[12]
    S8[1,4]=(C241[1]-S241[17])*S[3]+(C241[9]-S241[9])*S[9]-(C241[17]-S241[1])*S[15]
    S8[2,4]=(C241[10]+C241[2])*S[6]-(S241[10]-S241[2])*S[12]-C241[18]
    S8[3,4]=(C241[3]+S241[3])*S[9]-(S241[11]+C241[19])*S[15]+(C241[11]+S241[19])*S[3]
    S8[4,4]=(C241[4]-C241[20])*S[12]+(S241[4]+S241[20])*S[6]-S241[12]
    S8[5,4]=(C241[5]-S241[13])*S[15]+(S241[21]-C241[21])*S[9]-(C241[13]-S241[5])*S[3]
    S8[6,4]=C241[6]-(C241[14]+C241[22])*S[6]-(S241[14]-S241[22])*S[12]
    S8[7,4]=(C241[7]+S241[23])*S[15]-(S241[15]+C241[15])*S[9]-(C241[23]+S241[7])*S[3]
    S8[8,4]=(C241[8]-C241[16])*S[12]-(S241[8]+S241[16])*S[6]+S241[24]
    for N in range(0, 5):
        C41[N]=C8[1,N]+C8[5,N]
        C42[N]=C8[2,N]+C8[6,N]
        C43[N]=C8[3,N]+C8[7,N]
        C44[N]=C8[8,N]+C8[4,N]
    for N in range(1, 5):
        S41[N]=S8[1,N]+S8[5,N]
        S42[N]=S8[2,N]+S8[6,N]
        S43[N]=S8[3,N]+S8[7,N]
        S44[N]=S8[8,N]+S8[4,N]
        C41[9-N]=(C8[1,N]-C8[5,N]+S8[1,N]-S8[5,N])*C[9]
        C42[9-N]=S8[2,N]-S8[6,N]
        C43[9-N]=(C8[7,N]-C8[3,N]+S8[3,N]-S8[7,N])*C[9]
        C44[9-N]=C8[8,N]-C8[4,N]
        S41[9-N]=(C8[1,N]-C8[5,N]+S8[5,N]-S8[1,N])*C[9]
        S42[9-N]=C8[2,N]-C8[6,N]
        S43[9-N]=(C8[3,N]-C8[7,N]+S8[3,N]-S8[7,N])*C[9]
        S44[9-N]=S8[4,N]-S8[8,N]
    C41[9]=(C8[1,0]-C8[5,0])*C[9]
    C42[9]=0.
    C43[9]=(C8[7,0]-C8[3,0])*C[9]
    C44[9]=C8[8,0]-C8[4,0]
    S41[9]=(C8[1,0]-C8[5,0])*C[9]
    S42[9]=C8[2,0]-C8[6,0]
    S43[9]=(C8[3,0]-C8[7,0])*C[9]
    S44[9]=0.
    for N in range(0, 10):
        C21[N]=C41[N]+C43[N]
        C22[N]=C44[N]+C42[N]
    for N in range(1, 9):
        S21[N]=S41[N]+S43[N]
        S22[N]=S44[N]+S42[N]
    for N in range(1, 10):
        C21[18-N]=S41[N]-S43[N]
        C22[18-N]=C44[N]-C42[N]
        S21[18-N]=C41[N]-C43[N]
        S22[18-N]=S42[N]-S44[N]
    C21[18]=0.
    C22[18]=C44[0]-C42[0]
    S21[18]=C41[0]-C43[0]
    S22[18]=0.
    return C21, C22, S21, S22


def fft72(Frows, C, S):
    """FFT: Frows (R, 72) gridpoint rows -> (A, B), each (R, 37) (index 0..36)."""
    Frows = np.asarray(Frows, dtype=float)
    R = Frows.shape[0]
    F = np.zeros((KM + 1, R))
    F[1:] = Frows.T
    C21, C22, S21, S22 = _docalc(F, C, S)
    A = np.zeros((IMH + 1, R))
    B = np.zeros((IMH + 1, R))
    A[0] = (C22[0] + C21[0]) * BYKM
    B[0] = 0.
    for N in range(1, KM // 4 + 1):
        A[N] = (C22[N] + C21[N]) * BYKMH
        B[N] = (S22[N] + S21[N]) * BYKMH
    for N in range(1, KM // 4):
        A[KM // 2 - N] = (C22[N] - C21[N]) * BYKMH
        B[KM // 2 - N] = (S21[N] - S22[N]) * BYKMH
    A[KM // 2] = (C22[0] - C21[0]) * BYKM
    B[KM // 2] = 0.
    return A.T.copy(), B.T.copy()


def ffti72(Arows, Brows, C, S):
    """FFTI: spectra (R, 37) -> gridpoint rows (R, 72)."""
    A = np.asarray(Arows, dtype=float).T
    Bq = np.asarray(Brows, dtype=float).T
    R = A.shape[1]
    z = lambda *sh: np.zeros(sh + (R,))
    C240 = z(25); C241 = z(25); S241 = z(25)
    C8 = z(9, 5); S8 = z(9, 5)
    C41 = z(10); C42 = z(10); C43 = z(10); C44 = z(10)
    S41 = z(10); S42 = z(10); S43 = z(10); S44 = z(10)
    C21 = z(19); C22 = z(19); S21 = z(19); S22 = z(19)
    F = z(KM + 1)
    B = Bq
    C22[0] = (A[0] + A[36]) * 2.
    C21[0] = (A[0] - A[36]) * 2.
    for N in range(1, 19):
        C22[N] = A[N] + A[36 - N]
        C21[N] = A[N] - A[36 - N]
        S22[N] = B[N] - B[36 - N]
        S21[N] = B[N] + B[36 - N]
    for N in range(0, 10):
        C42[N] = C22[N] - C22[18 - N]
        C44[N] = C22[N] + C22[18 - N]
    for N in range(0, 9):
        C41[N] = C21[N] + S21[18 - N]
        C43[N] = C21[N] - S21[18 - N]
    for N in range(1, 10):
        S42[N] = S22[N] + S22[18 - N]
        S44[N] = S22[N] - S22[18 - N]
        S41[N] = S21[N] + C21[18 - N]
        S43[N] = S21[N] - C21[18 - N]
    for N in range(0, 5):
        C8[8,N]=C44[N]+C44[9-N]
        C8[4,N]=C44[N]-C44[9-N]
        C8[2,N]=C42[N]+S42[9-N]
        C8[6,N]=C42[N]-S42[9-N]
    for N in range(1, 5):
        S8[2,N]=S42[N]+C42[9-N]
        S8[6,N]=S42[N]-C42[9-N]
        S8[8,N]=S44[N]-S44[9-N]
        S8[4,N]=S44[N]+S44[9-N]
        S8[5,N]=S41[N]+(S41[9-N]-C41[9-N])*S[9]
        S8[1,N]=S41[N]-(S41[9-N]-C41[9-N])*S[9]
        S8[3,N]=S43[N]+(S43[9-N]+C43[9-N])*S[9]
        S8[7,N]=S43[N]-(S43[9-N]+C43[9-N])*S[9]
        C8[5,N]=C41[N]-(S41[9-N]+C41[9-N])*S[9]
        C8[1,N]=C41[N]+(S41[9-N]+C41[9-N])*S[9]
        C8[3,N]=C43[N]+(S43[9-N]-C43[9-N])*S[9]
        C8[7,N]=C43[N]-(S43[9-N]-C43[9-N])*S[9]
    C8[5,0]=C41[0]-S41[9]*RT2
    C8[1,0]=C41[0]+S41[9]*RT2
    C8[3,0]=C43[0]+S43[9]*RT2
    C8[7,0]=C43[0]-S43[9]*RT2
    C240[24]=C8[8,0]+C8[8,3]*2.
    C240[16]=C8[8,0]-C8[8,3]-S8[8,3]*RT3
    C240[8]=C8[8,0]-C8[8,3]+S8[8,3]*RT3
    C240[18]=C8[2,0]-S8[2,3]*2.
    C240[2]=C8[2,0]+S8[2,3]+C8[2,3]*RT3
    C240[10]=C8[2,0]+S8[2,3]-C8[2,3]*RT3
    C240[12]=C8[4,0]-C8[4,3]*2.
    C240[4]=C8[4,0]+C8[4,3]+S8[4,3]*RT3
    C240[20]=C8[4,0]+C8[4,3]-S8[4,3]*RT3
    C240[6]=C8[6,0]+S8[6,3]*2.
    C240[14]=C8[6,0]-S8[6,3]-C8[6,3]*RT3
    C240[22]=C8[6,0]-S8[6,3]+C8[6,3]*RT3
    C240[9]=C8[1,0]-(C8[1,3]-S8[1,3])*RT2
    C240[1]=C8[1,0]+C8[1,3]*2.*S[15]+S8[1,3]*2.*S[3]
    C240[17]=C8[1,0]-C8[1,3]*2.*S[3]-S8[1,3]*2.*S[15]
    C240[3]=C8[3,0]+(C8[3,3]+S8[3,3])*RT2
    C240[11]=C8[3,0]-C8[3,3]*2.*S[15]+S8[3,3]*2.*S[3]
    C240[19]=C8[3,0]+C8[3,3]*2.*S[3]-S8[3,3]*2.*S[15]
    C240[21]=C8[5,0]+(C8[5,3]-S8[5,3])*RT2
    C240[13]=C8[5,0]-C8[5,3]*2.*S[15]-S8[5,3]*2.*S[3]
    C240[5]=C8[5,0]+C8[5,3]*2.*S[3]+S8[5,3]*2.*S[15]
    C240[15]=C8[7,0]-(C8[7,3]+S8[7,3])*RT2
    C240[23]=C8[7,0]+C8[7,3]*2.*S[15]-S8[7,3]*2.*S[3]
    C240[7]=C8[7,0]-C8[7,3]*2.*S[3]+S8[7,3]*2.*S[15]
    C241[24]=C8[8,2]+C8[8,4]+C8[8,1]
    S241[24]=-S8[8,2]+S8[8,4]+S8[8,1]
    C241[8]=-(C8[8,2]+C8[8,4])*S[6]+(S8[8,2]+S8[8,4])*S[12]+C8[8,1]
    C241[16]=-(C8[8,2]+C8[8,4])*S[6]-(S8[8,2]+S8[8,4])*S[12]+C8[8,1]
    S241[8]=(C8[8,2]-C8[8,4])*S[12]+(S8[8,2]-S8[8,4])*S[6]+S8[8,1]
    S241[16]=-(C8[8,2]-C8[8,4])*S[12]+(S8[8,2]-S8[8,4])*S[6]+S8[8,1]
    C241[18]=-S8[2,2]-S8[2,4]+C8[2,1]
    S241[18]=-C8[2,2]+C8[2,4]+S8[2,1]
    C241[2]=(C8[2,2]+C8[2,4])*S[12]+(S8[2,2]+S8[2,4])*S[6]+C8[2,1]
    C241[10]=-(C8[2,2]+C8[2,4])*S[12]+(S8[2,2]+S8[2,4])*S[6]+C8[2,1]
    S241[2]=(C8[2,2]-C8[2,4])*S[6]-(S8[2,2]-S8[2,4])*S[12]+S8[2,1]
    S241[10]=(C8[2,2]-C8[2,4])*S[6]+(S8[2,2]-S8[2,4])*S[12]+S8[2,1]
    S241[12]=S8[4,2]-S8[4,4]+S8[4,1]
    C241[12]=-C8[4,2]-C8[4,4]+C8[4,1]
    S241[4]=(C8[4,2]-C8[4,4])*S[12]-(S8[4,2]-S8[4,4])*S[6]+S8[4,1]
    S241[20]=-(C8[4,2]-C8[4,4])*S[12]-(S8[4,2]-S8[4,4])*S[6]+S8[4,1]
    C241[4]=(C8[4,2]+C8[4,4])*S[6]+(S8[4,2]+S8[4,4])*S[12]+C8[4,1]
    C241[20]=(C8[4,2]+C8[4,4])*S[6]-(S8[4,2]+S8[4,4])*S[12]+C8[4,1]
    C241[6]=S8[6,2]+S8[6,4]+C8[6,1]
    S241[6]=C8[6,2]-C8[6,4]+S8[6,1]
    C241[22]=(C8[6,2]+C8[6,4])*S[12]-(S8[6,2]+S8[6,4])*S[6]+C8[6,1]
    C241[14]=-(C8[6,2]+C8[6,4])*S[12]-(S8[6,2]+S8[6,4])*S[6]+C8[6,1]
    S241[22]=-(C8[6,2]-C8[6,4])*S[6]-(S8[6,2]-S8[6,4])*S[12]+S8[6,1]
    S241[14]=-(C8[6,2]-C8[6,4])*S[6]+(S8[6,2]-S8[6,4])*S[12]+S8[6,1]
    C241[9]=(-C8[1,2]-C8[1,4]+S8[1,2]+S8[1,4])*S[9]+C8[1,1]
    S241[9]=(C8[1,2]-C8[1,4]+S8[1,2]-S8[1,4])*S[9]+S8[1,1]
    C241[1]=(C8[1,2]+C8[1,4])*S[15]+(S8[1,2]+S8[1,4])*S[3]+C8[1,1]
    C241[17]=-(C8[1,2]+C8[1,4])*S[3]-(S8[1,2]+S8[1,4])*S[15]+C8[1,1]
    S241[1]=(C8[1,2]-C8[1,4])*S[3]-(S8[1,2]-S8[1,4])*S[15]+S8[1,1]
    S241[17]=-(C8[1,2]-C8[1,4])*S[15]+(S8[1,2]-S8[1,4])*S[3]+S8[1,1]
    C241[3]=(C8[3,2]+C8[3,4]+S8[3,2]+S8[3,4])*S[9]+C8[3,1]
    S241[3]=(C8[3,2]-C8[3,4]-S8[3,2]+S8[3,4])*S[9]+S8[3,1]
    C241[11]=-(C8[3,2]+C8[3,4])*S[15]+(S8[3,2]+S8[3,4])*S[3]+C8[3,1]
    S241[19]=-(C8[3,2]-C8[3,4])*S[15]-(S8[3,2]-S8[3,4])*S[3]+S8[3,1]
    C241[19]=(C8[3,2]+C8[3,4])*S[3]-(S8[3,2]+S8[3,4])*S[15]+C8[3,1]
    S241[11]=(C8[3,2]-C8[3,4])*S[3]+(S8[3,2]-S8[3,4])*S[15]+S8[3,1]
    C241[21]=(C8[5,2]+C8[5,4]-S8[5,2]-S8[5,4])*S[9]+C8[5,1]
    S241[21]=(-C8[5,2]+C8[5,4]-S8[5,2]+S8[5,4])*S[9]+S8[5,1]
    C241[13]=-(C8[5,2]+C8[5,4])*S[15]-(S8[5,2]+S8[5,4])*S[3]+C8[5,1]
    S241[13]=-(C8[5,2]-C8[5,4])*S[3]+(S8[5,2]-S8[5,4])*S[15]+S8[5,1]
    C241[5]=(C8[5,2]+C8[5,4])*S[3]+(S8[5,2]+S8[5,4])*S[15]+C8[5,1]
    S241[5]=(C8[5,2]-C8[5,4])*S[15]-(S8[5,2]-S8[5,4])*S[3]+S8[5,1]
    C241[15]=-(C8[7,2]+C8[7,4]+S8[7,2]+S8[7,4])*S[9]+C8[7,1]
    S241[15]=(-C8[7,2]+C8[7,4]+S8[7,2]-S8[7,4])*S[9]+S8[7,1]
    C241[23]=(C8[7,2]+C8[7,4])*S[15]-(S8[7,2]+S8[7,4])*S[3]+C8[7,1]
    C241[7]=-(C8[7,2]+C8[7,4])*S[3]+(S8[7,2]+S8[7,4])*S[15]+C8[7,1]
    S241[23]=-(C8[7,2]-C8[7,4])*S[3]-(S8[7,2]-S8[7,4])*S[15]+S8[7,1]
    S241[7]=(C8[7,2]-C8[7,4])*S[15]+(S8[7,2]-S8[7,4])*S[3]+S8[7,1]
    for IQ in range(1, 25):
        F[IQ+24]=(S241[IQ]*S[IQ+24]+C241[IQ]*C[IQ+24])+C240[IQ]*.5
        F[IQ]=(S241[IQ]*C[IQ+48]-C241[IQ]*S[IQ+48])*RT3+F[IQ+24]
        F[IQ+48]=-(S241[IQ]*C[IQ]-C241[IQ]*S[IQ])*RT3+F[IQ+24]
    return F[1:].T.copy()


# ---------------------------------------------------------------------------------------------
# AVRX
def avrx_tables(drat_dxp, bydyp3, dlon, xavrx=1.0):
    """AVRX one-time init (ATMDYN.f:1367-1387) from DXP(1:JM), BYDYP(3), DLON.
    Returns DRAT (JM,), NMIN (JM,) (0 where the Fortran leaves NMIN undefined), BYSN (1:36, index 0 unused).
    Index convention: arrays are indexed by Fortran J-1 (rows) and Fortran N (BYSN[N])."""
    jm = len(drat_dxp)
    bysn = np.zeros(IMH + 1)
    for N in range(1, IMH + 1):
        bysn[N] = xavrx / np.sin(.5 * dlon * N)
    drat = drat_dxp * bydyp3
    nmin = np.zeros(jm, dtype=int)
    for j in range(jm):
        for N in range(IMH, 0, -1):
            if bysn[N] * drat[j] > 1.:
                nmin[j] = N + 1
                break
    return drat, nmin, bysn


def avrx_rows(X, jrows, drat, nmin, bysn, C, S, return_spec=False):
    """AVRX applied to the rows X (R, 72) taken at Fortran latitude indices jrows (1-based, R,).
    Rows with DRAT(J) > 1 are skipped (returned unchanged).  Fortran order per row:
    FFT; AN(N)=BYSN(N)*DRAT(J)*AN(N) for N=NMIN..IMH-1; AN(IMH) likewise; FFTI."""
    X = np.array(X, dtype=float)
    out = X.copy()
    act = [r for r, j in enumerate(jrows) if not (drat[j - 1] > 1)]
    if not act:
        return (out, None, None) if return_spec else out
    A, B = fft72(X[act], C, S)
    A0, B0 = A.copy(), B.copy()
    for k, r in enumerate(act):
        j = jrows[r]
        d = drat[j - 1]
        for N in range(nmin[j - 1], IMH):
            A[k, N] = (bysn[N] * d) * A[k, N]
            B[k, N] = (bysn[N] * d) * B[k, N]
        A[k, IMH] = (bysn[IMH] * d) * A[k, IMH]
    out[act] = ffti72(A, B, C, S)
    if return_spec:
        return out, A0, B0
    return out


# ---------------------------------------------------------------------------------------------
# alternative: real FFT with the same normalisation (tolerance-validated only)
def fft72_np(Frows):
    x = np.asarray(Frows, dtype=float)
    X = np.fft.rfft(x, axis=-1)
    n = np.arange(IMH + 1)
    zz = np.conj(X) * np.exp(2j * np.pi * n / KM)     # sum_K F(K) exp(+i 2 pi N K / KM), K=1..KM
    sc = np.full(IMH + 1, 2.0 / KM)
    sc[0] = sc[IMH] = 1.0 / KM
    A = zz.real * sc
    B = zz.imag * sc
    B[..., 0] = 0.
    B[..., IMH] = 0.
    return A, B


def ffti72_np(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = np.arange(IMH + 1)
    zq = A + 1j * B
    zq[..., 0] = A[..., 0]
    zq[..., IMH] = A[..., IMH]
    y = np.conj(zq) * np.exp(2j * np.pi * n / KM)     # F(K)=Re sum_N z_N e^{-i theta}
    y[..., 1:IMH] *= KM / 2.0
    y[..., 0] *= KM
    y[..., IMH] *= KM
    return np.fft.irfft(y, n=KM, axis=-1)


def avrx_rows_np(X, jrows, drat, nmin, bysn):
    """AVRX with the real-FFT transforms (same truncation logic)."""
    X = np.array(X, dtype=float)
    out = X.copy()
    act = [r for r, j in enumerate(jrows) if not (drat[j - 1] > 1)]
    if not act:
        return out
    A, B = fft72_np(X[act])
    for k, r in enumerate(act):
        j = jrows[r]
        d = drat[j - 1]
        for N in range(nmin[j - 1], IMH):
            A[k, N] = (bysn[N] * d) * A[k, N]
            B[k, N] = (bysn[N] * d) * B[k, N]
        A[k, IMH] = (bysn[IMH] * d) * A[k, IMH]
    out[act] = ffti72_np(A, B)
    return out
