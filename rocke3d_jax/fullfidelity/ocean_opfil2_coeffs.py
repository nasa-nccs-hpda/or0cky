"""D138: numpy port of calc_opfil2_coeffs (OCNDYN2.f:887-1063, module opfil2_coeffs): the OPFIL2 setup tables that
were read from the recorded ffo_opcoef.bin. Analytic from LMU and the ocean geometry (DXPO, DYPO(3)).
Returns the same flat float64 vector layout as the instrumented dump (ffo_opcoef.bin), so
opfil2_ff.coef_from_dump(calc_opfil2_coeffs(lmu)) is a drop-in for the recorded file.
Serial decomposition assumed (js0=2, js1=JM-1, i.e. size(nmin)=JM-2): the recorded file has nmn=44 and agrees.
nbyzu/i1yzu/i2yzu are the contiguous runs of LMU(:,J)>=L without wraparound (OCNDYN.f:505-518, get_i1i2)."""
import numpy as np
from ocean_odiff import geometry, IM, JM, LMO, TWOPI

IMZ2 = IM // 2
HWID_MAX = 15
HUGE_I4 = 2147483647


def runs(q):
    """get_i1i2: 1-based (start, end) of each contiguous True run of q (no wraparound)."""
    out = []; i = 0; n = len(q)
    while i < n:
        if q[i]:
            j = i
            while j + 1 < n and q[j + 1]:
                j += 1
            out.append((i + 1, j + 1)); i = j + 1
        else:
            i += 1
    return out


def calc_opfil2_coeffs(lmu):
    g = geometry(); dxpo, dypo = g['dxpo'], g['dypo']
    js0, js1 = 2, JM - 1
    nmn = js1 - js0 + 1
    smooth = np.zeros((IMZ2, nmn)); nmin = np.zeros(nmn, dtype=np.int64)
    smooth_set = np.zeros((IMZ2, nmn), dtype=bool)
    for j in range(js0, js1 + 1):
        drat = dxpo[j] / dypo[3]
        n = IMZ2
        while n >= 1:
            smooth[n - 1, j - js0] = IMZ2 * drat / n; smooth_set[n - 1, j - js0] = True
            if smooth[n - 1, j - js0] >= 1.0:
                break
            n -= 1
        nmin[j - js0] = n + 1                 # n=0 after a full loop, as in the Fortran
    n1fft = np.full(LMO, HUGE_I4, dtype=np.int64); n2fft = np.full(LMO, -1, dtype=np.int64)
    n1fil = np.full(LMO, HUGE_I4, dtype=np.int64); n2fil = np.full(LMO, -1, dtype=np.int64)
    jfft, jfil, i1fil, i2fil, indx_fil = [], [], [], [], []
    indx_sv = {}
    reduco = []
    for l in range(1, LMO + 1):
        for j in range(js0, js1 + 1):
            if dxpo[j] >= dypo[3]:
                continue
            ja = min(j, JM + 1 - j)
            iwmin = int(np.ceil(dypo[3] / (dypo[3] - dxpo[ja])))
            seg = runs(lmu[:, j - 1] >= l)
            if len(seg) == 1 and seg[0] == (1, IM):
                n = len(jfft) + 1
                jfft.append(j); n1fft[l - 1] = min(n1fft[l - 1], n); n2fft[l - 1] = max(n2fft[l - 1], n)
            else:
                if l <= lmu[IM - 1, j - 1] and seg[0][0] == 1:        # IDL crossing
                    nbas = len(seg) - 1
                    bas = list(seg[1:nbas]) if nbas > 1 else []
                    bas.append((seg[-1][0], seg[0][1] + IM))
                else:
                    bas = list(seg)
                for (iw, ie) in bas:
                    iwide = ie - iw + 2
                    if iwide < iwmin:
                        continue
                    n = len(jfil) + 1
                    if (iwide, ja) not in indx_sv:
                        indx_sv[(iwide, ja)] = len(reduco)
                        k1 = [max(1, i - HWID_MAX) for i in range(1, iwide)]
                        k2 = [min(iwide - 1, i + HWID_MAX) for i in range(1, iwide)]
                        km = 2 * iwide
                        reduc = np.zeros((iwide - 1, iwide - 1))
                        for nn in range(iwide - 1, 0, -1):
                            reducn = (1 - dxpo[ja] * iwide / (dypo[3] * nn)) * 4 / km
                            if reducn <= 0:
                                continue
                            sintab = np.sin(TWOPI * nn * np.arange(1, iwide) / km)
                            for i in range(1, iwide):
                                for k in range(k1[i - 1], k2[i - 1] + 1):
                                    reduc[k - 1, i - 1] = reduc[k - 1, i - 1] + sintab[k - 1] * (sintab[i - 1] * reducn)
                        for i in range(1, iwide):
                            for k in range(k1[i - 1], k2[i - 1] + 1):
                                reduco.append(reduc[k - 1, i - 1])
                    jfil.append(j); i1fil.append(iw); i2fil.append(ie); indx_fil.append(indx_sv[(iwide, ja)])
                    n1fil[l - 1] = min(n1fil[l - 1], n); n2fil[l - 1] = max(n2fil[l - 1], n)
    nred, nfft, nfil = len(reduco), len(jfft), len(jfil)
    nsm = IMZ2 * nmn
    v = np.concatenate([[nred, nfft, nfil, nmn, nsm], smooth.reshape(-1, order='F'), nmin, jfft, n1fft, n2fft,
                        jfil, i1fil, i2fil, indx_fil, n1fil, n2fil, reduco]).astype(np.float64)
    return v, smooth_set.reshape(-1, order='F')
