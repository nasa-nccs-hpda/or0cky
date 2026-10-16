"""Reader for the stream dumps written by the instrumented ModelE (ffdump in
ATM_DRV.f). Big-endian (runtime_opts sets F_UFMTENDIAN=big). File format, per field: char*16 name, int32 rank, int32 lbound(3),
int32 ubound(3), then float64 data in Fortran (column-major) order.
ATURB dumps (ffa_*.bin) begin with one float64 (dtime): use read_dump(path, header_f8=1)
and read out["header"][0].
Returns dict name -> ndarray with shape (I, J, [L]) *including halo bounds*
(as allocated: e.g. J may be 0..JM+1); use .lbound to locate interior."""
import numpy as np


def read_dump(path, header_f8=0):
    out = {}
    with open(path, "rb") as f:
        buf = f.read()
    out = {}
    off = 8 * header_f8
    if header_f8:
        out['header'] = np.frombuffer(buf, '>f8', header_f8, 0).copy()
    while off < len(buf):
        name = buf[off:off + 16].decode().strip(); off += 16
        rank = int(np.frombuffer(buf, ">i4", 1, off)[0]); off += 4
        lb = np.frombuffer(buf, ">i4", 3, off); off += 12
        ub = np.frombuffer(buf, ">i4", 3, off); off += 12
        shape = tuple(int(ub[i] - lb[i] + 1) for i in range(rank))
        n = int(np.prod(shape))
        arr = np.frombuffer(buf, ">f8", n, off).astype(np.float64).reshape(shape, order="F"); off += 8 * n
        out[name] = arr
        out[name + "__lb"] = tuple(int(x) for x in lb[:rank])
    return out
