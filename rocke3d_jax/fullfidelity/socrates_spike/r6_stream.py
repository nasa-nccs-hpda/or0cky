"""R6: streaming reader of ffsoc_dump.bin (format of ffsoc/ffsoc_mod.F90), no 3 GB in memory.
   for rec in stream(path, tags=None): rec = dict(tag, iso, ncall, band, k, shape, a)   (tag 900 timer records are skipped)"""
import struct
import numpy as np


def stream(path, tags=None):
    with open(path, "rb", buffering=1 << 20) as f:
        while True:
            h = f.read(32)
            if len(h) < 32:
                return
            tag, iso, nc, ib, k, rank, n1, n2 = struct.unpack(">8i", h)
            if tag == 900:
                f.seek(8 + 96 + 48, 1)
                continue
            ext = struct.unpack(">%di" % (2 * rank), f.read(8 * rank))
            shape = ext[:rank]
            n = int(np.prod(shape))
            if tags is not None and tag not in tags:
                f.seek(8 * n, 1)
                continue
            a = np.frombuffer(f.read(8 * n), ">f8").reshape(shape, order="F").astype(float)
            yield dict(tag=tag, iso=iso, ncall=nc, band=ib, k=k, shape=shape, a=a)
