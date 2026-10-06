"""D158: build_batch for ghy_jax.advnc that is correct for cells with ffnit > 11.

ghy_advnc_test.build_batch gives each cell only the <= 11 Ent/dts records the ffg writer can hold, and dt_total = sum(recorded dts) < 900 s for a cell with
ffnit >= 12, so those cells are advanced over too short a time. Here, for such cells only, the missing iterations are generated with the real loop
(ghy_ref_nit.advnc_full: gdtm -> dts) on the numpy reference, and the Ent exports of iterations > 11 reuse those of iteration 11 (not recorded).
All other cells are returned exactly as AT.build_batch builds them (same arrays, padded to the wider width)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghy_advnc_test as AT
import ghy_compare as GC
import ghy_ref_nit as N

DT = 900.0


def build_batch_nit(rec):
    out = list(AT.build_batch(rec))
    (static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm, ws_can, shc_can, refs) = out
    ffnit = np.round(np.asarray(rec)[:, 289]).astype(int)
    width = max(ent_dts.shape[1], int(ffnit.max()))
    if width > ent_dts.shape[1]:
        pad = width - ent_dts.shape[1]
        ent_dts = np.pad(ent_dts, ((0, 0), (0, pad)))
        ent_cnc = np.pad(ent_cnc, ((0, 0), (0, pad)))
        ent_lai = np.pad(ent_lai, ((0, 0), (0, pad)))
        ent_betadl = np.pad(ent_betadl, ((0, 0), (0, pad), (0, 0)))
    dt_total = np.array(dt_total, dtype=float)
    n_substeps = np.array(n_substeps)
    for i in np.where(ffnit > 11)[0]:
        col, r, info = N.run_cell_full(rec[i], dt=DT, use_recorded_dts=False)
        assert info['nit'] == ffnit[i], (i, info['nit'], ffnit[i])
        u = GC.unpack(rec[i])[3]
        for j in range(info['nit']):
            src = u[min(j, len(u) - 1)]
            ent_dts[i, j] = info['dts'][j]; ent_cnc[i, j] = src['cnc']; ent_lai[i, j] = src['lai']; ent_betadl[i, j] = src['betadl']
        n_substeps[i] = info['nit']
        dt_total[i] = DT
    return (static0, dynamic0, forcing, ent_dts, ent_cnc, ent_betadl, ent_lai, n_substeps, dt_total, snowm, ws_can, shc_can, refs)
