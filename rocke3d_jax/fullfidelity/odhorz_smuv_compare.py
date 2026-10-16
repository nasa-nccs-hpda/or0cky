"""Validate odhorz_ff.py's D44 SMU/SMV accumulation (OCEAN_DYN's integrated horizontal mass
fluxes) against real Fortran dumps. Two checks per date:
1. Per-call: each of the 5 ODHORZ calls' own smu0->smu1 / smv0->smv1 delta, replayed through
   odhorz()'s qeven/smu0/smv0 arguments and compared to the real dumped smu1/smv1.
2. End-to-end: chaining all 5 calls together starting from SMU=SMV=0 (matching the real reset
   at the top of the "Do NO=1,NOCEAN" loop, NOCEAN=1 for this rundeck) using the port's own
   accumulated output as each next call's input, checked against ffz_smfinal_<itime>.bin -- the
   real, fully-integrated SMU/SMV recorded right before OFLUXV is called.
"""
import sys
import numpy as np
from odhorz_ff import odhorz, IM, JM, LMO
from odhorz_compare import load_records, load_lmm, load_lmv, load_lmu, load_hocean, FF_DEFAULT


def load_smuv_records(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    per_rec = 3 + 4 * n3
    assert raw.size % per_rec == 0, (raw.size, per_rec)
    nrec = raw.size // per_rec
    records = []
    off = 0
    for _ in range(nrec):
        rec = {}
        rec["itime"] = raw[off]; off += 1
        rec["dt"] = raw[off]; off += 1
        rec["xeven"] = raw[off]; off += 1
        for name in ["smu0", "smv0", "smu1", "smv1"]:
            a = np.zeros((IM + 1, JM + 1, LMO + 1))
            a[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
            rec[name] = a
        records.append(rec)
    assert off == raw.size
    return records


def load_smfinal(path):
    raw = np.fromfile(path, dtype='>f8')
    n3 = IM * JM * LMO
    assert raw.size == 1 + 2 * n3, (raw.size, 1 + 2 * n3)
    off = 1
    smu = np.zeros((IM + 1, JM + 1, LMO + 1))
    smu[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
    smv = np.zeros((IM + 1, JM + 1, LMO + 1))
    smv[1:, 1:, 1:] = raw[off:off + n3].reshape(LMO, JM, IM).transpose(2, 1, 0); off += n3
    return smu, smv


def run_one(date, itime):
    ff = f"{FF_DEFAULT}/{date}"
    lmm = load_lmm(f"{ff}/ffz_odhorz0_geom.bin")
    lmv = load_lmv(f"{ff}/ffz_polerelax_geom.bin")
    lmu = load_lmu(f"{ff}/ffz_ostres2_geom.bin")
    hocean = load_hocean(f"{ff}/ffz_odhorz_hocean.bin")
    records = load_records(f"{ff}/ffz_odhorz_{itime}.bin")
    smuv_records = load_smuv_records(f"{ff}/ffz_odhorz_smuv_{itime}.bin")
    smu_final_real, smv_final_real = load_smfinal(f"{ff}/ffz_smfinal_{itime}.bin")
    assert len(records) == len(smuv_records), (len(records), len(smuv_records))

    per_call = []
    smu_chain = np.zeros((IM + 1, JM + 1, LMO + 1))
    smv_chain = np.zeros((IM + 1, JM + 1, LMO + 1))
    for rec, srec in zip(records, smuv_records):
        qeven = srec["xeven"] > 0.5
        mo, uo, vo, uod, vod, opbot, smu, smv = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"],
            qeven=qeven, smu0=srec["smu0"], smv0=srec["smv0"])
        du = np.abs(smu - srec["smu1"]).max()
        dv = np.abs(smv - srec["smv1"]).max()
        per_call.append((du, dv,
                          np.allclose(smu, srec["smu1"], atol=1e-6, rtol=1e-6),
                          np.allclose(smv, srec["smv1"], atol=1e-6, rtol=1e-6)))
        # chained (end-to-end) accumulation, using the port's own prior output
        _, _, _, _, _, _, smu_chain, smv_chain = odhorz(
            lmm, lmu, lmv, hocean, rec["dt"],
            rec["moh"], rec["uoh"], rec["voh"], rec["uodh"], rec["vodh"], rec["opboth"],
            rec["mo0"], rec["uo0"], rec["vo0"], rec["uod0"], rec["vod0"], rec["opbot0"],
            rec["vbar"], rec["dzgdp"], rec["usmooth"], rec["pgfx"],
            qeven=qeven, smu0=smu_chain, smv0=smv_chain)

    du_f = np.abs(smu_chain - smu_final_real).max()
    dv_f = np.abs(smv_chain - smv_final_real).max()
    final_ok = (np.allclose(smu_chain, smu_final_real, atol=1e-6, rtol=1e-6) and
                np.allclose(smv_chain, smv_final_real, atol=1e-6, rtol=1e-6))
    return per_call, (du_f, dv_f, final_ok)


if __name__ == "__main__":
    dates = [("nov26", 33312), ("dec01", 33552), ("jan01", 17520)]
    all_ok = True
    for date, itime in dates:
        per_call, (du_f, dv_f, final_ok) = run_one(date, itime)
        print(f"== {date} (itime={itime}) ==")
        for k, (du, dv, ok_u, ok_v) in enumerate(per_call):
            print(f"  call {k}: SMU max_abs_diff={du:.3e} [{'OK' if ok_u else 'FAIL'}]  "
                  f"SMV max_abs_diff={dv:.3e} [{'OK' if ok_v else 'FAIL'}]")
            all_ok = all_ok and ok_u and ok_v
        print(f"  chained final vs ffz_smfinal: SMU={du_f:.3e} SMV={dv_f:.3e} "
              f"[{'OK' if final_ok else 'FAIL'}]")
        all_ok = all_ok and final_ok
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    sys.exit(0 if all_ok else 1)
