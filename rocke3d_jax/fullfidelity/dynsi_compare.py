"""Compare icedyn_dynsi_ff.vpicedyn against the real Fortran ffy_<itime>_{in,out}.bin dumps (D29).
Validates VPICEDYN as one unit (FORM+PLAST+RELAX, iterated to convergence): given the real recorded
inputs (post atm/ocean regrid: GAIRX/GAIRY/GWATX/GWATY/PGFUB/PGFVB, plus HEFF/AREA/AMASS/COR and the
previous step's UICE/VICE), does our plain-Python port reproduce the real UICE(:,:,1)/VICE(:,:,1)/
DMU/DMV/USI/VSI outputs?
"""
import sys
import numpy as np

import icedyn_dynsi_ff as D
from icedyn_geom_ff import geomicdyn, icdyn_masks
from icedyn_geom_compare import read_geom

NX1, NY1, IMICDYN = 74, 46, 72

IN_FIELDS = ["irsi", "imsi", "gairx", "gairy", "gwatx", "gwaty", "pgfub", "pgfvb",
             "heff", "area", "amass", "cor", "uice0", "vice0"]
OUT_FIELDS = ["uice1", "vice1", "dmu", "dmv", "usi", "vsi", "dmui", "dmvi"]


def _read_padded_2d(raw, pos, nx, ny):
    n = nx * ny
    arr = raw[pos:pos + n].reshape(ny, nx).T  # Fortran column-major (nx,ny)
    pos += n
    padded = np.zeros((nx + 1, ny + 1))
    padded[1:, 1:] = arr
    return padded, pos


def read_dynsi_in(path):
    raw = np.fromfile(path, dtype=">f8")
    pos = 1  # skip itime header
    out = {}
    out["irsi"], pos = _read_padded_2d(raw, pos, IMICDYN, NY1)
    out["imsi"], pos = _read_padded_2d(raw, pos, IMICDYN, NY1)
    for name in IN_FIELDS[2:]:
        out[name], pos = _read_padded_2d(raw, pos, NX1, NY1)
    assert pos == raw.size, f"in: consumed {pos}, have {raw.size}"
    return out


def read_dynsi_out(path):
    raw = np.fromfile(path, dtype=">f8")
    pos = 1
    out = {}
    for name in OUT_FIELDS[:4]:
        out[name], pos = _read_padded_2d(raw, pos, NX1, NY1)
    for name in OUT_FIELDS[4:]:
        out[name], pos = _read_padded_2d(raw, pos, IMICDYN, NY1)
    assert pos == raw.size, f"out: consumed {pos}, have {raw.size}"
    return out


def relerr(ref, mine, floor=1e-6):
    return np.abs(ref - mine) / np.maximum(np.abs(ref), floor)


def compare_one(geom_path, in_path, out_path, sinwat, coswat, dts, verbose=True):
    gd = read_geom(geom_path)
    dlon = 2.0 * np.pi / gd["imicdyn"]
    radius = gd["dxt"][1] / dlon
    D.RADIUS = radius
    D.BYRAD2 = 1.0 / (radius * radius)
    geom = geomicdyn(gd["imicdyn"], gd["jmicdyn"], radius)
    heffm, uvm = icdyn_masks(gd["gfocean"], gd["nx1"], gd["jmicdyn"])
    D.init_geometry(geom, heffm, uvm)

    din = read_dynsi_in(in_path)
    dout = read_dynsi_out(out_path)

    # unpad uice0/vice0 (NX1+1,NY1+1) -> (IMICDYN,NY1) real-column slice, matching usi0/vsi0 layout
    usi0 = din["uice0"][2:NX1, 1:NY1 + 1].T.copy()  # (NY1, IMICDYN-1)? need (IMICDYN,NY1) [i,j]
    # uice0 columns 2..NX1-1 (0-based fortran 2..73) correspond to physical i=1..IMICDYN (USI index)
    usi0 = din["uice0"][2:NX1, :].copy()  # shape (IMICDYN, NY1+1) with col0 unused
    vsi0 = din["vice0"][2:NX1, :].copy()
    usi0_2d = np.zeros((IMICDYN, NY1))
    vsi0_2d = np.zeros((IMICDYN, NY1))
    for i in range(IMICDYN):
        for j in range(1, NY1 + 1):
            usi0_2d[i, j - 1] = usi0[i, j]
            vsi0_2d[i, j - 1] = vsi0[i, j]

    uice1, vice1, kki = D.vpicedyn(
        NX1, NY1, usi0_2d, vsi0_2d, din["gairx"], din["gairy"], din["gwatx"], din["gwaty"],
        din["heff"], din["area"], din["amass"], din["cor"], sinwat, coswat, dts, osurf_tilt=0,
        pgfub=din["pgfub"], pgfvb=din["pgfvb"])

    nx1, ny1 = NX1, NY1
    dwatn = D.form(nx1, ny1, uice1, vice1, din["gairx"], din["gairy"], din["gwatx"], din["gwaty"],
                   din["heff"], din["area"], din["amass"], din["cor"], (sinwat, coswat), 0,
                   din["pgfub"], din["pgfvb"])["dwatn"]
    dmu = D._pad(nx1, ny1)
    dmv = D._pad(nx1, ny1)
    half = ny1 // 2
    for j in range(1, ny1 + 1):
        hemi = -1.0 if j <= half else 1.0
        for i in range(1, nx1):
            dmu[i, j] = dts * dwatn[i, j] * (coswat * (uice1[i, j] - din["gwatx"][i, j])
                        - hemi * sinwat * (vice1[i, j] - din["gwaty"][i, j]))
            dmv[i, j] = dts * dwatn[i, j] * (hemi * sinwat * (uice1[i, j] - din["gwatx"][i, j])
                        + coswat * (vice1[i, j] - din["gwaty"][i, j]))
    dmu[nx1, :] = dmu[2, :]
    dmu[1, :] = dmu[nx1 - 1, :]

    usi = np.zeros((IMICDYN + 1, ny1 + 1))
    vsi = np.zeros((IMICDYN + 1, ny1 + 1))
    for j in range(1, ny1):  # J_0S..J_1S = 1..NY1-1
        for i in range(1, IMICDYN + 1):
            u = uice1[i + 1, j]
            v = vice1[i + 1, j]
            usi[i, j] = 0.0 if abs(u) < 1e-10 else u
            vsi[i, j] = 0.0 if abs(v) < 1e-10 else v

    results = {}
    for name, mine in (("uice1", uice1), ("vice1", vice1), ("dmu", dmu), ("dmv", dmv)):
        ref = dout[name]
        mask = heffm[1:, :] if False else None
        e = relerr(ref, mine)
        e = e[1:nx1, 1:ny1]  # interior only (ghost cols/rows not meaningfully compared)
        results[name] = (float(np.max(e)), float(np.mean(e)))

    for name, mine in (("usi", usi), ("vsi", vsi)):
        ref = dout[name]
        e = relerr(ref[1:IMICDYN + 1, 1:ny1], mine[1:IMICDYN + 1, 1:ny1])
        results[name] = (float(np.max(e)), float(np.mean(e)))

    if verbose:
        print(f"kki={kki}")
        for k, (mx, mean) in results.items():
            print(f"{k:8s} max_relerr={mx:.3e} mean_relerr={mean:.3e}")
    return results, kki


if __name__ == "__main__":
    base = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data/nov26"
    oiphi = np.deg2rad(25.0)  # ICEDYN.f PARAMETER OIPHI=25d0*radian
    sinwat, coswat = np.sin(oiphi), np.cos(oiphi)
    dts = 900.0
    compare_one(f"{base}/ffz_geom.bin", f"{base}/ffy_33312_in.bin", f"{base}/ffy_33312_out.bin",
                sinwat, coswat, dts)
