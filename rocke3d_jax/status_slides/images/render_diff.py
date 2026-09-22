import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RJAX = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/imvi/rocke3d_jax"
DATA_DIR = os.path.join(RJAX, "compare_data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # status_slides/images/, alongside this script

IM, JM, LM = 72, 46, 40
SURFACE_LAYER = 0

_ref = nc.Dataset(os.path.join(RJAX, "ANN4099.aijP2SAoM40.nc"))
lat = _ref.variables["lat"][:].data
lon = _ref.variables["lon"][:].data
_ref.close()

NAVY = "#0F172A"
CARD = "#1E293B"
INK = "#F8FAFC"
INK2 = "#E2E8F0"
MUTED = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": CARD, "axes.edgecolor": MUTED,
    "text.color": INK2, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10,
})


def load_fortran_3d(field):
    path = os.path.join(DATA_DIR, f"drycnv_fortran_{field}_out.dat")
    return np.fromfile(path, dtype=np.float64).reshape((IM, JM, LM), order="F")


def load_fortran_1d(field):
    path = os.path.join(DATA_DIR, f"pbl_fortran_{field}_out.dat")
    return np.fromfile(path, dtype=np.float64)


def load_jax(module, device, field):
    path = os.path.join(DATA_DIR, f"{module}_jax_{device}_{field}_out.npy")
    return np.load(path) if os.path.exists(path) else None


def to_lat_lon(arr_im_jm):
    return np.asarray(arr_im_jm).T


# Same fields/values verify_diff will cross-check against outputs/p2saom40_kernel_*_diff_*.html
fields = [
    ("drycnv", "T", to_lat_lon(load_fortran_3d("T")[:, :, SURFACE_LAYER]),
     to_lat_lon(load_jax("drycnv", "cpu", "T")[:, :, SURFACE_LAYER]),
     to_lat_lon(load_jax("drycnv", "gpu", "T")[:, :, SURFACE_LAYER])),
    ("pbl", "u", to_lat_lon(load_fortran_1d("u").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "cpu", "u").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "gpu", "u").reshape(IM, JM))),
    ("pbl", "dpsih", to_lat_lon(load_fortran_1d("dpsih").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "cpu", "dpsih").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "gpu", "dpsih").reshape(IM, JM))),
    ("pbl", "dpsiq", to_lat_lon(load_fortran_1d("dpsiq").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "cpu", "dpsiq").reshape(IM, JM)),
     to_lat_lon(load_jax("pbl", "gpu", "dpsiq").reshape(IM, JM))),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
fig.suptitle("JAX − Fortran (difference maps, P2SAoM40 grid, synthetic test values)",
             color=INK, fontsize=16, fontweight="bold")

for col, (group, field, fortran, jax_cpu, jax_gpu) in enumerate(fields):
    for row, (dev, jax_arr) in enumerate([("cpu", jax_cpu), ("gpu", jax_gpu)]):
        diff = jax_arr - fortran
        vabs = max(abs(diff.min()), abs(diff.max()), 1e-12)
        ax = axes[row, col]
        im = ax.pcolormesh(lon, lat, diff, shading="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.set_title(f"{group}.{field}\nJAX-{dev.upper()} − Fortran", color=INK2, fontsize=10)
        if row == 1:
            ax.set_xlabel("Longitude", fontsize=8)
        if col == 0:
            ax.set_ylabel("Latitude", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.22, fraction=0.06)
        cbar.ax.tick_params(labelsize=7, colors=MUTED)
        print(f"{group}.{field} jax_{dev} max|diff|={np.max(np.abs(diff)):.3e}")

out_path = os.path.join(OUT_DIR, "diff_grid.png")
fig.savefig(out_path, dpi=150, facecolor=NAVY)
plt.close(fig)
print("wrote", out_path)
