"""Phase 2a proof-of-concept: call the REAL, unmodified SOCRATES library (`libsocrates.a`) from Python.

Architecture note (found empirically, 2026-09-27): `libsocrates.a`'s object files were compiled without
`-fPIC`, so `ld` refuses to link them into a shared object (`relocation R_X86_64_32 ... can not be used
when making a shared object`) -- confirmed by attempting exactly that. A `ctypes`/`cffi` `.so` load is
therefore not available without a full SOCRATES source recompile with `-fPIC` (source is present under
`ModelE_Support/socrates/src/`, but recompiling the ~150-file library is separate, larger work).

This module instead drives a small statically-linked Fortran executable (`gauss_angle_driver`, built
here from real SOCRATES source, `gauss_angle.F90`, linked against `libsocrates.a`) over stdin/stdout.
Python never reimplements the physics; it only marshals arrays to/from the real compiled routine.

Scope: this wraps ONE real SOCRATES kernel (`gauss_angle`, the Gaussian-quadrature IR flux solver used
inside `monochromatic_ir_radiance`) as a template for the eventual column-radiation driver. It is NOT
yet the full `RCOMPX`/`run_planet_rad` radiation call (see FULL_FIDELITY_PLAN.md Phase 2 for that scope)
-- this proves the calling architecture, not full radiation fidelity.
"""
import subprocess
import numpy as np

DRIVER = __file__.rsplit("/", 1)[0] + "/gauss_angle_driver"


def gauss_angle(tau, diff_planck, flux_inc_down, source_ground, albedo_surface_diff, n_order_gauss=2):
    """tau, diff_planck: 1-D arrays (n_layer,). Returns flux_diffuse (2*n_layer+2,) from the real
    SOCRATES gauss_angle routine (up/down flux at each layer edge, non-scattering IR quadrature)."""
    tau = np.asarray(tau, dtype=np.float64)
    diff_planck = np.asarray(diff_planck, dtype=np.float64)
    n_layer = tau.shape[0]
    assert diff_planck.shape[0] == n_layer
    inp = (f"{n_layer} {n_order_gauss}\n"
           + " ".join(f"{x:.17e}" for x in tau) + "\n"
           + " ".join(f"{x:.17e}" for x in diff_planck) + "\n"
           + f"{flux_inc_down:.17e} {source_ground:.17e} {albedo_surface_diff:.17e}\n")
    out = subprocess.run([DRIVER], input=inp, capture_output=True, text=True, check=True)
    return np.array([float(x) for x in out.stdout.split()])


if __name__ == "__main__":
    flux = gauss_angle(tau=[0.5, 0.8, 1.2], diff_planck=[10.0, 8.0, 5.0],
                       flux_inc_down=0.0, source_ground=300.0, albedo_surface_diff=0.1)
    print("Real SOCRATES gauss_angle() called from Python, flux_diffuse:", flux)
