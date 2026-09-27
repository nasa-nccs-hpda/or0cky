# Phase 2a proof-of-concept: calling the real SOCRATES library from Python

`gauss_angle_driver.f90` links against the real, precompiled `libsocrates.a` (no reimplementation)
and exposes ONE real SOCRATES kernel, `gauss_angle` (the Gaussian-quadrature IR-flux solver used
inside `monochromatic_ir_radiance`), over stdin/stdout. `socrates_py.py` drives it as a subprocess.

**Why subprocess, not ctypes/.so:** `libsocrates.a`'s object files are not `-fPIC`
(`ld: relocation R_X86_64_32 ... can not be used when making a shared object` -- confirmed by
attempting `ifort -shared -Wl,--whole-archive libsocrates.a`). A `ctypes` `.so` load needs a full
SOCRATES source recompile with `-fPIC` (source is at `ModelE_Support/socrates/src/`, ~150 files) --
separate, larger work. The subprocess architecture avoids that and still calls unmodified SOCRATES
object code with the same toolchain (`ifort` 19.1.3) the real ModelE binary uses.

Build: `source ../env_modele.sh && ./build.sh`. Run: `LD_LIBRARY_PATH=<socrates libdir> python3 socrates_py.py`.

**Scope:** this validates the calling architecture for ONE kernel, not full radiation. The real
radiation entry point (`RCOMPX`/`run_planet_rad`, ~100 module-level inputs populated by ~1,500 lines
of `RAD_DRV.f`) is scoped in `FULL_FIDELITY_PLAN.md` Phase 2 but not yet built this way.
