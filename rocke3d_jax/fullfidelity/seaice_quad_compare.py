"""D173: measure how much the float64 Ti/Ti2b shortcut matters for the ff sea-ice ports, against the real-Fortran dumps.

For each port (SEA_ICE/GROUND_SI ffi, ADDICE ffn, SIMELT ffm, seaice_to_atmgrid ffz_s2ag) the existing float64 port and the same port with the
binary128-emulated Ti/Ti2b (seaice_quad_ff.patched()) are compared with the real outputs; per output field: elements, bitwise-equal count,
max abs residual, max residual / field scale (scale = max |ref| over the field, floor 1e-6).  Also counts, over every Ti/Ti2b call made, how often
the float64 and the binary128 result differ in the last bits.
Usage: python seaice_quad_compare.py [ff_data_dir]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seaice_core_ff as S  # noqa: E402
import seaice_quad_ff as Q  # noqa: E402
import addice_compare as AD  # noqa: E402
import simelt_compare as SM  # noqa: E402
import seaice_compare as SC  # noqa: E402

FF = os.environ.get("FF_DATA", "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ff_data")


class Acc:
    def __init__(self):
        self.f = {}

    def add(self, name, got, ref):
        got = np.atleast_1d(np.asarray(got, float)); ref = np.atleast_1d(np.asarray(ref, float))
        d = self.f.setdefault(name, dict(n=0, eq=0, maxabs=0.0, scale=1e-6, maxrel_pt=0.0))
        diff = np.abs(got - ref)
        d["n"] += diff.size; d["eq"] += int(np.sum(got == ref)); d["maxabs"] = max(d["maxabs"], float(diff.max()))
        d["scale"] = max(d["scale"], float(np.abs(ref).max()))

    def rows(self):
        return {k: (v["n"], v["eq"], v["maxabs"], v["maxabs"] / v["scale"]) for k, v in self.f.items()}


def files(pat):
    return sorted(glob.glob(f"{FF}/*/{pat}"))


def port_ffi():
    A1, A2 = Acc(), Acc()
    for fn in files("ffi_*.bin"):
        rec = SC.load(fn)
        for r in rec:
            o = SC.run_row(r)
            for k, rr in (("snow", r[25]), ("msi2", r[34]), ("run", r[35]), ("erun", r[36]), ("srun", r[37]), ("melt12", r[39]), ("cmprs", r[40]),
                          ("srox2", r[41]), ("hsil", r[26:30]), ("ssil", r[30:34])):
                A1.add("sea_ice." + k, o[k], rr)
            o = SC.run_full(r)
            for k, rr in (("snow", r[42]), ("msi2", r[51]), ("runosi", r[52]), ("erunosi", r[53]), ("srunosi", r[54]), ("hsil", r[43:47]), ("ssil", r[47:51])):
                A2.add("ground_si." + k, o[k], rr)
    return {**A1.rows(), **A2.rows()}


def port_ffn():
    A = Acc()
    for fn in files("ffn_*.bin"):
        for r in AD.load(fn):
            o = AD.run_row(r)
            for k, rr in (("snow", r[22]), ("roice", r[23]), ("msi2", r[32]), ("dmimp", r[33]), ("dhimp", r[34]), ("dsimp", r[35]), ("hsil", r[24:28]), ("ssil", r[28:32])):
                A.add("addice." + k, o[k], rr)
    return A.rows()


def port_ffm():
    A = Acc()
    for fn in files("ffm_*.bin"):
        for r in SM.load(fn):
            o = SM.run_row(r)
            for k, rr in (("roice", r[18]), ("snow", r[19]), ("msi2", r[20]), ("enrgused", r[29]), ("hsil", r[21:25]), ("ssil", r[25:29])):
                A.add("simelt." + k, o[k], rr)
    return A.rows()


def port_s2ag():
    import seaice_to_atmgrid_compare as C
    from seaice_to_atmgrid_ff import seaice_to_atmgrid_cell
    A = Acc()
    for fn in files("ffz_s2ag_*.bin"):
        d = C.read_s2ag(fn)
        n = len(d["i"])
        got = {k: np.empty(n) for k in ("gtemp", "gtemp2", "gtempr", "zsnowi", "zsi", "fwsim")}
        for i in range(n):
            o = seaice_to_atmgrid_cell(d["rsi"][i], d["snowi"][i], d["msi"][i], d["hsi1"][i], d["hsi2"][i], d["ssi1"][i], d["ssi2"][i], d["ssi3"][i], d["ssi4"][i])
            for k in got:
                got[k][i] = o[k]
        for k in got:
            A.add("s2ag." + k, got[k], d[k])
    return A.rows()


PORTS = dict(ffi_GROUND_SI=port_ffi, ffn_ADDICE=port_ffn, ffm_SIMELT=port_ffm, ffz_s2ag=port_s2ag)


class CallStats:
    """Wrap the quad Ti/Ti2b to count calls whose float64 and binary128 results differ."""
    def __init__(self):
        self.ti = [0, 0, 0.0]; self.ti2b = [0, 0, 0.0]

    def install(self):
        import seaice_to_atmgrid_ff as A
        st = self

        def ti(e, s):
            q = Q.ti_quad(e, s); f = _ti64(e, s)
            st.ti[0] += 1; st.ti[1] += (q != f); st.ti[2] = max(st.ti[2], abs(q - f)); return q

        def ti2b(e, s, sn, mi):
            q = Q.ti2b_quad(e, s, sn, mi); f = _ti2b64(e, s, sn, mi)
            st.ti2b[0] += 1; st.ti2b[1] += (q != f); st.ti2b[2] = max(st.ti2b[2], abs(q - f)); return q
        S.Ti, S.Ti2b, A.Ti, A.Ti2b = ti, ti2b, ti, ti2b


_ti64, _ti2b64 = S.Ti, S.Ti2b


def main():
    import seaice_to_atmgrid_ff as A
    out = {}
    for name, fn in PORTS.items():
        base = fn()
        stats = CallStats()
        saved = (S.Ti, S.Ti2b, A.Ti, A.Ti2b)
        stats.install()
        try:
            quad = fn()
        finally:
            S.Ti, S.Ti2b, A.Ti, A.Ti2b = saved
        out[name] = (base, quad, stats)
        print(f"== {name}: Ti calls {stats.ti[0]} (f64!=quad {stats.ti[1]}, max |d| {stats.ti[2]:.2e}); Ti2b calls {stats.ti2b[0]} (f64!=quad {stats.ti2b[1]}, max |d| {stats.ti2b[2]:.2e})")
        print(f"   {'field':22s} {'n':>8s} {'eq f64':>8s} {'eq quad':>8s} {'maxabs f64':>11s} {'maxabs quad':>11s} {'rel f64':>9s} {'rel quad':>9s}")
        for k in base:
            n, e0, a0, r0 = base[k]; _, e1, a1, r1 = quad[k]
            print(f"   {k:22s} {n:8d} {e0:8d} {e1:8d} {a0:11.3e} {a1:11.3e} {r0:9.2e} {r1:9.2e}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        FF = sys.argv[1]
    main()
