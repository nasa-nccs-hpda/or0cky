"""Generate Reports/PROJECT_CHRONOLOGY.md from chronology_data.py (git log + ledger ids).
Usage: python3 build_chronology_md.py ../Reports/PROJECT_CHRONOLOGY.md"""
import sys
import chronology_data as C

out = sys.argv[1] if len(sys.argv) > 1 else "PROJECT_CHRONOLOGY.md"
ph = {k: (lab, s, e) for k, lab, s, e, col in C.PHASES}
L = []
L.append("# Project chronology: ROCKE-3D to JAX (2026-08-28 to 2026-10-08)\n")
L.append("Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code agent on 2026-10-08 from the repository only. Project-local per `projects/imvi/AGENTS.md`.")
L.append("Status: generated file; do not edit by hand. Source data and generator: `status_slides/chronology_data.py`, `status_slides/build_chronology_md.py`. The figures and the slide deck built from the same data are `status_slides/status_deck.pdf` (superseded deck kept as `status_deck_2026-10-01.pdf`).")
L.append("Limits: dates and short commit ids are from `git log` on branch `full-fidelity-port` (HEAD `ec36779`, 273 commits); 'what was learned' is quoted or condensed from the ledger entry named in the last column, not re-measured. The ledger (`FULL_FIDELITY_DELTAS.md`) ends at D198; there is no D199 entry. Commit times are local commit times; the ledger entries carry their own time stamps.")
L.append("Review by: when a GPU run exists, when the day criterion status changes, or when the owner records a relaxation (ACCEPTANCE_CRITERIA section 9).\n")
L.append("## 1. Phases\n")
L.append("| Phase | Dates | Name |\n|---|---|---|")
for k, lab, s, e, col in C.PHASES:
    L.append("| %s | %s to %s | %s |" % (k, s[5:], e[5:], lab))
L.append("\n## 2. Owner decisions on the timeline\n")
L.append("| Date and time | Decision (recorded in) |\n|---|---|")
for d, t, lab in C.DECISIONS:
    L.append("| %s %s | %s |" % (d, t, lab))
L.append("\nAlso recorded 2026-10-07: Ent exports stay recorded for runs up to one day and the Ent port starts (commit `367674a`); one writer per working tree (`ccf16b8`); test and push rule: large batches, no full-suite re-run per code change (`d16fba8`). Recorded 2026-10-08: the owner said the criteria may be relaxed later only to advance to the GPU run; if so it must be recorded as a dated decision made after the results, with the unrelaxed status next to it (ACCEPTANCE_CRITERIA section 9.3).\n")
L.append("## 3. Chronology\n")
L.append("| Date | Time | Commit | Phase | Ledger | Event | What was learned | Source |\n|---|---|---|---|---|---|---|---|")
for d, t, c, p, i, ev, ln, src in C.ROWS:
    esc = lambda s: s.replace("|", "/")
    L.append("| %s | %s | `%s` | %s | %s | %s | %s | %s |" % (d, t or "-", c, p, i or "-", esc(ev), esc(ln), esc(src)))
L.append("\n## 4. Activity (commits per day)\n")
L.append("| Date | Commits |\n|---|---|")
for d in sorted(C.COMMITS_PER_DAY):
    L.append("| %s | %d |" % (d, C.COMMITS_PER_DAY[d]))
L.append("\nTotal: %d commits. No commits on the days not listed.\n" % sum(C.COMMITS_PER_DAY.values()))
L.append("## 5. Regression 'passed' counts recorded in the repository\n")
L.append("| Date | Commit | Passed | Note |\n|---|---|---|---|")
for d, t, c, n, note in C.TESTS:
    L.append("| %s | `%s` | %s | %s |" % (d, c, format(n, ","), note or "-"))
L.append("\nThe entry `264283d` (2,471 passed + 206 post-fix) is left out because the two numbers are not additive as written. The one failure at `51acc2f` and `eda8789` is the export-artifact test (`test_jax_harness::test_header_complete`), changed afterwards in `1036056` to tolerate a missing `.git`; the suite was not re-run after that change.\n")
L.append("## 6. Where the documents disagree (newer ledger entry used)\n")
for s in [
    "Step time: `GOAL.md` (10-06) says about 6 s per step plus clouds; D187 44.3 s; D191 18.8 s steady. D191 is used.",
    "Remaining effort: `GOAL.md` about 100-170 h to a month comparison; `README_START_HERE.md` says that is too high but gives no new total. No total is claimed.",
    "Noise floor at steps < 3: D192, D193 and D195 said about 1e-9; the correction in commit `8a649f7` shows QCL step 1 largest member distance 4.06e-7. The correction is used.",
    "SURFACE rebuild cause: D193 said tile-set change; D195 shows `max_substeps`. D195 is used.",
    "Day table counts: D193 (Q 43/11/0, QCL 49/4/1) versus D195 (Q 41/13/0, QCL 47/6/1). D195 is used.",
    "C1 array count: D187 559/559; D191 558/558 (the host index array `land/p4_ij` exists only in the reference). 558 used for D191.",
    "`Project_Summary_and_Conclusions.md` stops at D51 (680 tests); `STATUS.md` describes Track A. The ledger and `README_START_HERE.md` are used for later facts.",
    "GPU host: `ACCEPTANCE_CRITERIA.md` lists it as to be named; `README_START_HERE.md` and `gpu/DISCOVER_RUN.md` record an A100 smoke test (job 58776063) and probe (job 58778982) on Discover. Both are stated.",
]:
    L.append("- " + s)
L.append("\n## 7. Status in one paragraph (2026-10-08)\n")
L.append("A hybrid, device-resident JAX coupled step exists (D191): C1 bitwise to the NumPy chain on three dates at step 0 and for steps 0-22 of the nov26 day; C2 at step 0 is MET with one named column on nov26 and PARTLY MET on dec01 and jan01; 18.8 s steady per step on CPU (was 44.3 s), 99 jit executions (was 262). The 54-step nov26 day is NOT MET under ACCEPTANCE_CRITERIA section 9 (QCL step 1, ratio 2.84, one bistable cell, D197); steps >= 3 have worst ratio 1.13. Radiation is the original Fortran or replayed; libimf math, QUS, pole columns and the OADVT2 pre-pass are host callbacks; Ent exports, land forcing and tile radiation columns are recorded. SOCRATES is not ported. No GPU run of the step exists (D198 runner validated on CPU only); the A100 probe shows launch-bound behavior and 1-ulp device exp/sin/pow differences (6-12% of values), so a GPU result cannot be bitwise.\n")
open(out, "w").write("\n".join(L))
print("wrote", out)
