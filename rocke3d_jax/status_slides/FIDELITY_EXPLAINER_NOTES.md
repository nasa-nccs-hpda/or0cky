# Deck notes: how fidelity is measured (draft for the next briefing-deck update)

Status: DRAFT, 2026-10-16. Not yet in any built deck or PDF. Requested by the user on 2026-10-14; paragraph drafted by a Claude Code agent from `Reports/ACCEPTANCE_CRITERIA.md` s2-4, s9-10 and the project memory note. Owner of the criteria: G. Tamkin. Review before use: the wording has not been checked by the owner or a domain expert.

Placement when the decks are next rebuilt (`build_briefing_decks_20261014.py` or its successor): one slide, or a card on the evidence slide. Keep the dated Oct 13 and Oct 14 PDFs as snapshots.

## Slide paragraph (plain language, about 110 words, no results stated)

**How we measure fidelity.** Fidelity asks whether the new JAX version of the model behaves like the original Fortran model, including its physics. The climate model is chaotic, so even a one-digit nudge to its starting point gives a slightly different answer later. We use that natural variation as our ruler: we re-run the original model from five such nudged starts and measure how far apart the answers spread. Then we ask how far the JAX answer is from the original run, compared with that spread. Our rule is that it never exceeds about twice the spread. Matching to many decimal places is a separate, secondary check, because tiny rounding differences grow over time without meaning any physics is missing.

## Points the paragraph deliberately leaves out (carry in speaker comments, not on the slide)

- A ratio such as "QCI 2.37" means cloud ice differs from the reference by 2.37 times the reference spread. It does NOT mean 2.37 times more ice (user correction 2026-10-14).
- Strict rule: never beyond 2 times the largest member distance, at every step and every scored field. Relaxed experiment rule (owner decision 2026-10-11): at most 3 for steps >= 3, always shown beside the strict status.
- The A/B/C/D categories (bitwise / <= 1e-12 / <= 1e-6 / worse) compare one JAX end state with one reference end state from the same start. They are a single-step precision gate (C1 port consistency), not the fidelity test (C2, against the real Fortran members). A large D count after several steps is expected when ulp-level differences grow, and is not by itself evidence of missing physics.
- No verdict is stated here. State the current status only from the latest dated report.

## Sources

- `Reports/ACCEPTANCE_CRITERIA.md` s2-4 (reference spread, criterion), s9-10 (strict and relaxed rules)
- Project memory note `project-slides-add-fidelity-explainer` (user request and corrections, 2026-10-14)
- GPU baseline of 2026-10-13/14 (jobs 37942390, 37942591): deterministic across nodes at `061486d`; its A/B/C/D counts are a precision diagnostic only.
