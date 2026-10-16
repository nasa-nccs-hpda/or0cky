# Deck notes: how fidelity is measured (draft for the next briefing-deck update)

Status: DRAFT, 2026-10-16. Not yet in any built deck or PDF. Requested by the user on 2026-10-14; paragraph drafted by a Claude Code agent from `Reports/ACCEPTANCE_CRITERIA.md` s2-4, s9-10 and the project memory note. Owner of the criteria: G. Tamkin. Review before use: the wording has not been checked by the owner or a domain expert.

Placement when the decks are next rebuilt (`build_briefing_decks_20261014.py` or its successor): one slide, or a card on the evidence slide. Keep the dated Oct 13 and Oct 14 PDFs as snapshots.

## Slide paragraph (plain language, about 110 words, no results stated)

**How we measure fidelity.** Fidelity asks whether the new JAX version of the model behaves like the original Fortran model, including its physics. The climate model is chaotic, so even a one-digit nudge to its starting point gives a slightly different answer later. We use that natural variation as our ruler: we re-run the original model from five such nudged starts and measure how far apart the answers spread. Then we ask how far the JAX answer is from the original run, compared with that spread. Our rule is that it never exceeds about twice the spread. Matching to many decimal places is a separate, secondary check, because tiny rounding differences grow over time without meaning any physics is missing.

## Goal statement (from an agent reply relayed by the user, 2026-10-16; wording not yet reviewed by the owner)

"Verify that JAX implements the intended physics, then assess whether its multi-day results remain within the accepted variability of the Fortran model."

- The one-step A/B/C/D categories are a diagnostic for numerical agreement, not the definition of overall fidelity.
- Neither the one-step check nor the multi-day spread check proves the physics is complete. Physics coverage needs its own explicit check; whether one exists in the repo has not been established.
- The same reply said the strict threshold is not met because QCI exceeds the allowed spread at step 3. That is the agent's statement and is unverified here. Under the documented rules, a ratio of 2.37 would fail the strict rule (2x) and pass the relaxed rule (3x, steps >= 3); report both.

## Points the paragraph deliberately leaves out (carry in speaker comments, not on the slide)

- A ratio such as "QCI 2.37" means cloud ice differs from the reference by 2.37 times the reference spread. It does NOT mean 2.37 times more ice (user correction 2026-10-14).
- Strict rule: never beyond 2 times the largest member distance, at every step and every scored field. Relaxed experiment rule (owner decision 2026-10-11): at most 3 for steps >= 3, always shown beside the strict status.
- The A/B/C/D categories (bitwise / <= 1e-12 / <= 1e-6 / worse) compare one JAX end state with one reference end state from the same start. They are a single-step precision gate (C1 port consistency), not the fidelity test (C2, against the real Fortran members). A large D count after several steps is expected when ulp-level differences grow, and is not by itself evidence of missing physics.
- No verdict is stated here. State the current status only from the latest dated report.

## Sources

- `Reports/ACCEPTANCE_CRITERIA.md` s2-4 (reference spread, criterion), s9-10 (strict and relaxed rules)
- Project memory note `project-slides-add-fidelity-explainer` (user request and corrections, 2026-10-14)
- GPU baseline of 2026-10-13/14 (jobs 37942390, 37942591): deterministic across nodes at `061486d`; its A/B/C/D counts are a precision diagnostic only.

## Prompt for the fidelity-report agent (revised 2026-10-16)

Purpose: have an agent re-explain the fidelity assessment with fidelity (ensemble-spread ratio) kept separate from the A/B/C/D precision diagnostic. Drafted by a Claude Code agent after an expert reader found the report's A/B/C/D section confusing. The prompt has not been run, and the section numbers of `ACCEPTANCE_CRITERIA.md` beyond s2-4 and s9-10 were not checked.

```
Re-explain the fidelity assessment of the JAX port, separating two different questions. Use only what is documented in the repo (mainly Reports/ACCEPTANCE_CRITERIA.md) or in your own run outputs. Where something is not documented, say so and don't infer it.

1. FIDELITY (primary question): does the JAX model reproduce the Fortran model's behavior, including its physics?
   - Fortran is chaotic, so a tiny perturbation to the initial state grows into a different end state after N days. The Fortran model's own variation is the yardstick.
   - State the documented criterion first, before any results, quoting the source section:
     a. Reference spread: how it is built (the five real one-ulp Fortran members for the nov26 day; the longer-window members if used), and which variables and fields are scored.
     b. The ratio: JAX distance from the reference run divided by the largest member distance.
     c. The strict rule (never beyond 2x at every step and every scored field) and the relaxed experiment rule (at most 3x for steps >= 3, owner decision 2026-10-11, always shown beside the strict status). Say who set each rule and where it is recorded.
     d. If anything you used is NOT in the documented criteria (your own thresholds, an envelope test, a different ratio definition), label it as yours and say so plainly.
   - Then report the result. For every ratio give the field, the step, and the run it comes from. Interpret a ratio such as "QCI 2.37" correctly: cloud ice differs from the reference by 2.37 times the reference spread; it does NOT mean 2.37 times more ice. State against which rule (strict or relaxed) each figure is judged, and by how much it passes or fails.
   - Say what this test can and cannot detect. In particular, state whether it would reveal a missing piece of physics, and what other checks would be needed.

2. ACCURACY / PRECISION (separate diagnostic, clearly labeled as not a fidelity measure):
   - The A/B/C/D categories compare ONE JAX end state against ONE Fortran or CPU reference end state from an identical start state (A bitwise, B rel <= 1e-12, C <= 1e-6, D worse). They are the single-step gate (port consistency, C1), not the fidelity test against the real Fortran members (C2).
   - State plainly that this measures numerical agreement, such as precision, device exp/pow rounding, and operation order, and that large D counts after several steps are expected when ulp-level differences are amplified by chaos.
   - Do not use these categories as evidence for or against fidelity. If an earlier statement did, correct it.

3. Format: put the fidelity criterion and result first, then the accuracy diagnostic, then open questions and what you could not determine. Name the source file or run for every number.

4. SLIDE EXPLAINER: write one short paragraph (about 80-120 words) for a slide, in plain language for a reader who is not a modeler.
   - Say what "fidelity" means here: does the port behave like the original model, including its physics?
   - Use the chaos idea: the original model gives slightly different answers if you nudge its starting point, so we measure that natural spread and ask how far the port's answer is from the original run compared with that spread (the rule is that it never exceeds about twice the spread).
   - Say in one sentence that matching to many decimal places (accuracy) is a different, secondary check, and why the two can disagree.
   - Use no jargon (no "ulp", "category D", "bitwise") and do not state any result that is not in your section 1. If the result is not settled, say the check is still open instead of giving a verdict.
```
