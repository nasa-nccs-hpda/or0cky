### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4. Members: p1, p2, p3, p4, p5. Steps scored: 6 (0-based [0]..[5]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 6 | 0 | 0 | 0.66 | 5 | 0.66 | 5 |
| U | 6 | 0 | 0 | 0.52 | 5 | 0.52 | 5 |
| V | 6 | 0 | 0 | 0.62 | 5 | 0.62 | 5 |
| Q | 6 | 0 | 0 | 0.86 | 5 | 0.86 | 5 |
| P | 6 | 0 | 0 | 0.45 | 5 | 0.45 | 5 |
| QCL | 4 | 1 | 1 | 0.58 | 4 | 2.83 | 1 |
| QCI | 6 | 0 | 0 | 0.23 | 5 | 0.23 | 5 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (0.8640315714217486, 'Q', 5).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step -- SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4

radiation computed by the original Fortran (hybrid component).
