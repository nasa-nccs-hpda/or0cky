### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4. Members: p1, p2, p3, p4, p5. Steps scored: 41 (0-based [0]..[40]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 41 | 0 | 0 | 0.90 | 18 | 0.90 | 18 |
| U | 40 | 1 | 0 | 1.02 | 4 | 1.02 | 4 |
| V | 41 | 0 | 0 | 0.90 | 8 | 0.90 | 8 |
| Q | 41 | 0 | 0 | 0.98 | 19 | 0.98 | 19 |
| P | 41 | 0 | 0 | 0.98 | 37 | 0.98 | 37 |
| QCL | 34 | 6 | 1 | 1.13 | 21 | 2.84 | 1 |
| QCI | 41 | 0 | 0 | 0.98 | 8 | 0.98 | 8 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.1270319740471522, 'QCL', 21).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step -- SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4

radiation computed by the original Fortran (hybrid component).
