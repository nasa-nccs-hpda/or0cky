### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.97 | 49 | 0.97 | 49 |
| U | 54 | 0 | 0 | 0.96 | 29 | 0.96 | 29 |
| V | 53 | 1 | 0 | 1.03 | 7 | 1.03 | 7 |
| Q | 54 | 0 | 0 | 1.00 | 19 | 1.00 | 19 |
| P | 52 | 2 | 0 | 1.01 | 52 | 1.01 | 52 |
| QCL | 52 | 1 | 1 | 0.99 | 53 | 2.83 | 1 |
| QCI | 49 | 5 | 0 | 1.08 | 45 | 1.08 | 45 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.0791048470902147, 'QCI', 45).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
