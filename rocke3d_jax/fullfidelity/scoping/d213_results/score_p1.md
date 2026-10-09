### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 53 | 1 | 0 | 1.09 | 3 | 1.09 | 3 |
| U | 53 | 1 | 0 | 0.85 | 29 | 1.29 | 1 |
| V | 54 | 0 | 0 | 0.85 | 28 | 0.85 | 28 |
| Q | 49 | 5 | 0 | 1.01 | 48 | 1.01 | 48 |
| P | 54 | 0 | 0 | 1.00 | 52 | 1.00 | 52 |
| QCL | 50 | 3 | 1 | 1.07 | 21 | 2.89 | 1 |
| QCI | 41 | 13 | 0 | 1.84 | 41 | 1.84 | 41 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.8361234215978322, 'QCI', 41).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
