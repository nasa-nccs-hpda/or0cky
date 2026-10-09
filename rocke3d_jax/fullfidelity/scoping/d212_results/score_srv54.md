### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.97 | 46 | 0.97 | 46 |
| U | 53 | 1 | 0 | 1.01 | 29 | 1.01 | 29 |
| V | 54 | 0 | 0 | 0.99 | 28 | 0.99 | 28 |
| Q | 54 | 0 | 0 | 0.99 | 23 | 0.99 | 23 |
| P | 51 | 3 | 0 | 1.05 | 6 | 1.05 | 6 |
| QCL | 41 | 12 | 1 | 1.41 | 8 | 2.83 | 1 |
| QCI | 42 | 12 | 0 | 1.18 | 47 | 1.18 | 47 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.4063090836111016, 'QCL', 8).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
