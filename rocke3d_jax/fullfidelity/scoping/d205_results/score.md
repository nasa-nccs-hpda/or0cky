### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.94 | 41 | 0.94 | 41 |
| U | 53 | 1 | 0 | 1.03 | 5 | 1.03 | 5 |
| V | 54 | 0 | 0 | 0.93 | 10 | 0.93 | 10 |
| Q | 40 | 14 | 0 | 1.05 | 46 | 1.05 | 46 |
| P | 54 | 0 | 0 | 1.00 | 21 | 1.00 | 21 |
| QCL | 50 | 3 | 1 | 1.01 | 14 | 2.84 | 1 |
| QCI | 49 | 5 | 0 | 1.12 | 8 | 1.12 | 8 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.1218486589208323, 'QCI', 8).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
