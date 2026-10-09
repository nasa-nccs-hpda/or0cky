### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.94 | 15 | 0.94 | 15 |
| U | 54 | 0 | 0 | 0.89 | 21 | 0.89 | 21 |
| V | 54 | 0 | 0 | 0.97 | 10 | 0.97 | 10 |
| Q | 42 | 12 | 0 | 1.03 | 46 | 1.03 | 46 |
| P | 52 | 2 | 0 | 1.04 | 33 | 1.04 | 33 |
| QCL | 52 | 1 | 1 | 0.95 | 9 | 2.83 | 1 |
| QCI | 37 | 17 | 0 | 1.69 | 47 | 1.69 | 47 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.6880431876504731, 'QCI', 47).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
