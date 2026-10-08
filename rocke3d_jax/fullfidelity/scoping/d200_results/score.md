### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 38 | 1 | 0 | 1.08 | 38 | 1.08 | 38 |
| U | 35 | 4 | 0 | 1.04 | 5 | 1.04 | 5 |
| V | 36 | 3 | 0 | 1.09 | 5 | 1.09 | 5 |
| Q | 29 | 10 | 0 | 1.05 | 38 | 1.05 | 38 |
| P | 35 | 4 | 0 | 1.28 | 38 | 1.28 | 38 |
| QCL | 36 | 2 | 1 | 1.09 | 20 | 2.84 | 1 |
| QCI | 36 | 3 | 0 | 1.15 | 8 | 1.15 | 8 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.2817248859651476, 'P', 38).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
