### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

one model day (54 steps). Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 53 | 1 | 0 | 1.11 | 3 | 1.11 | 3 |
| U | 53 | 1 | 0 | 0.91 | 22 | 1.41 | 1 |
| V | 54 | 0 | 0 | 0.90 | 19 | 0.90 | 19 |
| Q | 47 | 7 | 0 | 1.01 | 51 | 1.03 | 1 |
| P | 54 | 0 | 0 | 0.99 | 32 | 1.00 | 2 |
| QCL | 50 | 3 | 1 | 1.04 | 18 | 2.92 | 1 |
| QCI | 52 | 2 | 0 | 1.07 | 8 | 1.85 | 2 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.1089383094827772, 'T', 3).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).
