### multi-day acceptance score (ACCEPTANCE 4, criterion for the day)

SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4. Members: p1, p2, p3, p4, p5. Steps scored: 45 (0-based [0]..[44]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 45 | 0 | 0 | 0.94 | 15 | 0.94 | 15 |
| U | 45 | 0 | 0 | 0.89 | 21 | 0.89 | 21 |
| V | 45 | 0 | 0 | 0.97 | 10 | 0.97 | 10 |
| Q | 40 | 5 | 0 | 1.02 | 44 | 1.02 | 44 |
| P | 44 | 1 | 0 | 1.03 | 33 | 1.03 | 33 |
| QCL | 43 | 1 | 1 | 0.95 | 39 | 2.83 | 1 |
| QCI | 43 | 2 | 0 | 1.10 | 8 | 1.10 | 8 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.0989702980996512, 'QCI', 8).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step -- SHORT WINDOW (fewer than 54 steps): not the one-model-day criterion of ACCEPTANCE 4

radiation computed by the original Fortran (hybrid component).
