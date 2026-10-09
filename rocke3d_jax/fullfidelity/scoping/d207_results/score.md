### D207 day1 (steps 0-53)

day1 (steps 0-53); D207 wrapper over multiday_score.py, members p1-p5 run for 108 steps, daydir nov26_day2. Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [0]..[53]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

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

### D207 day2 (steps 54-107)

day2 (steps 54-107); D207 wrapper over multiday_score.py, members p1-p5 run for 108 steps, daydir nov26_day2. Members: p1, p2, p3, p4, p5. Steps scored: 54 (0-based [54]..[107]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 54 | 0 | 0 | 0.93 | 54 | 0.93 | 54 |
| U | 54 | 0 | 0 | 0.82 | 56 | 0.82 | 56 |
| V | 54 | 0 | 0 | 0.82 | 54 | 0.82 | 54 |
| Q | 17 | 37 | 0 | 1.06 | 107 | 1.06 | 107 |
| P | 54 | 0 | 0 | 0.95 | 62 | 0.95 | 62 |
| QCL | 53 | 1 | 0 | 1.04 | 87 | 1.04 | 87 |
| QCI | 52 | 2 | 0 | 1.10 | 56 | 1.10 | 56 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: True (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.1047484736149926, 'QCI', 56).
Verdict text: never beyond 2x of the largest member, but NOT within at every step (near at some steps)

radiation computed by the original Fortran (hybrid component).

### D207 both days (steps 0-107)

both days (steps 0-107); D207 wrapper over multiday_score.py, members p1-p5 run for 108 steps, daydir nov26_day2. Members: p1, p2, p3, p4, p5. Steps scored: 108 (0-based [0]..[107]). Thresholds fixed (atm_day_report): within <= 1, near <= 2, beyond > 2 times the largest member rms(member - real).

| field | within (incl. below) | near | beyond | worst ratio (steps >= 3) | at step | worst ratio (all steps) | at step |
|---|---|---|---|---|---|---|---|
| T | 108 | 0 | 0 | 0.94 | 15 | 0.94 | 15 |
| U | 108 | 0 | 0 | 0.89 | 21 | 0.89 | 21 |
| V | 108 | 0 | 0 | 0.97 | 10 | 0.97 | 10 |
| Q | 59 | 49 | 0 | 1.06 | 107 | 1.06 | 107 |
| P | 106 | 2 | 0 | 1.04 | 33 | 1.04 | 33 |
| QCL | 105 | 2 | 1 | 1.04 | 87 | 2.83 | 1 |
| QCI | 89 | 19 | 0 | 1.69 | 47 | 1.69 | 47 |

Criterion (a) within at every scored step, all fields: False.  (b) never beyond 2x of the largest member: False (steps >= 3 only: True).  Worst ratio (steps >= 3): (1.6880431876504731, 'QCI', 47).
Verdict text: beyond 2x only at steps < 3 (floor ~1e-9, not quoted in D157/D171), never beyond at steps >= 3; NOT within at every step

radiation computed by the original Fortran (hybrid component).

NOTE (D207): the sentence "radiation computed by the original Fortran" printed by multiday_score.report_markdown does NOT apply to these runs: radiation (SRHR/TRHR/COSZ1) is REPLAYED from the records.
