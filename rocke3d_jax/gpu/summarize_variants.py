#!/usr/bin/env python
"""Score the two fidelity runs and write one summary table for a gpu/run_variants.sh run.

usage (from fullfidelity/, graphcast-env):  python ../gpu/summarize_variants.py RUN_DIR
Environment: FORTRAN_RUNS (where the Fortran timing runs wrote run_<name>_<kind><i>/), FF_DATA (for the scorer).
Writes RUN_DIR/SUMMARY.md and RUN_DIR/score_<kind>.md.  A variant whose output is missing is reported as MISSING, never skipped silently.
Ratios are JAX distance from the Fortran run divided by the spread of the five one-ulp members (within <= 1, near <= 2, beyond > 2); repeats are not bitwise.
"""
import glob, json, os, re, statistics, subprocess, sys

# the runs use --rad native; the scorer's own default sentence names the Fortran callback (ACCEPTANCE_CRITERIA section 4 asks for the radiation actually used)
NATIVE_SENTENCE = 'radiation computed by the JAX port of RADIA + SOCRATES (socrates_spike, NativeHandoff)'
run = os.path.abspath(sys.argv[1])
name = os.path.basename(run)
runs_dir = os.environ.get('FORTRAN_RUNS', '')
out = []


def w(s=''):
    out.append(s)


def read(p):
    try:
        return open(p, errors='replace').read()
    except OSError:
        return None


def job_log(nm):
    t = read(os.path.join(run, nm + '.out'))
    if t is None:
        return 'MISSING (no job output)'
    m = re.findall(r'^exit: (\d+)', t, re.M)
    h = re.search(r'^host: (\S+)', t, re.M)
    return 'exit %s on %s' % (m[-1] if m else 'unknown (no exit line: job still running, failed or killed)', h.group(1) if h else '?')


w('# Variant run %s' % name)
w()
w(read(os.path.join(run, 'JOBS.txt')) or 'JOBS.txt missing')
w()

# ---- fidelity -------------------------------------------------------------------------------------------------------------------------------------------------------
w('## Fidelity (day 1, 54 steps, nov26): worst ratio at steps >= 3')
w()
rows = {}
rule = {}
for kind in ('release', 'fast'):
    ours = os.path.join(run, 'fid_%s_jax' % kind, 'ours_d193')
    md = os.path.join(run, 'score_%s.md' % kind)
    if not os.path.isdir(ours):
        rows[kind] = None
        continue
    r = subprocess.run([sys.executable, 'd207_score.py', '--ours', ours, '--daydir', 'nov26_day2', '--nsteps', '54', '--md', md],
                       capture_output=True, text=True, env={**os.environ, 'SCORE_RADIATION_SENTENCE': NATIVE_SENTENCE})
    open(os.path.join(run, 'score_%s.out' % kind), 'w').write(r.stdout + r.stderr)
    t = read(md)
    if r.returncode != 0 or t is None:
        rows[kind] = None
        continue
    sec = t.split('### D207 both days')[0]
    d = {}
    for m in re.finditer(r'^\| (T|U|V|Q|P|QCL|QCI) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| (\d+) \|', sec, re.M):
        d[m.group(1)] = (float(m.group(5)), int(m.group(6)))
    mb = re.search(r'\(b\) never beyond 2x of the largest member: (\w+) \(steps >= 3 only: (\w+)\)', t)
    rule[kind] = (mb.group(1), mb.group(2)) if mb else None
    rows[kind] = d
w('| field | release | at step | fast | at step |')
w('|---|---|---|---|---|')
for f in ('T', 'U', 'V', 'Q', 'P', 'QCL', 'QCI'):
    c = []
    for kind in ('release', 'fast'):
        d = rows.get(kind)
        c += ['MISSING', ''] if not d or f not in d else ['%.2f' % d[f][0], str(d[f][1])]
    w('| %s | %s | %s | %s | %s |' % ((f,) + tuple(c)))
w()
worst = {}
for kind, d in rows.items():
    if d:
        worst[kind] = max(v[0] for v in d.values())


def met(x):
    return {'True': 'MET', 'False': 'NOT MET'}.get(x, 'UNKNOWN')


for kind in ('release', 'fast'):
    if kind in worst and rule.get(kind):
        allsteps, k3 = rule[kind]
        w('* %s: approved pass rule (never beyond 2x the largest member distance, every step 0-53, every scored field): %s. Secondary line, steps >= 3 only: %s; worst ratio at steps >= 3 is %.2f. Relaxed experiment rule (ratio part only: steps >= 3 within 3x): %s; its other condition, all steps finite, is not checked here.'
          % (kind, met(allsteps), met(k3), worst[kind], 'MET' if worst[kind] <= 3 else 'NOT MET'))
    else:
        w('* %s: MISSING (run absent or scorer failed; see score_%s.out)' % (kind, kind))
w('* Job status: fid_release: %s; fid_fast: %s.' % (job_log('fid_release'), job_log('fid_fast')))
w('* The scorer was given the native-radiation sentence for these runs (`--rad native`). Reference values from the earlier packages: release T 1.18-1.20, U 1.24, V 1.23-1.24, Q 1.15, P 1.21-1.22, QCL 1.58, QCI 2.37; fast T 1.06, U 1.07, V 1.05, Q 1.04, P 1.05, QCL 1.59, QCI 2.47.')
w()

# ---- timing -------------------------------------------------------------------------------------------------------------------------------------------------------
w('## Timing (matched pairs, 108 steps, one exclusive node each; node-dependent, not merged)')
w()
w('| pair | node | Fortran wall per run (3 runs) | JAX step 0 (cold) | JAX steps 1+ median | JAX steps 1+ mean | completed |')
w('|---|---|---|---|---|---|---|')
for kind in ('release', 'fast'):
    js = os.path.join(run, 'pair_%s_jax' % kind, 'd193_run.json')
    try:
        d = json.load(open(js))
        st = [s['wall'] for s in d['steps']]
        node = (d.get('header', {}).get('host', {}) or {}).get('hostname', '?')
        jax = ('%.1f s' % st[0], '%.3f s' % statistics.median(st[1:]), '%.3f s' % statistics.mean(st[1:]), '%s/%s' % (d.get('completed'), d.get('nsteps_requested')))
    except (OSError, ValueError, KeyError, IndexError):
        node, jax = '?', ('MISSING',) * 3 + ('',)
    fw = []
    for i in (1, 2, 3):
        t = read(os.path.join(runs_dir, 'run_%s_%s%d' % (name, kind, i), 'run.time')) if runs_dir else None
        m = re.search(r'real\s+(\d+)m([\d.]+)s', t or '')
        fw.append('%.0f s' % (int(m.group(1)) * 60 + float(m.group(2))) if m else 'MISSING')
    w('| %s | %s | %s | %s |' % (kind, node, ', '.join(fw), ' | '.join(jax)))
w()
w('* Fortran wall is the whole process (`time`), including start-up; divide by 108 only as an upper bound on seconds per step. The frozen benchmark quoted 2.285 s/step for Fortran and 0.524 s/step for the fast JAX configuration (job 37941875); this table is a new measurement, not a confirmation of it.')
w('* Job status: pair_release: %s; pair_fast: %s.' % (job_log('pair_release'), job_log('pair_fast')))
w()

# ---- tests --------------------------------------------------------------------------------------------------------------------------------------------------------
w('## Regression subset (sharded pytest, CPU)')
w()
w('| configuration | files passed/failed (rc) | passed | failed | skipped | job |')
w('|---|---|---|---|---|---|')
for nm in ('tests_off', 'tests_on'):
    d = os.path.join(run, nm)
    rc = read(os.path.join(d, 'rc.txt'))
    if rc is None:
        w('| %s | MISSING | | | | %s |' % (nm, job_log(nm)))
        continue
    rcs = [int(x) for x in re.findall(r'rc=(\d+)', rc)]
    tot = {'passed': 0, 'failed': 0, 'skipped': 0}
    for lf in glob.glob(os.path.join(d, '*.log')):
        last = [l for l in (read(lf) or '').splitlines() if re.search(r'\d+ (passed|failed|skipped|error)', l)]
        for k in tot:
            m = re.search(r'(\d+) %s' % k, last[-1]) if last else None
            tot[k] += int(m.group(1)) if m else 0
    nok = sum(1 for x in rcs if x == 0)
    w('| %s | %d of %d jobs rc=0 | %d | %d | %d | %s |' % (nm, nok, len(rcs), tot['passed'], tot['failed'], tot['skipped'], job_log(nm)))
w()
w('* Reference (2026-10-16): flags off 3281 passed, 1 failed (harness check, since fixed), 40 skipped; flags on 3282 passed, 0 failed, 40 skipped. `test_jax_state` collects no tests (rc=5), so one job with rc=5 is expected in each row.')
w('* Skips are tests that declare absent data, binaries or opt-in slow runs; 17 of the 40 have not been traced to a reason.')
w()
open(os.path.join(run, 'SUMMARY.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
