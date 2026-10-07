"""Fold a GPU job directory (written by run_gpu_job.sbatch) into one results file.

Usage: python gpu/collect_gpu_results.py gpu_runs/<jobid> [more dirs ...]
Writes <dir>/results.json (metadata, exit code, wall time, every line of stdout containing 'speedup' or 'ms' or 's/step', and the count of warnings in stderr)
and prints a short markdown table. It never edits the repository's reports: you decide what to paste where.
DRAFT: the output format of benchmark_all.py is not known to this author (the script is not in this checkout), so the parsing keeps raw lines and extracts speedups by pattern only.
"""
import json
import os
import re
import sys


def read(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def parse_meta(txt):
    meta = {}
    for ln in txt.splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", ln)
        if m:
            meta[m.group(1)] = m.group(2)
    return meta


def collect(d):
    meta = parse_meta(read(os.path.join(d, "metadata.txt")))
    out = read(os.path.join(d, "stdout.txt"))
    err = read(os.path.join(d, "stderr.txt"))
    tim = read(os.path.join(d, "time.txt"))
    wall = re.search(r"m:ss\):\s*(\S+)", tim)
    lines = [ln.strip() for ln in out.splitlines() if re.search(r"speedup|x\s*faster|\bms\b|s/step|gpu|cpu|\d(\.\d+)?\s*x\b", ln, re.I)]
    speed = []
    for ln in lines:
        for m in re.finditer(r"([A-Za-z0-9_\-\. /\(\)]+?)\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*x\b", ln):
            speed.append({"label": m.group(1).strip(), "value": float(m.group(2)), "line": ln})
    res = {
        "dir": d,
        "metadata": meta,
        "wall_clock": wall.group(1) if wall else None,
        "n_stdout_lines": len(out.splitlines()),
        "selected_lines": lines,
        "speedups_parsed": speed,
        "n_stderr_warning_lines": len([ln for ln in err.splitlines() if "Warning" in ln]),
        "stderr_tail": err.splitlines()[-5:],
    }
    json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for d in sys.argv[1:]:
        r = collect(d)
        m = r["metadata"]
        print(f"### {d}\n- job {m.get('job_id')} on {m.get('node')}, exit {m.get('exit_code')}, wall {r['wall_clock']}, git {m.get('git_head','?')[:10]} (dirty files: {m.get('git_dirty_files')})")
        print(f"- GPU: {m.get('nvidia_smi','')}".rstrip())
        print(f"- command: {m.get('command')}")
        print("| label | speedup |\n|---|---|")
        for s in r["speedups_parsed"]:
            print(f"| {s['label']} | {s['value']}x |")
        print(f"- stderr warning lines: {r['n_stderr_warning_lines']}")
