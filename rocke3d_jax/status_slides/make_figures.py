"""Figures for the 2026-10-19 briefing decks. Every number is typed in from STATUS_20261016.md section 5 / the runs it lists; nothing is read from a data file.
usage: python3 make_figures.py images"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "images"
os.makedirs(OUT, exist_ok=True)
NAVY, CARD, GREEN, ORANGE, AMBER, BLUE = "#0F172A", "#1E293B", "#4ADE80", "#F97316", "#FBBF24", "#38BDF8"
INK, MUTED, RULE = "#F8FAFC", "#94A3B8", "#334155"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": RULE, "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)


# 1. seconds per step: Fortran, JAX release, JAX fast (ranges over the runs listed in the status document)
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
rows = [("Fortran, 1 CPU core", 2.24, 2.35, BLUE), ("JAX release, 1 V100", 1.73, 2.21, MUTED), ("JAX fast, 1 V100", 0.51, 0.84, GREEN)]
for i, (name, lo, hi, col) in enumerate(rows[::-1]):
    ax.barh(i, hi - lo, left=lo, height=0.5, color=col)
    ax.text(hi + 0.04, i, "%.2f-%.2f s" % (lo, hi), va="center", fontsize=9, color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
ax.set_xlim(0, 3.1); ax.set_xlabel("seconds per model step (30 min of model time), steady state", fontsize=8.5)
clean(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_step_time.png")); plt.close(fig)

# 2. fidelity: worst ratio at steps >= 3, run e (2026-10-18)
fields = ["T", "U", "V", "Q", "P", "QCL", "QCI"]
rel = [1.18, 1.24, 1.24, 1.15, 1.22, 1.58, 2.37]
fast = [1.08, 1.10, 1.07, 1.03, 1.11, 1.60, 2.47]
fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
x = range(len(fields)); w = 0.38
ax.bar([i - w / 2 for i in x], rel, w, color=MUTED, label="release")
ax.bar([i + w / 2 for i in x], fast, w, color=GREEN, label="fast")
ax.axhline(2, color=ORANGE, lw=1.4); ax.text(-0.45, 2.04, "strict limit 2x", color=ORANGE, fontsize=8)
ax.axhline(3, color=AMBER, lw=1.0, ls="--"); ax.text(-0.45, 3.04, "relaxed experiment limit 3x", color=AMBER, fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(fields, fontsize=9); ax.set_ylim(0, 3.4)
ax.set_ylabel("distance from Fortran / member spread", fontsize=8)
for i, v in enumerate(rel):
    ax.text(i - w / 2, v + 0.04, "%.2f" % v, ha="center", fontsize=7, color=INK)
for i, v in enumerate(fast):
    ax.text(i + w / 2, v + 0.04, "%.2f" % v, ha="center", fontsize=7, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.8), labelcolor=INK)
ax.grid(axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fidelity.png")); plt.close(fig)

# 3. fast configuration, mean s/step over JAX steps 2-107, nine runs, coloured by the number of Fortran runs before JAX on that node
runs = [("tag, no Fortran (gpu011)", 0.508, 0), ("branch, profile wrapper (gpu012)", 0.515, 0), ("branch (gpu011)", 0.524, 0), ("branch (gpu010)", 0.530, 3),
        ("branch (gpu010)", 0.684, 1), ("run e pair (gpu012)", 0.800, 3), ("run c pair (gpu012)", 0.835, 3), ("run d pair (gpu012)", 0.836, 3), ("branch (gpu011)", 0.841, 0)]
col = {0: BLUE, 1: AMBER, 3: ORANGE}
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
for i, (name, v, nf) in enumerate(runs):
    ax.barh(i, v, color=col[nf], height=0.62)
    ax.text(v + 0.012, i, "%.3f  %s" % (v, name), va="center", fontsize=7.4, color=INK)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 1.5)
ax.set_xlabel("mean seconds per step, JAX steps 2-107, fast configuration, exclusive V100 node", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="0 Fortran runs first"), Patch(color=AMBER, label="1 Fortran run first"), Patch(color=ORANGE, label="3 Fortran runs first")],
          frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
clean(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fast_runs.png")); plt.close(fig)
print("wrote 3 figures to", OUT)
