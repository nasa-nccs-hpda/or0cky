"""Matplotlib figures for the status deck (dark theme matching the deck).
Run: python make_figures.py OUTDIR   (conda env python; matplotlib only)
Data: chronology_data.py (git log + ledger ids) and the numbers quoted in
FULL_FIDELITY_DELTAS.md D191/D195 (cited in each caption on the slide).
"""
import sys, os, datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import chronology_data as C

for f in ("Regular", "Bold"):
    p = "/usr/share/fonts/liberation-sans/LiberationSans-%s.ttf" % f
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Liberation Sans", "text.color": "#E2E8F0",
                     "axes.labelcolor": "#E2E8F0", "xtick.color": "#94A3B8", "ytick.color": "#94A3B8",
                     "axes.edgecolor": "#334155", "figure.facecolor": "#0F172A", "axes.facecolor": "#0F172A"})
GREEN, ORANGE, AMBER, BLUE, MUTED, INK = "#4ADE80", "#F97316", "#FBBF24", "#38BDF8", "#94A3B8", "#F8FAFC"
PH = {k: (lab, col) for k, lab, s, e, col in C.PHASES}


def D(d, t=""):
    h, m = (t.replace("~", "") or "12:00").split(":")
    return dt.datetime.strptime(d, "%Y-%m-%d") + dt.timedelta(hours=int(h), minutes=int(m))


def fig_timeline_full(out):
    fig = plt.figure(figsize=(13.2, 5.6), dpi=200)
    ax = fig.add_axes([0.25, 0.40, 0.73, 0.52])
    ax2 = fig.add_axes([0.25, 0.08, 0.73, 0.24], sharex=ax)
    x0, x1 = D("2026-08-26"), D("2026-10-10")
    for i, (k, lab, s, e, col) in enumerate(C.PHASES):
        a, b = D(s, "00:00"), D(e, "23:59")
        ax.barh(i, (b - a).total_seconds() / 86400, left=matplotlib.dates.date2num(a), height=0.55, color=col, alpha=0.9)
        ax.text(-0.01, i, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.6, color=INK)
    ax.set_ylim(len(C.PHASES) - 0.4, -0.7)
    ax.set_yticks([])
    # milestones on the phase lanes
    keyc = {"af7bdfb", "abd805b", "4ec6056", "f648263", "ec5a323", "d7c871a", "3ccf563", "eb8cdb8", "02c80a8",
            "8909e4e", "cc429f8", "0ffa8f5", "f045127", "ec36779", "d320221", "bf212f8"}
    lanes = {k: i for i, (k, *_r) in enumerate(C.PHASES)}
    for r in C.ROWS:
        if r[2] in keyc:
            t = D(r[0], r[1])
            ax.plot([t], [lanes[r[3]]], marker="D", ms=4.5, color=INK, mec="#0F172A", mew=0.6, zorder=5)
    for d_, t_, lab in C.DECISIONS:
        ax.axvline(D(d_, t_), color="#F472B6", lw=0.8, ls=(0, (2, 2)), alpha=0.9)
    ax.text(D("2026-10-07", "09:00"), -0.62, "owner decisions 10-07 / 10-08", color="#F472B6", fontsize=7.2, ha="right", va="center")
    ax.xaxis_date()
    ax.set_xlim(x0, x1)
    ax.tick_params(labelbottom=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#1E293B", lw=0.6)
    ax.set_title("ROCKE-3D -> JAX: chronology, 2026-08-28 to 2026-10-08 (273 commits; diamonds = milestones listed in PROJECT_CHRONOLOGY.md)",
                 fontsize=10, color=INK, loc="left", pad=8, x=-0.33)
    # commits per day
    ds = sorted(C.COMMITS_PER_DAY)
    ax2.bar([D(d) for d in ds], [C.COMMITS_PER_DAY[d] for d in ds], width=0.8, color=BLUE)
    for d in ("2026-09-28", "2026-10-06", "2026-10-07"):
        ax2.text(D(d), C.COMMITS_PER_DAY[d] + 1.5, str(C.COMMITS_PER_DAY[d]), ha="center", fontsize=7.5, color=INK)
    ax2.set_ylabel("commits / day", fontsize=8)
    ax2.set_ylim(0, 70)
    ax2.grid(axis="y", color="#1E293B", lw=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    import matplotlib.dates as md
    ax2.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
    ax2.xaxis.set_major_formatter(md.DateFormatter("%m-%d"))
    ax2.tick_params(labelsize=8)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_timeline_zoom(out):
    import textwrap as tw
    rows = [r for r in C.ROWS if r[0] >= "2026-10-06" and r[2] not in ("f21e007", "6e4859c", "eb8cdb8", "7bd7a14", "c7c42c8", "5c5b7a4", "bd9b16c", "b71eb22", "978672e", "45e0d20", "9602dc3")]
    nstrip = 3
    per = -(-len(rows) // nstrip)
    fig, axs = plt.subplots(nstrip, 1, figsize=(12.6, 6.0), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.04)
    fig.suptitle("Zoom: 2026-10-06 to 2026-10-08, events in commit order (not to time scale); date/time = commit time; colors = phase", fontsize=10, color=INK, x=0.01, ha="left", y=0.99)
    for si, ax in enumerate(axs):
        chunk = rows[si * per:(si + 1) * per]
        ax.set_xlim(-0.75, per - 0.2); ax.set_ylim(-1.75, 1.75); ax.axis("off")
        ax.axhline(0, color=MUTED, lw=1)
        for n, r in enumerate(chunk):
            col = PH[r[3]][1]
            up = n % 2 == 0
            ax.plot([n, n], [0, 0.18 if up else -0.18], color=col, lw=1.2)
            ax.plot([n], [0], "o", color=col, ms=5)
            head = r[2] + ("  " + r[4] if r[4] else "")
            body = r[5] if len(r[5]) < 135 else r[5][:132] + "..."
            txt = head + "\n" + r[0][5:] + " " + r[1] + "\n" + "\n".join(tw.wrap(body, 44))
            ax.text(n, 0.26 if up else -0.26, txt, fontsize=7.6, color=INK, ha="center",
                    va="bottom" if up else "top", linespacing=1.18)
    handles = [matplotlib.patches.Patch(color=c, label=k + " " + l.split(" (")[0]) for k, (l, c) in PH.items() if k in "DEFG"]
    fig.legend(handles=handles, loc="lower left", fontsize=6.8, frameon=False, ncol=4, bbox_to_anchor=(0.01, 0.0))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def textwrap(head, body, width=34):
    import textwrap as tw
    body = body if len(body) < 120 else body[:117] + "..."
    return head + "\n" + "\n".join(tw.wrap(body, width))


def fig_perf(out):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.3), dpi=200)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.14, wspace=0.28)
    data = [("Steady step, seconds (nov26, CPU)", ["D187 hybrid\n(3 cores)", "D191 assembled\n(5 cores)", "NumPy libimf chain\n(D191, 22.6-23.3 s)"], [44.3, 18.8, 23.0], [ORANGE, GREEN, MUTED], ["44.3", "18.8", "22.6-23.3"]),
            ("jit executions per step", ["D187", "D191"], [262, 99], [ORANGE, GREEN], ["262", "99"]),
            ("Eager primitive dispatches per step", ["D187 (~)", "D191"], [15700, 490], [ORANGE, GREEN], ["~15,700", "490"])]
    for ax, (ttl, labs, vals, cols, txt) in zip(axs, data):
        b = ax.bar(labs, vals, color=cols, width=0.6)
        for x, v, t in zip(range(len(vals)), vals, txt):
            ax.text(x, v * 1.02, t, ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
        ax.set_title(ttl, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.tick_params(labelsize=7.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#1E293B", lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_day(out):
    f = ["T", "U", "V", "Q", "P", "QCL", "QCI"]
    within = [54, 54, 54, 42, 52, 52, 37]
    near = [0, 0, 0, 12, 2, 1, 17]
    beyond = [0, 0, 0, 0, 0, 1, 0]
    r_ge3 = [0.94, 0.89, 0.97, 1.03, 1.04, 0.95, 1.69]
    r_all = [0.94, 0.89, 0.97, 1.03, 1.04, 2.83, 1.69]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 3.7), dpi=200)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.14, wspace=0.18)
    y = list(range(len(f)))
    a1.barh(y, within, color=GREEN, label="within")
    a1.barh(y, near, left=within, color=AMBER, label="near")
    a1.barh(y, beyond, left=[w + n for w, n in zip(within, near)], color=ORANGE, label="beyond")
    for i in y:
        a1.text(55, i, "%d / %d / %d" % (within[i], near[i], beyond[i]), va="center", fontsize=8.5, color=INK)
    a1.set_yticks(y); a1.set_yticklabels(f, fontsize=9); a1.invert_yaxis()
    a1.set_xlim(0, 72); a1.set_xlabel("steps of 54 (within / near / beyond)", fontsize=8.5)
    a1.set_title("Steps per class, nov26 day, assembled step (D209, current defaults)", fontsize=10, color=INK, loc="left")
    a1.legend(fontsize=8, frameon=False, loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.32))
    a2.axvline(1, color=GREEN, lw=1, ls="--"); a2.axvline(2, color=ORANGE, lw=1.4)
    a2.text(2.03, -0.55, "2x: pass rule (ACCEPTANCE s9)", color=ORANGE, fontsize=8, va="center")
    a2.text(1.02, -0.55, "1x", color=GREEN, fontsize=8, va="center")
    for i in y:
        a2.plot([r_ge3[i], r_all[i]], [i, i], color=MUTED, lw=1.5)
        a2.plot(r_ge3[i], i, "o", color=GREEN, ms=6)
        a2.plot(r_all[i], i, "D", color=(ORANGE if r_all[i] > 2 else GREEN), ms=6)
    a2.text(2.92, 5.0, "2.83 (step 1)", fontsize=8.5, color=ORANGE, va="center")
    a2.text(1.74, 6.42, "1.69", fontsize=8, color=INK, va="center")
    a2.set_yticks(y); a2.set_yticklabels(f, fontsize=9); a2.invert_yaxis()
    a2.set_xlim(0.6, 3.4); a2.set_ylim(6.6, -0.9)
    a2.set_xlabel("worst ratio ours / largest member distance (circle: steps >= 3, diamond: all steps)", fontsize=8.2)
    a2.set_title("Worst ratio per field", fontsize=10, color=INK, loc="left")
    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_tests(out):
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=200)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.14)
    xs = [D(d, t) for d, t, *_ in C.TESTS]
    ys = [r[3] for r in C.TESTS]
    ax.plot(xs, ys, "-o", color=GREEN, ms=4, lw=1.4)
    for i, (x, y, r) in enumerate(zip(xs, ys, C.TESTS)):
        if i in (0, 1, 2, len(ys) - 1):
            ax.text(x, y + 90, "%s" % format(y, ","), fontsize=7.5, color=INK, ha="center")
    ax.set_title("Full-regression 'passed' counts quoted in the repository", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, 3500)
    import matplotlib.dates as md
    ax.xaxis.set_major_formatter(md.DateFormatter("%m-%d"))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#1E293B", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    o = sys.argv[1] if len(sys.argv) > 1 else "images"
    os.makedirs(o, exist_ok=True)
    fig_timeline_full(os.path.join(o, "chronology_full.png"))
    fig_timeline_zoom(os.path.join(o, "chronology_zoom.png"))
    fig_perf(os.path.join(o, "perf_steps.png"))
    fig_day(os.path.join(o, "day_score.png"))
    fig_tests(os.path.join(o, "tests_growth.png"))
    print("figures written to", o)
