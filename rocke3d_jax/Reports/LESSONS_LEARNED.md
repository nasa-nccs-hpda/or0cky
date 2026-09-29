# Lessons learned — plain-language summary

This page is written for anyone following this project who isn't a software engineer — a
program manager, a scientist, a reviewer. It skips code details and focuses on what we learned
about doing this kind of work well. For the technical version, see `Project_Summary_and_Conclusions.md`.

**What this project is doing, in one sentence:** rewriting a large, decades-old climate-model
computer program (written in a language called Fortran) into a modern one (Python, using a
library called JAX that runs efficiently on the same kind of chips used for AI), while checking,
piece by piece, that every rewritten part produces *exactly* the same numbers as the original —
not "close enough," but checked against the real program actually running.

## Lessons about trust and verification

- **"It ran without crashing" is not the same as "it worked."** Several times, a test appeared
  to succeed — no error message, a normal-looking exit — but had actually done nothing at all,
  because it silently started from the wrong point. The only way we caught this was by checking
  the actual output numbers every time, never by trusting a clean-looking run on its own.
- **A progress counter can lie.** The underlying program keeps an internal counter of how many
  times each piece of code has run, saved between runs. We found that counter can carry over a
  stale, old number, making it look like something ran recently when it didn't. Lesson: don't
  trust an internal counter as proof; check the actual output.
- **Read before you estimate.** Early on, a rough guess at "how much code is left to convert"
  turned out to be significantly wrong once we actually read that code carefully — a large chunk
  of it turned out to be old, retired code that isn't even used anymore, so it didn't need
  converting at all. We now insist on reading code fully before giving a time estimate for it,
  even when that's slower up front.
- **Check for unused, leftover code before spending time on it.** The original program had grown
  over many years, and an entire major section — the equivalent of an old build of a car engine —
  had been quietly replaced by a newer version elsewhere in the codebase, with the old version left
  in place but disconnected. We almost spent significant effort faithfully converting code that
  would never actually run in the real program. Catching this saved real time and avoided
  delivering something that couldn't be tested against reality (because it never executes).

## Lessons about doing careful, repeatable work

- **Keep a "known good" copy of your starting point, and never let it get overwritten.**
  Several times, a test run accidentally altered the very starting files the next test needed,
  because the two were sharing the same workspace. Once we started keeping permanent, protected
  backup copies of the correct starting point for every test, this stopped being a problem —
  restoring took seconds instead of, in one case, over ten minutes of extra computer time to
  regenerate a lost starting point from scratch.
- **Watch out for near-identical file names.** One especially costly mistake: the actual program
  that runs and a freshly rebuilt update of that program had almost the same file name, and were
  not automatically kept in sync. For a while, we were testing an old, unchanged version and not
  realizing it — every "new" test was silently using stale software. This is now a standing
  checklist item before every test.
- **Small typos in translated formulas matter and are easy to miss by eye.** When converting
  mathematical formulas from the old code to the new code, we found several small transcription
  slips (the equivalent of swapping two very similarly-named variables). None were visible by
  reading the code — they only showed up because we checked the new code's output against the old
  code's real output, number by number, every single time.
- **Big rewritten sections deserve extra caution, not less.** A few pieces of code were large and
  complex enough that converting them properly (rather than a quick approximation) would take
  several hours by itself. We chose to set those aside temporarily and do the smaller,
  well-understood pieces first, rather than rushing the big ones — a decision that kept the overall
  error rate at zero across everything actually finished.

## Lessons about how the work was directed

- **Some things are permanently off-limits, and that has to stay true throughout.** One part of
  the original program (a separate, specialized piece handling how sunlight and heat move through
  the atmosphere) belongs to a different team and must never be modified or rewritten — only
  called as-is, like using a tool without taking it apart. This rule was set once, early on, and
  had to be actively remembered and respected through every subsequent piece of work.
- **A "good enough" shortcut was considered and explicitly rejected.** Early in the project, a
  simpler, faster version of the ocean-modeling portion was on the table (approximating rather
  than fully replicating the ocean's behavior). That path was deliberately turned down twice, in
  favor of a slower but fully faithful conversion, because the accuracy of the full ocean model
  mattered more here than the speed of finishing.
- **Working steadily without stopping to check in produced faster, more consistent progress** —
  but only because every single step was still independently verified against the real program's
  actual output. Working continuously did not mean working carelessly; if anything, the checking
  process became more disciplined the longer the work continued, since every new piece had to
  fit cleanly with everything already verified.
- **Writing down what went wrong, in detail, as it happens, pays off quickly.** Each mistake
  above — the stale progress counter, the near-identical file names, the overwritten starting
  files — was recorded in detail the first time it happened, with plain instructions for how to
  avoid it again. Later instances of the *same* category of mistake (there were a few) were caught
  and fixed in minutes instead of requiring the original, longer investigation.

## The bottom line

None of these lessons are exotic — they amount to "check the real output, don't trust
appearances, keep good backups, watch for easy-to-confuse names, write down what you learn." What
made them matter here is scale and stakes: a program this large and this old, being converted for
scientific use, has many chances for a small, invisible mistake to go unnoticed. The discipline of
checking every single converted piece against the real, unmodified original — every time, without
exception — is what caught every mistake in this list before it could become a hidden error in the
final result.
