# Status briefings (2026-10-19)

Three concise PDF briefings that summarize the whole project, each for one audience. 16:9, source-cited footers, same navy/green/orange style as the earlier decks in the archive.

| File | Audience | Slides | Content |
|---|---|---|---|
| `management_briefing_20261019.pdf` | management | 5 | goal and status in three numbers, the journey, evidence and limits, decisions needed, proposed gates |
| `science_briefing_20261019.pdf` | scientists | 6 | what is ported and what is still host or recorded, how fidelity is measured, the one-day result, radiation, what the evidence cannot say, open questions |
| `developer_briefing_20261019.pdf` | the next developer | 8 | orientation, step anatomy, running `gpu/run_variants.sh`, environment and data, benchmark arithmetic, why the fast configuration is faster, the step-time puzzle, tests and open work |

The developer deck keeps the technical content of the archived `technical_briefing_2026101[34].pdf` (benchmark arithmetic, optimization table, test status) and updates it with the results of this branch.

Owner: G. Tamkin. Written by a Claude Code session on 2026-10-19 from `../STATUS_20261016.md`, `../Reports/` and the documents of the archive tag (`archive/full-fidelity-port-20261016`); every slide footer names its sources. Status: DRAFT, not reviewed by the owner, the in-house climate scientist or the external reviewer. The plain-language fidelity wording on the science deck's second slide is drafted from `ACCEPTANCE_CRITERIA.md` and the archived `FIDELITY_EXPLAINER_NOTES.md`; it has not been checked by a domain expert. Numbers are typed into the builder from those documents and were not re-measured for the decks. Review by: when the fast step-time variation is explained, a second developer has reproduced the handoff, or the fidelity status changes.

Rebuild (reportlab and Pillow; matplotlib for the figures; no network):
```
cd rocke3d_jax/status_slides
python3 make_figures.py images
python3 build_briefing_decks_20261019.py          # three PDFs here, PNG previews in preview_*/ (not tracked)
```
The builder raises if any text block does not fit its box. Only WinAnsi characters are used (built-in Helvetica and Courier).

Files: `deck_layout.py` (layout layer, from the archive, footer date made a parameter), `build_briefing_decks_20261019.py` (all three decks), `make_figures.py` and `images/` (three figures), the three PDFs. Earlier decks (chronology deck, 13/14 Oct briefings, the 2026-09-24 artifact content) stay in the archive tag.
