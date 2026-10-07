# Documentation review: the two sources given by the project lead

Owner: project owner of `rocke3d_jax` (G. Tamkin). Drafted by a Claude Code session on 2026-10-07. Project-local per `projects/imvi/AGENTS.md`.
Review by: the next session that changes the F3 validation target or `Reports/GOAL.md`.

## 1. What was reviewed, and how

The project lead gave exactly two links as the only starting information for this project:

1. The ROCKE-3D 2.0 paper, anchored at its section 13: <https://gmd.copernicus.org/articles/18/5825/2025/#section13>
2. The NCCS publication-supplement directory for the P2SAoM40_003 run: <https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/publication-supplements/Tsigaridis2025GMD-planet_2.0/P2SAoM40_003/>

Both pages were fetched on 2026-10-07 with a web-fetch tool that returns a summary produced by a small model, not the raw page. Everything below marked **[summary]** comes from such a summary and has not been checked against the paper itself.

## 2. Limitations of this review (read first)

- **Section 13 of the paper was NOT read.** The fetched page text is cut off in section 4.4 ("Creating a new planet"), so the Conclusions, the Code and data availability section and any section 12 to 14 were not seen. The anchor `#section13` in the lead's link may point at one of those; this needs a human read.
- The paper's PDF (about 12 MB) is too large for the fetch tool, and no PDF reader (`pdftotext`, `pypdf`, `poppler`) is installed on this node, so it could not be read locally either. A copy is in the session scratchpad only (not in the repository).
- The page for the supplement directory contains no description. What "_003" means (a run index, an ensemble member, a version) is **not stated** anywhere that was seen.
- The repository's own rundeck is `P2SAoM40`; whether it is identical to the rundeck behind the supplement run was **not checked**.

## 3. What the sources establish

### 3.1 The paper
- Title and venue: Tsigaridis et al., "ROCKE-3D 2.0: an updated general circulation model for simulating the climates of rocky planets", Geoscientific Model Development 18, 5825 (2025). The first author is Kostas Tsigaridis (Columbia University / NASA GISS).
- It describes version 2.0 of ROCKE-3D, a three-dimensional GCM for rocky planets, a descendant of the NASA GISS ModelE2.1. **[summary]**
- Intended applications named in the fetched text: paleoclimate studies, exoplanet habitability assessment and parameter exploration; the emphasis is on flexibility across planetary conditions rather than one configuration. **[summary]**
- Configuration naming: `P2SAoM40` is read as SOCRATES radiation, anoxic atmosphere, dynamic ocean, medium resolution (40 layers). **[summary]** This agrees with the repository (`projects/imvi/rocke3d_jax`, rundeck `P2SAoM40`, SOCRATES as a third-party library that is never ported).
- Code and data: the search results report that the model code and simulation output are on a public Zenodo archive, DOI 10.5281/zenodo.14721184. **[search result, not read in the paper]**

### 3.2 The Zenodo archive (<https://zenodo.org/records/14721184>)
- Title "ROCKE-3D v2"; contents: model code, rundecks, restart files, topographies and simulation outputs; license CC BY 4.0; 5.0 GB for this version.
- Stated content: "equilibrated climatologies spanning either 20 years (prescribed ocean simulations) or 100 years (Q-flux or dynamic ocean runs), plus time series from initial conditions through equilibration", with nearly complete atmospheric output from each simulation. **[summary]**
- Its description does not describe P2SAoM40 individually.

### 3.3 The NCCS supplement directory (P2SAoM40_003)
- It lists 100 pairs of files, `ANN4000` through `ANN4099`: `*.aijP2SAoM40.nc` (6.9 MiB each) and `*.aijlP2SAoM40.nc` (7.6 MiB each), all dated 2025-02-26.
- Reading: 100 consecutive **annual-mean** output files of a P2SAoM40 run, matching the "100 years for dynamic-ocean runs" statement of the Zenodo record. This is an inference from the file names, sizes and the Zenodo text; the directory itself says nothing about its meaning.

## 4. What this means for the project

1. **The published reference for the configuration is a 100-year equilibrated annual-mean climatology**, not a month. Cost of reproducing that period with the port: 100 years is about 1.75 million 30-minute steps (48 per day, 365 days). At the per-step cost measured so far on CPU (tens of seconds for the coupled surface step; see `README_START_HERE.md`) that is out of reach on this node, which is consistent with the project's stated purpose (speed on accelerators).
2. **The current F3 target (one model month from the January restart, scored against the 8 real JAN1950 members) is a short-run proxy**, not the published comparison. The month-scale noise floor (ensemble D165) is valid for monthly means only.
3. **The 100 annual files would give a much better noise floor for annual means** (interannual spread over 100 years) once the port can run a model year. They cannot score a single month: the directory has annual files only.
4. **Consistency with the repository's existing choices:** the AIJ/AIJL accumulators (`f3_diagnostics.py`) produce the same diagnostic families (`aij`, `aijl`) as these files, so annual means built from them could be compared directly (file format and field names would need a mapping check; not done).

## 5. Questions for the project lead

1. Is the intended final comparison the published 100-year annual-mean climatology (`ANN4000`-`ANN4099`), or a shorter run? What does "_003" denote?
2. What do the paper's section 13 and its Conclusions say about the intended use and about data availability? (Not readable here.)
3. Should carbon-cycle outputs (Ent) be part of the acceptance? The water/energy loop does not need them (see D169/D171); they would add roughly 20 to 30 hours.
4. Is the rundeck in the repository identical to the one behind this supplement run?

## 6. The candidate wordings considered for `GOAL.md` ("intended science use")

Drafted only from what the two sources say, and offered to the owner on 2026-10-07:

1. **Ensembles** of ROCKE-3D simulations (perturbed-initial-condition or parameter-variation runs). Consistent with the 100-year runs and their interannual spread; not stated in the sources.
2. **Parameter exploration for rocky-planet studies**, reproducing the published P2SAoM40 climatology at much lower cost. Best supported: "parameter exploration", exoplanet habitability and paleoclimate are the applications named in the fetched text, and the published reference is a long run that only an accelerator makes cheap.
3. **Gradient-based or machine-learning-assisted work** (calibration, emulation). Nothing in the two sources supports it; not recommended.

The owner chose wording 2, now in `Reports/GOAL.md`.

## 7. Sources

- Tsigaridis, K. et al. (2025), ROCKE-3D 2.0, GMD 18, 5825: <https://gmd.copernicus.org/articles/18/5825/2025/> (HTML page, truncated before section 13); PDF <https://gmd.copernicus.org/articles/18/5825/2025/gmd-18-5825-2025.pdf> (downloaded, unreadable here).
- NCCS supplement directory: <https://portal.nccs.nasa.gov/GISS_modelE/ROCKE-3D/publication-supplements/Tsigaridis2025GMD-planet_2.0/P2SAoM40_003/>
- Zenodo record "ROCKE-3D v2": <https://zenodo.org/records/14721184> (DOI 10.5281/zenodo.14721184)
