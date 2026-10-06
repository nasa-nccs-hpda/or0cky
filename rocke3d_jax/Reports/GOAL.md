ROCKE-3D Update - 20261006 (status as of 17:30 EDT; detailed evidence in README_START_HERE.md and FULL_FIDELITY_DELTAS.md)

-----

**The goal, in plain terms:** ROCKE-3D is a computer climate model that simulates a planet's atmosphere, ocean and ice. It is written in Fortran and runs the way weather and climate models traditionally do. The project is rewriting it in Python/JAX so it can run on modern accelerators such as GPUs. The rewrite has to give the same answers as the original, which is checked piece by piece against the real model's own output. Speed on new hardware is the stated purpose in the project's documents.

**Intended science use of the finished port:** *not documented in the repository; to be stated by the project owner.* (The model family is built for planetary climate studies; what this project will do with a faster copy, for example parameter sweeps or ensembles, is not recorded.)

**What the finished run does:** steps a 3-D planet forward in 30-minute increments (winds, temperature, humidity, clouds and rain, ocean currents and temperature, sea ice, all interacting). The configuration being ported is the rundeck `P2SAoM40`: a Sun-like star, a dynamic ocean and a 40-layer atmosphere on a 72x46 grid, with Earth's radius, gravity and rotation.

**Where we are:**
- **Done and checked against the real model:** the ocean (all pieces, plus a chained whole-ocean step that matches the real model over 12 steps on 3 dates), sea ice, the atmosphere's air-motion code (a chained whole dynamics step that matches the real model bit for bit on 18 steps, when using the same math library the real build uses), and the cloud and convection code (including the column driver, matching at rounding level with a few percent of cloud columns sensitive to last-digit differences when a different math library is used).
- **Found and fixed while joining the pieces:** a wrong planet-rotation constant in the ocean code that had been hidden by a loose test tolerance. The other constants I checked match the original.
- **Joined into one atmosphere step (new today):** dynamics, surface, clouds, and the rest of one 30-minute step now run chained from the real starting state, with radiation values recorded rather than computed. Verdict against the pre-set pass criteria for the first step (using the same math library as the real build): **met on all three test dates** when the land-surface part is taken from recorded values; with **our own ported land-surface code, met on one date and only partly met on the other two** (every field within about one part in a hundred million of the real model, but not within the stricter one-in-a-trillion bound). Without the real build's math library the cloud code flips near thresholds in 3-5% of columns, so a multi-step run drifts chaotically from the real model (about 0.1-0.2 K after 6 steps).
- **Speed, first fix landed:** the cloud code now runs about 22 times faster (about 5 seconds per step instead of about 111) by processing all columns at once.
- **Two real porting errors in the land-surface code were found and fixed today** (precipitation conditioning, and a dropped irrigation term); with them the JAX land model now matches the real model on every cell tested to about 2e-13 of field scale. These two fixes moved the first-step verdict with our own land code from not met / partly met to met / partly met.
- **Ocean:** two more recorded inputs are now computed by our code (momentum diffusion and the polar-filter coefficients); the ocean-atmosphere exchange fluxes and the straits start state are still recorded.
- **GPU caveat:** the development node has no GPU (12 CPU cores only), so the speed-up on accelerators, the project's stated purpose, cannot be measured here. Everything is validated for correctness on CPU.
- **Not done:**
  - Radiation (how sunlight and heat move through the air) is not ported, by a standing rule that the third-party library is never ported. A finished version needs that library, or recorded outputs from it.
  - Our own land-surface code does not yet meet the pass criteria on every date (see above).
  - The batched cloud code is not yet wired into the joined step, and nothing is yet in a GPU (JAX) form for the atmosphere; per step the joined run is about 6 seconds plus clouds (about 5 s batched).
  - Only single steps (up to 12 steps) are validated, not days or months of simulated time, and a multi-step run will drift from the real model chaotically once a threshold flips.

**Is the path realistic soon? Partly, and the scale depends on the target.** (Estimates; the radiation plan in `fullfidelity/scoping/RADIATION_AND_F2_PLAN.md` has the basis.)
- **Reached:** a validated one-step atmosphere (radiation recorded) and ocean step; the cloud, dynamics and land pieces are fast enough to run a step in about 10 s on CPU.
- **About 10 more hours:** one model day replayed with recorded radiation.
- **About 40 more hours in total:** a free-running atmosphere day with the real radiation code called as a black box through a small server built from the original model (the surface, ocean and vegetation still replayed).
- **About 100-170 more hours:** a one-month comparison with the real run's monthly output, the plan's top validation level, including closing the surface, ocean, ice and vegetation loop.
- **Not measurable on this node:** the GPU speed-up the project is for.

**For management:** real, steady progress and a credible route to each level above; the earlier "30 to 65 hours" figure was for the near-term items and understated the full one-month target. The radiation dependency, the vegetation (Ent) inputs and the missing GPU host are the biggest open questions.
