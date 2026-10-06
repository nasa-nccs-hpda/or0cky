ROCKE-3D Update - 20261006 (status as of 12:00 EDT; detailed evidence in README_START_HERE.md and FULL_FIDELITY_DELTAS.md)

-----

**The goal, in plain terms:** ROCKE-3D is a computer climate model that simulates a planet's atmosphere, ocean and ice. It is written in Fortran and runs the way weather and climate models traditionally do. The project is rewriting it in Python/JAX so it can run on modern accelerators such as GPUs. The rewrite has to give the same answers as the original, which is checked piece by piece against the real model's own output. Speed on new hardware is the stated purpose in the project's documents.

**Intended science use of the finished port:** *not documented in the repository; to be stated by the project owner.* (The model family is built for planetary climate studies; what this project will do with a faster copy, for example parameter sweeps or ensembles, is not recorded.)

**What the finished run does:** steps a 3-D planet forward in 30-minute increments (winds, temperature, humidity, clouds and rain, ocean currents and temperature, sea ice, all interacting). The configuration being ported is the rundeck `P2SAoM40`: a Sun-like star, a dynamic ocean and a 40-layer atmosphere on a 72x46 grid, with Earth's radius, gravity and rotation.

**Where we are:**
- **Done and checked against the real model:** the ocean (all pieces, plus a chained whole-ocean step that matches the real model over 12 steps on 3 dates), sea ice, the atmosphere's air-motion code (a chained whole dynamics step that matches the real model bit for bit on 18 steps, when using the same math library the real build uses), and the cloud and convection code (including the column driver, matching at rounding level with a few percent of cloud columns sensitive to last-digit differences when a different math library is used).
- **Found and fixed while joining the pieces:** a wrong planet-rotation constant in the ocean code that had been hidden by a loose test tolerance. The other constants I checked match the original.
- **Not done:**
  - Radiation (how sunlight and heat move through the air) is not ported, by a standing rule that the third-party library is never ported. A finished version needs that library, or recorded outputs from it.
  - The atmosphere pieces are not yet joined into one validated atmosphere step (the plan and most of the code are written; the comparison against the real model has not been run). That is the next milestone.
  - The Python is too slow to run the model: the cloud code takes about two minutes per 30-minute step, the ocean about 4 seconds and the dynamics about 3 seconds. Speed work (batched and GPU forms) is mostly still to do.
  - Only single steps (up to 12 steps) are validated, not days or months of simulated time.

**Is the path realistic soon? Partly.**
- **Within days:** a validated one-step demonstration (atmosphere with radiation recorded, plus the ocean). The first full-chain atmosphere test has not reported, so this is not yet certain.
- **Not soon:** a complete replacement that runs a real simulation faster than the original. Estimate: about 30 to 65 more focused hours (central about 45), down from 150-280 on 2026-10-05 because the dynamics, cloud and ocean chains are now done. It is an estimate, not a measurement; earlier estimates had to be revised upward once, and the speed and multi-step validation items have not been tried.

**For management:** real, steady progress with a credible route to a validated one-step model. "Finished in a day" is not realistic. The radiation dependency and the speed work are the two biggest open questions.
