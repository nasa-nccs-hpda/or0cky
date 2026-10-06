# IMVI agent instructions

- Treat documents in this directory as project-local unless they explicitly reference a shared canonical entry.
- Do not infer IMVI requirements, model behavior, interfaces, or validation criteria that are not documented.
- Prefer adding draft knowledge under this project before creating shared repository guidance.
- Record sources and identify the responsible owner for substantive technical claims.
- Never include credentials, protected datasets, or sensitive operational details.
- When content becomes useful to another project, propose promoting it into `knowledge/` or `skills/` and leave a link here.

# Repository instructions for agents

## Purpose

Treat this repository as a shared agent platform and institutional knowledge base. Preserve the distinction between project-local material and reusable components.

## Placement rules

- Put new, project-specific information in `projects/<project>/`.
- Put information reused by multiple projects in `knowledge/`.
- Put repeatable agent procedures in `skills/`.
- Put reusable role configurations in `agents/`.
- Put external service and platform instructions in `integrations/`.
- Put behavioral verification in `evals/`.

When placement is unclear, prefer the relevant project directory. Do not generalize speculative or project-specific practices into shared guidance.

## Editing rules

- Preserve owners, sources, limitations, and review dates.
- Do not invent project facts, model behavior, dataset characteristics, or validation results.
- Never add secrets, credentials, restricted data, or production endpoints without explicit authorization.
- Use lowercase, hyphen-separated directory names.
- Keep skills small, task-oriented, and self-contained.
- Validate scripts and links in proportion to their risk.
- Remove `.gitkeep` when a directory gains real content.

