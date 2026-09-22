---
id: 41
title: 'Enforce the workflow: CI, required checks and squash-only merges'
type: chore
status: doing
milestone: write
assignee: Oddur Sigurdsson
claimed: 2026-09-22
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: s
area: infra
---

Every change reaches `main` through a PR that names its cairn item and passes CI, and GitHub enforces it (the repo is public for this; branch protection on a private repo needs GitHub Pro).

- [ ] CI on every PR and on `main`: `ruff check`, `ruff format --check`, `pytest` (fast tests; the weight tests stay local, they need Apple Silicon and model downloads)
- [ ] CI runs `cairn check` and fails if `ROADMAP.md` is not what `cairn render` produces
- [ ] CI fails a PR whose title does not name its item as `(cairn NNNN)`
- [x] squash merges only, PR title and body as the commit, branches deleted on merge
- [ ] branch protection on `main`: PR required, the CI checks required and up to date, linear history, no force pushes, admins included
