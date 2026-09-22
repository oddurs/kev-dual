---
id: 41
title: 'Enforce the workflow: CI, required checks and squash-only merges'
type: chore
status: done
milestone: write
assignee: Oddur Sigurdsson
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p0
effort: s
area: infra
---

Every change reaches `main` through a PR that names its cairn item and passes CI, and GitHub enforces it (the repo is public for this; branch protection on a private repo needs GitHub Pro).

- [x] CI on every PR and on `main`: `ruff check`, `ruff format --check`, `pytest` (fast tests; the weight tests stay local, they need Apple Silicon and model downloads)
- [x] CI runs `cairn check` and fails if `ROADMAP.md` is not what `cairn render` produces
- [x] CI fails a PR whose title does not name its item as `(cairn NNNN)`
- [x] squash merges only, PR title and body as the commit, branches deleted on merge
- [x] branch protection on `main`: PR required, the CI checks required and up to date, linear history, no force pushes, admins included

## 2026-09-22

Verified on PR #1: all four checks green. Retitling the PR without `(cairn NNNN)` fails `pr names its item`, and restoring the title passes it. The GitHub API confirms protection on `main`: required checks lint, test, backlog and 'pr names its item' (strict, up to date), admins included, linear history, no force pushes or deletions, PR required with 0 approvals (a solo repo can't approve its own PRs). The repo was made public 2026-09-22, at the user's choice, to get protection without GitHub Pro.
