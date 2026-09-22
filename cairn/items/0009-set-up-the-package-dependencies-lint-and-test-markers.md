---
id: 9
title: 'Set up the package: dependencies, lint and test markers'
type: chore
status: planned
milestone: load
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: s
area: infra
---

The package is a stub with no dependencies.

## Acceptance criteria

- [ ] Kev as a git dependency pinned to a commit (`jaredpalmer/kev@250330b`, the commit the context docs were written against), not the local clone
- [ ] `mlx` and `mlx-lm` behind a `sys_platform == 'darwin'` marker; `torch`, `transformers>=5.17` (Kev's floor for Qwen3.5) and `peft` for the CUDA path
- [ ] `ruff` configured and passing
- [ ] a `weights` pytest marker, skipped by default, for tests that download models; `uv run pytest` is green without network
