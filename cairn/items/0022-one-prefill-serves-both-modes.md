---
id: 22
title: One prefill serves both modes
type: feature
status: backlog
milestone: share
depends_on:
- 20
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: model
---

## Problem

With question-side placement the state's cache is adapter-agnostic, but nothing yet reuses it across modes.

## Proposal

`DualModel.prefill(state)` returns a cache. Decide rows (adapter on from `<q>`) and text generation (adapter off) both continue from copies of it. The DeltaNet recurrent state must be copied, not mutated, as Kev's MLX branch batching already does (`type(c).merge`).

## Acceptance criteria

- [ ] generation from the shared cache equals generation from a fresh prefill, token for token
- [ ] decide from the shared cache equals decide from a fresh prefill (max |dp| ≤ 1e-3 in bf16)
- [ ] the cache is unchanged after a decide call and after a generate call
