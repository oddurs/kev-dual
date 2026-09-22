---
id: 42
title: Fix the two edge cases from the code review
type: bug
status: done
milestone: write
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p2
effort: s
area: model
---

## What happens

1. `TorchDualModel`'s `_Decider` skips `DecisionModel.__init__`, including its guard, so a checkpoint with `option_isolation=True` on a hybrid (Qwen3.5) base loads and scores silently instead of failing.
2. `kev_dual.suites.records(n=0)` raises `ZeroDivisionError`, and a negative `n` returns an empty list without an error.

## What should happen

- [x] `_Decider` raises the same `ValueError` Kev does for `option_isolation` on a hybrid backbone
- [x] `records` rejects `n < 1` with a `ValueError`
- [x] tests for both

## 2026-09-22

Both found by a code review of main on 2026-09-22. Gates: ruff clean; pytest 8 passed (3 new tests in tests/test_edges.py); pytest -m weights 9 passed.
