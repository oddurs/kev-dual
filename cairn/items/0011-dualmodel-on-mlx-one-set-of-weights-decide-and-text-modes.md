---
id: 11
title: 'DualModel on MLX: one set of weights, decide and text modes'
type: feature
status: planned
milestone: load
depends_on:
- 10
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: model
---

## Problem

Kev's `DecisionModel` keeps only the backbone (`AutoModelForCausalLM(...).model`, `kev/model.py:179`), so the vocabulary head is thrown away, and its serving paths merge the adapter. Qwen3.5-4B ties its output head to the embedding matrix, and the LoRA doesn't touch the embeddings, so the head is recoverable at no cost.

## Proposal

`kev_dual.DualModel`: load the full mlx-lm model once, attach Kev's adapter unmerged (per the answer to the MLX LoRA spike), load Kev's `PointerHead` and temperature, and expose `decide(state, questions)` (System One request and response shapes, reusing `kev.model.encode` and `rows_of`) and `generate(prompt, …)` with the adapter off.

## Acceptance criteria

- [ ] one copy of the base weights in memory; peak memory reported for each mode
- [ ] decide mode matches Kev's MLX path: max |dp| ≤ 0.02, no argmax flips, on 40 `transfer-v4` dev records (Kev's own parity bar)
- [ ] text mode matches plain `mlx_lm.generate` on the base checkpoint token for token (greedy, 5 prompts)
- [ ] switching decide → text → decide returns the same probabilities as before the switch
- [ ] tests for all of the above under the `weights` marker
