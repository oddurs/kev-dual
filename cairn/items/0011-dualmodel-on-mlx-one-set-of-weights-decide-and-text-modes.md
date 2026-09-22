---
id: 11
title: 'DualModel on MLX: one set of weights, decide and text modes'
type: feature
status: done
milestone: load
assignee: Oddur Sigurdsson
depends_on:
- 10
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p0
effort: m
area: model
---

## Problem

Kev's `DecisionModel` keeps only the backbone (`AutoModelForCausalLM(...).model`, `kev/model.py:179`), so the vocabulary head is thrown away, and its serving paths merge the adapter. Qwen3.5-4B ties its output head to the embedding matrix, and the LoRA doesn't touch the embeddings, so the head is recoverable at no cost.

## Proposal

`kev_dual.DualModel`: load the full mlx-lm model once, attach Kev's adapter unmerged (per the answer to the MLX LoRA spike), load Kev's `PointerHead` and temperature, and expose `decide(state, questions)` (System One request and response shapes, reusing `kev.model.encode` and `rows_of`) and `generate(prompt, …)` with the adapter off.

## Acceptance criteria

- [x] one copy of the base weights in memory; peak memory reported for each mode
- [x] decide mode matches Kev's MLX path: max |dp| ≤ 0.02, no argmax flips, on 40 `transfer-v4` dev records (Kev's own parity bar)
- [x] text mode matches plain `mlx_lm.generate` on the base checkpoint token for token (greedy, 5 prompts)
- [x] switching decide → text → decide returns the same probabilities as before the switch
- [x] tests for all of the above under the `weights` marker

## 2026-09-22

Verified 2026-09-22 on the M-series Mac (48 GB). `uv run pytest -m weights` → 6 passed:
- decide parity vs Kev's merged MLX path on 40 transfer-v4 dev records: ≤ 0.02, 0 argmax flips (the probe measured max |dp| 0.012)
- greedy text over 5 prompts × 32 tokens is identical to plain mlx-lm on the base
- probabilities are bit-identical after a decide → text → decide round trip
- the model holds the base parameters once, plus the low-rank tensors only

Memory (reports/dual-memory.json): weights 8.48 GB (adapter 0.065 GB), active after load 8.48 GB, peak 8.79 GB in decide mode and 8.57 GB in text mode. As expected from a Base checkpoint, the text sample is fluent but generic (a numbered plan of trivial steps). That is the Write milestone's question.
