---
id: 16
title: Text mode on Base vs Instruct
type: experiment
status: planned
milestone: write
depends_on:
- 15
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: text-eval
---

## Question

How much worse is Qwen3.5-4B-Base with the Kev adapter off than Qwen3.5-4B (the post-trained model; `Qwen/Qwen3.5-4B`, there is no `-Instruct` repo) at the text an agent step needs?

## Hypothesis

The gap is small on copied or derived values and large on free text and plans. Evidence: Kev's PLAN.md finds the post-trained Qwen3.6 base keeps its date arithmetic through training (0.88 → 0.95), while Base checkpoints lose it. An untrained Qwen3.5-4B (the post-trained model; `Qwen/Qwen3.5-4B`, there is no `-Instruct` repo) readout also scores 0.747 on `transfer-v4`, against 0.692 for the Base probe.

## Setup

Arm A: Kev-4B's base with the adapter off (few-shot). Arm B: Qwen3.5-4B (the post-trained model; `Qwen/Qwen3.5-4B`, there is no `-Instruct` repo) (chat template). The frozen text-mode set; greedy decoding; fp32 torch, or MLX bf16 if the parity holds.

## Gate (fixed before the run)

If A is within 5 points of B overall **and** within 10 on every category → stay on Base: text mode is free with the released Kev-4B, and the retrain is dropped. Otherwise → the Instruct path: run the retrain experiment.

## Result

## Verdict
