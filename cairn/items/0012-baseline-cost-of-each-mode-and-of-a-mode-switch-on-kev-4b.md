---
id: 12
title: Baseline cost of each mode and of a mode switch on Kev-4B
type: experiment
status: planned
milestone: load
depends_on:
- 11
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: s
area: model
---

## Question

What does each mode cost on this machine (M-series, 48 GB) before any optimisation, and what does a mode switch cost?

## Hypothesis

Decide mode is close to Kev's published MLX numbers (Kev-4B: 721 ms new state, 136 ms with the prefix cached, five questions on a ~270-token state, M5 32 GB). The unmerged adapter adds some overhead; the switch itself is free.

## Setup

Kev-4B through `DualModel`. States of 270, 1k and 2k tokens (Mind2Web pages are long). Five three-option questions; 32-token greedy generation. 20 repetitions after warm-up; report the median and p90.

## Gate (fixed before the run)

Informational: no adopt decision. It becomes the baseline every later latency claim is compared against.

## Result

## Verdict
