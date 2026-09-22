---
id: 23
title: 'Per-step latency: shared prefill vs two prefills vs two models'
type: experiment
status: backlog
milestone: share
depends_on:
- 21
- 22
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: s
area: model
---

## Question

What does one shared prefill buy per agent step, at Mind2Web page sizes?

## Hypothesis

Prefill of the page dominates a step, so sharing it approaches 2× over two prefills on steps that need both modes.

## Setup

States of 1k, 2k and 4k tokens; a step with 2 decide questions (element, operation) plus 16 generated tokens. Three configurations: (1) DualModel with the shared prefill; (2) DualModel with whole-adapter switching (two prefills); (3) Kev-4B plus a separate `Qwen/Qwen3.5-4B` (two models, the jev-ultrafast shape). Median and p90 over 20 steps, plus peak memory.

## Gate (fixed before the run)

The "one prefill, two modes" headline is used only if (1) is ≥ 1.5× faster than (3) at 2k tokens. Otherwise the headline is memory ("one 4B model, two modes"), stated with the latency as measured.

## Result

## Verdict
