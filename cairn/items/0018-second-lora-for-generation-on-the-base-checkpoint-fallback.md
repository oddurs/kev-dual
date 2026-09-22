---
id: 18
title: Second LoRA for generation on the Base checkpoint (fallback)
type: experiment
status: backlog
milestone: write
depends_on:
- 17
created: 2026-09-22
updated: 2026-09-22
priority: p3
effort: m
area: training
---

## Question

If the Instruct retrain fails its gate: can a small generation adapter on Qwen3.5-4B-Base close the text gap, hot-swapped against Kev's decision adapter?

## Hypothesis

A rank-16 SFT adapter on short agent-style instructions reaches Arm B's text-mode score within 5 points.

## Setup

Only runs if the retrain experiment fails. Details to be written before the run.

## Gate (fixed before the run)

Adopt if within 5 points of Instruct on the text-mode set.

## Result

## Verdict
