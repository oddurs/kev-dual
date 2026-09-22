---
id: 17
title: Kev-4B retrained on Qwen3.5-4B Instruct
type: experiment
status: planned
milestone: write
depends_on:
- 16
- 39
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: m
area: training
---

## Question

Does Kev's recipe on an Instruct base keep decision quality, so text mode can come from the Instruct weights for free?

## Hypothesis

Accuracy is within noise of Kev-4B, and `deadline` is higher: the post-trained base keeps its date skill, as with the Qwen3.6-35B-A3B trial in Kev's PLAN.md.

## Setup

Kev's Kev-4B recipe unchanged except `--base` (`Qwen/Qwen3.5-4B`, the post-trained model, pinned revision): `decision-v7`, 2 epochs, lr 5e-5, batch 4, accum 2, bf16, checkpointing, `--p_none_pair 0.25`, then the dates + unknowable delta. One H100 via Modal, about 1 h. Temperature fitted per `scripts/calibrate_checkpoint.py`.

## Gate (fixed before the run)

Adopt if `transfer-v4` dev accuracy is within 1 pp of Kev-4B (0.797, paired bootstrap on the same records) and Brier is no worse by more than 0.02. Otherwise fall back to a second generation adapter on Base.

## Result

## Verdict
