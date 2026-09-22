---
id: 29
title: Fine-tune decide mode on Mind2Web train
type: experiment
status: backlog
milestone: act
depends_on:
- 28
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: m
area: training
---

## Question

Does a short `--init_from` fine-tune on Mind2Web train make element choice competitive without costing general decision quality?

## Setup

`kev.train --data` on Mind2Web train steps converted by the step converter, `--init_from` the chosen checkpoint, lr 2e-5 (Kev's recommended delta rate), replay a sample of `decision-v7`. Refit the temperature on held-out steps.

## Gate (fixed before the run)

Adopt if dev step success beats the two-model baseline at lower median per-step latency, **and** `transfer-v4` dev stays within 2 pp of the checkpoint it started from.

## Result

## Verdict
