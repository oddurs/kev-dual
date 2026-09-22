---
id: 21
title: Question-side Kev-4B
type: experiment
status: backlog
milestone: share
depends_on:
- 16
- 20
- 39
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: training
---

## Question

How much decision quality does moving the adapter off the state cost?

## Hypothesis

A small loss on transfer (the state is read by the unadapted base) and possibly a gain on `deadline`: Kev's PLAN_27b.md suggests date erosion may be a document-side effect.

## Setup

Kev's 4B recipe on the base chosen in the Write milestone, `placement="question"`, 2 seeds. Compare against the whole-model adapter on the same base.

## Gate (fixed before the run)

Adopt if `transfer-v4` dev is within 2 pp of the whole-model adapter (point estimate; paired bootstrap reported). Otherwise the Share milestone falls back to whole-adapter switching, and the headline moves to the agent demo.

## Result

## Verdict
