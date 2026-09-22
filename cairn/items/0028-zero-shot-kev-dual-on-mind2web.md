---
id: 28
title: Zero-shot kev-dual on Mind2Web
type: experiment
status: backlog
milestone: act
depends_on:
- 27
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: s
area: agent
---

## Question

How well does kev-dual act on Mind2Web without training on it?

## Hypothesis

Operation choice is good. Element choice among many candidates is weak: Kev never saw web pages, and JevForge needed Mind2Web training.

## Setup

Frozen dev slice; kev-dual (the chosen configuration from Share) against both baselines.

## Gate (fixed before the run)

If kev-dual's step success rate is below the single-LLM baseline's → run the Mind2Web fine-tune. Otherwise go straight to the test-slice read.

## Result

## Verdict
