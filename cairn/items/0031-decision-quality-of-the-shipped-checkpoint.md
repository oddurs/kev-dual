---
id: 31
title: Decision quality of the shipped checkpoint
type: experiment
status: backlog
milestone: bench
depends_on:
- 21
- 30
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: m
area: decide-eval
---

## Question

Did the changes for two modes cost decision quality?

## Setup

The checkpoint chosen for launch vs Kev-4B: jevbench, `transfer-v4` and `transfer-v9` dev, then one locked test read (Kev's rule: test data is read once per released checkpoint, after selection).

## Gate (fixed before the run)

Ship if `transfer-v4` dev is within 2 pp of Kev-4B. Report every number either way; the comparison table in the README shows it as measured.

## Result

## Verdict
