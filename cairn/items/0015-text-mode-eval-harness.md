---
id: 15
title: Text-mode eval harness
type: feature
status: planned
milestone: write
depends_on:
- 11
- 14
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: text-eval
---

## Problem

There is no fixed measurement of text-mode quality.

## Proposal

A frozen text-mode set built per the spike's answer, and a runner that scores any text path (a checkpoint plus a mode) against it.

## Acceptance criteria

- [ ] the set is frozen with a manifest: source, split, record count, SHA-256
- [ ] it runs a Base checkpoint few-shot and an Instruct checkpoint through its chat template, with the prompt format recorded in the report
- [ ] the report is JSON, with per-category scores and bootstrap CIs
