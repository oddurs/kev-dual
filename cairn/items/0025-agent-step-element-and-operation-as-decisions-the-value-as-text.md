---
id: 25
title: 'Agent step: element and operation as decisions, the value as text'
type: feature
status: backlog
milestone: act
depends_on:
- 11
- 24
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: agent
---

## Proposal

For each step: a Choice over the top-K candidates (plus "none"), a Choice over CLICK / TYPE / SELECT, then generate the value when the operation needs one, all from one DualModel. The previous actions go into the state.

## Acceptance criteria

- [ ] runs one Mind2Web task end to end, offline, from the frozen slice
- [ ] records per-step timings for prefill, decide and generate
