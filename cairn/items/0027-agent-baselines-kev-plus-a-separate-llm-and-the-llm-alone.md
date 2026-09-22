---
id: 27
title: 'Agent baselines: Kev plus a separate LLM, and the LLM alone'
type: feature
status: backlog
milestone: act
depends_on:
- 25
- 26
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: m
area: agent
---

## Acceptance criteria

- [ ] two-model baseline: Kev-4B for element and operation, `Qwen/Qwen3.5-4B` for the value, as two loaded models (the jev-ultrafast shape)
- [ ] single-LLM baseline: `Qwen/Qwen3.5-4B` does everything by generating, MindAct-style multiple choice
- [ ] both scored by the same scorer on the same slice
