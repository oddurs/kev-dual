---
id: 32
title: Serve both modes from one process
type: feature
status: backlog
milestone: launch
depends_on:
- 22
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: serving
---

## Proposal

Kev's `/v1/systemone` (unchanged request and response, so the TypeSafe SDK works) plus a generate endpoint, both served by one DualModel. `uvx`/`uv run` one-line start.

## Acceptance criteria

- [ ] Kev's API tests (`tests/test_api.py` shape) pass against it
- [ ] a generate endpoint (OpenAI-compatible chat completions, so existing agents can point at it)
- [ ] `/v1/models` reports the backend, dtype, adapter placement and both modes
