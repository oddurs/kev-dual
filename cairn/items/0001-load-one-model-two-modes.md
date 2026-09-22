---
id: 1
key: load
title: 'Load: one model, two modes'
type: milestone
status: backlog
created: 2026-09-22
updated: 2026-09-22
priority: p2
due: 2026-09-25
---

Kev-4B's decide mode and Qwen3.5-4B's text mode run from one loaded set of weights on Apple Silicon (MLX).

**Exit:** decide-mode probabilities match Kev's own MLX path (max |dp| ≤ 0.02 on 40 dev records), text-mode output matches plain `mlx_lm` generation token for token, and the per-mode latency baseline is recorded.
