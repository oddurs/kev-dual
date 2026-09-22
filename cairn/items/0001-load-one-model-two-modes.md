---
id: 1
key: load
title: 'Load: one model, two modes'
type: milestone
status: done
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p2
due: 2026-09-25
---

Kev-4B's decide mode and Qwen3.5-4B's text mode run from one loaded set of weights on Apple Silicon (MLX).

**Exit:** decide-mode probabilities match Kev's own MLX path (max |dp| ≤ 0.02 on 40 dev records), text-mode output matches plain `mlx_lm` generation token for token, and the per-mode latency baseline is recorded.

## 2026-09-22

Exit met 2026-09-22 on an M5 Pro (48 GB):
- **Decide parity:** max |dp| 0.012 against Kev's own MLX path on 40 transfer-v4 dev records, 0 argmax flips (0010, 0011).
- **Text parity:** token-for-token identical to plain mlx-lm on 5 greedy prompts (0011).
- **Latency baseline:** recorded in 0012.
- **Beyond the exit condition:** the torch backend reproduces Kev's fp32 transfer-v4 numbers exactly on MPS (0013).

What the next milestones inherit:
- One 8.48 GB copy of the weights serves both modes, and the switch itself is free.
- The unmerged adapter costs 23–41% in decide mode.
- 32 generated tokens take about 1 s, at the same speed as plain mlx-lm.
- CUDA is still unrun (0040, needs a Modal token).
- Kev's `encode` silently truncates states to 384 tokens unless it is given the serving limits. DualModel passes them, and anything that calls `encode` directly must do the same.
