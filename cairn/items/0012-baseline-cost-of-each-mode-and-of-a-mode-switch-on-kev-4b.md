---
id: 12
title: Baseline cost of each mode and of a mode switch on Kev-4B
type: experiment
status: done
milestone: load
assignee: Oddur Sigurdsson
depends_on:
- 11
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p1
effort: s
area: model
outcome: pass
cost: '0'
run: reports/bench-modes.json
---

## Question

What does each mode cost on this machine (M-series, 48 GB) before any optimisation, and what does a mode switch cost?

## Hypothesis

Decide mode is close to Kev's published MLX numbers (Kev-4B: 721 ms new state, 136 ms with the prefix cached, five questions on a ~270-token state, M5 32 GB). The unmerged adapter adds some overhead; the switch itself is free.

## Setup

Kev-4B through `DualModel`. States of 270, 1k and 2k tokens (Mind2Web pages are long). Five three-option questions; 32-token greedy generation. 20 repetitions after warm-up; report the median and p90.

## Gate (fixed before the run)

Informational: no adopt decision. It becomes the baseline every later latency claim is compared against.

## Result

**Setup:**

- Apple M5 Pro, 48 GB, macOS 26.5.2; Kev-4B; 20 repetitions after one warm-up.
- The states are real text from Kev's dev records, cut to exactly 270, 1,024 and 2,048 tokens. They are encoded at the **serving** limits: `encode` defaults to the 384-token training limit and truncates silently, so the script asserts no truncation.
- Committed report: `reports/bench-modes.json`, from `scripts/bench_modes.py`.

**Decide**, five 3-option questions, median ms (p90):

| state tokens | Kev merged, new | DualModel, new | overhead | Kev merged, cached | DualModel, cached | overhead |
|---|---|---|---|---|---|---|
| 270 | 217 (221) | 267 (269) | +23% | 82 (84) | 101 (102) | +24% |
| 1,024 | 499 (501) | 645 (658) | +29% | 85 (85) | 109 (110) | +29% |
| 2,048 | 939 (958) | 1,235 (1,311) | +32% | 90 (92) | 128 (175) | +41% |

**Generate**, adapter off: the state as the prompt, then exactly 32 tokens (EOS masked):

| state tokens | total ms (p90) | prefill tok/s | decode tok/s |
|---|---|---|---|
| 270 | 1,266 (1,303) | 801 | 31.3 |
| 1,024 | 1,878 (1,918) | 1,901 | 29.8 |
| 2,048 | 2,135 (2,585) | 1,836 | 21.2 |

**The switch**, decide on a new state timed straight after a generate vs straight after another decide: 309 vs 305 ms, 644 vs 763 ms, 1,096 vs 1,140 ms. There is no measurable cost; the differences are noise in both directions.

**Is the decode speed caused by the wrappers?** No. A follow-up on the 270-token state, 64 tokens, median of 5, gave plain mlx-lm on the base 30.7 tok/s and DualModel with the adapter off 31.0 tok/s. The 248 switch wrappers cost nothing when off, and about 31 tok/s is simply this model's decode speed on this machine.

## Verdict

The baseline is recorded (the gate was informational). What later items should take from it:

1. **A mode switch is free.** Nothing is reloaded or recomputed except the prefill.
2. **The unmerged adapter costs 23–41% in decide mode**, growing with state length. It is the price of switchability and of question-side placement later. `mx.compile` on the adapted layers is the first thing to try if it matters. The 0023 latency comparison should include it.
3. **Generation dominates a step that uses both modes.** About 32 decoded tokens take roughly 1 s, against 0.3–1.2 s for decide. Prefill at 2k tokens is about 1.1 s in each mode. So a shared prefill (Share milestone) saves up to about 1 s per step at 2k tokens, and **short value outputs matter as much as the shared prefill does**. The text-mode spike (0014) should favour formats where the value is a few tokens.
4. **Kev's published MLX figure isn't comparable here.** Its 721 ms for 270 tokens was on an M5 (32 GB); this M5 Pro measures 217 ms for Kev merged. Compare only numbers taken on the same machine.
