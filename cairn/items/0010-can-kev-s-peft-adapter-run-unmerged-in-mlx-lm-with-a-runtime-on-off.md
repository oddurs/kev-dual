---
id: 10
title: Can Kev's PEFT adapter run unmerged in mlx-lm with a runtime on/off?
type: spike
status: done
milestone: load
assignee: Oddur Sigurdsson
depends_on:
- 9
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p0
effort: s
area: model
---

## Question

Kev's MLX path always merges the LoRA into the base weights (`kev/checkpoint.py:164`, `kev/mlx_model.py: merge_lora`). A merged adapter cannot be switched off. Can the same PEFT adapter be applied as unmerged low-rank layers in mlx-lm, with a scale that can be set to 0 at runtime, and at what latency cost?

## Why it has to be answered first

It decides how `DualModel` is built: unmerged layers with a runtime scale, two weight copies (merged and base; about 8 GB each for 4B in bf16), or reapplying the merge delta on each switch.

## What would settle it

Load `jaredpalmer/kev-4b` with mlx-lm's LoRA layers (key mapping as in `merge_lora`: `base_model.model.<…>` → `language_model.model.<…>`, including the DeltaNet projections `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`). Compare decide-mode probabilities against Kev's merged MLX path on 40 dev records, and time both.

## Answer


**Yes.** `kev_dual.lora` wraps each of the 248 adapted linears in a `SwitchLoRALinear`. The base weight stays untouched, and the layer adds `scale·(x·Aᵀ)·Bᵀ` while a shared `Switch` is on.

`scripts/probe_mlx_lora.py` compares it against Kev's merged MLX path on Kev-4B, using 40 evenly spaced `transfer-v4` dev records. The committed report is `reports/mlx-lora-probe.json`. Timings are the median of 10 runs on the longest sampled state, 195 tokens:

| | max abs dp | mean abs dp | argmax flips | new state | cached state |
|---|---|---|---|---|---|
| Kev, merged (reference) | – | – | – | 147 ms | 56 ms |
| unmerged, adapter in fp32 | 0.013 | 0.002 | 0 | 194 ms | 71 ms |
| unmerged, adapter in bf16 | 0.012 | 0.002 | 0 | 180 ms | 67 ms |
| unmerged, switch off (base) | – | – | – | 149 ms | 56 ms |

- Switching off and back on reproduces the probabilities bit for bit.
- An earlier run of the same probe measured the bf16 adapter at 215 ms against 154 ms merged. Run-to-run variance is large at this size, so the overhead is somewhere between 20% and 40%.

**Decision: use unmerged layers with a runtime switch, and keep the adapter in bf16.**

- It is within the ≤ 0.02 parity bar. bf16 loses nothing against fp32 and is faster.
- It is the only option that allows **per-position** placement. Question-side placement needs the adapter on for some tokens of a sequence and off for others, and a merged weight can't do that.
- The alternatives were ruled out on design, not speed:
  - Keeping two weight copies would cost about 8 GB extra at 4B, because the adapter touches nearly every linear.
  - Re-merging on every switch was also rejected.
- The overhead comes from 248 extra rank-16 matmul pairs per forward pass. Item 0012 measures it at realistic state sizes. If it matters, the first thing to try is `mx.compile` on the adapted layers.
- The remaining difference from the merged path is rounding. Kev's merge rounds W+ΔW to bf16 once; here ΔW is never folded into W. Neither is the fp32 truth. The torch backend (item 0013) is where fp32 gets checked.
