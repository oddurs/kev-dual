---
id: 10
title: Can Kev's PEFT adapter run unmerged in mlx-lm with a runtime on/off?
type: spike
status: planned
milestone: load
depends_on:
- 9
created: 2026-09-22
updated: 2026-09-22
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
