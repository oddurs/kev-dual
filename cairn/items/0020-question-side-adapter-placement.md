---
id: 20
title: Question-side adapter placement
type: feature
status: backlog
milestone: share
depends_on:
- 11
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: training
---

## Problem

A whole-model adapter changes the state's hidden states, so the page must be prefilled twice when the mode switches, once with the adapter and once without.

## Proposal

Scale the LoRA output by a per-position mask: 0 over the state, 1 from `<q>` onward. The mechanism already exists, so reuse it rather than write a new one:

- **torch:** PEFT's Activated LoRA (`LoraConfig(alora_invocation_tokens=[…])`, arXiv 2504.12397). The adapter turns on from the last occurrence of the invocation sequence. Kev's `<q>` delimiter is `<|fim_middle|>`, but every question row repeats it, so check where PEFT puts the start when there are several. Limits: causal LMs only, cannot be merged, no beam search. First check that it covers the DeltaNet projections.
- **MLX:** a masked LoRA layer on the same principle; Solomon's `SwitchLoRA` (`engine_contract.py`) is a reference.
- Kev's PLAN_27b.md A1 sketches the same switch (`lora_placement="question"`, not started as of Kev HEAD `dfe16ca`). Offer it upstream rather than keep a fork.

See docs/research/prior-art.md.

## Acceptance criteria

- [ ] `placement="full"` reproduces today's Kev numbers exactly (parity test)
- [ ] with `placement="question"`, the state's hidden states and cache equal the base model's bit for bit
- [ ] training (`kev.train`) and both DualModel backends honour the placement
