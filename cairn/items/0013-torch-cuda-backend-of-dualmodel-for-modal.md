---
id: 13
title: Torch/CUDA backend of DualModel for Modal
type: feature
status: planned
milestone: load
depends_on:
- 11
created: 2026-09-22
updated: 2026-09-22
priority: p2
effort: m
area: serving
---

## Problem

Kev's published numbers use the fp32 torch path, and training and long evals run on Modal GPUs. MLX only covers the Mac.

## Proposal

The same `DualModel` interface on PyTorch: `AutoModelForCausalLM` kept whole, the Kev adapter through PEFT unmerged, and text mode inside `with lm.disable_adapter():`.

## Acceptance criteria

- [ ] decide mode reproduces Kev's fp32 benchmark on `transfer-v4` dev exactly (same accuracy and Brier)
- [ ] text mode matches plain `transformers` generation token for token
- [ ] MLX and torch agree to bf16 rounding on the parity records
