---
id: 13
title: Torch/CUDA backend of DualModel for Modal
type: feature
status: done
milestone: load
assignee: Oddur Sigurdsson
depends_on:
- 11
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p2
effort: m
area: serving
---

## Problem

Kev's published numbers use the fp32 torch path, and training and long evals run on Modal GPUs. MLX only covers the Mac.

## Proposal

The same `DualModel` interface on PyTorch: `AutoModelForCausalLM` kept whole, the Kev adapter through PEFT unmerged, and text mode inside `with lm.disable_adapter():`.

## Acceptance criteria

- [x] decide mode reproduces Kev's fp32 benchmark on `transfer-v4` dev exactly (same accuracy and Brier)
- [x] text mode matches plain `transformers` generation token for token
- [x] MLX and torch agree to bf16 rounding on the parity records

## 2026-09-22

Verified 2026-09-22 on MPS in fp32.

Whole transfer-v4 dev split (764 records), through Kev's own `evaluate_records`, same records and context (reports/torch-parity.json, scripts/torch_parity.py):

| path | accuracy | Brier | ECE |
|---|---|---|---|
| Kev LocalPredictor, fp32, merged | 0.79726 | 0.263741 | 0.04048 |
| TorchDualModel, fp32, unmerged | 0.79726 | 0.263740 | 0.04048 |

- Largest probability difference 2.5e-5, 0 argmax flips.
- Accuracy equals the 0.797 in Kev's README and model card.

Weight tests (`pytest -m weights`, 9 passed):
- text mode matches plain transformers greedy generation over 5 prompts × 32 tokens
- the switch round trip is exact
- MLX (bf16) and torch (fp32) agree within 0.03 on the 40 parity records

One near-tie flips under bf16: record 1, MMLU, fp32 0.3216 vs 0.3167. Kev's own merged MLX path flips on the same record, so it is rounding, not an implementation difference. The test allows a flip only when the fp32 top-2 margin is below the rounding bound.

**Not verified: CUDA.** There is no CUDA device or Modal token on this machine. The code is device-agnostic (SDPA on CUDA, as in Kev), but that has not been run; item created for it.
