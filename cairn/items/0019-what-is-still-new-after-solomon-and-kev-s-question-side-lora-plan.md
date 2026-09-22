---
id: 19
title: What is still new after Solomon and Kev's question-side LoRA plan?
type: spike
status: done
milestone: share
created: 2026-09-22
updated: 2026-09-22
closed_at: 2026-09-22
priority: p0
effort: s
area: research
---

## Question

Question-side LoRA (adapter off over the document, on from the question) is not new. Solomon (DoccyHealth, on the Hub) ships it as `SwitchLoRA` with a `placement` setting, and Kev's `PLAN_27b.md` pre-registers it as experiment A1. Activated LoRA (aLoRA) is related. What is left for kev-dual to claim?

## Why it has to be answered first

The launch headline rests on it. The notes already warn that "two adapters on one model may not read as new".

## What would settle it

For each of Solomon, aLoRA and Kev's A1: does the same prefix cache also serve text generation, in an agent loop, measured end to end? See docs/research/prior-art.md.

## Answer

Answered from primary sources on 2026-09-22; details and links in docs/research/prior-art.md.

**The mechanism is taken.** Activated LoRA (Greenewald et al., IBM, arXiv 2504.12397) applies the adapter only from an invocation sequence onward, so the prefix cache is the base model's and can be reused. It is in PEFT as `alora_invocation_tokens`. Solomon (DoccyHealth, Qwen3.8-27B, Apache-2.0) uses question placement (`SwitchLoRA`), and its code (`engine_reasoning.py`) generates base-model reasoning text with the adapter off from the same cached document state as its decision branches. So "one prefill serves typed decisions and generated text" exists in code, though Solomon's card says "It does not generate text" and does not advertise that path.

**What is still unclaimed:**

1. Open weights at a laptop size (4B) with text mode as a general product: the value typed into a field, a tool call, an answer, not only hidden reasoning before a forced letter.
2. Measured in an agent loop (Mind2Web) against the two-model shape jev-ultrafast uses, with end-to-end step latency and memory.
3. Apple Silicon (MLX) serving of both modes from one cache.

**Consequence for the headline:** do not claim the architecture. Credit aLoRA and Solomon, and claim the measured result: "one 4B model, one prefill per step, N× faster than Kev plus an LLM at equal step success on Mind2Web". This confirms the risk in the notes: the demo carries the launch. Re-check in the pre-launch scan.
