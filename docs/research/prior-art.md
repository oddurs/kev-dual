# Prior art

*Checked 2026-09-22 against primary sources (papers, READMEs, model cards, code). Re-check before launch (cairn item 0036).*

## Summary

The mechanism kev-dual needs is published and shipped. An adapter that is off over the prefix and on from a marker token, with the prefix cache shared with the base model, is Activated LoRA, and it is in PEFT. Solomon already runs decision branches and adapter-off text generation from one cached document state. What remains open is a measured result: an open, laptop-sized model whose text mode is general output, evaluated in an agent loop against the two-model setup. The reasoning is in cairn item 0019.

## Closest: shared prefix, adapter only after a marker

### Activated LoRA (aLoRA)

Greenewald et al., IBM Research, [arXiv 2504.12397](https://arxiv.org/html/2504.12397).

- The adapter changes Q/K/V only for tokens from the start of an invocation sequence onward.
- Every earlier position uses exactly the base weights, so the prefix KV "coincide with the base model's" and the base model's cache is reused without recomputation.
- It often needs a higher rank (r≈32).
- Reported speedups: 2–7× on 250-token prompts, over 20× on longer ones.

**In PEFT** ([docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora)):

- Configured with `LoraConfig(alora_invocation_tokens=[…])`.
- The adapter activates from the **last** occurrence of the invocation sequence. Generated tokens after it also use the adapter.
- The docs say the prefix cache "is interchangeable with base model cache".
- Limits: causal LMs only, cannot be merged, no beam search.
- Unverified: whether it covers Qwen3.5's DeltaNet projections, and which vLLM version reuses the cache.

### Solomon

DoccyHealth, [huggingface.co/DoccyHealth/Solomon](https://huggingface.co/DoccyHealth/Solomon).

- `Qwen/Qwen3.8-27B` (base weights not redistributed), LoRA r=64, Apache-2.0.
- Decisions come from trained linear heads over letter logits: yes/no, single choice, ordered, multi-label. The card says "It does not generate text".
- `adapter/config.json` sets `"placement": "question"`: "adapter is OFF while the document prefix is prefilled and ON from the question branch onward".
- `SwitchLoRA` in `src/solomon/engine_contract.py` adds the LoRA term only at positions ≥ `CTL['start']`.
- **Shared prefill with text, in code:** `src/solomon/engine_reasoning.py` generates greedy base-model text with the adapter off. It works from a deep copy of the same cached document state (it requires `placement=='question'`), inside a `<think>` block capped at 3,072 tokens. It returns `reused_prefix_tokens`, then reads a forced answer letter. It is used as an escalation stage (`routing.py`). This is reasoning in service of a decision, not general text output, and the card doesn't advertise it.
- v1.1 numbers:
  - 88.0% on 802 real-document questions (labels are AI-generated and not checked by people).
  - 72.9% on MMLU plus MMLU-Pro.
  - BF16 ECE 0.018.
  - Cached and full passes differ by at most 0.0108.

### Kev's own plan

[`PLAN_27b.md`](https://github.com/jaredpalmer/kev/blob/main/PLAN_27b.md) item A1 pre-registers question-side LoRA (`lora_placement="question"`) at 4B and 9B. The motivation: the base may keep its reading skills, such as date arithmetic and MMLU, and the prefix cache becomes adapter-agnostic. As of Kev HEAD `dfe16ca` (2026-09-22), the plan is marked "Nothing here has been started", and no code, PR or issue exists.

## One model generating and deciding

| Work | What it does | Relation |
|---|---|---|
| [GenRM](https://arxiv.org/abs/2408.15240) (Aug 2024, ICLR 2025) | Trains verifiers by next-token prediction, jointly on verification and on solution generation. One LLM generates and scores. Best-of-N on GSM8K 73% → 93.4%. | The verifier emits tokens (Yes/No), not a head. No shared-prefix serving. |
| [SwiftSage](https://arxiv.org/abs/2305.17390) (May 2023) | A fast small model (Swift) and GPT-4 (Sage) on ScienceWorld. | Fast and slow modes, but two separate models. |
| [System 1 and 2 communication](https://arxiv.org/abs/2510.00494) (Oct 2025) | A Base model exchanges latent messages with a coprocessor. A single unified model with soft embeddings nearly matches the dual design. | Supports "one model" over two. |

## Decision models near this space

- **[SemIf](https://github.com/TheoLeeCJ/SemIf)** (formerly OpenJev):
  - Frozen Qwen3.5-4B, *untrained*. Probabilities come from the declared option letters' logits in one forward pass.
  - Prefills a shared state once and branches per criterion, with an MLX backend.
  - RTX 3090: 21 decisions in 1.023 s vs 5.332 s for generating JSON. Prefix reuse takes throughput from 2.33 to 20.03 decisions/s.
  - Second on jevbench at 73.1.
  - It is the obvious "free" dual-mode baseline: the same post-trained model generates text and reads letter logits, with no adapter at all. **kev-dual must beat it on decisions** to justify the adapter.
- **[Laya](https://huggingface.co/convaiinnovations/laya):** a bidirectional encoder (ModernBERT-large) with a 2-layer head over per-option `[MASK]` tokens. It runs one sequence per question and cannot generate. (From Kev's PLAN.md.)
- **[PlayJev](https://github.com/OmniJev/PlayJev):**
  - Qwen3.5-0.8B-Base fine-tuned on raw game pixels. One frame plus a shuffled move list gives per-move probabilities in one pass, 43 ms on an H200.
  - Trained on 2.2M frames with behaviour cloning, then 3 rounds of DAgger.
  - Mean vs-teacher score 0.57. Served at `/v1/systemone`. Apache-2.0.

## Web agents built on decision models

- **[jev-ultrafast](https://github.com/browser-use/jev-ultrafast):**
  - Each observation is one DOM snapshot turned into an indexed table (`[n] role name · value`), with no screenshots.
  - One Jev request returns an operation (CLICK, TYPE_TEXT, SELECT, SCROLL_UP/DOWN, WAIT, DONE, BLOCKED) plus speculative target heads, and only the matching head is used.
  - A separate LLM (`inception/mercury-2.5` over OpenRouter, reasoning off) is called only for TYPE_TEXT, and its output must parse as JSON.
  - Google Flights Zürich→London in 7,073 ms. The median fell from 9.450 s to 7.092 s over 3 runs of one task.
  - **This is the two-model setup kev-dual replaces.**
- **[JevForge](https://github.com/zwliJay/jev-forge):**
  - Qwen3.5-0.8B trained on Mind2Web: gold `pos_candidates` share the probability mass equally, and sampled negatives get zero. Loss is cross-entropy plus Brier, then calibration.
  - Splits are website-disjoint. Each question expands into K rows with a pooled 2-layer scorer.
  - "Choice top-1 in positive set" is 0.579 on test (2,400 questions) and 0.637 on OOD (1,158). K is not stated; the caller supplies candidates.
- **[GUI_JEV](https://github.com/ZihuaEvan/GUI_JEV)** (from the notes, not re-checked): a vision model captions grid tiles, then Jev chooses. 9/12 on a ScreenSpot subset.
