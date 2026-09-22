# Architecture

*Written 2026-09-22. This is the design as planned. Where it depends on an experiment, the cairn item is named, and this file is updated when the item closes.*

## One step of the agent

```
page state (cleaned DOM, ~1–4k tokens) + task + action history
        │
        ▼
  prefill ONCE with the base weights ──► cache  (KV for 8 attention layers,
        │                                        conv + recurrent state for 24 DeltaNet layers)
        ├── copy ─► <q> which element? <opt>…</opt>×K <decide>   adapter ON  ─► pointer head ─► p(element)
        ├── copy ─► <q> which operation? CLICK / TYPE / SELECT   adapter ON  ─► pointer head ─► p(op)
        └── copy ─► "Value to type:" …                           adapter OFF ─► LM head       ─► text
```

The decision branches run as one batch on copies of the cache, as Kev's MLX path already does. Text generation continues from its own copy, and only runs when the chosen operation needs a value.

## The three pieces

1. **One set of base weights.** Qwen3.5-4B with its output head kept. The head is tied to the embeddings, which the adapter doesn't touch ([context/kev-internals.md](../context/kev-internals.md)).
2. **Kev's adapter, unmerged and switchable.** Rank-16 LoRA over the attention, MLP and DeltaNet projections, plus Kev's `PointerHead` and fitted temperature. On MLX this needs unmerged LoRA layers with a runtime scale (item 0010). On torch, PEFT, with `disable_adapter()` or aLoRA.
3. **Adapter placement.** It decides whether the prefill can be shared:

| Placement | Adapter active on | Prefill per step | Needs retraining |
|---|---|---|---|
| **Full** (released Kev) | every token | twice when both modes are used: adapter on for decisions, off for text | no |
| **Question-side** (aLoRA / Solomon) | from `<q>` onward | **once**; the state cache is the base model's | yes: Kev's recipe with the placement mask (items 0020, 0021) |

The **Load** milestone ships Full, which works with the released Kev-4B. The **Share** milestone adds question-side placement. If question-side costs more than 2 pp on `transfer-v4`, the project keeps Full and the headline becomes memory and one model, not one prefill (item 0021's gate).

## Which base

| Option | Text mode | Decide mode | Cost |
|---|---|---|---|
| A. Qwen3.5-4B-Base (released Kev-4B), adapter off | Base model writing few-shot: weak on free text, unknown on short values | Kev-4B as released | none |
| B. `Qwen/Qwen3.5-4B` (post-trained) + Kev adapter retrained | a proper chat model | unknown; retrain and gate within 1 pp | ~1 H100-hour |
| C. Base + a second generation adapter | an SFT adapter, hot-swapped | Kev-4B as released | an SFT run; only if B fails |

- The **Write** milestone chooses between them on the text an agent step actually needs (items 0014–0018).
- Evidence leaning to B, from Kev's PLAN.md: a post-trained base keeps date arithmetic through Kev's training (Qwen3.6-35B-A3B: 0.88 → 0.95), and Base checkpoints lose it. An untrained post-trained 4B readout (SemIf's prompt) scores 0.747 on `transfer-v4`, against 0.692 for the Base probe.
- B is also the only option where question-side placement and a good text mode come together without two adapters.

## The prompt-format problem a shared prefill creates

- Kev's state prefix is `<|fim_prefix|>` + state text: a FIM token, not a chat template.
- With a shared prefill, text mode continues from exactly that prefix, so the post-trained model's chat format is not there at the start.
- Options, to be settled in the Write milestone:
  1. Keep Kev's format and measure text quality from it.
  2. Wrap the state in the chat template *inside* the state (the adapter is trained on whatever state format it sees, so a retrain on B can adopt it).
  3. Change the state delimiter.
- Option 2 fits best with retraining on B. The state format is part of the retrain's setup and must be fixed before it runs.

## Serving surface

One process (item 0032):

- Kev's `/v1/systemone`, unchanged, so the TypeSafe SDK and jevbench's harness work.
- An OpenAI-compatible chat-completions endpoint for text mode.
- `/v1/models` reports the backend, dtype, placement and modes.
- The server's LRU of state prefixes serves both modes when placement is question-side.

## What is deliberately out of scope for v1

- **Pixels.** Kev loads text-only, and screenshots belong to kev-vision (item 0037).
- **Sizes other than 4B** (item 0038).
- **Concurrent requests or batching across callers.** Kev's server is single-request, and so is this one.
