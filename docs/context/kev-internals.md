# Kev internals that matter for a second mode

*Read 2026-09-22 from `~/Code/jevv/kev` at commit `250330b`. Line numbers refer to that commit.*

## The output head survives, but Kev throws it away

- `DecisionModel` loads `AutoModelForCausalLM.from_pretrained(...).model` (`kev/model.py:179`), with the comment "backbone only (no vocab head): we never generate text". The LM head object is discarded at load time.
- Qwen3.5-4B-Base has `tie_word_embeddings: true` (top-level and `text_config`, from the cached HF config). The output head *is* the embedding matrix.
- Kev's LoRA targets the attention, MLP and DeltaNet projections, not `embed_tokens` (`kev/model.py:189-196`). The exception is `special_embeddings=True`, which trains the delimiter rows through PEFT's `trainable_token_indices`. The released `jaredpalmer/kev-4b` (snapshot `485ace8`) doesn't use it: its `adapter_config.json` has r=16, no `trainable_token_indices`, and targets `q/k/v/o_proj`, `gate/up/down_proj` and the DeltaNet projections `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`. Its fitted temperature is 2.378 (`result.json`). So with the adapter off, the base model's text path is intact. Loading the whole `CausalLM` and reading `.model` for the pointer head recovers it.

## The adapter is merged in serving

- **MLX:** always merged, `W + (B @ A)·α/r` computed in fp32 on the CPU and rounded once to bf16 (`kev/mlx_model.py:28-54`). `KEV_MERGE=0` is rejected (`kev/checkpoint.py:164`).
- **torch:** merged by default (`merge_and_unload()` in fp32, `kev/checkpoint.py:189`). It stays unmerged for bf16 without merge, and for adapters with token-trained embeddings.
- A merged adapter cannot be switched off. kev-dual needs it unmerged (cairn item 0010).
- The LoRA key mapping from PEFT to mlx-lm is `base_model.model.<layers…>` → `language_model.model.<layers…>` (comment at `kev/mlx_model.py:44`).

## Qwen3.5 is a hybrid, and that shapes the cache

- Qwen3.5-4B has 32 layers: 8 full attention and 24 Gated DeltaNet (`linear_attention`), with hidden size 2560 and a vocabulary of 248,320.
- DeltaNet layers are recurrent and ignore attention masks. So on hybrid bases Kev doesn't use its block-causal packed mask. Each question instead runs as **its own row**: the state followed by that question's branch (`rows_of`, `kev/model.py:133`). Isolation is exact.
- The MLX path prefills the state once into an mlx-lm prompt cache. That cache holds KV for the attention layers, and conv plus recurrent state for the DeltaNet layers. It then runs every branch as a batch on a **copy** (`type(c).merge([c] * n)`, `kev/mlx_model.py:112-116`), so the prefix stays untouched. This is the same move kev-dual needs for text generation from a shared state.
- The server keeps an LRU of state prefixes keyed by the state's token ids (`kev/serve.py:28-60`). This is exact because "the state's activations do not depend on the branches".

## The token format

`kev/model.py:10` reuses rarely used Qwen special tokens as delimiters, so no embedding rows are added:

| Role | Token |
|---|---|
| `<state>` | `<\|fim_prefix\|>` |
| `<q>` | `<\|fim_middle\|>` |
| `<opt>` | `<\|box_start\|>` |
| `</opt>` | `<\|box_end\|>` |
| `<decide>` | `<\|fim_suffix\|>` |

- A record is packed as `<state> …state… <q> instructions <opt> o1 </opt> … <decide>` for each question (`encode`, `kev/model.py:50`).
- Branch position ids restart just after the state.
- Caller text goes through `user_tokens`, which rewrites `<|name|>` to `<¦name¦>` so user input can never forge a delimiter (`kev/model.py:41`).

**Consequence for a shared prefill:** the state prefix is `<|fim_prefix|>` plus the state text. It is not a chat-template prompt. Text generation that continues from that cache sees a FIM-style opening rather than its usual format. Whether that hurts the post-trained model's text is a question the Write milestone has to measure (see [design/architecture.md](../design/architecture.md)).

## The pointer head

`PointerHead` (`kev/model.py:150`) has two linear projections, d→256, for `<decide>` and each `</opt>`. The logit is a scaled dot product, divided by the checkpoint's fitted temperature at inference. Probabilities are a softmax over a question's options. `noul` is a two-option choice, and `score` is a choice over ordered levels.

## Context limits

- Training: 384 state tokens, 1,024 per branch, 2,048 packed (`MAX_STATE`, `MAX_BRANCH`, `MAX_PACKED`).
- Serving: allows 8,192 for state plus branch. Longer contexts than training are untested.
- A cleaned Mind2Web page averages about 580 elements (see [research/benchmarks.md](../research/benchmarks.md)), well past 384 tokens. Long states are therefore out of Kev's training distribution, which is one reason the Act milestone expects a fine-tune.

## Training entry points

- `kev.train`: `--suite` or `--data` JSONL, plus `--init_from` to warm-start. It checks base, revision, LoRA rank and head size against the checkpoint before loading.
- `kev.benchmark`: fp32. Reports accuracy, Brier, calibration, coverage at a 5% error budget, option-order changes and isolation. `--remote` scores any System One endpoint.
- `kev.compare`: paired bootstrap between two runs.
- Modal: `modal_app.py::study` / `::pull` / `::locked_test` (one read, ever).
