# Jev, System One and Kev

*Checked 2026-09-22 against Kev's README at commit `250330b` and the notesnake notes.*

## Jev and the System One API

**Jev** is TypeSafe AI's "System One" model, in early access since the week of 2026-09-15. It takes a state plus typed questions and returns calibrated probabilities instead of text. API: [docs.typesafe.ai/api](https://docs.typesafe.ai/api).

| Type | Criteria | Answer |
|---|---|---|
| `noul` | optional descriptions for `true` / `false` | `noul`: probability of yes |
| `choice` | 1–255 options, each with a description or `null` | `choice`, `probabilities`, `confidence` |
| `score` | 2–255 ordered levels | `score` (mean level index), `legend`, `probabilities`, `confidence` |

Request: `POST /v1/systemone` with `{state, model, questions: {<id>: {type, instructions, criteria}}}`. The question id is never shown to the model. Questions share the state but can't see each other.

## Kev

[github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0) is an open family of Jev-compatible models. Each checkpoint is a **rank-16 LoRA adapter plus a small pointer head** on a Qwen3.5 *Base* model. It serves the same API, so TypeSafe's Python SDK can point at it.

| Model | Base | Accuracy, trained sources (dev / test) | Accuracy, new sources (dev / test) | Brier, new sources |
|---|---|---|---|---|
| Kev-0.8B | Qwen3.5-0.8B-Base | 0.825 / 0.834 | 0.652 / 0.684 | 0.499 / 0.460 |
| **Kev-4B** | Qwen3.5-4B-Base | 0.872 / 0.871 | **0.797 / 0.837** | 0.299 / 0.255 |
| Kev-9B | Qwen3.5-9B-Base | 0.872 / 0.874 | 0.822 / 0.852 | 0.286 / 0.237 |
| Jev | hosted | 0.845 / – | 0.857 / – | 0.211 / – |

- "New sources" means `transfer-v4`: datasets and policy rule types Kev wasn't trained on. Accuracy is from fp32 evaluation. Test sets are read once per released checkpoint.
- Probabilities are calibrated with a single temperature per checkpoint (about 2.1–2.4) fitted in distribution.
- Known gaps against Jev:
  - Knowledge (MMLU-Pro 0.52 vs 0.84). This comes from the base model.
  - Date arithmetic, which LoRA training on Base checkpoints erodes.
  - Calibration: 0.45–0.57 of decisions can be automated at a 5% error budget, against Jev's 0.70.
- On jevbench (v1.2 results), Kev 4B ranks #27 at 59.7. Jev leads at 74.4 and SemIf (an untrained Qwen3.5-4B readout) is second at 73.1. The score combines intelligence, calibration, speed and cost (see [research/benchmarks.md](../research/benchmarks.md)). The per-axis breakdown for Kev's rows hasn't been read yet, so the reason for its rank is not known.

## Kev on this machine

On 2026-09-22 Kev-4B ran locally on a 48 GB Apple Silicon Mac from `~/Code/jevv/kev`. It answered a three-question request in about 0.3 s after warm-up. Kev's published MLX timings for Kev-4B on an M5 (32 GB), five questions on a ~270-token state: 721 ms for a new state, 136 ms with the state prefix cached. The PyTorch MPS path is 3.3 s / 0.85 s, because MPS has no Gated DeltaNet kernels.

## Training recipe, for retraining

The Kev-4B recipe from Kev's README: `decision-v7` (10,000 examples from ten public datasets, 896 generated policy examples, 1,680 from 60 generated rule structures), 2 epochs, LoRA r=16, lr 5e-5, batch 4, accum 2, bf16, gradient checkpointing, `--p_none_pair 0.25`. That is about 1 h on one H100 via Modal. The released weights also include a second pass on dates and unknowable examples. `--init_from` warm-starts from a released checkpoint: use a smaller lr (2e-5), and keep the base, revision, rank and head size identical.
