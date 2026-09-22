# The project

*Written 2026-09-22 from the notesnake notes (`11-kev-fresh-takes`), updated with the prior-art check of the same day.*

## The pitch

One open model with two modes. It writes when a step needs text, and it returns a typed, calibrated decision when that is all the step needs. Both modes run from one loaded set of weights.

## Why it matters

Today's agent stacks pair a decision model with a separate LLM. [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) has Jev choose the operation and target, then calls a separate LLM (`inception/mercury-2.5` over OpenRouter) only to type text. Merging the two removes a model load, a network hop and a second bill. With the adapter kept off the page state, it also removes a second pass over the page: see [design/architecture.md](../design/architecture.md).

## How this idea was chosen

Four directions were weighed on 2026-09-22 (notesnake `00-overview.md`):

| Idea | Novelty on 2026-09-22 | Outcome |
|---|---|---|
| One model that thinks and decides | Open at the time of the notes | **Chosen**. This repo |
| Kev that sees (pixels → UI actions) | Crowded, one gap left: no pixel-only Kev adapter for UI actions evaluated on Multimodal-Mind2Web | `~/Code/kev/kev-vision`; joins this project later (cairn item 0037) |
| Make your own Kev | Crowded ([Luce](https://github.com/scienthoon/luce)) | `~/Code/kev/kev-distill` |
| Kev in the browser | Taken ([kev.js](https://github.com/ai-ecoverse/kev.js), 2026-09-21) | Dropped |

**Correction on 2026-09-22:** the novelty was narrower than the notes assumed. Activated LoRA and Solomon already share a base-model prefix between an adapter-on readout and adapter-off generation (see [research/prior-art.md](../research/prior-art.md)). What is left to claim is a measured result, not an architecture: an open 4B model with general text output, evaluated in an agent loop against the two-model setup, running on a laptop. Cairn item 0019 has the reasoning.

## Known risks

- **Base checkpoints write poorly.** Kev is trained on Qwen3.5 *Base* checkpoints. The fix is either to rebase on the post-trained `Qwen/Qwen3.5-4B` and retrain Kev's adapter, or to train a second adapter for generation. The **Write** milestone measures this before choosing.
- **"Two adapters on one model" may not read as new.** It doesn't, and prior art confirms it. The launch has to rest on the demo and one measured number.
- **The window is short.** The Jev ecosystem was one week old on 2026-09-22, and the awesome-jev lists already tracked 200+ projects. Ship within weeks. Re-run the prior-art scan the day before launch (item 0036).

## Launch playbook

From notesnake `00-overview.md`:

- **A 30-second video** that shows the speed difference side by side.
- **One repeatable number**, e.g. "N× faster per agent step" or "one 4B model, two modes".
- **Open weights and a one-line install** on day one.
- **An honest comparison against Jev and jevbench.** Kev's README sets the bar: every published number is checked against a committed report (`docs/claims.json`, `scripts/verify_claims.py`).
- **Timing:** weeks, not months.

## Ecosystem trackers

[cobanov/awesome-jev](https://github.com/cobanov/awesome-jev), [yibie/awesome-jev](https://github.com/yibie/awesome-jev), [jevbench](https://github.com/fstandhartinger/jevbench).
