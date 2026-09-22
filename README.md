# kev-dual

One model that thinks and decides. A single loaded model that generates text when a step needs it and returns Kev-style typed, calibrated decisions when that is all the step needs.

- **First prototype:** Kev-4B with the adapter toggled or hot-swapped; the base model's LM head stays available.
- **Open question:** Kev uses Base checkpoints, which write poorly. Rebase on Instruct, or train a second adapter for generation.
- **Eval:** agent loop on [Multimodal-Mind2Web](https://osu-nlp-group.github.io/Mind2Web/), and [jevbench](https://github.com/fstandhartinger/jevbench) for decisions.
- **Notes:** `~/Code/notesnake/01-projects/11-kev-fresh-takes/04-think-and-decide.md`
- **Research and context:** [`docs/`](docs/README.md). **Plan:** [`ROADMAP.md`](ROADMAP.md), generated from the [cairn](https://oddurs.github.io/cairn) items under `cairn/items` (`cairn roadmap`, `cairn next`).

## Develop

```sh
uv sync
uv run pytest
```
