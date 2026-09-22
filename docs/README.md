# docs

Research and context for kev-dual. The plan itself lives in cairn (`cairn roadmap`, `ROADMAP.md`); these files hold what the plan rests on.

| File | What it holds |
|---|---|
| [context/project.md](context/project.md) | Why this project, which ideas were considered and dropped, the launch playbook |
| [context/jev-and-kev.md](context/jev-and-kev.md) | Jev, TypeSafe's System One API, and the Kev model family with its numbers |
| [context/kev-internals.md](context/kev-internals.md) | What Kev's code does that matters for a second mode, with file and line references |
| [research/prior-art.md](research/prior-art.md) | Work that already does part of this: aLoRA, Solomon, SemIf, jev-ultrafast, and others |
| [research/benchmarks.md](research/benchmarks.md) | Multimodal-Mind2Web, jevbench and Kev's own suites: splits, metrics, baselines to beat |
| [design/architecture.md](design/architecture.md) | How kev-dual is built: the modes, adapter placement, the shared cache, and the open decisions |

## Rules for these files

- **Facts come from sources, dated.** Every number links to where it came from. The ecosystem is moving daily, so each file says when it was checked.
- **Research answers go here; decisions go in cairn.** A spike's answer is recorded in its item. When the answer will matter for more than one item, it is also written up here, and the item links to it.
- **Pinned references.** Kev is cited at commit `250330b` (the local clone at `~/Code/jevv/kev`); upstream HEAD on 2026-09-22 was `dfe16ca`. Line numbers refer to `250330b`.
- **Origin.** The idea and first prior-art scan came from the notes at `~/Code/notesnake/01-projects/11-kev-fresh-takes/` (`00-overview.md`, `01-kev-that-sees.md`, `04-think-and-decide.md`), written 2026-09-22.
