# kev-dual

One model that thinks and decides: Kev's typed, calibrated decisions and text generation from one loaded set of Qwen3.5-4B weights, ideally from one prefill of the state. Start with `docs/README.md`; the design is in `docs/design/architecture.md`.

- **Kev is upstream, not ours.** Depend on `jaredpalmer/kev` pinned to a commit; the local clone at `~/Code/jevv/kev` is for reading. Changes Kev would want (question-side placement) go upstream as PRs rather than living in a fork.
- **Experiments are pre-registered.** Write an experiment item's gate before the run starts and never edit it afterwards. Record `outcome`, `cost` and `run` when it finishes, and write the verdict even when it failed. This is Kev's own discipline (its PLAN.md), and the launch depends on numbers people can trust.
- **Numbers come from committed reports.** Accuracy claims use fp32 evaluation. Select on dev, read test once per candidate, and compare with paired bootstraps (`kev.compare`).
- **Facts come from sources, dated.** The ecosystem changes daily. Research answers go in the spike's body; lasting ones also go in `docs/`, with links.
- **No weights or dataset copies in git.** `runs/` is ignored. Mind2Web asks that its data not be redistributed, so commit manifests and hashes only.
- **On a Mac, one training or eval job on the GPU at a time.** Long runs go to Modal.
- **Every change lands through a PR, and GitHub enforces it.** The flow is:
  1. One cairn item per branch (`<type>/<NNNN>-<slug>`), cut from fresh `origin/main`.
  2. Run the gates locally before pushing: `ruff check`, `ruff format --check`, `pytest`, `pytest -m weights` when model code changed, and `cairn check`.
  3. Open a PR titled `<type>: <summary> (cairn NNNN)`.
  4. CI must be green. `main` is protected: PRs only, required checks, linear history.
  5. Squash-merge.

  Never tick a criterion or close an item in a commit whose gates failed.

<!-- cairn:begin -->
## Roadmap and issues

This project tracks its roadmap and issues with `cairn`. Every item is a Markdown file under `cairn/items`, described by the schema in `cairn.toml`.

**Do not create ad-hoc TODO, PLAN or NOTES files.** Create a cairn item instead, so the work appears on the board and in the generated roadmap.

### The loop

1. `cairn next` — what is ready to start. It excludes anything blocked by unfinished dependencies and puts work already in progress first.
2. `cairn claim <ID>` — take it before you start, so no one duplicates the work. `cairn claim --next` picks and claims the top-ranked unclaimed item in one step, and prints its body so you can begin immediately.
3. Do the work. Record what you learn: `cairn set <ID> <field>=<value>` for fields, `cairn note <ID> "<TEXT>"` for anything that needs a sentence — why you chose something, what you tried, what to watch for.
4. `cairn tick <ID> <N>` as each acceptance criterion becomes true — `cairn show <ID> --criteria` lists them numbered. Tick what is true, not what would let you close.
5. `cairn close <ID>` when it is done, or `cairn release <ID>` to hand it back.
6. `cairn check` before you report finished. It must pass.

### Commands

```sh
cairn next --json                 # ready work, ranked
cairn claim --next                # take the next ready item
cairn search <TEXT> --json        # titles, bodies and labels
cairn list --json                 # all open items
cairn list --filter 'blocked=false,priority=p0'
cairn show <ID> --json            # one item, including its body
cairn new "<TITLE>" --type <TYPE> --milestone <MILESTONE>
cairn set <ID> status=<STATUS>    # also labels+=x, or any field below
cairn note <ID> "<TEXT>"          # append reasoning; never replaces
cairn show <ID> --criteria        # acceptance criteria, numbered
cairn tick <ID> <N>               # tick one; --all for every one
cairn close <ID>
cairn check                       # validate; run before finishing
cairn render                      # regenerate ROADMAP.md
```

### Schema

- **Types**: `milestone`, `experiment`, `spike`, `feature`, `bug`, `chore`, `docs`
- **Statuses**: `backlog` (open), `planned` (open), `doing` (active), `running` (active), `blocked` (active), `done` (done), `dropped` (dropped)
- **`due`**: date, YYYY-MM-DD — target date for a milestone
- **`part_of`**: names any items, by id, several allowed — a larger piece of work this belongs to
- **`priority`**: one of p0, p1, p2, p3 — p0 blocks the current milestone
- **`effort`**: one of s, m, l, xl — Rough size, not an estimate
- **`area`**: one of model, serving, training, decide-eval, text-eval, agent, data, infra, research, launch — Subsystem this touches
- **`outcome`**: one of pass, fail, inconclusive — an experiment's result against its pre-registered gate
- **`cost`**: number — GPU spend in USD, recorded when an experiment finishes
- **`run`**: free text — where the results live: runs/<dir> or a Modal study name
- **Milestones**: `load` (due 2026-09-25), `write` (due 2026-09-30), `share` (due 2026-10-09), `act` (due 2026-10-16), `bench` (due 2026-10-19), `launch` (due 2026-10-23), `later`
- **Saved views** (`cairn list --view NAME`): `now`, `next`, `experiments`, `questions`, `triage`

### Rules

1. Before starting work, find or create the item and set it to an active status.
2. Use the fields above rather than inventing new ones; add new fields to `cairn.toml` first.
3. Never hand-edit the generated roadmap file — change items and run `cairn render`.
4. `cairn check` must pass before the work is considered done.

<!-- cairn:end -->
