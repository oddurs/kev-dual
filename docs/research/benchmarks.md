# Benchmarks

*Checked 2026-09-22 against the Mind2Web paper, the Hugging Face dataset card, the jevbench repo and Kev's README.*

## Multimodal-Mind2Web: the agent loop

Sources: [paper, arXiv 2306.06070](https://arxiv.org/pdf/2306.06070), [HF card](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web), [repo](https://github.com/OSU-NLP-Group/Mind2Web).

### Data

- **Overall:** 2,350 tasks, 137 websites, 31 domains. 7.3 actions per task and about 1,135 elements per page on average. HTML cleaning brings a page to about 580 elements while keeping 94.7% recall of the target.
- **HF splits** (these are the ones to report against):

| Split | Tasks | Actions | Paper name |
|---|---|---|---|
| train | 1,009 | 7,775 | Train |
| test_task | 177 | 1,339 | Cross-Task (the paper has 252 tasks; the gap is unexplained) |
| test_website | 142 | 1,019 | Cross-Website (paper: 177) |
| test_domain | 694 | 4,060 | Cross-Domain (paper: 912) |

- **Per-row fields:** `annotation_id`, `website`, `domain`, `subdomain`, `action_uid`, `confirmed_task`, `screenshot`, `raw_html`, `cleaned_html`, `operation` (CLICK/TYPE/SELECT plus a value), `action_reprs`, `target_action_index`, `pos_candidates`, `neg_candidates`.
  - Each candidate has a tag, `backend_node_id`, attributes and a relevance flag.
  - `pos_candidates` can be empty when preprocessing dropped the target.
- **Train screenshots** may be rendered badly; people checked the test splits.
- **License:** the card says OpenRAIL, and the original Mind2Web is CC-BY-4.0 with MIT code. The repo asks **not to redistribute the unzipped data online** (test zips use the password `mind2web`). Only manifests and hashes go in git.

### Candidates

- MindAct stage 1 is a DeBERTa-base cross-encoder (86M) that scores each element against the task and previous actions.
- Recall@50: 88.9% cross-task, 85.3% cross-website, 85.7% cross-domain.
- Stage 2 runs multiple choice over groups of 5 plus "None", repeated until one element is left. SeeAct uses the same ranker, with top 50 in groups of 17.
- kev-dual can ask a single Choice over the top-K (Kev allows up to 255 options) instead of the tournament. The shared-state cost makes this cheap, but a 50-option branch is far outside Kev's training.

### Metrics (paper §4.2)

- **Element accuracy:** the chosen element matches any acceptable element.
- **Operation F1:** token-level F1 on the predicted operation string. For CLICK this equals accuracy; for TYPE and SELECT it also scores the value. **This is where text mode is scored.**
- **Step success rate:** element *and* operation are correct. Each step is scored on its own, given the ground-truth history.
- **Success rate:** every step of the task succeeds.
- Step metrics are macro-averaged over tasks.

### Published numbers to place against (Ele.Acc / Op.F1 / Step SR / SR)

| System | Cross-Task | Cross-Website | Cross-Domain |
|---|---|---|---|
| MindAct Flan-T5-XL (fine-tuned) | 55.1 / 75.7 / 52.0 / 5.2 | 42.0 / 65.2 / 38.9 / 5.1 | 42.1 / 66.5 / 39.6 / 2.9 |
| MindAct GPT-4 (50-task subset, top-10) | 41.6 / 60.6 / 36.2 / 2.0 | 35.8 / 51.1 / 30.1 / 2.0 | 37.1 / 46.5 / 26.4 / 2.0 |
| SeeAct GPT-4V, Choice grounding ([arXiv 2401.01614](https://arxiv.org/html/2401.01614)) | 46.4 / 73.4 / 40.2 / – | 38.0 / 67.8 / 32.4 / – | 42.4 / 69.3 / 36.8 / – |
| SeeAct FLAN-T5-XL re-run | 57.1 / 75.7 / 53.5 / – | 43.8 / 67.7 / 41.1 / – | 41.4 / 65.9 / 38.9 / – |

- JevForge (Qwen3.5-0.8B) reports 0.579 top-1 on its own website-disjoint test. That split is not the paper's, so it isn't directly comparable.
- The paper's numbers were computed on its own splits. The HF test_task split has fewer tasks, so comparisons against the table are indicative, not exact.

## jevbench: decision quality

Source: [github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench).

- **What it measures:** 534 decisions for "Jev-class decision models", including 220 hard ones (111 public, 109 held out).
- **Score (v1.3.0):** the geometric mean of four equally weighted axes.
  - Intelligence: chance-corrected accuracy, weighted hard 30%, easy 14%, standard 28%, judge 28%.
  - Calibration: ECE plus fidelity to the gold distributions on the hard tier. Label-only systems get 0.
  - Speed: 100 − 20·log10(s / 0.1 s), averaged over p50 and p95.
  - Cost: 100 − 30·log10($ per 1k decisions / $0.001).
  - If Intelligence is below 50, the score is multiplied by (I/50)².
- **Harness:** `POST {endpoint}/v1/systemone`, one request at a time, question id `decision` (`jevbench/adapters/typesafe.py`). Kev's API already answers it.
- **Submission:**
  - Requests go through GitHub issues, and most entrants are run by the authors on their own GPUs.
  - Self-hosted endpoints get latency ×2 + 0.15 s, see only public items, and are not ranked.
- **Standing** ([RESULTS-v1.2.md](https://github.com/fstandhartinger/jevbench/blob/main/RESULTS-v1.2.md)):
  1. Jev 1.13.0: 74.4
  2. SemIf (Qwen3.5-4B): 73.1
  3. djev: 73.0
  4. Winnow-12B Q8: 71.2
  5. reflex 4B: 70.3
  - Kev: kev 0.6B #19 at 62.5, kev 4B #27 at 59.7.
  - classifier.dev scores 83.6 but is unranked, because it calls Jev.

## Kev's suites: decision quality, controlled

Frozen under `evals/` in the Kev repo, with manifests and hashes. The large files are mirrored at [jaredpalmer/kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites).

| Suite | Use |
|---|---|
| `decision-v7` | Training data, plus an in-distribution dev set |
| `transfer-v4` | Out of domain: the headline "new sources" column, and the gate for every experiment here |
| `transfer-v9` | Adds 10-way MMLU-Pro, buried states and unknowable items |
| `evals/external/` | SemIf's 144 decisions, scienthoon's 900 tickets, WANLI, the typesafe-v1 102 questions |

Rules carried over from Kev:

- Select on dev only.
- Read test once per candidate.
- Compare runs with record-clustered paired bootstraps (`kev.compare`).
- Report fp32 numbers for accuracy claims.
