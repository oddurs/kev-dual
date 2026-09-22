---
id: 24
title: Freeze the Mind2Web eval split and step format
type: feature
status: backlog
milestone: act
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: m
area: data
---

## Problem

The agent loop needs a fixed, reproducible slice of Multimodal-Mind2Web and a step format that turns a step into a System One request.

## Proposal

Load the dataset from the Hub (docs/research/benchmarks.md has the splits and fields). Use the dataset's pre-ranked candidate elements (top-K, as MindAct does) so element choice is a Choice question, and cleaned HTML as the state for v1 (pixels are later). Freeze a dev slice and a test slice with a manifest.

## Acceptance criteria

- [ ] manifest with the dataset revision, split, task and step counts, SHA-256
- [ ] a step → request converter with a test on a known step
- [ ] a documented token budget for the state, and the share of steps whose positive element survives the top-K cut

## 2026-09-22

From the HF card (2026-09-22): Multimodal-Mind2Web splits are train 1,009 tasks / 7,775 actions; test_task 177 / 1,339; test_website 142 / 1,019; test_domain 694 / 4,060. test_task has fewer tasks than the paper's Cross-Task (252), with no stated reason, so report against the HF split names. License: the card says OpenRAIL, and the Mind2Web README asks that the unzipped data not be redistributed online. Keep the frozen slices out of git: commit only the manifest and hashes. Candidates: MindAct's DeBERTa ranker, top-50 (recall@50 of 85–89%). Check whether the HF rows carry ranker scores or only pos/neg candidates. If only pos/neg, the top-K has to be rebuilt with the published ranker.
