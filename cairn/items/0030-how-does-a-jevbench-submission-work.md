---
id: 30
title: How does a jevbench submission work?
type: spike
status: backlog
milestone: bench
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: s
area: decide-eval
---

## Question

What does jevbench (github.com/fstandhartinger/jevbench) measure, how is a model scored, and how is an entry submitted? Does it call a System One HTTP endpoint that `kev.serve`-style serving can answer? See docs/research/benchmarks.md.

## Why it has to be answered first

The Bench milestone's exit depends on it, and so does whether the server needs anything beyond Kev's API.

## What would settle it

The repo's README and harness code, and one local run against Kev-4B.

## Answer

## 2026-09-22

From the jevbench repo (2026-09-22): the authors' harness calls `POST {endpoint}/v1/systemone` one request at a time, with a single question id `decision` (adapter: jevbench/adapters/typesafe.py). Kev's API already has this shape. Score v1.3.0 = geometric mean of Intelligence, Calibration, Speed and Cost, 25% each. Self-hosted endpoints get latency x2 + 0.15 s and only public items, so those rows are unranked. Most entrants are run by the authors on their GPUs, and requests go through GitHub issues. Standing (RESULTS-v1.2.md): Jev 1.13.0 74.4, SemIf (Qwen3.5-4B) 73.1, Kev 4B 59.7 (#27). Left open: one local run against Kev-4B, and whether to ask the authors to run kev-dual. docs/research/benchmarks.md has the full detail.
