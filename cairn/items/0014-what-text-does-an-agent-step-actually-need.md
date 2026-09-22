---
id: 14
title: What text does an agent step actually need?
type: spike
status: planned
milestone: write
created: 2026-09-22
updated: 2026-09-22
priority: p0
effort: s
area: text-eval
---

## Question

"Base checkpoints write poorly" is a claim about essays. An agent step mostly needs short, exact text: a value to type into a field, an option to select, sometimes a one-line plan. What exactly must text mode produce, and how is it scored?

## Why it has to be answered first

The Base-or-Instruct decision in this milestone rests on it. Measuring the wrong kind of text would pick the wrong base.

## What would settle it

Sample Mind2Web TYPE and SELECT steps (see docs/research/benchmarks.md) and classify what the value depends on: copied from the task, derived from it, or free text. Add a small set of short reasoning prompts (next subgoal given the task and history). Settle the metric: exact match after normalisation for values, a rubric or LLM judge for plans.

## Answer
