---
id: 39
title: What state format serves both modes?
type: spike
status: planned
milestone: write
depends_on:
- 14
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: s
area: text-eval
---

## Question

Kev's state prefix is `<|fim_prefix|>` + state text. With a shared prefill, text mode continues from exactly that prefix, not from a chat template. Which state format lets the decision adapter and text mode both work from one cache?

## Why it has to be answered first

The state format is part of the retrain's setup (item 0017) and of question-side training (0021), and it can't change after those runs without redoing them.

## What would settle it

On the text-mode set, compare text quality from (1) Kev's FIM-style prefix, (2) the state wrapped in the chat template inside the state, and (3) a changed delimiter. Use the post-trained model with no adapter. Pick the best format that Kev's encoder can produce; the retrain then trains on it. See docs/design/architecture.md.

## Answer
