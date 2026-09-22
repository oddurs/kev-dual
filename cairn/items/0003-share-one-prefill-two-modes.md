---
id: 3
key: share
title: 'Share: one prefill, two modes'
type: milestone
status: backlog
created: 2026-09-22
updated: 2026-09-22
priority: p2
due: 2026-10-09
---

Whole-model LoRA changes the state's hidden states, so switching modes means prefilling the page twice. A question-side adapter (off over the state, on from `<q>` onward) makes the state cache shared by both modes.

**Exit:** a question-side Kev-4B within 2 pp of the whole-model adapter on `transfer-v4` dev, and a measured per-step latency for one shared prefill against two — or a recorded fail and a fallback to whole-adapter switching.
