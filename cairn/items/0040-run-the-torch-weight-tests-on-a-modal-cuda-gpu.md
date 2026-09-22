---
id: 40
title: Run the torch weight tests on a Modal CUDA GPU
type: chore
status: backlog
milestone: write
depends_on:
- 13
created: 2026-09-22
updated: 2026-09-22
priority: p1
effort: s
area: infra
---

TorchDualModel was verified on MPS in fp32 only (item 0013); this machine has no CUDA device and no Modal token. Before the first training or eval run on Modal depends on it:

- [ ] `uv run modal token new` (needs the user, in a browser)
- [ ] run `KEV_DUAL_TORCH_DEVICE=cuda pytest -m weights tests/test_torch_dual.py` on a Modal GPU
- [ ] run `scripts/torch_parity.py --device cuda` and confirm accuracy 0.797 and Brier 0.2637 on transfer-v4 dev
