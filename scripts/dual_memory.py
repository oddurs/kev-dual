"""Memory of one DualModel in each mode, and a text-mode sample.

uv run python scripts/dual_memory.py --out reports/dual-memory.json
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from kev_dual.model import DualModel
from kev_dual.suites import records

GB = 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs = records(n=10)
    m = DualModel(a.run)
    params = tree_flatten(m.lm.parameters())
    lora = sum(v.nbytes for n, v in params if n.endswith((".lora_a", ".lora_b")))
    report = {
        "run": a.run,
        "weights_gb": round(sum(v.nbytes for _, v in params) / GB, 3),
        "adapter_gb": round(lora / GB, 3),
        "active_after_load_gb": round(mx.get_active_memory() / GB, 3),
    }
    mx.reset_peak_memory()
    for r in recs:
        m.probs(r)
    report["peak_decide_gb"] = round(mx.get_peak_memory() / GB, 3)
    mx.reset_peak_memory()
    prompt = "Task: book a flight from Zurich to London on Friday.\nFirst step:"
    report["sample"] = {"prompt": prompt, "text": m.generate(prompt, max_tokens=64)}
    report["peak_generate_gb"] = round(mx.get_peak_memory() / GB, 3)
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
