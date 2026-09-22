"""Item 0013: TorchDualModel's decide mode against Kev's own fp32 benchmark, on the whole transfer-v4 dev split.

    uv run python scripts/torch_parity.py --device mps --out reports/torch-parity.json

Both sides go through Kev's `evaluate_records` with the same records and context: the reference is Kev's
`LocalPredictor` (fp32 torch, adapter merged, as Kev's benchmark scores), the candidate is the same predictor with
TorchDualModel's decider (fp32, adapter unmerged) swapped in. Full per-record outputs go to runs/torch-parity/.
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import load_split, read_manifest

from kev_dual.suites import KEV_REPO
from kev_dual.torch_model import TorchDualModel


def dual_predictor(m, device, context):
    """Kev's LocalPredictor, scoring with TorchDualModel's decider instead of the model it would load."""
    p = LocalPredictor.__new__(LocalPredictor)
    p.run, p.tok, p.model, p.device, p.context = m.run, m.tok, m.decider, device, context
    p.temperature = m.decider.head.temperature
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--suite", default="evals/v4/transfer-v4")
    ap.add_argument("--work", default="runs/torch-parity")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    suite = KEV_REPO / a.suite
    manifest = read_manifest(suite)
    recs = load_split(suite, "development")
    context, heldout = manifest["context"], tuple(manifest["holdout_sources"])
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)

    kev = LocalPredictor(a.run, a.device, LoadOptions(backend="torch", dtype=torch.float32), context=context)
    ref, ref_rows = evaluate_records(recs, kev, work / "kev", heldout_sources=heldout)
    del kev
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()

    m = TorchDualModel(a.run, device=a.device, dtype=torch.float32)
    got, got_rows = evaluate_records(recs, dual_predictor(m, a.device, context), work / "dual", heldout_sources=heldout)

    pairs = list(zip(ref_rows, got_rows, strict=True))
    assert all(r["id"] == s["id"] and r["question"] == s["question"] for r, s in pairs)
    diffs = [max(abs(x - y) for x, y in zip(r["p"], s["p"], strict=True)) for r, s in pairs]
    flips = sum(int(r["p"].index(max(r["p"])) != s["p"].index(max(s["p"]))) for r, s in pairs)
    keys = ("acc", "brier", "ece", "nll", "n")
    report = {
        "run": a.run,
        "device": a.device,
        "suite": a.suite,
        "records": len(recs),
        "kev_merged_fp32": {k: ref["clean"][k] for k in keys},
        "dual_unmerged_fp32": {k: got["clean"][k] for k in keys},
        "questions": len(diffs),
        "max_dp": max(diffs),
        "argmax_flips": flips,
    }
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
