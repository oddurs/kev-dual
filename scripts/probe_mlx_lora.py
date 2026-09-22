"""Spike 0010: does Kev's PEFT adapter run unmerged on mlx-lm with a runtime on/off, and what does it cost?

    uv run python scripts/probe_mlx_lora.py --out reports/mlx-lora-probe.json

Reference: Kev's own MLX path (adapter merged into bf16 weights). Candidates: the same base with the adapter as
switchable low-rank layers, computed in fp32 and in bf16. Probabilities on 40 transfer-v4 dev records, and median
latency for a new state and a cached state (prefix hit), adapter on and off.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import torch
from kev.checkpoint import Checkpoint, LoadOptions, resolve_run
from kev.mlx_model import MLXDecisionModel
from kev.model import load_tokenizer, pad_id

from kev_dual.lora import Switch, SwitchLoRALinear, attach_lora
from kev_dual.suites import records


def timed(fn, reps=10):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


def latency(m, tok, rec):
    enc = m.encode(tok, rec)
    _, prefix = m.probs_and_prefix(enc)
    return {
        "tokens": len(enc["ids"]),
        "miss_ms": timed(lambda: m.probs(enc)),
        "hit_ms": timed(lambda: m.probs_with_prefix(enc, prefix)),
    }


def pairs(got, want):
    """(p, t) per question across records, insisting both sides have the same shape."""
    return [(p, t) for g, w in zip(got, want, strict=True) for p, t in zip(g, w, strict=True)]


def compare(got, want):
    pt = pairs(got, want)
    dps = [float((p - t).abs().max()) for p, t in pt]
    flips = sum(int(p.argmax() != t.argmax()) for p, t in pt)
    return {
        "questions": len(dps),
        "max_dp": max(dps),
        "mean_dp": statistics.mean(dps),
        "argmax_flips": flips,
        "over_0.02": sum(d > 0.02 for d in dps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs = records(n=a.n)
    long_rec = max(recs, key=lambda r: len(json.dumps(r["state"])))
    ck = Checkpoint(a.run)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)

    _, ref = ck.load("mps", LoadOptions(backend="mlx"))
    want = [ref.probs(ref.encode(tok, r)) for r in recs]
    report = {
        "run": a.run,
        "records": len(recs),
        "reference": "kev MLX, adapter merged (bf16)",
        "latency": {"merged": latency(ref, tok, long_rec)},
    }
    del ref
    gc.collect()
    mx.clear_cache()

    base_dir = resolve_run(f"{ck.meta.base}@{ck.meta.base_revision or ''}")
    m = MLXDecisionModel(base_dir, pad_id(tok), head_dim=ck.meta.head_dim)
    m.head.load_state_dict(ck.meta.head)
    m.eval()
    m.head.temperature = ck.meta.temperature
    switch = Switch()
    report["adapted_layers"] = attach_lora(m.lm, ck.path, switch, dtype=mx.float32)
    layers = [mod for _, mod in m.lm.named_modules() if isinstance(mod, SwitchLoRALinear)]
    for dtype in (mx.float32, mx.bfloat16):
        for layer in layers:
            layer.lora_a, layer.lora_b = layer.lora_a.astype(dtype), layer.lora_b.astype(dtype)
        mx.eval(m.lm.parameters())
        name = f"unmerged_{str(dtype).removeprefix('mlx.core.')}"
        got = [m.probs(m.encode(tok, r)) for r in recs]
        report[name] = compare(got, want)
        report["latency"][name] = latency(m, tok, long_rec)
    switch.on = False
    report["latency"]["adapter_off"] = latency(m, tok, long_rec)
    switch.on = True
    again = [m.probs(m.encode(tok, r)) for r in recs[:5]]
    report["switch_round_trip_identical"] = all(torch.equal(p, q) for p, q in pairs(again, got[:5]))
    report["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
