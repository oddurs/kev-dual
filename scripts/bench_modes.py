"""Experiment 0012: what each mode costs on this machine, before any optimisation, and what a mode switch costs.

    uv run python scripts/bench_modes.py --out reports/bench-modes.json

For states of 270, 1k and 2k tokens: decide (five three-option questions) on a new state and with the state prefix
cached, for Kev's merged MLX path and for DualModel; 32 generated tokens from DualModel with the adapter off (prefill of
the state included); and decide straight after a generate vs straight after a decide (the switch). Median and p90 of
20 repetitions after one warm-up.
"""

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from kev_dual.model import DualModel
from kev_dual.suites import records

SIZES = (270, 1024, 2048)
QUESTIONS = {
    f"q{i}": {"type": "choice", "instructions": text, "criteria": {"a": None, "b": None, "c": None}}
    for i, text in enumerate(
        [
            "Which option best describes the main topic?",
            "Which option best describes the tone?",
            "Which option is the most likely next step?",
            "Which option is mentioned first?",
            "Which option would a reviewer pick?",
        ]
    )
}
GEN_TOKENS = 32


def stats(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 1), "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 1)}


def state_of(tok, n):
    """Real text from Kev's dev states, cut to exactly n tokens."""
    text = "\n\n".join(json.dumps(r["state"], ensure_ascii=False) for r in records())
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return tok.decode(ids)


def record(state):
    rec, _ = to_record(SystemOneRequest.model_validate({"state": state, "model": "kev", "questions": QUESTIONS}))
    return rec


def decide_cost(model, tok, rec, reps):
    # serving limits: encode's defaults are the 384-token training limits and truncate the state silently
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    assert not enc["state_truncated"]
    _, prefix = model.probs_and_prefix(enc)
    return {
        "tokens": len(enc["ids"]),
        "new_state": stats(lambda: model.probs(enc), reps),
        "cached_state": stats(lambda: model.probs_with_prefix(enc, prefix), reps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    report = {"run": a.run, "machine": f"{chip}, {mem:.0f} GB, macOS {platform.mac_ver()[0]}", "reps": a.reps}

    ck = Checkpoint(a.run)
    tok, kev = ck.load("mps", LoadOptions(backend="mlx"))
    states = {n: state_of(tok, n) for n in SIZES}
    report["kev_merged"] = {n: decide_cost(kev, tok, record(s), a.reps) for n, s in states.items()}
    del kev
    gc.collect()
    mx.clear_cache()

    m = DualModel(a.run)
    eos = mx.array(sorted(set(m.tok.all_special_ids)))

    def no_eos(_, logits):  # force exactly GEN_TOKENS tokens so every run does the same work
        logits[:, eos] = -mx.inf
        return logits

    report["dual_decide"], report["dual_generate"], report["switch"] = {}, {}, {}
    for n, s in states.items():
        rec = record(s)
        report["dual_decide"][n] = decide_cost(m.decider, m.tok, rec, a.reps)

        def gen(state=s):
            with m.adapter(False):
                kw = {"max_tokens": GEN_TOKENS, "sampler": make_sampler(temp=0.0), "logits_processors": [no_eos]}
                out = list(stream_generate(m.lm, m.tok, state, **kw))
            return out[-1]

        last = gen()
        report["dual_generate"][n] = {
            **stats(gen, a.reps),
            "prompt_tokens": last.prompt_tokens,
            "generated_tokens": last.generation_tokens,
            "prompt_tps": round(last.prompt_tps, 1),
            "generation_tps": round(last.generation_tps, 1),
        }

        # the switch: decide timed right after a generate vs right after another decide (both new state)
        after_gen, after_dec = [], []
        for _ in range(a.reps):
            gen()
            t = time.perf_counter()
            m.probs(rec)
            after_gen.append((time.perf_counter() - t) * 1000)
            m.probs(rec)
            t = time.perf_counter()
            m.probs(rec)
            after_dec.append((time.perf_counter() - t) * 1000)
        report["switch"][n] = {
            "decide_after_generate_ms": round(statistics.median(after_gen), 1),
            "decide_after_decide_ms": round(statistics.median(after_dec), 1),
        }
    report["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
