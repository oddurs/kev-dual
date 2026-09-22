"""TorchDualModel on Kev-4B in fp32 (run with `uv run pytest -m weights`; set KEV_DUAL_TORCH_DEVICE, default mps).

Text mode against plain transformers generation, and decide mode against the MLX DualModel on the parity records. The
whole-split check against Kev's own fp32 benchmark is scripts/torch_parity.py (too slow for a test on a Mac).
"""

import gc
import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.weights

from kev.checkpoint import Checkpoint  # noqa: E402
from kev.model import load_tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from kev_dual.suites import records  # noqa: E402
from kev_dual.torch_model import TorchDualModel  # noqa: E402

RUN = "jaredpalmer/kev-4b"
DEVICE = os.environ.get("KEV_DUAL_TORCH_DEVICE", "mps")
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Task: book a flight from Zurich to London on Friday.\nFirst step:",
    "Q: What is 17 + 25?\nA:",
    "The three primary colours are",
]
MAX_TOKENS = 32


def free():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def recs():
    return records(n=40)


@pytest.fixture(scope="module")
def mlx_probs(recs):
    """Decide mode from the MLX DualModel (bf16), released before the torch model loads."""
    pytest.importorskip("mlx.core")
    from kev_dual.model import DualModel

    m = DualModel(RUN)
    out = [m.probs(r) for r in recs]
    del m
    free()
    return out


@pytest.fixture(scope="module")
def base_text():
    """Greedy continuations from plain transformers on the base, no adapter code involved."""
    meta = Checkpoint(RUN).meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    lm = AutoModelForCausalLM.from_pretrained(
        meta.base, revision=meta.base_revision, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE)
    out = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").to(DEVICE)
            gen = lm.generate(**ids, max_new_tokens=MAX_TOKENS, do_sample=False)
            out.append(tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))
    del lm
    free()
    return out


@pytest.fixture(scope="module")
def dual(mlx_probs, base_text):
    return TorchDualModel(RUN, device=DEVICE, dtype=torch.float32)


def test_text_matches_plain_transformers(dual, base_text):
    assert [dual.generate(p, max_tokens=MAX_TOKENS) for p in PROMPTS] == base_text


def test_decide_agrees_with_mlx_to_bf16_rounding(dual, recs, mlx_probs):
    """Kev's own MLX-vs-fp32 parity on Kev-4B was max |dp| 0.024; hold this pair to 0.03. The argmax may flip only
    where fp32's top two options are within that rounding bound: record 1 (MMLU, fp32 0.3216 vs 0.3167) flips under
    both bf16 paths, this one and Kev's merged MLX, so it is rounding, not a difference between the implementations."""
    bound = 0.03
    got = [dual.probs(r) for r in recs]
    pairs = [(p, t) for g, w in zip(got, mlx_probs, strict=True) for p, t in zip(g, w, strict=True)]
    assert max(float((p - t).abs().max()) for p, t in pairs) <= bound
    for p, t in pairs:
        top2 = p.topk(2).values
        if float(top2[0] - top2[1]) > bound:
            assert p.argmax() == t.argmax()


def test_switch_round_trip(dual, recs):
    before = [dual.probs(r) for r in recs[:3]]
    dual.generate(PROMPTS[0], max_tokens=4)
    after = [dual.probs(r) for r in recs[:3]]
    for g, w in zip(after, before, strict=True):
        for p, t in zip(g, w, strict=True):
            assert torch.equal(p, t)
