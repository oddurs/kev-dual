"""DualModel on Kev-4B (downloads the base and adapter once; Apple Silicon; run with `uv run pytest -m weights`).

Decide mode against Kev's own MLX path, text mode against plain mlx-lm on the same base, and the switch between them.
"""

import gc
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.weights

from kev.checkpoint import Checkpoint, LoadOptions, resolve_run  # noqa: E402
from kev.model import load_tokenizer  # noqa: E402
from mlx_lm.generate import generate as mlx_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402

from kev_dual.model import DualModel  # noqa: E402
from kev_dual.suites import records  # noqa: E402

RUN = "jaredpalmer/kev-4b"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Task: book a flight from Zurich to London on Friday.\nFirst step:",
    "Q: What is 17 + 25?\nA:",
    "The three primary colours are",
]
MAX_TOKENS = 32


@pytest.fixture(scope="module")
def recs():
    return records(n=40)


@pytest.fixture(scope="module")
def kev_reference(recs):
    """Kev's MLX path (adapter merged), released before DualModel loads."""
    ck = Checkpoint(RUN)
    tok, ref = ck.load("mps", LoadOptions(backend="mlx"))
    out = [ref.probs(ref.encode(tok, r)) for r in recs]
    del ref
    gc.collect()
    mx.clear_cache()
    return out


@pytest.fixture(scope="module")
def base_text():
    """Greedy continuations from the plain base model, no adapter code involved."""
    meta = Checkpoint(RUN).meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    lm, _ = load_model(Path(resolve_run(f"{meta.base}@{meta.base_revision or ''}")))
    out = [mlx_generate(lm, tok, p, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0)) for p in PROMPTS]
    del lm
    gc.collect()
    mx.clear_cache()
    return out


@pytest.fixture(scope="module")
def dual(kev_reference, base_text):
    return DualModel(RUN)


def dps(got, want):
    return [
        (float((p - t).abs().max()), int(p.argmax() != t.argmax()))
        for g, w in zip(got, want, strict=True)
        for p, t in zip(g, w, strict=True)
    ]


def test_decide_matches_kev(dual, recs, kev_reference):
    d = dps([dual.probs(r) for r in recs], kev_reference)
    assert max(x for x, _ in d) <= 0.02
    assert sum(f for _, f in d) == 0


def test_text_matches_plain_mlx_lm(dual, base_text):
    assert [dual.generate(p, max_tokens=MAX_TOKENS) for p in PROMPTS] == base_text


def test_switch_round_trip(dual, recs):
    before = [dual.probs(r) for r in recs[:5]]
    dual.generate(PROMPTS[0], max_tokens=8)
    after = [dual.probs(r) for r in recs[:5]]
    for g, w in zip(after, before, strict=True):
        for p, t in zip(g, w, strict=True):
            assert (p == t).all()


def test_generate_leaves_the_adapter_on(dual):
    dual.generate(PROMPTS[0], max_tokens=4)
    assert dual.switch.on


def test_one_copy_of_the_weights(dual):
    """The adapted model holds the base parameters once, plus only the low-rank tensors."""
    from mlx.utils import tree_flatten

    names = [n for n, _ in tree_flatten(dual.lm.parameters())]
    lora = [n for n in names if n.endswith((".lora_a", ".lora_b"))]
    assert len(lora) == 2 * dual.adapted_layers
    assert not any(n.endswith(".linear.linear.weight") for n in names)


def test_decide_response_shape(dual):
    res = dual.decide(
        {
            "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
            "model": "kev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges and refunds", "shipping": "Delivery", "billing": "Charges"},
                },
                "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
                "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Angry"]},
            },
        }
    )
    a = res["answers"]
    assert a["department"]["choice"] in {"returns", "shipping", "billing"}
    assert abs(sum(a["department"]["probabilities"].values()) - 1) < 0.02
    assert 0 <= a["escalate"]["noul"] <= 1
    assert 0 <= a["frustration"]["score"] <= 1
