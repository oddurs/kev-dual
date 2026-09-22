"""Guards that need no weights: the torch decider's option_isolation check, and the suite sampler's `n`."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kev_dual.suites import records  # noqa: E402
from kev_dual.torch_model import _Decider  # noqa: E402


class Backbone(torch.nn.Module):
    def __init__(self, layer_types):
        super().__init__()
        self.config = SimpleNamespace(layer_types=layer_types, hidden_size=8)


TOK = SimpleNamespace(pad_token_id=0)


def test_decider_refuses_option_isolation_on_a_hybrid_backbone():
    with pytest.raises(ValueError, match="option_isolation"):
        _Decider(Backbone(["linear_attention", "full_attention"]), TOK, "cpu", head_dim=4, option_isolation=True)


def test_decider_accepts_option_isolation_on_an_attention_only_backbone():
    d = _Decider(Backbone(["full_attention"]), TOK, "cpu", head_dim=4, option_isolation=True)
    assert d.option_isolation and not d.hybrid


@pytest.mark.parametrize("n", [0, -3])
def test_records_rejects_a_sample_size_below_one(n):
    with pytest.raises(ValueError, match="at least 1"):
        records(n=n)
