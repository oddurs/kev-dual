"""kev_dual.lora on a toy module: the switch gives W + (B @ A) * alpha / r when on and exactly W when off."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from safetensors.numpy import save_file  # noqa: E402

from kev_dual.lora import Switch, SwitchLoRALinear, attach_lora  # noqa: E402


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 6, bias=False)


class Model(nn.Module):
    """mlx-lm's nesting: language_model.model.layers[i]."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = [Leaf(), Leaf()]


@pytest.fixture
def adapted(tmp_path):
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 8), dtype=np.float32), rng.standard_normal((6, 4), dtype=np.float32)
    save_file(
        {"base_model.model.layers.1.q_proj.lora_A.weight": a, "base_model.model.layers.1.q_proj.lora_B.weight": b},
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")
    lm, switch = Model(), Switch()
    w = np.array(lm.language_model.model.layers[1].q_proj.weight)
    n = attach_lora(lm, tmp_path, switch)
    return lm, switch, n, w, a, b


def test_wraps_only_targeted_layers(adapted):
    lm, _, n, *_ = adapted
    assert n == 1
    assert isinstance(lm.language_model.model.layers[1].q_proj, SwitchLoRALinear)
    assert isinstance(lm.language_model.model.layers[0].q_proj, nn.Linear)


@pytest.fixture
def cpu():
    """The GPU fp32 matmul is a reduced-precision fast path (~1e-3 relative); exact checks run on the CPU."""
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(mx.gpu)


def test_switch_on_is_peft_arithmetic_and_off_is_the_base(adapted, cpu):
    lm, switch, _, w, a, b = adapted
    layer = lm.language_model.model.layers[1].q_proj
    x = np.random.default_rng(1).standard_normal((3, 8), dtype=np.float32)
    on = np.array(layer(mx.array(x)))
    np.testing.assert_allclose(on, x @ (w + (b @ a) * 2.0).T, rtol=1e-5, atol=1e-5)
    switch.on = False
    np.testing.assert_array_equal(np.array(layer(mx.array(x))), np.array(layer.linear(mx.array(x))))
    switch.on = True
    np.testing.assert_array_equal(np.array(layer(mx.array(x))), on)


def test_rejects_an_adapter_the_model_does_not_have(tmp_path):
    save_file(
        {
            "base_model.model.layers.0.k_proj.lora_A.weight": np.zeros((4, 8), np.float32),
            "base_model.model.layers.0.k_proj.lora_B.weight": np.zeros((6, 4), np.float32),
        },
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="no weight"):
        attach_lora(Model(), tmp_path, Switch())
