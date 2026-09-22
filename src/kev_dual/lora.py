"""Kev's PEFT adapter as switchable low-rank layers on an mlx-lm model.

Kev's MLX backend folds the adapter into the base weights (`kev.mlx_model.merge_lora`), which makes it permanent. Here
each adapted linear keeps the base weight untouched and adds `scale * (x @ A.T) @ B.T` only while the shared `Switch`
is on, so one loaded set of weights serves both the decision model (adapter on) and the base language model (off).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class Switch:
    """Shared by every adapted layer of one model: flipping `on` changes mode for all of them at once."""

    on: bool = True


class SwitchLoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, a: mx.array, b: mx.array, scale: float, switch: Switch):
        super().__init__()
        self.linear = linear
        self.lora_a = a.T  # [in, r]
        self.lora_b = b.T  # [r, out]
        self.scale = scale
        self._switch = switch  # underscore: not a parameter, not saved

    def __call__(self, x):
        y = self.linear(x)
        if not self._switch.on:
            return y
        a, b = self.lora_a, self.lora_b
        z = (x.astype(a.dtype) @ a) @ b
        return y + (self.scale * z).astype(y.dtype)


def _parent(root, path):
    obj = root
    for part in path:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def attach_lora(lm: nn.Module, adapter_dir, switch: Switch, dtype=mx.float32) -> int:
    """Wrap every linear the PEFT adapter in `adapter_dir` targets. `dtype` is the adapter's compute dtype (the base
    weights keep their own). Same key mapping and alpha as `kev.mlx_model.merge_lora`. Returns the layer count."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("trainable_token_indices"):
        raise ValueError("adapters with trained token embeddings are not supported on MLX")
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(tree_flatten(lm.parameters()))
    n = 0
    for name, a in weights.items():
        if not name.endswith(".lora_A.weight"):
            continue
        stem = name[: -len(".lora_A.weight")]
        # peft names the text model `base_model.model.<layers...>`; mlx-lm nests it as `language_model.model.<...>`
        target = stem.replace("base_model.model.", "language_model.model.", 1)
        if target + ".weight" not in params:
            raise ValueError(f"adapter tensor {stem} has no weight in the mlx-lm model (looked for {target}.weight)")
        *path, leaf = target.split(".")
        parent = _parent(lm, path)
        b = weights[stem + ".lora_B.weight"]
        setattr(parent, leaf, SwitchLoRALinear(getattr(parent, leaf), a.astype(dtype), b.astype(dtype), scale, switch))
        n += 1
    mx.eval(lm.parameters())
    return n
