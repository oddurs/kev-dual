"""DualModel: Kev's typed decisions and the base model's text generation from one loaded set of weights (MLX).

Kev's MLX backend keeps only the backbone and merges the adapter into it. Here the whole mlx-lm model is kept (Qwen3.5
ties its output head to the embeddings, which the adapter does not touch) and the adapter stays unmerged behind a
switch (`kev_dual.lora`): on for `decide`, off for `generate`.
"""

from contextlib import contextmanager

import mlx.core as mx
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import Checkpoint, resolve_run
from kev.mlx_model import MLXDecisionModel
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, load_tokenizer, pad_id
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from .lora import Switch, attach_lora


class DualModel:
    def __init__(self, run="jaredpalmer/kev-4b", adapter_dtype=mx.bfloat16):
        ck = Checkpoint(run)
        meta = ck.meta
        self.run = run
        self.tok = load_tokenizer(meta.base, revision=meta.base_revision)
        self.base_dir = resolve_run(f"{meta.base}@{meta.base_revision or ''}")
        self.decider = MLXDecisionModel(self.base_dir, pad_id(self.tok), head_dim=meta.head_dim)
        self.decider.head.load_state_dict(meta.head)
        self.decider.eval()
        self.decider.head.temperature = meta.temperature
        self.lm = self.decider.lm  # the full model: backbone plus the (tied) output head
        self.switch = Switch()
        self.adapted_layers = attach_lora(self.lm, ck.path, self.switch, dtype=adapter_dtype)

    @contextmanager
    def adapter(self, on: bool):
        was = self.switch.on
        self.switch.on = on
        try:
            yield
        finally:
            self.switch.on = was

    def probs(self, rec):
        """Per-question probability tensors for an internal Kev record (adapter on)."""
        with self.adapter(True):
            enc = self.decider.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            return self.decider.probs(enc)

    def decide(self, request) -> dict:
        """A System One request (dict or `SystemOneRequest`) -> the /v1/systemone response body, as Kev serves it."""
        req = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        rec, meta = to_record(req)
        answers = to_answers([p.tolist() for p in self.probs(rec)], meta)
        return {"model": req.model, "answers": answers, "usage": {"output_tokens": output_tokens(self.tok, answers)}}

    def generate(self, prompt, max_tokens=256, temp=0.0, **kw) -> str:
        """Text from the base model (adapter off). `prompt` is a string or token ids; greedy unless `temp` > 0."""
        with self.adapter(False):
            return mlx_generate(self.lm, self.tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), **kw)
