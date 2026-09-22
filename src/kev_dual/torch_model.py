"""TorchDualModel: the DualModel interface on PyTorch, for CUDA (Modal) and for the fp32 path Kev's numbers use.

The whole causal LM is loaded, and Kev's adapter is attached with PEFT to its backbone, unmerged: PEFT injects the LoRA
layers in place, so the causal LM's `generate` goes through them too, and `disable_adapter()` turns them off for text.
The decision side is Kev's own `DecisionModel` code over that same backbone.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import Checkpoint
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, DecisionModel, PointerHead, is_hybrid, load_tokenizer, pad_id
from peft import PeftModel
from transformers import AutoModelForCausalLM


class _Decider(DecisionModel):
    """Kev's DecisionModel around a backbone that is already loaded (its own __init__ loads one and drops the head)."""

    def __init__(self, backbone, tok, device, head_dim, option_isolation):
        nn.Module.__init__(self)
        self.lm = backbone
        self.pad_id = pad_id(tok)
        self.hybrid = is_hybrid(backbone.config)
        if self.hybrid and option_isolation:  # the guard DecisionModel.__init__ applies
            raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        self.head = PointerHead(backbone.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)


class TorchDualModel:
    def __init__(self, run="jaredpalmer/kev-4b", device="cuda", dtype=torch.float32, attn=None):
        ck = Checkpoint(run)
        meta = ck.meta
        self.run, self.device = run, device
        self.tok = load_tokenizer(meta.base, revision=meta.base_revision)
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")  # Kev's choice per device
        self.lm = AutoModelForCausalLM.from_pretrained(
            meta.base, revision=meta.base_revision, dtype=dtype, attn_implementation=attn
        ).to(device)
        self.peft = PeftModel.from_pretrained(self.lm.model, ck.path, torch_device=str(device))
        self.decider = _Decider(self.peft, self.tok, device, meta.head_dim, meta.option_isolation)
        self.decider.head.load_state_dict(meta.head)
        self.decider.eval()
        self.lm.eval()
        self.decider.head.temperature = meta.temperature

    @contextmanager
    def adapter(self, on: bool):
        with nullcontext() if on else self.peft.disable_adapter():
            yield

    @torch.no_grad()
    def probs(self, rec):
        enc = self.decider.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        return [p.cpu() for p in self.decider.probs(enc)]

    def decide(self, request) -> dict:
        req = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        rec, meta = to_record(req)
        answers = to_answers([p.tolist() for p in self.probs(rec)], meta)
        return {"model": req.model, "answers": answers, "usage": {"output_tokens": output_tokens(self.tok, answers)}}

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens=256) -> str:
        """Greedy text from the base model (adapter off)."""
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.adapter(False):
            out = self.lm.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
