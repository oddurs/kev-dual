"""Kev's frozen evaluation suites, read from a local Kev checkout (the pip package ships only the code)."""

import os
from pathlib import Path

from kev.data import materialize
from kev.suite import read_jsonl

KEV_REPO = Path(os.environ.get("KEV_REPO", "~/Code/jevv/kev")).expanduser()


def records(suite="evals/v4/transfer-v4", split="development", n=None):
    """Materialized records of one split. With `n`, an evenly spaced sample so every source is represented."""
    path = KEV_REPO / suite / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set KEV_REPO to a Kev checkout at the pinned commit")
    recs = [materialize(r) for r in read_jsonl(path)]
    if n is not None and n < len(recs):
        step = len(recs) / n
        recs = [recs[int(i * step)] for i in range(n)]
    return recs
