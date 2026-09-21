"""Model geometry the cache, allocators and scheduler need, kept out of the model code.

GPT-2 small used to be hard-coded as module constants in cache.py / model_runner.py. A ModelSpec
carries the same numbers per model so a second architecture (Qwen2) can share the scheduler and
KV-cache machinery. GPT2_SPEC reproduces the old constants exactly; nothing about GPT-2 changes.
"""
from __future__ import annotations

from dataclasses import dataclass

FP32_BYTES = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str                      # "gpt2" | "qwen2"
    n_layers: int
    n_heads: int                   # query heads
    n_kv_heads: int                # key/value heads (== n_heads for MHA, fewer for GQA)
    head_dim: int
    max_context: int               # positions the engine will serve (may be below the model's own limit)
    eos_ids: tuple[int, ...]

    @property
    def hidden(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def bytes_per_token(self) -> int:
        """KV bytes for one token across all layers: K and V, fp32. GQA shrinks this by n_heads/n_kv_heads."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * FP32_BYTES


GPT2_SPEC = ModelSpec("gpt2", "gpt2", n_layers=12, n_heads=12, n_kv_heads=12, head_dim=64,
                      max_context=1024, eos_ids=(50256,))
