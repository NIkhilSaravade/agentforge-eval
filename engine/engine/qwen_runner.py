"""Our own Qwen2 forward pass (Qwen2.5-Coder 0.5B / 1.5B): RMSNorm, RoPE, GQA, SwiGLU.

Same contract as ModelRunner (GPT-2) so the scheduler, KV pools and API are shared: `spec`,
`tokenizer`, `step_tokens`, `generate_cached`, `last_pad_frac`. transformers supplies the tokenizer
ONLY. Weights are read straight from the safetensors file and upcast bf16 -> fp32 (exactly what
`from_pretrained(dtype=float32)` does), and no HF model object is kept, so RAM holds one copy of the
weights. The math below is ours; every op mirrors the HF reference in order and kernel so greedy
token ids can be compared exactly (docs/03-correctness.md): same F.linear, same fp32 RMSNorm, same
RoPE construction, same sdpa call shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.cache import ContiguousKVCache
from engine.sampling import pick_tokens
from engine.spec import ModelSpec

MODEL_IDS = {
    "qwen2.5-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}
# The model supports 32K positions. CPU prefill cost and per-slot KV reservation make that a poor
# default; the served cap is EngineConfig.max_context (this is only the fallback).
DEFAULT_MAX_CONTEXT = 8192


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2RMSNorm: statistics in fp32, then scale."""
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv, n, d = x.shape
    return x[:, :, None, :, :].expand(b, kv, n_rep, n, d).reshape(b, kv * n_rep, n, d)


def resolve_model_dir(model: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_IDS[model]))


class Qwen2Runner:
    def __init__(self, model: str = "qwen2.5-coder-0.5b", num_threads: int = 4,
                 max_context: int | None = None) -> None:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        torch.set_num_threads(num_threads)
        self.num_threads = num_threads
        path = resolve_model_dir(model)
        c = json.loads((path / "config.json").read_text())
        assert c["model_type"] == "qwen2" and c["hidden_act"] == "silu"
        assert not c.get("use_sliding_window"), "sliding-window attention is not implemented"
        head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        gen = json.loads((path / "generation_config.json").read_text())
        eos = gen["eos_token_id"]
        self.spec = ModelSpec(
            name=model, arch="qwen2", n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"], head_dim=head_dim,
            max_context=min(max_context or DEFAULT_MAX_CONTEXT, c["max_position_embeddings"]),
            eos_ids=tuple(eos if isinstance(eos, list) else [eos]))
        self.eps = c["rms_norm_eps"]
        self.n_rep = self.spec.n_heads // self.spec.n_kv_heads
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.last_pad_frac = 0.0

        self.inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        files = sorted(path.glob("*.safetensors"))
        sd: dict[str, torch.Tensor] = {}
        for f in files:
            sd.update(load_file(str(f)))
        g = lambda k: sd.pop(k).to(torch.float32).contiguous()  # noqa: E731  (pop: free bf16 as we go)
        self.embed = g("model.embed_tokens.weight")
        # Tied checkpoints omit lm_head; the head is then the embedding matrix itself.
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in sd else self.embed
        self.norm = g("model.norm.weight")
        self.layers = []
        for i in range(self.spec.n_layers):
            p = f"model.layers.{i}."
            self.layers.append({n: g(p + n) for n in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")})
        del sd

    # ------------------------------------------------------------------ shared math
    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [B, T] -> cos, sin [B, T, head_dim], built exactly as Qwen2RotaryEmbedding does."""
        inv = self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
        freqs = (inv @ positions[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _attend(self, q, k_all, v_all, mask, is_causal) -> torch.Tensor:
        """q [B,H,T,D]; k_all/v_all [B,KV,Lk,D]. Mirrors transformers' sdpa path: GQA is handed to
        sdpa natively when there is no mask, otherwise K/V are expanded first."""
        if self.n_rep > 1 and mask is None:
            return F.scaled_dot_product_attention(q, k_all, v_all, is_causal=is_causal, enable_gqa=True)
        k_all, v_all = _repeat_kv(k_all, self.n_rep), _repeat_kv(v_all, self.n_rep)
        return F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask, is_causal=is_causal)

    def _layer(self, x, w, cos, sin, li, kv_io):
        """One decoder layer. `kv_io(k, v) -> (k_all, v_all, mask, is_causal)` writes this step's
        K/V into whichever cache is in use and returns what attention should read."""
        b, t, _ = x.shape
        s = self.spec
        h = _rms_norm(x, w["input_layernorm.weight"], self.eps)
        q = F.linear(h, w["self_attn.q_proj.weight"], w["self_attn.q_proj.bias"])
        k = F.linear(h, w["self_attn.k_proj.weight"], w["self_attn.k_proj.bias"])
        v = F.linear(h, w["self_attn.v_proj.weight"], w["self_attn.v_proj.bias"])
        q = q.view(b, t, -1, s.head_dim).transpose(1, 2)
        k = k.view(b, t, -1, s.head_dim).transpose(1, 2)
        v = v.view(b, t, -1, s.head_dim).transpose(1, 2)
        c, sn = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * c) + (_rotate_half(q) * sn)
        k = (k * c) + (_rotate_half(k) * sn)          # cached K is post-RoPE, as in HF
        k_all, v_all, mask, is_causal = kv_io(li, k, v)
        a = self._attend(q, k_all, v_all, mask, is_causal)
        a = a.transpose(1, 2).contiguous().reshape(b, t, -1)
        x = x + F.linear(a, w["self_attn.o_proj.weight"])
        h = _rms_norm(x, w["post_attention_layernorm.weight"], self.eps)
        h = F.linear(F.silu(F.linear(h, w["mlp.gate_proj.weight"])) * F.linear(h, w["mlp.up_proj.weight"]),
                     w["mlp.down_proj.weight"])
        return x + h

    # ------------------------------------------------------------------ M1: single sequence
    @torch.inference_mode()
    def _forward_single(self, ids: list[int], start: int, cache: ContiguousKVCache) -> torch.Tensor:
        """Logits [vocab] of the last token; K/V for these tokens are written into `cache`."""
        t = len(ids)
        tok = torch.tensor([ids], dtype=torch.long)
        pos = torch.arange(start, start + t, dtype=torch.long).unsqueeze(0)
        cos, sin = self._rope(pos)
        x = F.embedding(tok, self.embed)

        def kv_io(li, k, v):
            cache.write(li, start, k, v)
            k_all, v_all = cache.read(li, start + t)
            return k_all, v_all, None, t > 1     # prefill is causal; one decode token sees everything

        for li, w in enumerate(self.layers):
            x = self._layer(x, w, cos, sin, li, kv_io)
        x = _rms_norm(x[:, -1], self.norm, self.eps)
        return F.linear(x, self.lm_head)[0]

    def generate_cached(self, prompt_token_ids: list[int], max_new_tokens: int,
                        ignore_eos: bool = False) -> list[int]:
        cache = ContiguousKVCache(self.spec.max_context, self.spec)
        out: list[int] = []
        logits = self._forward_single(prompt_token_ids, 0, cache)
        while True:
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
            seq_len = len(prompt_token_ids) + len(out)
            if len(out) >= max_new_tokens or seq_len >= self.spec.max_context:
                break
            if nxt in self.spec.eos_ids and not ignore_eos:
                break
            logits = self._forward_single([nxt], seq_len - 1, cache)
        return out

    # ------------------------------------------------------------------ M2+: batched, pooled
    @torch.inference_mode()
    def step_tokens(self, token_lists: list[list[int]], starts: list[int], pool,
                    block_tables: list[list[int]], samplers=None) -> list[int]:
        """One forward step for a batch; row i feeds token_lists[i] at positions starts[i]..
        Returns the next token per row: greedy unless that row has a Sampler. Same contract as
        ModelRunner.step_tokens."""
        ns = [len(t) for t in token_lists]
        acc = pool.access(block_tables, starts, ns)
        b, t = acc.b, acc.t
        ids = torch.zeros(b, t, dtype=torch.long)          # pad id 0; never attended
        for i, toks in enumerate(token_lists):
            ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        cos, sin = self._rope(acc.positions())
        x = F.embedding(ids, self.embed)

        def kv_io(li, k, v):
            acc.write(li, k, v)
            k_all, v_all = acc.gather(li)
            return k_all, v_all, acc.mask, acc.is_causal

        for li, w in enumerate(self.layers):
            x = self._layer(x, w, cos, sin, li, kv_io)
        last = x[torch.arange(b), torch.tensor(ns) - 1]    # last REAL token of each row
        last = _rms_norm(last, self.norm, self.eps)
        self.last_pad_frac = 1.0 - sum(acc.totals) / (b * acc.lk)
        return pick_tokens(F.linear(last, self.lm_head), samplers)

    # ------------------------------------------------------------------ M0
    def run(self, req, on_token=None):
        # The naive baseline is defined as "HF forward, recompute everything, no cache" and exists
        # only for the GPT-2 ablation. There is no Qwen2 M0 to compare against.
        raise NotImplementedError("backend='naive' (M0) is GPT-2 only")
