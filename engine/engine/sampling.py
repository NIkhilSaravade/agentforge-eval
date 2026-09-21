"""Temperature + top-p sampling with a seedable per-request RNG.

Scope: independent short generations (e.g. HumanEval completions). Greedy decoding is NOT routed through
here: a request with no Sampler takes the exact argmax branch the golden tests pin, so nothing in this
file can change greedy output. `pick_tokens` only calls a Sampler for rows that have one.

Reproducibility contract: the same seed and the same logits give the same token. Each Sampler owns its own
torch.Generator, so one request's draws never depend on which other requests share the batch, and the
generator state travels with the request across preemption-and-recompute.
LIMITATION: the logits themselves can differ in the last bits between batch compositions (padding and
kernel shapes), which can flip a draw that lands exactly on a probability boundary. Tests pin same-seed
reproducibility under identical scheduling; the golden tests pin that greedy is exact across batchings.
"""
from __future__ import annotations

import torch

TOP_K_FAST = 256   # see Sampler.nucleus


class Sampler:
    def __init__(self, temperature: float, top_p: float = 1.0, seed: int | None = None) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0; use no sampler for greedy decoding")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature, self.top_p = float(temperature), float(top_p)
        # An unseeded request still gets a concrete seed, so it can be reported and replayed.
        self.seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)

    def nucleus(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(token ids, renormalised probabilities) of the top-p nucleus, most probable first.

        Exact fast path: if the TOP_K_FAST most probable tokens already hold >= top_p of the mass, the nucleus
        (the smallest prefix reaching top_p) lies inside them, so the result equals the full-vocabulary sort
        while touching 256 elements instead of ~152K (measured: 11.8 ms -> ~1 ms per row per step, which at
        batch 16 was the difference between the sampler costing as much as the model step and being negligible).
        Otherwise (a very flat distribution) fall back to the full sort."""
        x = logits.to(torch.float64) / self.temperature   # float64: the boundary decision must not depend on
        k = min(TOP_K_FAST, x.numel())                    # fp32 rounding (cheap: only the top-k are summed)
        lse = torch.logsumexp(x, dim=-1)
        vals, idx = torch.topk(x, k)
        p = torch.exp(vals - lse)
        cum = torch.cumsum(p, dim=-1)
        if cum[-1] < self.top_p and k < x.numel():
            p, idx = torch.sort(torch.softmax(x, dim=-1), descending=True, stable=True)
            cum = torch.cumsum(p, dim=-1)
        keep = (cum - p) < self.top_p          # smallest prefix reaching top_p; the top token is always kept
        p = p * keep
        return idx, (p / p.sum()).to(torch.float32)

    def sample(self, logits: torch.Tensor) -> int:
        """logits: [vocab] (any float dtype). Returns one token id."""
        if self.top_p < 1.0:
            idx, p = self.nucleus(logits)
            return int(idx[torch.multinomial(p, 1, generator=self.gen)].item())
        probs = torch.softmax(logits.to(torch.float32) / self.temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.gen).item())


def pick_tokens(logits: torch.Tensor, samplers: list[Sampler | None] | None) -> list[int]:
    """logits [B, vocab] -> next token per row. Rows without a sampler are greedy (argmax), and if no
    row has one this is exactly the pre-existing `torch.argmax(logits, dim=-1).tolist()`."""
    if samplers is None or all(s is None for s in samplers):
        return torch.argmax(logits, dim=-1).tolist()
    greedy = torch.argmax(logits, dim=-1).tolist()
    return [g if s is None else s.sample(logits[i]) for i, (g, s) in enumerate(zip(greedy, samplers))]
