"""Real-money accounting for the hosted arms: prices, a persistent ledger, and a hard circuit-breaker.

Prices are USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing (fetched 2026-09-22).
Cost is computed from `response.usage` (input_tokens, output_tokens; thinking tokens are billed as output and are
included in output_tokens), not from LiteLLM's price map, which can lag new models.

The ledger is append-only JSONL shared by EVERY hosted call in this project (pilots and full runs), so the cap is a
project-wide total, not per run. The breaker trips when ledger total + a small in-flight allowance would exceed the cap;
once tripped, queued requests are skipped without calling the API.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

PRICES_USD_PER_MTOK = {           # (input, output)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5-1": (10.0, 50.0),
}
LEDGER = Path(__file__).resolve().parent.parent / "results" / "humaneval" / "spend_ledger.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES_USD_PER_MTOK[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


def ledger_total(path: Path = LEDGER) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in path.read_text().splitlines() if x.strip())


class SpendGuard:
    """Thread-safe. `cap_usd` is the project-wide hard cap. `reserve_usd` is held back for the requests already
    in flight when the breaker is checked (concurrency x worst-case cost of one call)."""

    def __init__(self, cap_usd: float, reserve_usd: float = 0.0, path: Path = LEDGER) -> None:
        self.cap, self.reserve, self.path = cap_usd, reserve_usd, path
        self._lock = threading.Lock()
        self.spent = ledger_total(path)          # everything spent so far, across all runs
        self.tripped = self.spent + reserve_usd >= cap_usd

    def check(self) -> None:
        if self.tripped:
            raise BudgetExceeded(f"spend guard tripped at ${self.spent:.4f} (cap ${self.cap:.2f})")

    def record(self, label: str, model: str, input_tokens: int, output_tokens: int) -> float:
        usd = cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            self.spent += usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"ts": time.time(), "label": label, "model": model,
                                     "input_tokens": input_tokens, "output_tokens": output_tokens,
                                     "usd": usd}) + "\n")
            if self.spent + self.reserve >= self.cap:
                self.tripped = True
        return usd
