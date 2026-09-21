"""Golden greedy-decoding fixtures for Qwen2.5-Coder, from the HuggingFace reference (fp32).

Same approach and cases as make_fixtures.py (GPT-2), plus two that matter for an agent workload:
a real bench agent prompt rendered through the chat template, and a ~2K-token prompt (RoPE at
larger positions). Like make_fixtures.py this is a place where model.generate() is allowed.

Usage:  python scripts/make_fixtures_qwen.py qwen2.5-coder-0.5b

PITFALL this script exists to avoid: the checkpoint's generation_config.json says do_sample=True,
temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05. Calling generate(do_sample=False)
alone would still apply the 1.05 repetition penalty, so "greedy" would silently not be greedy and the
engine could never match. Everything is overridden explicitly below and asserted in the fixture.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_fixtures import BASE_TEXT, BATCHES, SINGLES  # noqa: E402

from engine.qwen_runner import resolve_model_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PY = ROOT.parent / "bench" / "agent" / "prompts.py"

PROBLEM = ("`parse()` throws a TypeError when the input schema is nullable and the value is "
           "undefined; it should return a validation error instead. Reproduce with "
           "z.string().nullable().parse(undefined).")

# Extra cases beyond the GPT-2 set: name -> (kind, prompt_len_or_None, max_new_tokens, why)
EXTRA = {
    "long_2000_out20": ("long", 2000, 20, "RoPE and prefill at ~2K positions"),
    "agent_prompt_out64": ("agent", None, 64, "real bench system prompt through the chat template"),
    "chat_ok_eos_out64": ("chat", None, 64, "short chat reply that ends on EOS (<|im_end|>) before max_new_tokens"),
}


def load_bench_prompt() -> str:
    spec = importlib.util.spec_from_file_location("bench_prompts", PROMPTS_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SYSTEM_PROMPT.format(problem_statement=PROBLEM, submit_command=m.SUBMIT_COMMAND)


def main(model_key: str) -> None:
    path = resolve_model_dir(model_key)
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32).eval()
    gen = json.loads((path / "generation_config.json").read_text())
    eos = gen["eos_token_id"]
    pad = gen["pad_token_id"]
    out_dir = ROOT / "tests" / "fixtures" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    base = tok.encode(BASE_TEXT)
    long_base = tok.encode(" ".join([BASE_TEXT] * 8))
    cases: dict[str, tuple[list[int], int, str]] = {}
    for name, (off, plen, n_new, why) in SINGLES.items():
        prompt = base[off:off + plen]
        assert len(prompt) == plen, f"{name}: base text too short ({len(base)} tokens)"
        cases[name] = (prompt, n_new, why)
    for name, (kind, plen, n_new, why) in EXTRA.items():
        if kind == "long":
            assert len(long_base) >= plen
            cases[name] = (long_base[:plen], n_new, why)
        else:
            content = ("Reply with just the word OK and nothing else." if kind == "chat"
                       else load_bench_prompt())
            text = tok.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
            cases[name] = (tok.encode(text), n_new, why)

    for name, (prompt, n_new, why) in cases.items():
        ids = torch.tensor([prompt])
        with torch.inference_mode():
            out = model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0, eos_token_id=eos, pad_token_id=pad)
        expected = out[0, len(prompt):].tolist()
        (out_dir / f"{name}.json").write_text(json.dumps({
            "name": name, "why": why, "model": model_key, "prompt_token_ids": prompt,
            "max_new_tokens": n_new, "expected_token_ids": expected,
            "reference": {"snapshot": path.name, "dtype": "float32",
                          "transformers": transformers.__version__, "torch": torch.__version__,
                          "attn_implementation": model.config._attn_implementation,
                          "do_sample": False, "repetition_penalty": 1.0, "eos_token_id": eos},
        }, indent=1) + "\n")
        print(f"{name}: prompt={len(prompt)} out={len(expected)} "
              f"stopped_on_eos={expected[-1] in (eos if isinstance(eos, list) else [eos])}")

    batches = dict(BATCHES)
    batches["agent_and_short"] = ["agent_prompt_out64", "short_5_out20", "long_2000_out20"]
    (out_dir / "batches.json").write_text(json.dumps(batches, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder-0.5b")
