"""Prompt construction, code extraction from a chat reply, and program assembly.

The generation prompt is FIXED and identical for every model (same spirit as bench's fairness contract clause 2):
nothing here is tuned per model. It is recorded verbatim in every results file.
"""
from __future__ import annotations

import re

from humaneval.data import Problem

PROMPT_TEMPLATE = (
    "Complete the following Python function. Reply with the complete function implementation in a single "
    "```python code block and nothing else.\n\n```python\n{prompt}\n```"
)

_FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\r?\n(.*)\Z", re.DOTALL)


def build_user_message(problem: Problem) -> str:
    return PROMPT_TEMPLATE.format(prompt=problem.prompt.strip("\n"))


def extract_code(reply: str, entry_point: str) -> str:
    """Pick the code from a chat reply: the first fenced block that defines the entry point, else the first
    fenced block, else (an unclosed fence, e.g. cut off by max_tokens) the rest of the reply, else the raw text."""
    blocks = _FENCE.findall(reply)
    for b in blocks:
        if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", b, re.MULTILINE):
            return b
    if blocks:
        return blocks[0]
    m = _OPEN_FENCE.search(reply)
    if m:
        return m.group(1)
    return reply


def assemble_program(problem: Problem, code: str) -> tuple[str, str]:
    """(program_source, mode).

    mode "full_function": the reply defines the entry point. Program = the original prompt (keeps its imports
      and any helper functions the tests rely on, e.g. encode_cyclic for decode_cyclic) followed by the reply;
      the reply's later definition overrides the docstring-only stub in the prompt.
    mode "body_only": the reply has no `def <entry_point>`. Treat it as a function body continuing the prompt
      (indenting it if it came back flush-left).
    """
    if re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(problem.entry_point)}\b", code, re.MULTILINE):
        return problem.prompt.rstrip("\n") + "\n\n\n" + code.rstrip("\n") + "\n", "full_function"
    lines = code.rstrip("\n").split("\n")
    if lines and lines[0] and not lines[0][0].isspace():
        lines = ["    " + ln if ln.strip() else ln for ln in lines]
    return problem.prompt.rstrip("\n") + "\n" + "\n".join(lines) + "\n", "body_only"


def assemble_canonical(problem: Problem) -> str:
    """The dataset's own reference solution, for the gold gate."""
    return problem.prompt + problem.canonical_solution


def build_test_file(problem: Problem, program: str) -> str:
    """One module: the program, then the dataset's tests, then a single pytest test that calls check(entry_point).
    Program and tests share one namespace, exactly as in the original human-eval execution, because some tests
    call helpers that the prompt defines (HumanEval/32 `poly`, /33 `sort_third`, /38 `encode_cyclic`, /50
    `encode_shift`). An import/syntax error in the program surfaces as a pytest collection error (a scored fail)."""
    return (
        f"{program}\n\n\n"
        f"{problem.test}\n\n\n"
        "def test_humaneval():\n"
        f"    check({problem.entry_point})\n"
    )
