"""HumanEval harness: extraction/assembly logic (pure) and the sandboxed executor (needs Docker + the
agentforge-humaneval:py312 image, built from humaneval/Dockerfile; these tests fail loudly if it is missing)."""
from __future__ import annotations

import pytest

from humaneval.data import Problem, load_problems
from humaneval.executor import run_program
from humaneval.program import assemble_canonical, assemble_program, build_user_message, extract_code
from humaneval.sandbox import PythonDockerSandbox

ADD = Problem("HumanEval/x", 'def add(a, b):\n    """Add two numbers."""\n', "    return a + b\n",
              "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n", "add")


# --------------------------------------------------------------------------- extraction / assembly
def test_extract_prefers_the_block_that_defines_the_entry_point():
    reply = "Usage:\n```python\nprint(add(1,2))\n```\nImpl:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "def add" in extract_code(reply, "add")


def test_extract_falls_back_to_first_block_then_unclosed_fence_then_raw():
    assert extract_code("```python\nx = 1\n```", "add") == "x = 1\n"
    assert extract_code("here:\n```python\ndef add(a, b):\n    return a +", "add").startswith("def add")
    assert extract_code("def add(a, b): return a + b", "add") == "def add(a, b): return a + b"


def test_assemble_full_function_keeps_prompt_helpers_and_overrides_the_stub():
    prog, mode = assemble_program(ADD, "def add(a, b):\n    return a + b\n")
    assert mode == "full_function" and prog.startswith("def add(a, b):\n    \"\"\"Add two numbers.")
    ns: dict = {}
    exec(prog, ns)
    assert ns["add"](2, 3) == 5


def test_assemble_body_only_indents_flush_left_bodies():
    prog, mode = assemble_program(ADD, "return a + b")
    assert mode == "body_only"
    ns: dict = {}
    exec(prog, ns)
    assert ns["add"](2, 3) == 5


def test_canonical_assembly_is_prompt_plus_body():
    ns: dict = {}
    exec(assemble_canonical(ADD), ns)
    assert ns["add"](1, 1) == 2


def test_user_message_is_fixed_and_contains_the_prompt():
    m = build_user_message(ADD)
    assert "single ```python code block" in m and "def add(a, b):" in m


def test_dataset_has_164_problems_with_unique_ids():
    problems, prov = load_problems()
    assert len(problems) == 164 == prov["n_problems"] and len({p.task_id for p in problems}) == 164
    assert [p.index for p in problems] == list(range(164))


# --------------------------------------------------------------------------- sandboxed execution
@pytest.fixture(scope="module")
def sb():
    return PythonDockerSandbox()


def test_correct_solution_passes(sb):
    r = run_program(sb, ADD, assemble_program(ADD, "def add(a, b):\n    return a + b\n")[0])
    assert r.passed and r.outcome == "passed"


def test_wrong_solution_fails_not_errors(sb):
    r = run_program(sb, ADD, assemble_program(ADD, "def add(a, b):\n    return a - b\n")[0])
    assert not r.passed and r.outcome in ("failed", "error")


def test_syntax_error_is_a_scored_collection_failure_not_infra(sb):
    r = run_program(sb, ADD, "def add(a, b) _> int:\n    return a + b\n")
    assert not r.passed and r.outcome == "collection_error"


def test_missing_entry_point_fails(sb):
    r = run_program(sb, ADD, assemble_program(ADD, "def other():\n    return 1\n")[0].replace("def add", "def nope", 1))
    assert not r.passed


def test_infinite_loop_is_killed_and_reported_as_timeout(sb):
    r = run_program(sb, ADD, "def add(a, b):\n    while True:\n        pass\n", timeout=4)
    assert not r.passed and r.outcome == "timeout"


def test_module_level_exception_is_a_fail(sb):
    r = run_program(sb, ADD, assemble_program(ADD, "raise RuntimeError('boom')\ndef add(a, b):\n    return a + b\n")[0])
    assert not r.passed


PROBE = Problem("probe", "def f():\n    pass\n", "", (
    "def check(candidate):\n"
    "    import os, socket\n"
    "    try:\n"
    "        socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
    "        raise SystemExit('network reachable')\n"
    "    except OSError:\n"
    "        pass\n"
    "    for path in ('/etc/pwned', '/usr/pwned', '/home/runner/pwned'):\n"
    "        try:\n"
    "            open(path, 'w')\n"
    "            raise SystemExit('read-only root filesystem is writable: ' + path)\n"
    "        except OSError:\n"
    "            pass\n"
    "    assert os.geteuid() != 0\n"
), "f")


def test_sandbox_contains_hostile_code(sb):
    """The probe test PASSES only if the network is blocked, the root filesystem is read-only, and we are not root."""
    r = run_program(sb, PROBE, "def f():\n    pass\n")
    assert r.passed, r.detail


def test_sandbox_memory_bomb_is_contained(sb):
    r = run_program(sb, ADD, "def add(a, b):\n    x = bytearray(4 * 1024**3)\n    return a + b\n", timeout=20)
    assert not r.passed


# --------------------------------------------------------------------------- the gate, on the awkward problems
@pytest.mark.parametrize("idx", [0, 32, 33, 38, 50, 129, 163])
def test_gold_passes_and_empty_fails_on_selected_problems(sb, idx):
    """32/33/38/50 have tests that call helpers defined in the prompt (the bug the full gate caught)."""
    p = load_problems()[0][idx]
    assert run_program(sb, p, assemble_canonical(p)).passed
    assert not run_program(sb, p, assemble_program(p, "")[0]).passed
