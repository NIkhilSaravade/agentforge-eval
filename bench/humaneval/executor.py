"""Run one program against one problem's tests in the sandbox and classify the outcome."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.python_adapter import PythonAdapter
from humaneval.data import Problem
from humaneval.program import build_test_file

TEST_ID = "test_solution.py::test_humaneval"
# pytest's own exit codes: 1 = tests failed, 2 = interrupted (incl. collection errors), 4 = usage error
COLLECTION_EXITS = {2, 4}


@dataclass
class ExecResult:
    passed: bool
    outcome: str          # passed | failed | error | collection_error | timeout | no_report
    detail: str           # short error text (first failure line), for triage; never used for scoring
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _detail_from_report(report: dict) -> str:
    for t in report.get("tests", []):
        crash = (t.get("call") or {}).get("crash") or {}
        if crash.get("message"):
            return crash["message"][:300]
        if (t.get("call") or {}).get("longrepr"):
            return str(t["call"]["longrepr"])[-300:]
    for c in report.get("collectors", []):
        if c.get("outcome") == "failed" and c.get("longrepr"):
            return str(c["longrepr"])[-300:]
    return ""


def run_program(sandbox, problem: Problem, program: str, timeout: int = 30,
                work_root: Path | None = None) -> ExecResult:
    """Score = the dataset's own `check(candidate)` passes. Exceptions, assertion failures, syntax errors,
    timeouts and missing functions are all a fail, never an infra error; a container that produced no report at
    all is reported as `no_report` (an infra problem to investigate, and counted separately by callers)."""
    work = Path(tempfile.mkdtemp(prefix="he-", dir=work_root))
    t0 = time.perf_counter()
    try:
        os.chmod(work, 0o777)   # the container user (uid 10001) must be able to write the report
        (work / "test_solution.py").write_text(build_test_file(problem, program))
        cmd = ["pytest", "-p", "no:cacheprovider", "-q", "--json-report",
               "--json-report-file=report.json", "test_solution.py"]
        try:
            proc = sandbox.run(cmd, cwd=work, timeout=timeout)
        except TimeoutError:
            return ExecResult(False, "timeout", f"exceeded {timeout}s", time.perf_counter() - t0)
        report_path = work / "report.json"
        if not report_path.exists():
            if proc.returncode == 137:
                # SIGKILL: the container hit its memory cap (verified with `docker inspect`: OOMKilled=true on
                # the HumanEval/100 infinite-append loops). A model failure the sandbox contained, not infra.
                return ExecResult(False, "resource_killed", "killed by the sandbox memory limit (exit 137)",
                                  time.perf_counter() - t0)
            return ExecResult(False, "no_report", (proc.stderr or proc.stdout)[-300:], time.perf_counter() - t0)
        raw = report_path.read_text()
        report = json.loads(raw)
        results = PythonAdapter(work).parse_results(raw)      # bench's own pytest-report parser
        secs = time.perf_counter() - t0
        if results.get(TEST_ID) is True:
            return ExecResult(True, "passed", "", secs)
        if any(c.get("outcome") == "failed" for c in report.get("collectors", [])) or proc.returncode in COLLECTION_EXITS:
            return ExecResult(False, "collection_error", _detail_from_report(report), secs)
        outcome = "failed" if any((t.get("call") or {}).get("crash", {}).get("message", "").startswith("Assertion")
                                  or "assert" in str((t.get("call") or {}).get("longrepr", "")).lower()
                                  for t in report.get("tests", [])) else "error"
        return ExecResult(False, outcome, _detail_from_report(report), secs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
