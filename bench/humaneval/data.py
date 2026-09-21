"""Load the 164 HumanEval problems from the pinned HF dataset revision.

HumanEval (Chen et al. 2021, MIT license) is public and very likely present in the training data of code models,
including Qwen2.5-Coder. Scores here therefore overstate true out-of-distribution ability; the write-up must say so.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO = "openai/openai_humaneval"
FILENAME = "openai_humaneval/test-00000-of-00001.parquet"
EXPECTED_PROBLEMS = 164


@dataclass(frozen=True)
class Problem:
    task_id: str            # "HumanEval/0"
    prompt: str             # imports + signature + docstring, no body
    canonical_solution: str  # function body only (indented), to be appended to `prompt`
    test: str               # defines `check(candidate)`
    entry_point: str

    @property
    def index(self) -> int:
        return int(self.task_id.split("/")[1])


def load_problems() -> tuple[list[Problem], dict]:
    """Returns (problems, provenance) where provenance pins the dataset revision and file hash."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO, FILENAME, repo_type="dataset"))
    rows = pq.read_table(path).to_pylist()
    if len(rows) != EXPECTED_PROBLEMS:
        raise RuntimeError(f"expected {EXPECTED_PROBLEMS} HumanEval problems, got {len(rows)}")
    problems = sorted((Problem(r["task_id"], r["prompt"], r["canonical_solution"], r["test"],
                               r["entry_point"]) for r in rows), key=lambda p: p.index)
    provenance = {"dataset": REPO, "revision": path.parents[2].name, "file": FILENAME,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_problems": len(problems)}
    return problems, provenance
