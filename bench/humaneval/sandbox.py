"""An isolated sandbox for executing model-generated Python.

Same `run(cmd, cwd, timeout, extra_env)` shape as harness.local_sandbox.LocalSandbox / DockerSandbox, so it is a
drop-in. Why not those: LocalSandbox has no isolation (its own docstring says so) and DockerSandbox is
Node-specific (node: images, `corepack enable`). Generated code is untrusted, so this one runs each command in
a throwaway container: no network, non-root, all capabilities dropped, read-only root filesystem, capped memory /
pids / CPU / file size. Only `cwd` (a per-sample temp dir) is writable.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

IMAGE = "agentforge-humaneval:py312"


class PythonDockerSandbox:
    def __init__(self, image: str = IMAGE, memory: str = "1g", pids: int = 256, cpus: str = "1") -> None:
        self.image, self.memory, self.pids, self.cpus = image, memory, pids, cpus

    def run(self, cmd: list[str], cwd: Path, timeout: int = 30,
            extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        name = f"he-run-{uuid.uuid4().hex[:12]}"
        docker = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--user", "10001:10001", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--pids-limit", str(self.pids), "--cpus", self.cpus,
            "--ulimit", "fsize=50000000", "--ulimit", "nofile=256",
            "-v", f"{cwd}:{cwd}:rw", "-w", str(cwd),
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1",
        ]
        for k, v in (extra_env or {}).items():
            docker += ["-e", f"{k}={v}"]
        docker += [self.image, *cmd]
        try:
            r = subprocess.run(docker, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            raise TimeoutError(f"{' '.join(cmd)} exceeded {timeout}s in container {name}") from None
        return subprocess.CompletedProcess(cmd, r.returncode, r.stdout, r.stderr)
