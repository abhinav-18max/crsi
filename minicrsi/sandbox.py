"""A throwaway Docker container per rollout.

Everything the agent runs happens inside the container: no network, capped
memory and CPU, and it is deleted afterwards. The hidden verifier is copied in
only after the agent has finished, so neither the model nor the harness can
read it.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE = "python:3.12-slim"
OUTPUT_CAP = 20_000  # characters kept per stream, per command


@dataclass
class CmdResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    seconds: float


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    half = OUTPUT_CAP // 2
    return text[:half] + f"\n...[{len(text) - OUTPUT_CAP} characters dropped]...\n" + text[-half:]


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox:
    def __init__(self, files: dict[str, str], image: str = IMAGE, lifetime: int = 3600) -> None:
        self.image = image
        proc = subprocess.run(
            ["docker", "run", "-d", "--rm", "--network", "none", "--memory", "512m",
             "--cpus", "1", "--pids-limit", "256", "-w", "/work",
             "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1",
             image, "sleep", str(lifetime)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"could not start sandbox: {proc.stderr.strip()[:300]}")
        self.cid = proc.stdout.strip()
        if files:
            self.put_files(files, "/work")

    def put_files(self, files: dict[str, str], dest: str) -> None:
        with tempfile.TemporaryDirectory(prefix="minicrsi-files-") as tmp:
            for rel, content in files.items():
                path = Path(tmp) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            subprocess.run(["docker", "exec", self.cid, "mkdir", "-p", dest],
                           capture_output=True, timeout=60)
            proc = subprocess.run(["docker", "cp", f"{tmp}/.", f"{self.cid}:{dest}/"],
                                  capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"could not copy files into sandbox: {proc.stderr[:300]}")

    def run(self, cmd: str, timeout: int = 10, stdin: str | None = None) -> CmdResult:
        """Run one shell command in /work. Each call is a fresh bash process."""
        start = time.time()
        argv = ["docker", "exec", "-i", "-w", "/work", self.cid,
                "timeout", "-s", "KILL", f"{int(timeout)}s", "bash", "-c", cmd]
        try:
            proc = subprocess.run(argv, input=(stdin or "").encode(), capture_output=True,
                                  timeout=timeout + 20)
            code = proc.returncode
            out = proc.stdout.decode("utf-8", "replace")
            err = proc.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as exc:
            code = 137
            out = (exc.stdout or b"").decode("utf-8", "replace")
            err = (exc.stderr or b"").decode("utf-8", "replace")
        return CmdResult(_clip(out), _clip(err), code, code == 137, round(time.time() - start, 2))

    def verify(self, script: str, timeout: int = 90) -> tuple[bool, str]:
        """Copy the hidden verifier in and run it. Passing means exit code 0."""
        self.put_files({"verify.py": script}, "/opt/hidden")
        res = self.run("python /opt/hidden/verify.py", timeout=timeout)
        return res.exit_code == 0, (res.stdout + res.stderr)[-1500:]

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True, timeout=60)

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
