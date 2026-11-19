"""Model backends.

Inference goes through the locally installed `claude` (Claude Code) or `codex`
command-line tools, so it is billed to those subscriptions and needs no API key.
Each call is a plain completion: tools are switched off and the system prompt is
replaced, so the model behaves like a bare LLM rather than a coding agent.

Every reply is cached on disk, keyed by (backend, model, system, prompt, salt).
Re-running a notebook cell therefore costs nothing; change `salt` to get a fresh
sample for the same prompt.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class LLMError(RuntimeError):
    """The backend failed after all retries."""


class Usage:
    """Thread-safe counters shared by all calls of one backend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.cached = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0

    def add(self, *, cached: bool, input_tokens: int = 0, output_tokens: int = 0,
            cost_usd: float = 0.0, seconds: float = 0.0) -> None:
        with self._lock:
            if cached:
                self.cached += 1
                return
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost_usd
            self.seconds += seconds

    def summary(self) -> dict:
        return {
            "live_calls": self.calls,
            "cache_hits": self.cached,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_equivalent_usd": round(self.cost_usd, 4),
            "model_seconds": round(self.seconds, 1),
        }


class _DiskCache:
    def __init__(self, directory: str | Path | None) -> None:
        self.dir = Path(directory) if directory else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        if not self.dir:
            return None
        path = self._path(key)
        if path.exists():
            return json.loads(path.read_text())["text"]
        return None

    def put(self, key: str, text: str, meta: dict) -> None:
        if not self.dir:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp%d" % threading.get_ident())
        tmp.write_text(json.dumps({"text": text, **meta}))
        tmp.replace(path)


class _Backend:
    """Shared cache, retry and bookkeeping logic."""

    kind = "base"

    def __init__(self, model: str, *, cache_dir: str | Path | None = None,
                 timeout: int = 600, retries: int = 3) -> None:
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.usage = Usage()
        self._cache = _DiskCache(cache_dir)
        self._cwd = tempfile.mkdtemp(prefix="minicrsi-llm-")

    @property
    def ident(self) -> str:
        return f"{self.kind}:{self.model}"

    def _key(self, system: str, prompt: str, salt: str) -> str:
        blob = json.dumps([self.ident, self._extra_key(), system, prompt, salt])
        return hashlib.sha256(blob.encode()).hexdigest()

    def _extra_key(self) -> str:
        return ""

    def _call(self, system: str, prompt: str) -> tuple[str, dict]:
        raise NotImplementedError

    def complete(self, system: str, prompt: str, salt: str = "") -> str:
        """Return the model's reply text for one system + user prompt."""
        key = self._key(system, prompt, salt)
        hit = self._cache.get(key)
        if hit is not None:
            self.usage.add(cached=True)
            return hit
        last_error = ""
        for attempt in range(self.retries):
            start = time.time()
            try:
                text, meta = self._call(system, prompt)
            except (LLMError, subprocess.TimeoutExpired) as exc:
                last_error = str(exc)[:500]
                # Subscription rate limits clear with time; back off harder for them.
                limited = "limit" in last_error.lower() or "overloaded" in last_error.lower()
                time.sleep((60 if limited else 5) * (attempt + 1))
                continue
            self.usage.add(cached=False, seconds=time.time() - start,
                           input_tokens=meta.get("input_tokens", 0),
                           output_tokens=meta.get("output_tokens", 0),
                           cost_usd=meta.get("cost_usd", 0.0))
            self._cache.put(key, text, {"model": self.ident, "salt": salt})
            return text
        raise LLMError(f"{self.ident} failed after {self.retries} attempts: {last_error}")


class ClaudeCLI(_Backend):
    """A Claude model reached through `claude -p` (Claude Code subscription).

    model: an alias such as "haiku", "sonnet" or "opus", or a full model id.
    effort: optional reasoning effort ("low", "medium", "high", ...).
    """

    kind = "claude"

    def __init__(self, model: str = "haiku", *, effort: str | None = None, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.effort = effort

    def _extra_key(self) -> str:
        return self.effort or ""

    def _call(self, system: str, prompt: str) -> tuple[str, dict]:
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--tools", "",                      # no built-in tools: plain completion
            "--system-prompt", "SYSTEM PROMPT\n" + system,
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--no-session-persistence",
            "--disable-slash-commands",
            "--setting-sources", "",            # ignore user/project settings and memory
            "--output-format", "json",
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              timeout=self.timeout, cwd=self._cwd)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise LLMError(f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        if data.get("is_error") or data.get("subtype") != "success":
            raise LLMError(str(data.get("result") or data.get("subtype"))[:400])
        usage = data.get("usage") or {}
        meta = {
            "input_tokens": usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": data.get("total_cost_usd") or 0.0,
        }
        return data.get("result") or "", meta


class CodexCLI(_Backend):
    """A model reached through `codex exec` (Codex subscription).

    Codex has no flag to replace its system prompt, so `system` is sent as the
    first part of the prompt. It runs read-only, with no saved session.
    """

    kind = "codex"

    def _call(self, system: str, prompt: str) -> tuple[str, dict]:
        out_file = Path(self._cwd) / f"reply-{threading.get_ident()}-{time.time_ns()}.txt"
        cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check",
               "--ephemeral", "--color", "never", "-o", str(out_file)]
        if self.model and self.model != "default":
            cmd += ["-m", self.model]
        cmd.append("-")
        full = (
            "Act as a plain text-completion model. Do not run commands or read files. "
            "Reply with exactly the response the instructions below ask for.\n\n"
            f"# Instructions\n{system}\n\n# Input\n{prompt}"
        )
        proc = subprocess.run(cmd, input=full, capture_output=True, text=True,
                              timeout=self.timeout, cwd=self._cwd)
        if proc.returncode != 0 or not out_file.exists():
            raise LLMError(f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        text = out_file.read_text()
        out_file.unlink(missing_ok=True)
        return text, {}


class ScriptedLLM(_Backend):
    """A fake model for testing the plumbing without spending any inference.

    `fn(system, prompt) -> str` produces the reply.
    """

    kind = "scripted"

    def __init__(self, fn, name: str = "scripted") -> None:
        super().__init__(name, cache_dir=None)
        self._fn = fn

    def _call(self, system: str, prompt: str) -> tuple[str, dict]:
        return self._fn(system, prompt), {}
