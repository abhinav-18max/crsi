"""The evaluator: run a harness version on tasks and score it.

A harness is one Python file that defines

    run_task(task, policy, sandbox) -> dict

The evaluator hands it three objects and owns everything else:

* `task`    - {"instruction": str}
* `policy`  - the frozen task-solving model, with a hard cap on calls
* `sandbox` - a Docker container holding the task files

The turn and command budgets are enforced here, outside the harness, so an
edited harness cannot buy itself more attempts. A full trace of every model
call and every command (including output the harness chose not to show the
model) is recorded for the optimizer to read.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import inspect
import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import LLMError
from .sandbox import CmdResult, DockerSandbox
from .tasks import VERIFY_PRELUDE, Task

# What harness code may import. It gets all side effects through `policy` and `sandbox`.
ALLOWED_IMPORTS = {
    "json", "re", "time", "textwrap", "typing", "dataclasses", "collections", "itertools",
    "functools", "math", "string", "shlex", "difflib", "enum", "copy", "random", "base64",
    "hashlib", "ast", "abc", "operator", "uuid", "__future__",
}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
                   "globals", "locals", "vars", "exit", "quit"}

HARNESS_CONTRACT = """\
A harness is ONE Python file that defines:

    def run_task(task, policy, sandbox) -> dict

* task["instruction"]: the task text (a string). Nothing else is provided.
* policy.ask(system: str, messages: list[dict]) -> str
    Calls the frozen task-solving model. `messages` is a list of
    {{"role": "user"|"assistant", "content": str}}. Returns the reply text.
    policy.calls_left is the number of calls remaining; calling with none left
    raises an exception that ends the rollout, so check it in your loop.
    {reply_limit_note}
* sandbox.run(cmd: str, timeout: int = 10) -> result
    Runs one bash command in /work inside the task container. Every call is a
    fresh shell (cd and exported variables do not persist). The result has
    .stdout, .stderr, .exit_code, .timed_out. `timeout` is capped at {cmd_timeout_cap}
    seconds. sandbox.commands_left is the number of commands remaining.
* sandbox.write_file(path: str, content: str) -> result   (writes text exactly)
* sandbox.read_file(path: str) -> str                      ("" if it cannot be read)
* Return a small JSON-serialisable dict, for example {{"stop": "...", "turns": n}}.

Rules enforced before a harness is allowed to run:
* It may import only: {imports}.
* It may not use open/eval/exec/compile/__import__/input, or any double-underscore
  attribute. All file and process access goes through `sandbox`.
* It must be general. It never sees task names or tests, and it is scored on
  held-out tasks it was not tuned on, so task-specific answers are worthless.

Per-rollout budget (fixed, enforced outside the harness): {max_turns} model calls,
{max_commands} sandbox commands{reply_limit_short}.
"""


@dataclass(frozen=True)
class Budget:
    max_turns: int = 8          # model calls per rollout
    max_commands: int = 40      # sandbox commands per rollout
    cmd_timeout_cap: int = 30   # seconds; upper bound on any single command
    max_reply_chars: int | None = None   # each model reply is cut after this many characters
                                         # (an output-token limit); None means no limit

    def contract(self) -> str:
        if self.max_reply_chars:
            note = (f"Every reply is cut off after {self.max_reply_chars} characters, like an output-token "
                    "limit: anything beyond that never reaches the harness.\n    The limit is available as "
                    "policy.max_reply_chars. The policy does not know about it unless the harness says so.")
            short = f", replies cut at {self.max_reply_chars} characters"
        else:
            note, short = "policy.max_reply_chars is None (replies are not cut).", ""
        return HARNESS_CONTRACT.format(imports=", ".join(sorted(ALLOWED_IMPORTS - {"__future__"})),
                                       reply_limit_note=note, reply_limit_short=short, **asdict(self))


class BudgetExhausted(Exception):
    pass


# --------------------------------------------------------------------------- ports

def _flatten(messages: list[dict]) -> str:
    """Render a chat history as one prompt (the CLI backends take a single prompt)."""
    if len(messages) == 1:
        return str(messages[0].get("content", ""))
    parts = ["The conversation so far:\n"]
    for msg in messages:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        parts.append(f'<message role="{role}">\n{msg.get("content", "")}\n</message>\n')
    parts.append("Write the next assistant message only (no role tag, no commentary about the transcript).")
    return "\n".join(parts)


class PolicyPort:
    def __init__(self, llm, max_calls: int, salt: str, trace: list,
                 max_reply_chars: int | None = None) -> None:
        self._llm, self._max, self._salt, self._trace = llm, max_calls, salt, trace
        self._used = 0
        self.max_reply_chars = max_reply_chars

    @property
    def calls_left(self) -> int:
        return self._max - self._used

    def ask(self, system: str, messages: list[dict]) -> str:
        if self._used >= self._max:
            raise BudgetExhausted("model-call budget used up")
        self._used += 1
        system, prompt = str(system), _flatten(list(messages))
        reply = self._llm.complete(system, prompt, salt=self._salt)
        full_length = len(reply)
        if self.max_reply_chars and full_length > self.max_reply_chars:
            reply = reply[:self.max_reply_chars]
        event = {"type": "model", "turn": self._used, "reply": reply,
                 "cut_from": full_length if full_length > len(reply) else 0,
                 "last_input": str(messages[-1].get("content", "")) if messages else ""}
        if self._used == 1:
            event["system"] = system
        self._trace.append(event)
        return reply


class SandboxPort:
    def __init__(self, sandbox: DockerSandbox, budget: Budget, trace: list) -> None:
        self._sb, self._budget, self._trace = sandbox, budget, trace
        self._used = 0

    @property
    def commands_left(self) -> int:
        return self._budget.max_commands - self._used

    def _exec(self, cmd: str, timeout: int, stdin: str | None, shown_cmd: str) -> CmdResult:
        if self._used >= self._budget.max_commands:
            raise BudgetExhausted("sandbox-command budget used up")
        self._used += 1
        timeout = max(1, min(int(timeout), self._budget.cmd_timeout_cap))
        res = self._sb.run(cmd, timeout=timeout, stdin=stdin)
        self._trace.append({"type": "cmd", "cmd": shown_cmd, "exit": res.exit_code,
                            "timed_out": res.timed_out, "timeout": timeout,
                            "stdout": res.stdout[:3000], "stderr": res.stderr[:3000]})
        return res

    def run(self, cmd: str, timeout: int = 10) -> CmdResult:
        cmd = str(cmd)
        return self._exec(cmd, timeout, None, cmd)

    def write_file(self, path: str, content: str) -> CmdResult:
        path = str(path)
        script = 'mkdir -p "$(dirname -- "$0")" && cat > "$0"'
        return self._exec(f"bash -c '{script}' {_shquote(path)}", 10, str(content),
                          f"[write_file {path} ({len(str(content))} chars)]")

    def read_file(self, path: str) -> str:
        res = self._exec(f"cat -- {_shquote(str(path))}", 10, None, f"[read_file {path}]")
        return res.stdout if res.exit_code == 0 else ""


def _shquote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------- loading and screening

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"harness code may not import {name!r}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    safe = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN_NAMES}
    safe["__import__"] = _guarded_import
    return safe


def load_harness(path: str | Path):
    """Import a harness file as a fresh module with restricted builtins."""
    spec = importlib.util.spec_from_file_location(f"harness_{uuid.uuid4().hex}", str(path))
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = _safe_builtins()
    spec.loader.exec_module(module)
    return module


def screen_harness(path: str | Path) -> tuple[bool, str]:
    """Static checks a candidate must pass before any rollout is spent on it."""
    try:
        source = Path(path).read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        return False, f"does not parse: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative>"]
        else:
            roots = []
        for root in roots:
            if root not in ALLOWED_IMPORTS:
                return False, f"imports {root!r}, which is not allowed"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"uses forbidden name {node.id!r}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            return False, f"uses double-underscore attribute {node.attr!r}"
    try:
        module = load_harness(path)
    except Exception as exc:  # noqa: BLE001 - any import-time failure rejects the candidate
        return False, f"fails on import: {type(exc).__name__}: {exc}"
    fn = getattr(module, "run_task", None)
    if not callable(fn):
        return False, "does not define run_task(task, policy, sandbox)"
    if len(inspect.signature(fn).parameters) != 3:
        return False, "run_task must take exactly (task, policy, sandbox)"
    return True, "ok"


# --------------------------------------------------------------------------- rollouts

@dataclass
class Rollout:
    task: str
    rollout: int
    passed: bool
    stop: str
    turns: int
    commands: int
    error: str
    verify_tail: str
    seconds: float
    trace: list = field(default_factory=list)


def _rollout_key(source: str, task: Task, rollout: int, llm, budget: Budget) -> str:
    blob = json.dumps([source, task.digest, rollout, llm.ident, llm._extra_key(), asdict(budget)])
    return hashlib.sha256(blob.encode()).hexdigest()


def run_rollout(harness_path: str | Path, task: Task, llm, rollout: int, budget: Budget,
                cache_dir: str | Path | None = None) -> Rollout:
    source = Path(harness_path).read_text()
    cache_file = None
    if cache_dir:
        key = _rollout_key(source, task, rollout, llm, budget)
        cache_file = Path(cache_dir) / key[:2] / f"{key}.json"
        if cache_file.exists():
            return Rollout(**json.loads(cache_file.read_text()))

    start = time.time()
    trace: list = []
    info: dict = {}
    error = ""
    sandbox = DockerSandbox(task.files)
    try:
        policy = PolicyPort(llm, budget.max_turns, f"{task.name}#r{rollout}", trace,
                            budget.max_reply_chars)
        port = SandboxPort(sandbox, budget, trace)
        try:
            module = load_harness(harness_path)
            info = module.run_task({"instruction": task.instruction}, policy, port) or {}
            if not isinstance(info, dict):
                info = {"stop": str(info)[:200]}
        except BudgetExhausted as exc:
            info = {"stop": f"budget_exhausted ({exc})"}
        except LLMError:
            raise  # an inference outage is not the harness's fault: stop instead of scoring zero
        except Exception:  # noqa: BLE001 - a crashing harness is a scored failure
            error = traceback.format_exc()[-800:]
            info = {"stop": "harness_crashed"}
        passed, verify_out = sandbox.verify(VERIFY_PRELUDE + task.verify)
    finally:
        sandbox.close()

    result = Rollout(
        task=task.name, rollout=rollout, passed=passed,
        stop=str(info.get("stop", "returned"))[:200],
        turns=policy._used, commands=port._used, error=error,
        verify_tail=verify_out[-700:], seconds=round(time.time() - start, 1), trace=trace)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(asdict(result)))
    return result


@dataclass
class EvalResult:
    label: str
    rollouts: int
    results: list            # list[Rollout]
    seconds: float

    @property
    def per_task(self) -> dict[str, int]:
        """task name -> number of passing rollouts"""
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.task] = counts.get(r.task, 0) + int(r.passed)
        return dict(sorted(counts.items()))

    @property
    def passed(self) -> int:
        return sum(int(r.passed) for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def score(self) -> float:
        """pass@1 on a 0-100 scale: the mean rollout pass rate over tasks."""
        return round(100.0 * self.passed / self.total, 2) if self.total else 0.0

    def failing(self) -> list[str]:
        return [name for name, n in self.per_task.items() if n < self.rollouts]

    def summary(self) -> dict:
        return {"label": self.label, "score": self.score, "passed": self.passed,
                "total": self.total, "per_task": self.per_task, "seconds": self.seconds}


def evaluate(harness_path: str | Path, tasks: list[Task], llm, *, rollouts: int = 2,
             budget: Budget = Budget(), workers: int = 4, cache_dir: str | Path | None = None,
             label: str = "", verbose: bool = True) -> EvalResult:
    """Run `rollouts` attempts of every task and return the scored results."""
    start = time.time()
    jobs = [(task, r) for task in tasks for r in range(rollouts)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda job: run_rollout(harness_path, job[0], llm, job[1], budget, cache_dir), jobs))
    out = EvalResult(label or Path(harness_path).stem, rollouts, results, round(time.time() - start, 1))
    if verbose:
        print(f"  eval {out.label:<34} {out.score:6.2f}  ({out.passed}/{out.total} rollouts, {out.seconds:.0f}s)")
    return out


# --------------------------------------------------------------------------- traces for the optimizer

def _short(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + f" ...[{len(text) - limit} chars cut]... " + text[-(limit - head):]


def render_trace(rollout: Rollout, max_chars: int = 5000) -> str:
    """A compact, readable account of one rollout for the proposer."""
    lines = [f"### task `{rollout.task}` rollout {rollout.rollout}: "
             f"{'PASSED' if rollout.passed else 'FAILED'} "
             f"(stop={rollout.stop}, model calls={rollout.turns}, commands={rollout.commands})"]
    for event in rollout.trace:
        if event["type"] == "model":
            lines.append(f"[model call {event['turn']}] harness sent (tail): "
                         f"{_short(event['last_input'], 500)!r}")
            cut = (f" [REPLY CUT: the model wrote {event['cut_from']} chars, the harness received "
                   f"only the first {len(event['reply'])}]" if event.get("cut_from") else "")
            lines.append(f"[model reply {event['turn']}]{cut} {_short(event['reply'], 700)}")
        else:
            flag = " TIMED OUT" if event["timed_out"] else ""
            lines.append(f"  $ {_short(event['cmd'], 240)}   -> exit {event['exit']}{flag}")
            if event["stdout"].strip():
                lines.append(f"    real stdout: {_short(event['stdout'], 260)!r}")
            if event["stderr"].strip():
                lines.append(f"    real stderr: {_short(event['stderr'], 260)!r}")
    if rollout.error:
        lines.append(f"HARNESS EXCEPTION: {_short(rollout.error, 500)}")
    lines.append(f"hidden verifier said: {_short(rollout.verify_tail, 400)}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        head = max_chars * 3 // 5
        text = text[:head] + "\n...[middle of trace cut]...\n" + text[-(max_chars - head):]
    return text
